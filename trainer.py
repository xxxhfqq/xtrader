from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config import load_config
from data_sync import refresh_symbol_data
from env_stock import AStockTradingEnv
from features import load_symbol_frames
from policy import build_market_feature_cols, build_policy_kwargs
from sampler import MultiSymbolSampler
from validator import ModelValidator


def _artifact_dir(cfg: dict[str, Any]) -> Path:
    model_root = Path(cfg["paths"]["model_dir"])
    d = model_root / "multi_symbol"
    d.mkdir(parents=True, exist_ok=True)
    return d


class ValidateSaveBestCallback(BaseCallback):
    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        validator: ModelValidator,
        save_dir: Path,
        rollout_steps: int,
        validate_every_rollouts: int,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.cfg = cfg
        self.validator = validator
        self.save_dir = save_dir
        self.rollout_steps = int(rollout_steps)
        self.eval_interval_steps = int(rollout_steps * validate_every_rollouts)
        self.next_eval_step = int(self.eval_interval_steps)
        self.best_total_pnl = float("-inf")

        save_cfg = cfg["save"]
        self.best_model_name = str(save_cfg["best_model_name"])
        self.best_log_file = self.save_dir / str(save_cfg["best_log_file"])
        self.best_log_file.parent.mkdir(parents=True, exist_ok=True)

    def _append_log(self, payload: dict[str, Any]) -> None:
        with self.best_log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _evaluate_and_maybe_save(self) -> None:
        assert self.model is not None
        metrics = self.validator.evaluate(self.model)
        total_pnl = float(metrics["total_pnl"])
        total_pnl_pct = float(metrics["total_pnl_pct"])
        evaluated_count = int(metrics["evaluated_count"])
        rollout_idx = int(self.num_timesteps // self.rollout_steps)

        per_symbol_rounded = {
            k: {
                "pnl": round(float(v["pnl"]), 2),
                "pnl_pct": round(float(v["pnl_pct"]), 6),
                "final_equity": round(float(v["final_equity"]), 2),
            }
            for k, v in metrics["per_symbol"].items()
        }

        improved = evaluated_count > 0 and total_pnl > self.best_total_pnl
        if improved:
            self.best_total_pnl = total_pnl
            self.model.save(str(self.save_dir / self.best_model_name))

        log_payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "num_timesteps": int(self.num_timesteps),
            "rollout_index": rollout_idx,
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 6),
            "evaluated_count": evaluated_count,
            "skipped": list(metrics["skipped"]),
            "errors": dict(metrics.get("errors", {})),
            "per_symbol": per_symbol_rounded,
            "is_new_best": bool(improved),
        }
        self._append_log(log_payload)

        if self.verbose > 0:
            print(
                f"[VALID] steps={self.num_timesteps}, rollouts={rollout_idx}, "
                f"total_pnl={total_pnl:.2f}, total_pnl_pct={total_pnl_pct:.4%}, "
                f"evaluated={evaluated_count}, new_best={improved}"
            )
            if evaluated_count == 0:
                print("[WARN] 本轮验证全部失败，跳过 best_model 更新")

    def _on_step(self) -> bool:
        while self.num_timesteps >= self.next_eval_step:
            self._evaluate_and_maybe_save()
            self.next_eval_step += self.eval_interval_steps
        return True


def _maybe_compile_policy(model: PPO, cfg: dict[str, Any]) -> None:
    accel = cfg["model"]["acceleration"]
    if not bool(accel.get("use_torch_compile", False)):
        return
    if not hasattr(torch, "compile"):
        print("[WARN] 当前 torch 不支持 torch.compile, 已跳过")
        return

    mode = str(accel.get("torch_compile_mode", "reduce-overhead"))
    fullgraph = bool(accel.get("torch_compile_fullgraph", False))
    dynamic = bool(accel.get("torch_compile_dynamic", False))

    try:
        model.policy.features_extractor = torch.compile(
            model.policy.features_extractor,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        print(f"[INFO] torch.compile 已启用(仅特征提取器): mode={mode}, fullgraph={fullgraph}, dynamic={dynamic}")
    except Exception as exc:
        print(f"[WARN] features_extractor 编译失败, 回退未编译: {exc}")


def train_from_config(config_path: str | Path = "config.json") -> Path:
    cfg = load_config(config_path)

    train_codes = list(cfg["train_codes"])
    infer_codes = list(cfg["infer_codes"])
    all_symbols = list(dict.fromkeys([*train_codes, *infer_codes]))

    if bool(cfg["train"].get("update_data_before_train", True)):
        print(f"[INFO] 训练前更新数据, symbols={len(all_symbols)}")
        refresh_symbol_data(all_symbols, include_history=True, verbose=False)

    data_dir = Path(cfg["paths"]["data_dir"])
    indicators = list(cfg["features"]["indicators"])
    market_cols = build_market_feature_cols(indicators)

    train_data = load_symbol_frames(
        symbols=train_codes,
        data_dir=data_dir,
        indicators=indicators,
        prefer_infer=False,
    )
    infer_data = load_symbol_frames(
        symbols=infer_codes,
        data_dir=data_dir,
        indicators=indicators,
        prefer_infer=True,
    )

    sampler = MultiSymbolSampler(
        data_by_symbol=train_data,
        sampler_cfg=cfg["sampler"],
        seed=int(cfg["train"]["seed"]),
    )

    window_size = int(cfg["sampler"]["infer_window_bars"])
    env_cfg = cfg["env"]

    def make_env():
        return Monitor(
            AStockTradingEnv(
                data_by_symbol=train_data,
                sampler=sampler,
                market_feature_cols=market_cols,
                window_size=window_size,
                initial_cash=float(env_cfg["initial_cash"]),
                buy_fee_rate=float(env_cfg["buy_fee_rate"]),
                sell_fee_rate=float(env_cfg["sell_fee_rate"]),
                lot_size=int(env_cfg["lot_size"]),
                reward_type=str(env_cfg["reward_type"]),
                reward_epsilon=float(env_cfg["reward_epsilon"]),
            )
        )

    vec_env = DummyVecEnv([make_env])
    ppo_cfg = cfg["ppo"]
    policy_kwargs = build_policy_kwargs(cfg, market_feature_dim=len(market_cols))

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=float(ppo_cfg["learning_rate"]),
        n_steps=int(ppo_cfg["n_steps"]),
        batch_size=int(ppo_cfg["batch_size"]),
        n_epochs=int(ppo_cfg["n_epochs"]),
        gamma=float(ppo_cfg["gamma"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        clip_range=float(ppo_cfg["clip_range"]),
        ent_coef=float(ppo_cfg["ent_coef"]),
        policy_kwargs=policy_kwargs,
        device=str(cfg["train"]["device"]),
        verbose=1,
    )
    _maybe_compile_policy(model, cfg)

    save_dir = _artifact_dir(cfg)
    validator = ModelValidator(
        cfg=cfg,
        infer_data_by_symbol=infer_data,
        market_feature_cols=market_cols,
    )
    callback = ValidateSaveBestCallback(
        cfg=cfg,
        validator=validator,
        save_dir=save_dir,
        rollout_steps=int(cfg["sampler"]["rollout_steps"]),
        validate_every_rollouts=int(cfg["train"]["validate_every_rollouts"]),
        verbose=1,
    )

    total_timesteps = int(cfg["train"]["total_timesteps"])
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    # 训练结束保存 last 模型，便于排障。
    last_model_path = save_dir / "last_model"
    model.save(str(last_model_path))
    best_model_path = save_dir / f"{cfg['save']['best_model_name']}.zip"
    print(f"[INFO] 训练完成, last_model={last_model_path}.zip, best_model={best_model_path}")
    return best_model_path

