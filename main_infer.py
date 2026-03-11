from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from config import load_config
from data_sync import refresh_symbol_data
from env_stock import AStockTradingEnv
from features import load_symbol_frame, load_symbol_frames
from policy import build_market_feature_cols
from sampler import EpisodeSpec, FixedEpisodeSampler, compute_warmup_bars

_MODEL_CACHE_KEY: tuple[str, str, float] | None = None
_MODEL_CACHE_MODEL: PPO | None = None


def _artifact_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["model_dir"]) / "multi_symbol"


def _extract_trade_codes(cfg: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for item in cfg.get("trade_codes", []):
        if isinstance(item, dict):
            code = item.get("code")
            if isinstance(code, str) and code:
                out.append(code)
        elif isinstance(item, str) and item:
            out.append(item)
    return out


def _extract_all_codes(cfg: dict[str, Any]) -> list[str]:
    train_codes = [c for c in cfg.get("train_codes", []) if isinstance(c, str) and c]
    infer_codes = [c for c in cfg.get("infer_codes", []) if isinstance(c, str) and c]
    trade_codes = _extract_trade_codes(cfg)
    return list(dict.fromkeys([*train_codes, *infer_codes, *trade_codes]))


def _compute_latest_infer_index(total_rows: int, sampler_cfg: dict[str, Any]) -> tuple[int, int]:
    # 推理只做最新一步决策，不需要训练 rollout 长度约束。
    warmup = compute_warmup_bars(sampler_cfg)
    infer_window = int(sampler_cfg["infer_window_bars"])
    next_open_guard = int(sampler_cfg["next_open_guard_bars"])
    valid_start = (infer_window - 1) + warmup
    start_idx = total_rows - next_open_guard - 1
    return valid_start, start_idx


def _load_model_cached(cfg: dict[str, Any]) -> PPO:
    global _MODEL_CACHE_KEY
    global _MODEL_CACHE_MODEL

    save_cfg = cfg["save"]
    model_base = _artifact_dir(cfg) / str(save_cfg["best_model_name"])
    model_zip = model_base.with_suffix(".zip")
    if not model_zip.exists():
        raise FileNotFoundError(f"未找到 best model: {model_zip}")

    device = str(cfg["train"]["device"])
    cache_key = (str(model_base), device, float(model_zip.stat().st_mtime))
    if _MODEL_CACHE_MODEL is not None and _MODEL_CACHE_KEY == cache_key:
        return _MODEL_CACHE_MODEL

    _MODEL_CACHE_MODEL = PPO.load(str(model_base), device=device)
    _MODEL_CACHE_KEY = cache_key
    return _MODEL_CACHE_MODEL


def reset_model_cache() -> None:
    global _MODEL_CACHE_KEY
    global _MODEL_CACHE_MODEL
    _MODEL_CACHE_KEY = None
    _MODEL_CACHE_MODEL = None


def model_available(config_path: str | Path = "config.json") -> bool:
    cfg = load_config(config_path)
    save_cfg = cfg["save"]
    model_base = _artifact_dir(cfg) / str(save_cfg["best_model_name"])
    model_zip = model_base.with_suffix(".zip")
    return model_zip.exists()


def _infer_one_signal(
    model: PPO,
    cfg: dict[str, Any],
    symbol: str,
    data_by_symbol: dict[str, Any],
    market_feature_cols: list[str],
    *,
    cash: float | None = None,
    shares: int = 0,
    locked_today: int = 0,
    avg_cost: float | None = None,
) -> float:
    df = data_by_symbol[symbol]
    valid_start, start_idx = _compute_latest_infer_index(len(df), cfg["sampler"])
    if start_idx < valid_start:
        raise ValueError(f"{symbol} 数据不足，无法推理")

    sampler = FixedEpisodeSampler(
        EpisodeSpec(symbol=symbol, start_idx=int(start_idx), rollout_steps=1)
    )
    env = AStockTradingEnv(
        data_by_symbol={symbol: df},
        sampler=sampler,
        market_feature_cols=market_feature_cols,
        window_size=int(cfg["sampler"]["infer_window_bars"]),
        initial_cash=float(cfg["env"]["initial_cash"]),
        buy_fee_rate=float(cfg["env"]["buy_fee_rate"]),
        sell_fee_rate=float(cfg["env"]["sell_fee_rate"]),
        lot_size=int(cfg["env"]["lot_size"]),
        reward_type=str(cfg["env"]["reward_type"]),
        reward_epsilon=float(cfg["env"]["reward_epsilon"]),
    )
    init_state = {
        "cash": float(cfg["env"]["initial_cash"]) if cash is None else float(cash),
        "position": int(max(0, shares)),
        "locked": int(max(0, locked_today)),
    }
    if avg_cost is not None:
        init_state["avg_cost"] = float(avg_cost)

    obs, _ = env.reset(options={"initial_state": init_state})
    action, _ = model.predict(obs, deterministic=True)
    return float(np.asarray(action).reshape(-1)[0])


def signal_to_target_ratio(signal: float) -> float:
    ratio = (float(signal) + 1.0) / 2.0
    return float(np.clip(ratio, 0.0, 1.0))


def infer_signal_for_code(
    code: str,
    *,
    config_path: str | Path = "config.json",
    cash: float | None = None,
    shares: int = 0,
    locked_today: int = 0,
    avg_cost: float | None = None,
    refresh_data: bool = False,
) -> float:
    cfg = load_config(config_path)
    if refresh_data:
        refresh_symbol_data([code], include_history=True, verbose=False)

    indicators = list(cfg["features"]["indicators"])
    market_cols = build_market_feature_cols(indicators)
    df = load_symbol_frame(
        symbol=code,
        data_dir=Path(cfg["paths"]["data_dir"]),
        indicators=indicators,
        prefer_infer=True,
    )
    model = _load_model_cached(cfg)
    return _infer_one_signal(
        model=model,
        cfg=cfg,
        symbol=code,
        data_by_symbol={code: df},
        market_feature_cols=market_cols,
        cash=cash,
        shares=shares,
        locked_today=locked_today,
        avg_cost=avg_cost,
    )


def infer_target_ratio_for_code(
    code: str,
    *,
    config_path: str | Path = "config.json",
    cash: float | None = None,
    shares: int = 0,
    locked_today: int = 0,
    avg_cost: float | None = None,
    refresh_data: bool = False,
) -> float:
    signal = infer_signal_for_code(
        code=code,
        config_path=config_path,
        cash=cash,
        shares=shares,
        locked_today=locked_today,
        avg_cost=avg_cost,
        refresh_data=refresh_data,
    )
    return signal_to_target_ratio(signal)


def run_infer_from_config(config_path: str | Path = "config.json") -> None:
    cfg = load_config(config_path)
    infer_codes = list(cfg["infer_codes"])
    trade_codes = _extract_trade_codes(cfg)
    all_codes = _extract_all_codes(cfg)
    run_full = bool(cfg.get("run_full_infer", True))
    target_codes = all_codes if run_full else (trade_codes or infer_codes)
    if not target_codes:
        raise ValueError("未找到可推理股票（run_full_infer=false 时需至少配置 trade_codes 或 infer_codes）")

    if bool(cfg["train"].get("update_data_before_infer", True)):
        print(f"[INFO] 推理前更新数据, symbols={len(target_codes)}")
        refresh_symbol_data(target_codes, include_history=True, verbose=False)

    indicators = list(cfg["features"]["indicators"])
    market_cols = build_market_feature_cols(indicators)
    data_by_symbol = load_symbol_frames(
        symbols=target_codes,
        data_dir=Path(cfg["paths"]["data_dir"]),
        indicators=indicators,
        prefer_infer=True,
    )

    model = _load_model_cached(cfg)
    signals: list[tuple[str, float]] = []
    print(f"[INFO] 推理模式: {'全标的(train+infer+trade)' if run_full else 'trade_codes优先'}")

    for symbol in target_codes:
        signal = _infer_one_signal(model, cfg, symbol, data_by_symbol, market_cols)
        signals.append((symbol, signal))

    signals.sort(key=lambda x: x[1], reverse=True)
    top_k = int(cfg.get("topK", 5))

    print("\n[INFER] 全量信号（降序）")
    for symbol, sig in signals:
        print(f"- {symbol}: {sig:.6f} (target={signal_to_target_ratio(sig):.4f})")

    print(f"\n[INFER] Top{top_k}")
    for symbol, sig in signals[:top_k]:
        print(f"- {symbol}: {sig:.6f} (target={signal_to_target_ratio(sig):.4f})")

    if trade_codes:
        trade_set = set(trade_codes)
        trade_signals = [(s, v) for s, v in signals if s in trade_set]
        if trade_signals:
            print("\n[INFER] trade_codes 信号")
            for symbol, sig in trade_signals:
                print(f"- {symbol}: {sig:.6f} (target={signal_to_target_ratio(sig):.4f})")


def main() -> None:
    run_infer_from_config("config.json")


if __name__ == "__main__":
    main()

