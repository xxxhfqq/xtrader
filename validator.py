from __future__ import annotations

from typing import Any

import pandas as pd

from env_stock import AStockTradingEnv
from sampler import EpisodeSpec, FixedEpisodeSampler, compute_valid_range


class ModelValidator:
    def __init__(
        self,
        cfg: dict[str, Any],
        infer_data_by_symbol: dict[str, pd.DataFrame],
        market_feature_cols: list[str],
    ) -> None:
        self.cfg = cfg
        self.infer_data_by_symbol = infer_data_by_symbol
        self.market_feature_cols = market_feature_cols

        self.window_size = int(cfg["sampler"]["infer_window_bars"])
        self.initial_cash = float(cfg["env"]["initial_cash"])
        self.buy_fee_rate = float(cfg["env"]["buy_fee_rate"])
        self.sell_fee_rate = float(cfg["env"]["sell_fee_rate"])
        self.lot_size = int(cfg["env"]["lot_size"])
        self.reward_type = str(cfg["env"]["reward_type"])
        self.reward_epsilon = float(cfg["env"]["reward_epsilon"])

    def _evaluate_one(self, model: Any, symbol: str, df: pd.DataFrame) -> dict[str, float]:
        valid_start, _ = compute_valid_range(len(df), self.cfg["sampler"])
        next_open_guard = int(self.cfg["sampler"]["next_open_guard_bars"])
        rollout_steps = len(df) - valid_start - next_open_guard - 1
        if rollout_steps <= 0:
            raise ValueError(f"{symbol} 验证长度不足, rows={len(df)}, valid_start={valid_start}")

        sampler = FixedEpisodeSampler(
            EpisodeSpec(symbol=symbol, start_idx=valid_start, rollout_steps=rollout_steps)
        )
        env = AStockTradingEnv(
            data_by_symbol={symbol: df},
            sampler=sampler,
            market_feature_cols=self.market_feature_cols,
            window_size=self.window_size,
            initial_cash=self.initial_cash,
            buy_fee_rate=self.buy_fee_rate,
            sell_fee_rate=self.sell_fee_rate,
            lot_size=self.lot_size,
            reward_type=self.reward_type,
            reward_epsilon=self.reward_epsilon,
        )

        obs, _ = env.reset()
        done = False
        last_equity = self.initial_cash
        reward_sum = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            last_equity = float(info["equity"])
            done = bool(terminated or truncated)

        pnl = last_equity - self.initial_cash
        pnl_pct = pnl / self.initial_cash if self.initial_cash != 0 else 0.0
        return {
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reward_sum": reward_sum,
            "final_equity": last_equity,
        }

    def evaluate(self, model: Any) -> dict[str, Any]:
        per_symbol: dict[str, dict[str, float]] = {}
        total_pnl = 0.0
        total_equity = 0.0
        skipped: list[str] = []
        errors: dict[str, str] = {}

        for symbol, df in self.infer_data_by_symbol.items():
            try:
                metrics = self._evaluate_one(model, symbol, df)
                per_symbol[symbol] = metrics
                total_pnl += float(metrics["pnl"])
                total_equity += float(metrics["final_equity"])
            except Exception as exc:
                skipped.append(symbol)
                errors[symbol] = str(exc)

        evaluated_count = len(per_symbol)
        total_initial = self.initial_cash * evaluated_count
        total_pnl_pct = (total_pnl / total_initial) if total_initial > 0 else 0.0

        return {
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "total_equity": total_equity,
            "evaluated_count": evaluated_count,
            "skipped": skipped,
            "errors": errors,
            "per_symbol": per_symbol,
        }

