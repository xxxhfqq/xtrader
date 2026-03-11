from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from sampler import EpisodeSpec


class AStockTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data_by_symbol: dict[str, pd.DataFrame],
        sampler: Any,
        market_feature_cols: list[str],
        *,
        window_size: int,
        initial_cash: float,
        buy_fee_rate: float,
        sell_fee_rate: float,
        lot_size: int,
        reward_type: str = "log_return",
        reward_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.data_by_symbol = data_by_symbol
        self.sampler = sampler
        self.market_feature_cols = market_feature_cols
        self.window_size = int(window_size)
        self.initial_cash = float(initial_cash)
        self.buy_fee_rate = float(buy_fee_rate)
        self.sell_fee_rate = float(sell_fee_rate)
        self.lot_size = int(lot_size)
        self.reward_type = str(reward_type)
        self.reward_epsilon = float(reward_epsilon)

        self.market_feature_dim = len(self.market_feature_cols)
        self.account_feature_dim = 7
        self.observation_size = self.window_size * self.market_feature_dim + self.account_feature_dim

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_size,), dtype=np.float32
        )

        self._episode: EpisodeSpec | None = None
        self._symbol: str | None = None
        self._df: pd.DataFrame | None = None
        self._open: np.ndarray | None = None
        self._close: np.ndarray | None = None
        self._trade_days: np.ndarray | None = None
        self._market_features: np.ndarray | None = None

        self._idx = 0
        self._steps_done = 0
        self._cash = 0.0
        self._position = 0
        self._locked = 0
        self._avg_cost = 0.0
        self._current_trade_day: Any = None

    def _init_episode_arrays(self, symbol: str) -> None:
        if symbol not in self.data_by_symbol:
            raise KeyError(f"未知 symbol: {symbol}")
        df = self.data_by_symbol[symbol]
        self._df = df
        self._open = df["open"].astype(float).to_numpy()
        self._close = df["close"].astype(float).to_numpy()
        self._trade_days = pd.to_datetime(df["date"]).dt.date.to_numpy()
        self._market_features = df[self.market_feature_cols].to_numpy(dtype=np.float32)

    def _equity(self, close_price: float) -> float:
        return float(self._cash + self._position * close_price)

    def _max_buyable(self, next_open: float) -> int:
        if next_open <= 0:
            return 0
        per_share_cost = next_open * (1.0 + self.buy_fee_rate)
        raw = int(self._cash // per_share_cost)
        return (raw // self.lot_size) * self.lot_size

    def _max_sellable(self) -> int:
        sellable = max(0, self._position - self._locked)
        return (sellable // self.lot_size) * self.lot_size

    def _build_obs(self) -> np.ndarray:
        assert self._market_features is not None
        assert self._close is not None
        assert self._open is not None

        left = self._idx - self.window_size + 1
        if left < 0:
            raise RuntimeError("观测窗口越界, 请检查 window_size 与采样起点")
        seq = self._market_features[left : self._idx + 1]
        if seq.shape[0] != self.window_size:
            raise RuntimeError("观测窗口长度异常")

        next_idx = min(self._idx + 1, len(self._open) - 1)
        next_open = float(self._open[next_idx])
        close_now = float(self._close[self._idx])
        equity_now = self._equity(close_now)

        account = np.array(
            [
                close_now,
                float(self._cash),
                equity_now,
                float(self._max_buyable(next_open)),
                float(self._max_sellable()),
                float(self._locked),
                float(self._avg_cost),
            ],
            dtype=np.float32,
        )
        obs = np.concatenate([seq.reshape(-1), account], axis=0).astype(np.float32, copy=False)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        spec = self.sampler.sample() if options is None else options.get("episode_spec", self.sampler.sample())
        if not isinstance(spec, EpisodeSpec):
            raise TypeError("sampler.sample() 必须返回 EpisodeSpec")

        self._episode = spec
        self._symbol = spec.symbol
        self._init_episode_arrays(spec.symbol)
        assert self._close is not None

        if spec.start_idx < self.window_size - 1:
            raise ValueError(
                f"{spec.symbol} 起点 {spec.start_idx} 小于 window_size-1({self.window_size - 1}), 无法构造完整窗口"
            )
        if spec.start_idx + spec.rollout_steps >= len(self._close):
            raise ValueError(
                f"{spec.symbol} 采样区间越界: start={spec.start_idx}, steps={spec.rollout_steps}, rows={len(self._close)}"
            )

        self._idx = int(spec.start_idx)
        self._steps_done = 0
        self._cash = float(self.initial_cash)
        self._position = 0
        self._locked = 0
        self._avg_cost = 0.0
        if options and isinstance(options, dict):
            init_state = options.get("initial_state")
            if isinstance(init_state, dict):
                # 推理阶段允许注入账户状态，用于实盘持仓上下文。
                cash = init_state.get("cash")
                position = init_state.get("position")
                locked = init_state.get("locked")
                avg_cost = init_state.get("avg_cost")
                if cash is not None:
                    self._cash = float(cash)
                if position is not None:
                    self._position = max(0, int(position))
                if locked is not None:
                    self._locked = max(0, min(int(locked), self._position))
                if avg_cost is not None:
                    self._avg_cost = max(0.0, float(avg_cost))
                elif self._position > 0:
                    self._avg_cost = float(self._close[self._idx])
        self._current_trade_day = pd.to_datetime(self._df.iloc[self._idx]["date"]).date()  # type: ignore[index]

        obs = self._build_obs()
        info = {"symbol": self._symbol, "start_idx": self._idx}
        return obs, info

    def step(self, action: np.ndarray):
        if self._episode is None:
            raise RuntimeError("环境尚未 reset")
        assert self._open is not None
        assert self._close is not None
        assert self._trade_days is not None

        a = float(np.asarray(action).reshape(-1)[0])
        a = float(np.clip(a, -1.0, 1.0))

        prev_equity = self._equity(float(self._close[self._idx]))
        trade_idx = self._idx + 1
        trade_day = self._trade_days[trade_idx]
        if trade_day != self._current_trade_day:
            # T+1: 跨日后昨日买入仓位解锁
            self._locked = 0
            self._current_trade_day = trade_day

        next_open = float(self._open[trade_idx])
        executed_buy = 0
        executed_sell = 0

        if a > 0:
            max_buy = self._max_buyable(next_open)
            target_buy = int((max_buy * a) // self.lot_size) * self.lot_size
            if target_buy > 0:
                gross = target_buy * next_open
                fee = gross * self.buy_fee_rate
                total_cost = gross + fee
                if total_cost <= self._cash:
                    old_total_cost = self._avg_cost * self._position
                    self._cash -= total_cost
                    self._position += target_buy
                    self._locked += target_buy
                    self._avg_cost = (old_total_cost + gross) / max(1, self._position)
                    executed_buy = target_buy
        elif a < 0:
            max_sell = self._max_sellable()
            target_sell = int((max_sell * abs(a)) // self.lot_size) * self.lot_size
            if target_sell > 0:
                gross = target_sell * next_open
                fee = gross * self.sell_fee_rate
                proceeds = gross - fee
                self._cash += proceeds
                self._position -= target_sell
                if self._position == 0:
                    self._avg_cost = 0.0
                executed_sell = target_sell

        self._idx = trade_idx
        self._steps_done += 1

        now_equity = self._equity(float(self._close[self._idx]))
        if self.reward_type == "log_return":
            prev_safe = max(prev_equity, self.reward_epsilon)
            now_safe = max(now_equity, self.reward_epsilon)
            reward = float(np.log(now_safe / prev_safe))
        elif self.reward_type == "equity_delta":
            reward = float(now_equity - prev_equity)
        else:
            raise ValueError(f"未知 reward_type: {self.reward_type}")

        terminated = self._steps_done >= self._episode.rollout_steps
        if self._idx + 1 >= len(self._close):
            terminated = True
        truncated = False

        obs = self._build_obs()
        info = {
            "symbol": self._symbol,
            "equity": now_equity,
            "cash": float(self._cash),
            "position": int(self._position),
            "locked": int(self._locked),
            "avg_cost": float(self._avg_cost),
            "executed_buy": int(executed_buy),
            "executed_sell": int(executed_sell),
            "idx": int(self._idx),
        }
        return obs, reward, terminated, truncated, info

