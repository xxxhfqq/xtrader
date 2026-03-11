from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any


@dataclass(frozen=True)
class EpisodeSpec:
    symbol: str
    start_idx: int
    rollout_steps: int


@dataclass(frozen=True)
class SymbolRange:
    symbol: str
    valid_start: int
    valid_end: int
    total_rows: int


def compute_warmup_bars(sampler_cfg: dict[str, Any]) -> int:
    indicator_map = sampler_cfg["indicator_warmup_bars"]
    max_indicator_warmup = max(indicator_map.values()) if indicator_map else 0
    return max(max_indicator_warmup, int(sampler_cfg["extra_warmup_bars"]))


def compute_valid_range(total_rows: int, sampler_cfg: dict[str, Any]) -> tuple[int, int]:
    warmup = compute_warmup_bars(sampler_cfg)
    infer_window = int(sampler_cfg["infer_window_bars"])
    rollout_steps = int(sampler_cfg["rollout_steps"])
    next_open_guard = int(sampler_cfg["next_open_guard_bars"])

    valid_start = (infer_window - 1) + warmup
    valid_end = total_rows - rollout_steps - next_open_guard - 1
    return valid_start, valid_end


class MultiSymbolSampler:
    def __init__(self, data_by_symbol: dict[str, Any], sampler_cfg: dict[str, Any], seed: int) -> None:
        self.rollout_steps = int(sampler_cfg["rollout_steps"])
        self._rng = random.Random(seed)
        self._ranges: list[SymbolRange] = []
        for symbol, df in data_by_symbol.items():
            valid_start, valid_end = compute_valid_range(len(df), sampler_cfg)
            if valid_end >= valid_start:
                self._ranges.append(
                    SymbolRange(
                        symbol=symbol,
                        valid_start=valid_start,
                        valid_end=valid_end,
                        total_rows=len(df),
                    )
                )
        if not self._ranges:
            raise ValueError("没有任何股票具备可采样区间, 请检查窗口/预热/rollout 配置")

    @property
    def ranges(self) -> list[SymbolRange]:
        return self._ranges

    def sample(self) -> EpisodeSpec:
        symbol_range = self._rng.choice(self._ranges)
        start_idx = self._rng.randint(symbol_range.valid_start, symbol_range.valid_end)
        return EpisodeSpec(
            symbol=symbol_range.symbol,
            start_idx=start_idx,
            rollout_steps=self.rollout_steps,
        )


class FixedEpisodeSampler:
    def __init__(self, spec: EpisodeSpec) -> None:
        self.spec = spec

    def sample(self) -> EpisodeSpec:
        return self.spec

