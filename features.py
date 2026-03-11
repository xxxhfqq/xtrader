from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from stockstats import StockDataFrame


RAW_COLUMNS = ["date", "code", "time", "open", "high", "low", "close", "volume"]
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _parse_datetime(df: pd.DataFrame) -> pd.Series:
    if "time" in df.columns:
        dt = pd.to_datetime(df["time"].astype(str), format="%Y%m%d%H%M%S%f", errors="coerce")
        if dt.notna().any():
            return dt
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        if dt.notna().any():
            return dt
    raise RuntimeError("无法从 CSV 解析 datetime, 需要 time 或 date 列")


def resolve_data_path(data_dir: Path, symbol: str, prefer_infer: bool) -> Path:
    infer_path = data_dir / f"{symbol}_infer.csv"
    train_path = data_dir / f"{symbol}.csv"
    if prefer_infer:
        if infer_path.exists():
            return infer_path
        if train_path.exists():
            return train_path
    else:
        if train_path.exists():
            return train_path
        if infer_path.exists():
            return infer_path
    raise FileNotFoundError(f"{symbol} 缺少数据文件: {train_path.name} 或 {infer_path.name}")


def add_indicators(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    tmp = df[OHLCV_COLUMNS].copy()
    ss = StockDataFrame.retype(tmp)
    for ind in indicators:
        _ = ss[ind]

    out = df.copy()
    ss_df = pd.DataFrame(ss)
    missing = [ind for ind in indicators if ind not in ss_df.columns]
    if missing:
        raise KeyError(f"stockstats 未生成指标列: {missing}")
    for ind in indicators:
        out[ind] = ss_df[ind].astype(float)
    return out


def load_symbol_frame(
    symbol: str,
    data_dir: Path,
    indicators: list[str],
    prefer_infer: bool,
) -> pd.DataFrame:
    csv_path = resolve_data_path(data_dir, symbol, prefer_infer=prefer_infer)
    df = pd.read_csv(csv_path)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"{csv_path} 缺少必要列: {missing}")

    out = df[RAW_COLUMNS].copy()
    out["datetime"] = _parse_datetime(out)
    out = out.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime")
    out = out.reset_index(drop=True)

    for c in OHLCV_COLUMNS:
        out[c] = out[c].astype(float)

    out = add_indicators(out, indicators)
    return out


def load_symbol_frames(
    symbols: Iterable[str],
    data_dir: Path,
    indicators: list[str],
    prefer_infer: bool,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        result[symbol] = load_symbol_frame(
            symbol=symbol,
            data_dir=data_dir,
            indicators=indicators,
            prefer_infer=prefer_infer,
        )
    return result

