# -*- coding: utf-8 -*-
"""
value_stock_screener 的本地数据存储层。

数据目录默认 `app_data/`，包含：
- stocks.csv      股票池
- k_data.csv      K线/估值
- profit.csv      盈利能力
- balance.csv     资产负债
- stock_basic.csv 上市/退市信息
- dividend.csv    分红信息
- growth.csv      成长信息
- meta.json       各数据表最后更新日期
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import pandas as pd

DATA_DIR = "app_data"
META_FILE = os.path.join(DATA_DIR, "meta.json")
STOCKS_FILE = os.path.join(DATA_DIR, "stocks.csv")
K_DATA_FILE = os.path.join(DATA_DIR, "k_data.csv")
PROFIT_FILE = os.path.join(DATA_DIR, "profit.csv")
BALANCE_FILE = os.path.join(DATA_DIR, "balance.csv")
DIVIDEND_FILE = os.path.join(DATA_DIR, "dividend.csv")
GROWTH_FILE = os.path.join(DATA_DIR, "growth.csv")
STOCK_BASIC_FILE = os.path.join(DATA_DIR, "stock_basic.csv")

# 股票池建议每周更新
STOCKS_UPDATE_DAYS = 7
# K线保留最近 N 天
K_DATA_DAYS = 90


def _ensure_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def load_meta() -> dict:
    _ensure_dir()
    if not os.path.exists(META_FILE):
        return {}
    try:
        with open(META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_meta(meta: dict) -> None:
    _ensure_dir()
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _need_update(key: str, meta: dict, max_days: int | None = None) -> bool:
    last = meta.get(key)
    if not last:
        return True
    if max_days is None:
        return False
    try:
        dt = datetime.strptime(last[:10], "%Y-%m-%d")
        return (datetime.now() - dt).days >= max_days
    except Exception:
        return True


def load_stocks() -> list[dict] | None:
    if not os.path.exists(STOCKS_FILE):
        return None
    try:
        df = pd.read_csv(STOCKS_FILE, dtype=str)
        recs = df.to_dict("records")
        for r in recs:
            if "board" not in r or not r["board"]:
                r["board"] = "kcb" if str(r.get("code", "")).startswith("sh.688") else "main"
        return recs
    except Exception:
        return None


def save_stocks(stocks: list[dict]) -> None:
    _ensure_dir()
    df = pd.DataFrame(stocks)
    df.to_csv(STOCKS_FILE, index=False, encoding="utf-8-sig")


def load_stock_basic() -> pd.DataFrame:
    if not os.path.exists(STOCK_BASIC_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(STOCK_BASIC_FILE, dtype={"code": str})
    except Exception:
        return pd.DataFrame()


def save_stock_basic(df: pd.DataFrame) -> None:
    _ensure_dir()
    if df.empty:
        return
    df = df.drop_duplicates(subset=["code"], keep="last")
    df.to_csv(STOCK_BASIC_FILE, index=False, encoding="utf-8-sig")


def merge_stock_basic(old_df: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    if not new_rows:
        return old_df
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([old_df, new_df], ignore_index=True)
    return combined.drop_duplicates(subset=["code"], keep="last")


def load_k_data() -> pd.DataFrame:
    if not os.path.exists(K_DATA_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(K_DATA_FILE, dtype={"code": str})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        if "psTTM" not in df.columns:
            df["psTTM"] = None
        return df
    except Exception:
        return pd.DataFrame()


def save_k_data(df: pd.DataFrame) -> None:
    _ensure_dir()
    if df.empty:
        return
    df = df.drop_duplicates(subset=["code", "date"], keep="last")
    df = df.sort_values(["code", "date"])
    df.to_csv(K_DATA_FILE, index=False, encoding="utf-8-sig")


def get_k_max_date_per_code(df: pd.DataFrame) -> dict[str, str]:
    if df.empty:
        return {}
    g = df.groupby("code")["date"].max()
    return g.to_dict()


def merge_k_data(old_df: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    base_cols = ["code", "date", "close", "peTTM", "pbMRQ", "psTTM"]
    if not new_rows:
        return old_df if not old_df.empty else pd.DataFrame(columns=base_cols)
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["code", "date"], keep="last")
    cutoff = (datetime.now() - timedelta(days=K_DATA_DAYS)).strftime("%Y-%m-%d")
    combined = combined[combined["date"] >= cutoff].copy()
    return combined.sort_values(["code", "date"]).reset_index(drop=True)


def load_profit() -> pd.DataFrame:
    if not os.path.exists(PROFIT_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(PROFIT_FILE, dtype={"code": str})
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
        if "quarter" in df.columns:
            df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce").fillna(0).astype(int)
        return df
    except Exception:
        return pd.DataFrame()


def save_profit(df: pd.DataFrame) -> None:
    _ensure_dir()
    if df.empty:
        return
    df = df.drop_duplicates(subset=["code", "year", "quarter"], keep="last")
    df.to_csv(PROFIT_FILE, index=False, encoding="utf-8-sig")


def merge_profit(old_df: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    if not new_rows:
        return old_df
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([old_df, new_df], ignore_index=True)
    return combined.drop_duplicates(subset=["code", "year", "quarter"], keep="last")


def load_balance() -> pd.DataFrame:
    if not os.path.exists(BALANCE_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(BALANCE_FILE, dtype={"code": str})
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
        if "quarter" in df.columns:
            df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce").fillna(0).astype(int)
        return df
    except Exception:
        return pd.DataFrame()


def save_balance(df: pd.DataFrame) -> None:
    _ensure_dir()
    if df.empty:
        return
    df = df.drop_duplicates(subset=["code", "year", "quarter"], keep="last")
    df.to_csv(BALANCE_FILE, index=False, encoding="utf-8-sig")


def merge_balance(old_df: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    if not new_rows:
        return old_df
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([old_df, new_df], ignore_index=True)
    return combined.drop_duplicates(subset=["code", "year", "quarter"], keep="last")


def load_dividend() -> pd.DataFrame:
    if not os.path.exists(DIVIDEND_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(DIVIDEND_FILE, dtype={"code": str})
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
        if "dividPerShare" not in df.columns:
            df["dividPerShare"] = 0.0
        df["dividPerShare"] = pd.to_numeric(df["dividPerShare"], errors="coerce").fillna(0)
        return df
    except Exception:
        return pd.DataFrame()


def save_dividend(df: pd.DataFrame) -> None:
    _ensure_dir()
    if df.empty:
        return
    if "dividPerShare" not in df.columns:
        df["dividPerShare"] = 0.0
    df["dividPerShare"] = pd.to_numeric(df["dividPerShare"], errors="coerce").fillna(0)
    df = df.drop_duplicates(subset=["code", "year"], keep="last")
    df.to_csv(DIVIDEND_FILE, index=False, encoding="utf-8-sig")


def merge_dividend(old_df: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    if not new_rows:
        return old_df
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([old_df, new_df], ignore_index=True)
    return combined.drop_duplicates(subset=["code", "year"], keep="last")


def load_growth() -> pd.DataFrame:
    if not os.path.exists(GROWTH_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(GROWTH_FILE, dtype={"code": str})
        for c in ["year", "quarter"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        return df
    except Exception:
        return pd.DataFrame()


def save_growth(df: pd.DataFrame) -> None:
    _ensure_dir()
    if df.empty:
        return
    df = df.drop_duplicates(subset=["code", "year", "quarter"], keep="last")
    df.to_csv(GROWTH_FILE, index=False, encoding="utf-8-sig")


def merge_growth(old_df: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    if not new_rows:
        return old_df
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([old_df, new_df], ignore_index=True)
    return combined.drop_duplicates(subset=["code", "year", "quarter"], keep="last")
