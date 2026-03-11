# -*- coding: utf-8 -*-
"""
A股主板价值选股主程序（独立模块）。

策略概要（与实现一致）：
- 主板范围：仅 `sh.6*`（排除 `sh.688`）和 `sz.0*`
- 过滤条件：PE/PB 区间、ROE 下限、上市日期限制、ST 过滤
- 打分模型：格雷厄姆（E/P + PB + ROE + 负债）+ 巴菲特偏好（毛利率 + 行业分散）
- 数据来源：baostock 为主，akshare 作为补充
- 数据落盘：通过 `value_stock_screener.data_store` 做增量缓存
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any

import baostock as bs
import pandas as pd

try:
    from . import data_store as ds
except Exception:  # 兼容直接执行文件
    import data_store as ds  # type: ignore

try:
    import akshare as ak

    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False

# 可选启用 akshare 代理补丁（仅当环境变量完整配置时）
try:
    import akshare_proxy_patch

    _AK_PATCH_AVAILABLE = True
except Exception:
    _AK_PATCH_AVAILABLE = False

_PATCH_HOST = os.environ.get("AKSHARE_PROXY_HOST", "").strip()
_PATCH_TOKEN = os.environ.get("AKSHARE_PROXY_TOKEN", "").strip()
_PATCH_RETRY = int(os.environ.get("AKSHARE_PROXY_RETRY", "30") or 30)
if _AK_PATCH_AVAILABLE and _PATCH_HOST and _PATCH_TOKEN:
    try:
        akshare_proxy_patch.install_patch(_PATCH_HOST, _PATCH_TOKEN, _PATCH_RETRY)
    except Exception:
        pass


LOG_DIR = "app_logs"


def _build_log_file(prefix: str) -> str:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        return os.path.join(LOG_DIR, f"{prefix}_{ts}.txt")
    except Exception:
        return f"{prefix}_{os.getpid()}.txt"


_log_path = _build_log_file("select")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(_log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ============ 策略参数 ============
PE_MAX = 25
PE_MIN = 3
PB_MAX = 3
PB_MIN = 0.5
ROE_MIN = 8
DEBT_RATIO_MAX = 100
GP_MIN = 15

TOP_K = int(os.environ.get("TOP_K", "100")) or 100
IPO_DATE_CUTOFF = "2019-01-01"
CAP_FINANCE = int(0 * TOP_K / 100)
CAP_UTIL_CITY = int(5 * TOP_K / 100)

# ============ 下载参数 ============
REQUEST_DELAY = float(os.environ.get("REQUEST_DELAY", "0"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3") or 3)
BACKOFF_AFTER_FAIL = float(os.environ.get("BACKOFF_AFTER_FAIL", "5"))
MAX_SCAN = int(os.environ.get("MAX_SCAN", "0")) or 0
SKIP_UPDATE = os.environ.get("SKIP_UPDATE", "") in ("1", "true", "yes")


def normalize_code(code: Any) -> str:
    if not code:
        return ""
    code = str(code).strip()
    if "." in code and len(code) >= 9:
        return code.lower()
    dig = "".join(c for c in code if c.isdigit())
    if len(dig) >= 6:
        dig = dig[:6]
        return f"sh.{dig}" if dig.startswith(("6", "9")) else f"sz.{dig}"
    return code.lower()


def safe_float(val: Any, default: float | None = None) -> float | None:
    if val is None or val == "" or val == "nan":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _throttle(retry_count: int = 0) -> None:
    delay = REQUEST_DELAY + BACKOFF_AFTER_FAIL * max(0, retry_count)
    if delay > 0:
        time.sleep(delay)


def _is_main_board(code: str) -> bool:
    if not code or len(code) < 9:
        return False
    code = str(code).lower()
    if code.startswith("sh.6") and not code.startswith("sh.688"):
        return True
    if code.startswith("sz.0"):
        return True
    return False


def _parse_akshare_stocks(df: pd.DataFrame) -> list[dict[str, str]]:
    stocks: list[dict[str, str]] = []
    cols = df.columns.tolist()
    ckey = "代码" if "代码" in cols else "code"
    nkey = "名称" if "名称" in cols else "name"
    for _, row in df.iterrows():
        raw = str(row.get(ckey, "")).strip()
        code = raw.zfill(6) if raw.isdigit() else raw
        if len(code) != 6 or not code.isdigit():
            continue
        name = str(row.get(nkey, ""))
        if "ST" in name.upper() or "*" in name:
            continue
        code = normalize_code(code)
        if not _is_main_board(code):
            continue
        stocks.append({"code": code, "name": name})
    return stocks


def _fetch_stock_universe_akshare() -> list[dict[str, str]]:
    if not HAS_AKSHARE:
        return []
    try:
        df = ak.stock_info_a_code_name()
        if df is not None and not df.empty:
            out = _parse_akshare_stocks(df)
            if out:
                logger.info("akshare stock_info_a_code_name: %d 只", len(out))
                return out
    except Exception as exc:
        logger.warning("akshare stock_info_a_code_name 失败: %s", exc)
    try:
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            out = _parse_akshare_stocks(df)
            if out:
                logger.info("akshare stock_zh_a_spot_em: %d 只", len(out))
                return out
    except Exception as exc:
        logger.warning("akshare stock_zh_a_spot_em 失败: %s", exc)
    return []


def _fetch_stock_universe_baostock() -> list[dict[str, str]]:
    stocks: list[dict[str, str]] = []
    for day in [None, "2025-12-31", "2025-06-30"]:
        rs = bs.query_all_stock(day=day)
        if rs.error_code != "0":
            continue
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            code = row[0] if row else ""
            if len(str(code)) == 6 and str(code).replace(".", "").replace("-", "").isdigit():
                code = normalize_code(code)
            else:
                code = str(code).strip().lower()
            name = row[1] if len(row) > 1 else ""
            stype = row[2] if len(row) > 2 else ""
            status = row[5] if len(row) > 5 else "1"
            if len(code) != 9:
                continue
            if stype != "1" or status != "1":
                continue
            if "ST" in str(name).upper() or "*" in str(name):
                continue
            if not _is_main_board(code):
                continue
            stocks.append({"code": code, "name": name})
        if stocks:
            logger.info("baostock query_all_stock: %d 只", len(stocks))
            return stocks
    return []


def _fetch_stock_universe_fallback() -> list[dict[str, str]]:
    idx_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for fn in [bs.query_hs300_stocks, bs.query_zz500_stocks, bs.query_sz50_stocks]:
        rs = fn(date=idx_date)
        if rs.error_code != "0":
            rs = fn(date="")
        if rs.error_code != "0":
            continue
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            code = (row[1] if len(row) > 1 else row[0]) if row else ""
            name = row[2] if len(row) > 2 else ""
            if not code or len(code) != 9:
                continue
            if "ST" in str(name).upper():
                continue
            if not _is_main_board(code):
                continue
            if code in seen:
                continue
            seen.add(code)
            out.append({"code": code, "name": name})
    return out


def _fetch_stock_universe() -> list[dict[str, str]]:
    stocks = _fetch_stock_universe_baostock()
    if not stocks:
        stocks = _fetch_stock_universe_akshare()
    if not stocks:
        stocks = _fetch_stock_universe_fallback()
        if stocks:
            logger.info("回退至指数成分股: %d 只", len(stocks))
    return stocks


def _fetch_k_data(code: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    for attempt in range(MAX_RETRIES):
        try:
            _throttle(attempt)
            rs = bs.query_history_k_data_plus(
                code,
                "date,code,close,peTTM,pbMRQ",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",
            )
            if rs.error_code != "0":
                continue
            rows = []
            while rs.next():
                row = rs.get_row_data()
                if len(row) < 5:
                    continue
                close = safe_float(row[2])
                pe = safe_float(row[3])
                pb = safe_float(row[4])
                if close and close > 0:
                    rows.append(
                        {
                            "code": row[1],
                            "date": row[0],
                            "close": close,
                            "peTTM": pe,
                            "pbMRQ": pb,
                            "psTTM": None,
                        }
                    )
            return rows
        except Exception as exc:
            logger.warning("K线请求异常 %s 第%d次: %s", code, attempt + 1, exc)
    return []


def _fetch_profit(code: str, year: int, quarter: int) -> dict[str, Any] | None:
    for attempt in range(MAX_RETRIES):
        try:
            _throttle(attempt)
            rs = bs.query_profit_data(code=code, year=year, quarter=quarter)
            if rs.error_code != "0":
                continue
            fields = getattr(rs, "fields", [])
            while rs.next():
                row = rs.get_row_data()
                d = dict(zip(fields, row)) if fields else {}
                roe = safe_float(d.get("roeAvg") or (row[3] if len(row) > 3 else None))
                gp = safe_float(d.get("gpMargin"))
                if roe is not None and 0 < roe < 1:
                    roe *= 100
                if gp is not None and 0 < abs(gp) < 1:
                    gp *= 100
                if roe is not None or gp is not None:
                    return {"code": code, "year": year, "quarter": quarter, "roeAvg": roe, "gpMargin": gp}
            return None
        except Exception as exc:
            logger.warning("盈利数据请求异常 %s 第%d次: %s", code, attempt + 1, exc)
    return None


def _fetch_stock_basic(code: str) -> dict[str, str] | None:
    for attempt in range(MAX_RETRIES):
        try:
            _throttle(attempt)
            rs = bs.query_stock_basic(code=code)
            if rs.error_code != "0":
                continue
            while rs.next():
                row = rs.get_row_data()
                if len(row) >= 6:
                    return {
                        "code": code,
                        "ipo_date": str(row[2] or "").strip(),
                        "out_date": str(row[3] or "").strip(),
                        "status": str(row[5] or "").strip(),
                    }
            return None
        except Exception as exc:
            logger.warning("股票基本资料请求异常 %s 第%d次: %s", code, attempt + 1, exc)
    return None


def _fetch_balance(code: str, year: int, quarter: int) -> dict[str, Any] | None:
    for attempt in range(MAX_RETRIES):
        try:
            _throttle(attempt)
            rs = bs.query_balance_data(code=code, year=year, quarter=quarter)
            if rs.error_code != "0":
                continue
            fields = getattr(rs, "fields", [])
            while rs.next():
                row = rs.get_row_data()
                d = dict(zip(fields, row)) if fields else {}
                val = safe_float(d.get("liabilityToAsset"))
                if val is not None:
                    return {"code": code, "year": year, "quarter": quarter, "liabilityToAsset": val}
            return None
        except Exception as exc:
            logger.warning("资产负债请求异常 %s 第%d次: %s", code, attempt + 1, exc)
    return None


def update_all_data(stocks: list[dict[str, str]]) -> list[dict[str, str]] | None:
    meta = ds.load_meta()
    today = ds._today()
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=ds.K_DATA_DAYS)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    force = os.environ.get("FORCE_REFRESH_STOCKS", "") in ("1", "true", "yes")
    if force or ds._need_update("stocks", meta, ds.STOCKS_UPDATE_DAYS) or not stocks:
        stocks = _fetch_stock_universe()
        if stocks:
            ds.save_stocks(stocks)
            meta["stocks"] = today
            logger.info("股票池已更新: %d 只", len(stocks))

    if not stocks:
        return None

    basic_df = ds.load_stock_basic()
    basic_codes = set(basic_df["code"].tolist()) if not basic_df.empty else set()
    if not SKIP_UPDATE:
        scan_for_basic = stocks[:MAX_SCAN] if MAX_SCAN > 0 else stocks
        basic_new = []
        for i, s in enumerate(scan_for_basic):
            if s["code"] not in basic_codes:
                r = _fetch_stock_basic(s["code"])
                if r:
                    basic_new.append(r)
                    basic_codes.add(s["code"])
            if (i + 1) % 100 == 0:
                logger.info("股票基本资料更新进度: %d/%d", min(i + 1, len(scan_for_basic)), len(scan_for_basic))
        if basic_new:
            basic_df = ds.merge_stock_basic(basic_df, basic_new)
            ds.save_stock_basic(basic_df)
            meta["stock_basic"] = today
            logger.info("股票基本资料已更新: +%d 条", len(basic_new))

    if SKIP_UPDATE:
        logger.info("SKIP_UPDATE=1，仅使用本地缓存数据")
        return stocks

    k_df = ds.load_k_data()
    max_dates = ds.get_k_max_date_per_code(k_df)
    k_new: list[dict[str, Any]] = []
    for i, s in enumerate(stocks):
        if MAX_SCAN > 0 and i >= MAX_SCAN:
            break
        code = s["code"]
        last = max_dates.get(code, "")
        fetch_start = start_str
        if last:
            try:
                ld = datetime.strptime(last, "%Y-%m-%d")
                fetch_start = (ld + timedelta(days=1)).strftime("%Y-%m-%d")
            except Exception:
                pass
        if fetch_start <= end_str:
            rows = _fetch_k_data(code, fetch_start, end_str)
            if rows:
                k_new.extend(rows)
        if (i + 1) % 100 == 0:
            logger.info("K线更新进度: %d/%d", min(i + 1, len(stocks)), len(stocks))
    if k_new:
        k_df = ds.merge_k_data(k_df, k_new)
        ds.save_k_data(k_df)
        meta["k_data"] = today
        logger.info("K线已更新: +%d 条", len(k_new))

    cur = datetime.now()
    y, q = cur.year, (cur.month - 1) // 3 + 1
    if q == 1:
        y, q = y - 1, 4
    elif q == 2:
        q = 1

    profit_df = ds.load_profit()
    balance_df = ds.load_balance()
    profit_new: list[dict[str, Any]] = []
    balance_new: list[dict[str, Any]] = []
    scan_list = stocks[:MAX_SCAN] if MAX_SCAN > 0 else stocks
    for i, s in enumerate(scan_list):
        code = s["code"]
        has_p = not profit_df.empty and ((profit_df["code"] == code) & (profit_df["year"] == y) & (profit_df["quarter"] == q)).any()
        has_b = not balance_df.empty and ((balance_df["code"] == code) & (balance_df["year"] == y) & (balance_df["quarter"] == q)).any()
        for yy, qq in [(y, q), (y, max(1, q - 1)), (y - 1, 4)]:
            if yy < 2020:
                break
            if not has_p:
                r = _fetch_profit(code, yy, qq)
                if r:
                    profit_new.append(r)
                    has_p = True
            if not has_b:
                r = _fetch_balance(code, yy, qq)
                if r:
                    balance_new.append(r)
                    has_b = True
        if (i + 1) % 100 == 0:
            logger.info("财务更新进度: %d/%d", min(i + 1, len(scan_list)), len(scan_list))
    if profit_new:
        ds.save_profit(ds.merge_profit(profit_df, profit_new))
        meta["profit"] = today
    if balance_new:
        ds.save_balance(ds.merge_balance(balance_df, balance_new))
        meta["balance"] = today

    ds.save_meta(meta)
    return stocks


def get_latest_valuation_from_local(code: str, k_df: pd.DataFrame) -> tuple[float | None, float | None, float | None, float | None]:
    sub = k_df[k_df["code"] == code]
    if sub.empty:
        return None, None, None, None
    row = sub.sort_values("date", ascending=False).iloc[0]
    return (
        safe_float(row.get("peTTM")),
        safe_float(row.get("pbMRQ")),
        safe_float(row.get("close")),
        safe_float(row.get("psTTM")),
    )


def get_roe_from_local(code: str, year: int, quarter: int, profit_df: pd.DataFrame) -> float | None:
    sub = profit_df[(profit_df["code"] == code) & (profit_df["year"] == year) & (profit_df["quarter"] == quarter)]
    if sub.empty:
        return None
    return safe_float(sub.iloc[0].get("roeAvg"))


def get_gpmargin_from_local(code: str, year: int, quarter: int, profit_df: pd.DataFrame) -> float | None:
    sub = profit_df[(profit_df["code"] == code) & (profit_df["year"] == year) & (profit_df["quarter"] == quarter)]
    if sub.empty:
        return None
    return safe_float(sub.iloc[0].get("gpMargin"))


def _infer_industry(name: str) -> str:
    if not name:
        return "其他"
    n = str(name)
    if "银行" in n or "农商行" in n or "城商行" in n:
        return "金融_银行"
    if "保险" in n or "人寿" in n or "太保" in n or "人保" in n or "平安" in n:
        return "金融_保险"
    if "证券" in n or "券商" in n:
        return "金融_证券"
    if "城投" in n or "城建" in n or "交投" in n or "建投" in n or "能投" in n or "产投" in n:
        return "城投公用"
    if "电力" in n or "能源" in n or "环保" in n or "水务" in n or "环境" in n or "高速" in n or "港口" in n:
        return "公用事业"
    if "地产" in n or ("开发" in n and "房地产" in n):
        return "房地产"
    return "其他"


def _ipo_ok(code: str, basic_dict: dict[str, dict[str, Any]]) -> bool:
    if not basic_dict:
        return True
    r = basic_dict.get(code)
    if r is None:
        return False
    ipo = str(r.get("ipo_date", "") or "").strip()
    out = str(r.get("out_date", "") or "").strip()
    if out.lower() in ("nan", "nat", ""):
        out = ""
    status = str(r.get("status", "") or "").strip()
    if not ipo or ipo >= IPO_DATE_CUTOFF:
        return False
    if out:
        return False
    if status != "1":
        return False
    return True


def get_debt_from_local(code: str, year: int, quarter: int, balance_df: pd.DataFrame) -> float | None:
    sub = balance_df[(balance_df["code"] == code) & (balance_df["year"] == year) & (balance_df["quarter"] == quarter)]
    if sub.empty:
        return None
    val = safe_float(sub.iloc[0].get("liabilityToAsset"))
    if val is None:
        return None
    if 0 < val < 0.2:
        return round((1 - val) * 100, 2)
    return round(val * 100, 2) if val <= 1 else round(val, 2)


def calc_score(pe: float | None, pb: float | None, roe: float | None, debt_ratio: float | None, gp: float | None = None, is_finance: bool = False) -> float:
    score = 0.0
    if pe is not None and PE_MIN <= pe <= PE_MAX:
        score += min(100.0 / pe / 3.0, 14)
    if pb is not None and PB_MIN <= pb <= PB_MAX:
        score += max(0.0, (PB_MAX - pb) / (PB_MAX - PB_MIN) * 14)
    if roe is not None and roe >= ROE_MIN:
        score += min(roe / 6.0, 14)
    if debt_ratio is not None and debt_ratio <= DEBT_RATIO_MAX:
        score += max(0.0, (DEBT_RATIO_MAX - debt_ratio) / DEBT_RATIO_MAX * 8)
    if not is_finance and gp is not None and gp >= GP_MIN:
        score += min((gp - GP_MIN) / 15.0, 5)
    return round(score, 2)


def _apply_industry_diversification(results: list[dict[str, Any]], top_n: int | None = None) -> list[dict[str, Any]]:
    if top_n is None:
        top_n = TOP_K
    finance_keys = {"金融_银行", "金融_保险", "金融_证券"}
    util_keys = {"城投公用", "公用事业"}
    picked: list[dict[str, Any]] = []
    cnt_finance, cnt_util = 0, 0
    for r in results:
        if len(picked) >= top_n:
            break
        ind = r.get("industry", "其他")
        if ind in finance_keys:
            if cnt_finance >= CAP_FINANCE:
                continue
            cnt_finance += 1
        elif ind in util_keys:
            if cnt_util >= CAP_UTIL_CITY:
                continue
            cnt_util += 1
        picked.append(r)
    return picked


def _write_results(top_list: list[dict[str, Any]], top_k: int) -> tuple[str, str]:
    result_path = _build_log_file("result")
    with open(result_path, "w", encoding="utf-8", newline="\n") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        width = 72
        f.write("\n\n")
        f.write("+" + "=" * width + "+\n")
        f.write("|" + f" A股主板价值投资 Top {top_k} 选股结果 ".center(width) + "|\n")
        f.write("+" + "=" * width + "+\n")
        f.write("| 生成时间: " + ts.ljust(width - 13) + "|\n")
        strat = f"策略: 格雷厄姆+巴菲特 | 限制: 2019-01-01前上市且未退市 | 金融≤{CAP_FINANCE} 公用≤{CAP_UTIL_CITY}"
        f.write("| " + strat[: width - 2].ljust(width - 2) + "|\n")
        f.write("+" + "-" * width + "+\n\n")
        f.write("+------+------------+----------+------+------+-------+-------+------+--------+----------+\n")
        f.write("| 序号 |    代码    |   名称   |  PE   |  PB   | ROE%  | 负债% | 评分 | 最新价 |  行业    |\n")
        f.write("+------+------------+----------+------+------+-------+-------+------+--------+----------+\n")
        for idx, r in enumerate(top_list, 1):
            roe_str = f"{r['roe']:.1f}" if r.get("roe") is not None else "-"
            debt_val = r.get("debt_ratio")
            dr_str = f"{debt_val:.1f}" if debt_val is not None else "-"
            name = ((r.get("name") or "")[:8]).ljust(8)
            close_val = r.get("close") or 0
            ind = (r.get("industry") or "其他")[:8].ljust(8)
            line = "| %4d | %10s | %-8s | %5.2f | %5.2f | %5s | %5s | %5.2f | %6.2f | %-8s |\n" % (
                idx,
                r["code"],
                name,
                r["pe"],
                r["pb"],
                roe_str,
                dr_str,
                r["score"],
                close_val,
                ind,
            )
            f.write(line)
        f.write("+------+------------+----------+------+------+-------+-------+------+--------+----------+\n\n")
        f.write(f"  说明: 金融≤{CAP_FINANCE}只，公用/城投≤{CAP_UTIL_CITY}只。\n")
        f.write("  仅供参考，不构成投资建议。\n")

    json_path = result_path.replace(".txt", ".json")
    payload = {"codes": [{"code": r["code"], "name": (r.get("name") or "").strip()} for r in top_list]}
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(payload, jf, ensure_ascii=False, indent=2)
    return result_path, json_path


def run_screening(top_k: int | None = None) -> list[dict[str, Any]]:
    stocks = ds.load_stocks() or []
    stocks = update_all_data(stocks) or []
    if not stocks:
        raise RuntimeError("股票池为空")

    k_df = ds.load_k_data()
    profit_df = ds.load_profit()
    balance_df = ds.load_balance()
    basic_df = ds.load_stock_basic()
    basic_dict = basic_df.set_index("code").to_dict("index") if not basic_df.empty else {}

    cur = datetime.now()
    y, q = cur.year, (cur.month - 1) // 3 + 1
    if q == 1:
        y, q = cur.year - 1, 4
    elif q == 2:
        q = 1

    scan_list = [s for s in stocks if _is_main_board(s["code"]) and _ipo_ok(s["code"], basic_dict)]
    if MAX_SCAN > 0:
        scan_list = scan_list[:MAX_SCAN]
    logger.info("参与筛选股票: %d 只", len(scan_list))

    results: list[dict[str, Any]] = []
    for s in scan_list:
        code, name = s["code"], s.get("name", "")
        pe, pb, close, _ = get_latest_valuation_from_local(code, k_df)
        if pe is None or pb is None:
            continue
        if not (PE_MIN <= pe <= PE_MAX and PB_MIN <= pb <= PB_MAX):
            continue

        roe = get_roe_from_local(code, y, q, profit_df)
        if roe is None:
            roe = get_roe_from_local(code, y, max(1, q - 1), profit_df)
        if roe is None:
            roe = get_roe_from_local(code, y - 1, 4, profit_df)
        if roe is None or roe < ROE_MIN:
            continue

        debt = get_debt_from_local(code, y, q, balance_df)
        if debt is None:
            debt = get_debt_from_local(code, y, max(1, q - 1), balance_df)
        gp = get_gpmargin_from_local(code, y, q, profit_df) or get_gpmargin_from_local(code, y, max(1, q - 1), profit_df)
        industry = _infer_industry(name)
        is_finance = industry.startswith("金融_")
        score = calc_score(pe, pb, roe, debt, gp, is_finance)
        results.append(
            {
                "code": code,
                "name": name,
                "pe": pe,
                "pb": pb,
                "roe": roe,
                "debt_ratio": debt,
                "score": score,
                "close": close,
                "industry": industry,
                "gp": gp,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    pick_top = top_k if top_k is not None else TOP_K
    return _apply_industry_diversification(results, top_n=pick_top)


def main() -> None:
    logger.info("=" * 60)
    logger.info("A股主板价值投资选股 - 格雷厄姆+巴菲特偏好")
    logger.info("扫描范围: %s", "主板全量" if MAX_SCAN <= 0 else f"前{MAX_SCAN}只")
    logger.info("=" * 60)

    lg = bs.login()
    if lg.error_code != "0":
        logger.error("baostock 登录失败: %s", lg.error_msg)
        return

    try:
        top_list = run_screening(TOP_K)
        logger.info("主板 Top%d", TOP_K)
        for idx, r in enumerate(top_list, 1):
            roe_str = f"{r['roe']:.1f}" if r.get("roe") is not None else "-"
            logger.info(
                "%3d. %s %-8s PE:%.1f PB:%.1f ROE:%s 评分:%.1f",
                idx,
                r["code"],
                (r.get("name") or "")[:8],
                r["pe"],
                r["pb"],
                roe_str,
                r["score"],
            )

        result_path, json_path = _write_results(top_list, TOP_K)
        logger.info("选股结果已写入 %s", os.path.abspath(result_path))
        logger.info("Top%d 代码已导出 %s", TOP_K, os.path.abspath(json_path))
    finally:
        bs.logout()
        logger.info("完成")


if __name__ == "__main__":
    main()
