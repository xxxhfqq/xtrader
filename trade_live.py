from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from config import load_config
from data_sync import refresh_symbol_data
from main_infer import infer_target_ratio_for_code, model_available

CONFIG_PATH = "config.json"


def _parse_time_delta(time_text: str) -> pd.Timedelta:
    parts = [int(x) for x in str(time_text).strip().split(":")]
    if len(parts) == 2:
        hh, mm = parts
        ss = 0
    elif len(parts) == 3:
        hh, mm, ss = parts
    else:
        raise ValueError(f"时间格式错误: {time_text}")
    return pd.Timedelta(hours=hh, minutes=mm, seconds=ss)


def _load_trading_sessions(raw_sessions: Any) -> list[tuple[pd.Timedelta, pd.Timedelta]]:
    fallback = [
        (_parse_time_delta("09:35:00"), _parse_time_delta("11:30:00")),
        (_parse_time_delta("13:00:00"), _parse_time_delta("15:00:00")),
    ]
    if not isinstance(raw_sessions, list) or not raw_sessions:
        return fallback

    sessions: list[tuple[pd.Timedelta, pd.Timedelta]] = []
    for item in raw_sessions:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            start = _parse_time_delta(str(item[0]))
            end = _parse_time_delta(str(item[1]))
        except Exception:
            continue
        if start < end:
            sessions.append((start, end))
    return sessions or fallback


cfg = load_config(CONFIG_PATH)
main_config_path = Path.cwd() / "main_config.json"
trade_cfg = dict(cfg.get("trade", {}))

buy_fee_rate = float(cfg["env"]["buy_fee_rate"])
lot_size = int(cfg["env"]["lot_size"])
default_topk = int(cfg.get("topK", 4))
default_run_full_infer = bool(cfg.get("run_full_infer", True))
default_test_mode = False
min_fee = float(trade_cfg.get("min_fee", 5.0))
OPEN_FULL_THRESHOLD = float(trade_cfg.get("open_full_threshold", 0.7))
FLAT_THRESHOLD = float(trade_cfg.get("flat_threshold", 0.3))
OPEN_CANDIDATE_THRESHOLD = float(trade_cfg.get("open_candidate_threshold", 0.3))
REBALANCE_DIFF_THRESHOLD = float(trade_cfg.get("rebalance_diff_threshold", 0.1))
MIN_TRADE_VALUE = float(trade_cfg.get("min_trade_value", 100.0))
BAR_INTERVAL_MINUTES = max(1, int(trade_cfg.get("bar_interval_minutes", 5)))
ORDER_SETTLE_INITIAL_WAIT_SECONDS = max(0.0, float(trade_cfg.get("order_settle_initial_wait_seconds", 3)))
ORDER_SETTLE_MAX_WAIT_SECONDS = max(
    ORDER_SETTLE_INITIAL_WAIT_SECONDS,
    float(trade_cfg.get("order_settle_max_wait_seconds", 10)),
)
IN_BAR_POLL_SLEEP_SECONDS = max(0.0, float(trade_cfg.get("in_bar_poll_sleep_seconds", 0.1)))
OUT_OF_TRADING_SLEEP_SECONDS = max(0.0, float(trade_cfg.get("out_of_trading_sleep_seconds", 1.0)))
DEFAULT_XIADAN_PATH = str(trade_cfg.get("xiadan_path", r"C:\\同花顺软件\\同花顺\\xiadan.exe")).strip() or r"C:\\同花顺软件\\同花顺\\xiadan.exe"
TRADING_SESSIONS = _load_trading_sessions(trade_cfg.get("trading_sessions"))

trade_codes_in_cfg = []
trade_code_name_in_cfg: dict[str, str] = {}
for item in cfg.get("trade_codes", []):
    if isinstance(item, dict):
        code = item.get("code")
        if isinstance(code, str) and code:
            trade_codes_in_cfg.append(code)
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                trade_code_name_in_cfg[code] = name.strip()
    elif isinstance(item, str) and item:
        trade_codes_in_cfg.append(item)

code_name_map: dict[str, str] = {}
candidate_codes: list[str] = []
user: Any | None = None
today = pd.to_datetime("today").date()
shared_model_ready = False
holding_codes: set[str] = set()
code_states: dict[str, dict[str, Any]] = defaultdict(
    lambda: {
        "cash": 0.0,
        "shares": 0,
        "locked_today": 0,
        "actual_ratio": 0.0,
        "target_ratio": None,
        "total_asset": 0.0,
    }
)


def _load_runtime_cfg() -> dict[str, Any]:
    if not main_config_path.exists():
        return {}
    try:
        with open(main_config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    return {}


def get_topk() -> int:
    runtime_cfg = _load_runtime_cfg()
    value = runtime_cfg.get("topK", default_topk)
    try:
        k = int(value)
        if k >= 1:
            return k
    except Exception:
        pass
    return max(1, int(default_topk))


def get_run_full_infer() -> bool:
    runtime_cfg = _load_runtime_cfg()
    return bool(runtime_cfg.get("run_full_infer", default_run_full_infer))


def get_test_mode() -> bool:
    runtime_cfg = _load_runtime_cfg()
    return bool(runtime_cfg.get("test_mode", default_test_mode))


def _sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def _get_xiadan_path() -> str:
    runtime_cfg = _load_runtime_cfg()
    path = runtime_cfg.get("xiadan_path")
    if isinstance(path, str) and path.strip():
        return path.strip()
    return DEFAULT_XIADAN_PATH


def _load_candidate_codes() -> tuple[list[str], dict[str, str]]:
    out = list(dict.fromkeys(trade_codes_in_cfg))
    local_code_name_map = dict(trade_code_name_in_cfg)
    if out:
        print(f"[INFO] 从 config.trade_codes 读取候选标的: {len(out)} 个")
    return out, local_code_name_map


def init_runtime_state() -> None:
    global candidate_codes
    global code_name_map
    candidate_codes, code_name_map = _load_candidate_codes()
    if not candidate_codes:
        raise ValueError("候选池为空：请检查 config.trade_codes")
    topk = get_topk()
    if topk > len(candidate_codes):
        print(
            f"[WARNING] topK={topk} 大于 trade_codes 数量={len(candidate_codes)}，"
            f"实际最多管理 {len(candidate_codes)} 只"
        )


def init_user() -> None:
    global user
    import easytrader
    from easytrader import grid_strategies, refresh_strategies
    from x_fin import easytrader_patch as _easytrader_patch  # noqa: F401

    user = easytrader.use("universal_client")
    user.connect(_get_xiadan_path())
    user.grid_strategy = grid_strategies.Xls
    user.enable_type_keys_for_editor()
    user.refresh_strategy = refresh_strategies.Toolbar(refresh_btn_index=4)
    print("[INFO] 交易接口初始化成功")


def ensure_model_ready() -> None:
    global shared_model_ready
    shared_model_ready = bool(model_available(CONFIG_PATH))
    if not shared_model_ready:
        model_path = Path(cfg["paths"]["model_dir"]) / "multi_symbol" / f"{cfg['save']['best_model_name']}.zip"
        raise FileNotFoundError(f"未找到共享模型: {model_path}")


def refresh_user_data() -> bool:
    if user is None:
        return False
    try:
        user.refresh()
        return True
    except Exception as exc:
        print(f"[WARNING] 刷新用户数据失败: {exc}")
        return False


def get_total_asset() -> float:
    if not refresh_user_data():
        return 0.0
    try:
        balance = user.balance
        if balance is None:
            return 0.0
        return float(balance.get("总资产", 0.0))
    except Exception as exc:
        print(f"[WARNING] 获取总资产失败: {exc}")
        return 0.0


def get_available_cash() -> float:
    if not refresh_user_data():
        return 0.0
    try:
        balance = user.balance
        if balance is None:
            return 0.0
        return float(balance.get("可用金额", 0.0))
    except Exception as exc:
        print(f"[WARNING] 获取可用现金失败: {exc}")
        return 0.0


def get_all_positions() -> list[dict[str, Any]]:
    if not refresh_user_data():
        return []
    try:
        positions = user.position
        if positions is None:
            return []
        return positions if isinstance(positions, list) else []
    except Exception as exc:
        print(f"[WARNING] 获取持仓失败: {exc}")
        return []


def get_code_name(code: str) -> str:
    return code_name_map.get(code, code)


def _position_code(item: dict[str, Any]) -> str | None:
    code_suffix = str(item.get("证券代码", "")).strip()
    if not code_suffix:
        return None
    market = str(item.get("交易市场", ""))
    if "上海" in market or market == "上海":
        return f"sh.{code_suffix}"
    if "深圳" in market or market == "深圳":
        return f"sz.{code_suffix}"
    return None


def get_position_info(code: str, positions_cache: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    if positions_cache is None:
        positions_cache = get_all_positions()

    code_suffix = code[3:]
    for item in positions_cache:
        if str(item.get("证券代码", "")) != code_suffix:
            continue
        try:
            shares = int(float(item.get("股票余额", item.get("实际数量", 0))))
            available = int(float(item.get("可用余额", item.get("可用股份", 0))))
            market_value = float(item.get("市值", item.get("最新市值", 0.0)))
            current_price = float(item.get("市价", item.get("当前价", 0.0)))
            return {
                "shares": shares,
                "available_shares": available,
                "market_value": market_value,
                "current_price": current_price,
            }
        except Exception as exc:
            print(f"[WARNING] 解析持仓失败, code={code}, error={exc}")
            return None
    return None


def has_trained_model(code: str) -> bool:
    _ = code
    return bool(shared_model_ready)


def get_managed_holding_codes() -> list[str]:
    out = []
    for code in holding_codes:
        if code in candidate_codes or has_trained_model(code):
            out.append(code)
    return out


def update_symbol_data(code: str, *, include_history: bool, verbose: bool = False) -> None:
    try:
        refresh_symbol_data([code], include_history=include_history, verbose=verbose)
    except Exception as exc:
        print(f"[WARNING] 更新数据失败 {code}: {exc}")


def update_all_codes_data(codes: list[str], description: str) -> None:
    uniq = list(dict.fromkeys(codes))
    if not uniq:
        return
    print(f"[INFO] {description}，共 {len(uniq)} 个标的")
    for idx, code in enumerate(uniq, start=1):
        update_symbol_data(code, include_history=True, verbose=False)
        if idx % 20 == 0 or idx == len(uniq):
            print(f"[INFO] {description} 进度: {idx}/{len(uniq)}")


def update_all_states() -> None:
    total_asset = get_total_asset()
    if total_asset <= 0:
        print("[WARNING] 总资产为0，跳过状态更新")
        return

    all_positions = get_all_positions()
    holding_market_values: dict[str, float] = {}

    for code in list(holding_codes):
        pos = get_position_info(code, positions_cache=all_positions)
        if pos is None:
            holding_market_values[code] = 0.0
            code_states[code]["shares"] = 0
            code_states[code]["locked_today"] = 0
            continue
        holding_market_values[code] = float(pos["market_value"])
        code_states[code]["shares"] = int(pos["shares"])
        code_states[code]["locked_today"] = max(0, int(pos["shares"]) - int(pos["available_shares"]))

    available_cash = get_available_cash()
    topk = get_topk()
    per_slot_quota = (total_asset / topk) if topk > 0 else 0.0

    needs: dict[str, float] = {}
    for code in list(holding_codes):
        mv = holding_market_values.get(code, 0.0)
        needs[code] = max(0.0, per_slot_quota - mv)

    total_need = sum(needs.values())
    for code in list(holding_codes):
        market_value = holding_market_values.get(code, 0.0)
        need = needs[code]
        if need <= 0 or total_need <= 0:
            alloc_cash = 0.0
        elif available_cash >= total_need:
            alloc_cash = need
        else:
            alloc_cash = need * (available_cash / total_need)

        code_states[code]["cash"] = float(alloc_cash)
        code_states[code]["total_asset"] = float(market_value + alloc_cash)
        denom = code_states[code]["total_asset"]
        code_states[code]["actual_ratio"] = (market_value / denom) if denom > 1e-8 else 0.0


def _infer_target_ratio(code: str, cash: float, shares: int, locked_today: int) -> float | None:
    try:
        return infer_target_ratio_for_code(
            code=code,
            config_path=CONFIG_PATH,
            cash=float(cash),
            shares=int(shares),
            locked_today=int(locked_today),
            refresh_data=False,
        )
    except Exception as exc:
        print(f"[WARNING] 标 {code} infer失败: {exc}")
        return None


def move_candidate_to_end(code: str) -> None:
    global candidate_codes
    if code not in candidate_codes:
        return
    candidate_codes = [c for c in candidate_codes if c != code] + [code]


def clear_zero_position_codes() -> int:
    to_remove = []
    for code in list(holding_codes):
        if int(code_states[code].get("shares", 0)) == 0:
            to_remove.append(code)
    for code in to_remove:
        move_candidate_to_end(code)
        holding_codes.discard(code)
        if code in code_states:
            del code_states[code]
    if to_remove:
        update_all_states()
    return len(to_remove)


def cancel_all_entrusts() -> bool:
    try:
        if not refresh_user_data():
            return False
        today_entrust = user.today_entrusts
        if not today_entrust:
            return False
        has_pending = False
        for item in today_entrust:
            status = item.get("委托状态", item.get("状态说明", ""))
            if status != "已成交":
                has_pending = True
                break
        if not has_pending:
            return False
        user.cancel_all_entrusts()
        print("[INFO] 已取消所有未成交委托")
        return True
    except Exception as exc:
        print(f"[WARNING] 撤单失败: {exc}")
        return False


def trade_code(code: str, target_ratio: float | None, allow_buy: bool = True) -> None:
    from x_fin import get_bid_ask, get_price_ak

    if target_ratio is None:
        return

    state = code_states[code]
    actual_ratio = float(state["actual_ratio"])

    if target_ratio >= OPEN_FULL_THRESHOLD:
        target_ratio = 1.0
    elif target_ratio <= FLAT_THRESHOLD:
        target_ratio = 0.0

    if abs(target_ratio - actual_ratio) < REBALANCE_DIFF_THRESHOLD:
        return

    pos_info = get_position_info(code) or {
        "shares": 0,
        "available_shares": 0,
        "market_value": 0.0,
        "current_price": 0.0,
    }
    total_asset = float(state["total_asset"])
    if total_asset <= 0:
        return

    try:
        current_price = float(pos_info.get("current_price", 0.0))
        if current_price <= 0:
            current_price = float(get_price_ak(code))
        if current_price <= 0:
            return
    except Exception:
        return

    # 买入
    if target_ratio > actual_ratio:
        if not allow_buy:
            return
        target_value = total_asset * float(target_ratio)
        current_value = float(pos_info.get("market_value", 0.0))
        buy_value = target_value - current_value
        if buy_value <= MIN_TRADE_VALUE:
            return

        bid, ask = get_bid_ask(code)
        price = float(ask) if ask is not None and ask > 0 else float(current_price)
        if price <= 0:
            return

        allocated_cash = float(state.get("cash", 0.0))
        account_cash = float(get_available_cash())
        max_cash_to_use = min(allocated_cash, account_cash)
        if max_cash_to_use <= min_fee:
            return

        shares_by_rate = int(max_cash_to_use / (price * (1.0 + buy_fee_rate)))
        shares_by_min = int((max_cash_to_use - min_fee) / price) if max_cash_to_use > min_fee else 0
        # 资金约束取更保守上限，避免因手续费估算差异导致余额不足拒单。
        max_shares_by_cash = min(shares_by_rate, shares_by_min)
        max_shares_by_value = int(buy_value / price)
        buy_shares = min(max_shares_by_cash, max_shares_by_value)
        buy_shares = (buy_shares // lot_size) * lot_size
        if buy_shares < lot_size:
            return
        try:
            user.buy(code[3:], price=price, amount=buy_shares)
            print(
                f"{pd.Timestamp.now()} [买入] {code} {buy_shares}股 "
                f"价格:{price:.2f} 目标比例:{target_ratio:.2f}"
            )
        except Exception as exc:
            print(f"[ERROR] 买入失败 {code}: {exc}")
        return

    # 卖出
    available_shares = int(pos_info.get("available_shares", 0))
    if available_shares <= 0:
        return

    target_value = total_asset * float(target_ratio)
    current_value = float(pos_info.get("market_value", 0.0))
    sell_value = current_value - target_value
    if sell_value <= MIN_TRADE_VALUE:
        return

    bid, ask = get_bid_ask(code)
    price = float(bid) if bid is not None and bid > 0 else float(current_price)
    if price <= 0:
        return

    sell_shares = int(sell_value / price)
    sell_shares = min(sell_shares, available_shares)
    sell_shares = (sell_shares // lot_size) * lot_size
    if sell_shares < lot_size:
        return
    try:
        user.sell(code[3:], price=price, amount=sell_shares)
        print(
            f"{pd.Timestamp.now()} [卖出] {code} {sell_shares}股 "
            f"价格:{price:.2f} 目标比例:{target_ratio:.2f}"
        )
    except Exception as exc:
        print(f"[ERROR] 卖出失败 {code}: {exc}")


def sync_actual_holdings_to_holding_codes() -> int:
    all_positions = get_all_positions()
    added = []
    for item in all_positions:
        code_full = _position_code(item)
        if not code_full:
            continue
        try:
            shares = int(float(item.get("股票余额", item.get("实际数量", 0))))
        except Exception:
            continue
        if shares <= 0:
            continue
        if code_full in holding_codes:
            continue
        if code_full not in candidate_codes and not has_trained_model(code_full):
            continue
        holding_codes.add(code_full)
        added.append(code_full)
    if added:
        print(f"[INFO] 同步实际持仓到 holding_codes: {added}")
    return len(added)


def find_new_candidate() -> tuple[str | None, float | None]:
    total_asset = get_total_asset()
    managed_cnt = len(get_managed_holding_codes())
    if managed_cnt >= get_topk():
        return None, None

    all_positions = get_all_positions()
    actual_holding_codes: set[str] = set()
    for item in all_positions:
        code_full = _position_code(item)
        if not code_full:
            continue
        try:
            shares = int(float(item.get("股票余额", item.get("实际数量", 0))))
        except Exception:
            continue
        if shares > 0:
            actual_holding_codes.add(code_full)

    topk = get_topk()
    per_slot_quota = (total_asset / topk) if topk > 0 else 0.0
    available_cash = get_available_cash()
    infer_cash_new = min(per_slot_quota, max(0.0, available_cash))

    for code in candidate_codes:
        if code in holding_codes or code in actual_holding_codes:
            continue
        update_symbol_data(code, include_history=True, verbose=False)
        target_ratio = _infer_target_ratio(code, cash=infer_cash_new, shares=0, locked_today=0)
        if target_ratio is not None:
            print(f"[INFO] 候选标 {code} infer结果: 目标持仓比例={target_ratio:.3f}")
        if target_ratio is not None and target_ratio > OPEN_CANDIDATE_THRESHOLD:
            return code, target_ratio
    return None, None


def adjust_portfolio() -> None:
    if sync_actual_holdings_to_holding_codes() > 0:
        update_all_states()

    new_codes: list[str] = []
    while len(get_managed_holding_codes()) < get_topk():
        new_code, target_ratio = find_new_candidate()
        if new_code is None:
            break
        if new_code in holding_codes:
            continue
        holding_codes.add(new_code)
        print(f"[INFO] 新增持仓标: {new_code}, 目标比例: {target_ratio:.3f}")
        update_symbol_data(new_code, include_history=True, verbose=False)
        new_codes.append(new_code)

    if not new_codes:
        return

    update_all_states()
    for code in new_codes:
        update_symbol_data(code, include_history=False, verbose=False)
        state = code_states[code]
        target_ratio = _infer_target_ratio(
            code,
            cash=float(state.get("cash", 0.0)),
            shares=int(state.get("shares", 0)),
            locked_today=int(state.get("locked_today", 0)),
        )
        code_states[code]["target_ratio"] = target_ratio
        trade_code(code, target_ratio, allow_buy=True)


def _extract_all_config_codes() -> list[str]:
    infer_codes = [c for c in cfg.get("infer_codes", []) if isinstance(c, str) and c]
    train_codes = [c for c in cfg.get("train_codes", []) if isinstance(c, str) and c]
    return list(dict.fromkeys([*trade_codes_in_cfg, *infer_codes, *train_codes]))


def infer_full_universe_snapshot() -> None:
    full_codes = _extract_all_config_codes()
    if not full_codes:
        print("[WARNING] run_full_infer 开启，但未配置可推理标的")
        return

    print("\n" + "=" * 80)
    print("[INFO] 全标的空仓推理快照（按打分降序）")
    print("=" * 80)

    init_cash = float(cfg["env"]["initial_cash"])
    ranked: list[tuple[str, float]] = []
    failed: list[str] = []
    for code in full_codes:
        target_ratio = _infer_target_ratio(code, cash=init_cash, shares=0, locked_today=0)
        if target_ratio is None:
            failed.append(code)
            continue
        ranked.append((code, float(target_ratio)))

    ranked.sort(key=lambda x: x[1], reverse=True)
    print(f"{'排名':<6} {'代码':<12} {'中文名':<12} {'目标持仓':<10}")
    for rank, (code, target_ratio) in enumerate(ranked, start=1):
        print(f"{rank:<6d} {code:<12} {get_code_name(code):<12} {target_ratio:<10.3f}")

    print(f"[INFO] 推理完成: 成功 {len(ranked)} / 总计 {len(full_codes)}")
    if failed:
        print(f"[WARNING] 以下标的推理失败: {failed}")
    print("=" * 80 + "\n")


def _wait_orders_settle() -> None:
    if ORDER_SETTLE_INITIAL_WAIT_SECONDS > 0:
        time.sleep(ORDER_SETTLE_INITIAL_WAIT_SECONDS)
    waited = ORDER_SETTLE_INITIAL_WAIT_SECONDS
    max_wait = ORDER_SETTLE_MAX_WAIT_SECONDS
    while waited < max_wait:
        try:
            if not refresh_user_data():
                break
            today_entrust = user.today_entrusts
            if not today_entrust:
                break
            has_pending = False
            for item in today_entrust:
                status = item.get("委托状态", item.get("状态说明", ""))
                if status != "已成交":
                    has_pending = True
                    break
            if not has_pending:
                break
        except Exception:
            break
        time.sleep(1)
        waited += 1


def _in_trading_time(now: pd.Timestamp, test_mode: bool) -> bool:
    current_time = now.time()
    now_td = pd.Timedelta(
        hours=current_time.hour,
        minutes=current_time.minute,
        seconds=current_time.second,
    )
    if test_mode:
        return True
    for start_td, end_td in TRADING_SESSIONS:
        if start_td <= now_td <= end_td:
            return True
    return False


def _init_holdings_from_broker() -> None:
    positions = get_all_positions()
    print(f"[INFO] 从券商读取到 {len(positions)} 条持仓记录")
    for item in positions:
        code_full = _position_code(item)
        if not code_full:
            continue
        pos_info = get_position_info(code_full, positions_cache=positions)
        if pos_info and int(pos_info["shares"]) > 0:
            holding_codes.add(code_full)
            code_states[code_full]["shares"] = int(pos_info["shares"])
            code_states[code_full]["locked_today"] = max(
                0,
                int(pos_info["shares"]) - int(pos_info["available_shares"]),
            )
            print(f"[INFO] 加载实际持仓: {code_full}, 股数={pos_info['shares']}")


def trade_loop() -> None:
    global today

    ensure_model_ready()
    init_runtime_state()
    init_user()
    _init_holdings_from_broker()

    run_full_infer = get_run_full_infer()
    if run_full_infer:
        init_codes = list(set(_extract_all_config_codes()) | set(candidate_codes) | set(holding_codes))
        update_all_codes_data(init_codes, "初始化：更新全标的数据")
    else:
        init_codes = list(set(candidate_codes) | set(holding_codes))
        update_all_codes_data(init_codes, "初始化：更新候选标与持仓标数据")
    update_all_states()

    print(f"[INFO] 初始化完成：候选池={len(candidate_codes)}，当前持仓标数={len(holding_codes)}")
    if run_full_infer:
        infer_full_universe_snapshot()

    last_bar_key = None

    while True:
        now = pd.Timestamp.now()
        test_mode = get_test_mode()
        if _in_trading_time(now, test_mode):
            minute_bucket = (now.minute // BAR_INTERVAL_MINUTES) * BAR_INTERVAL_MINUTES
            bar_key = now.strftime("%Y%m%d%H") + f"{minute_bucket:02d}"
            if bar_key != last_bar_key:
                last_bar_key = bar_key
                print(f"\n[========== 新{BAR_INTERVAL_MINUTES}分钟Bar: {bar_key} ==========]")

                cancel_all_entrusts()
                update_all_states()
                cleared = clear_zero_position_codes()
                if cleared > 0:
                    print(f"[INFO] 已清除 {cleared} 个零持仓标")

                for code in list(holding_codes):
                    is_candidate = code in candidate_codes
                    update_symbol_data(code, include_history=False, verbose=False)

                    if (not is_candidate) and (not has_trained_model(code)):
                        print(f"[INFO] {code} 非候选且无模型，跳过推理与交易")
                        continue

                    state = code_states[code]
                    target_ratio = _infer_target_ratio(
                        code,
                        cash=float(state.get("cash", 0.0)),
                        shares=int(state.get("shares", 0)),
                        locked_today=int(state.get("locked_today", 0)),
                    )
                    state["target_ratio"] = target_ratio
                    status = "候选标" if is_candidate else "非候选标(仅可卖出)"
                    if target_ratio is not None:
                        print(
                            f"[{code}] infer目标={target_ratio:.3f} "
                            f"实际比例={state['actual_ratio']:.3f} 状态={status}"
                        )
                    else:
                        print(f"[{code}] infer失败，状态={status}")

                    trade_code(code, target_ratio, allow_buy=is_candidate)

                _wait_orders_settle()
                print("[INFO] 开始调仓检查")
                adjust_portfolio()
                print(f"[INFO] 调仓检查完成，当前持仓标数: {len(holding_codes)}/{get_topk()}")
            else:
                _sleep(IN_BAR_POLL_SLEEP_SECONDS)
        else:
            _sleep(OUT_OF_TRADING_SLEEP_SECONDS)

        day = pd.to_datetime("today").date()
        if day > today:
            today = day
            print(f"[INFO] 新的一天: {today}")
            if get_run_full_infer():
                day_codes = list(set(_extract_all_config_codes()) | set(candidate_codes) | set(holding_codes))
                update_all_codes_data(day_codes, "跨日：更新全标的数据")
            else:
                day_codes = list(set(candidate_codes) | set(holding_codes))
                update_all_codes_data(day_codes, "跨日：更新候选标与持仓标数据")
            for code in list(holding_codes):
                code_states[code]["locked_today"] = 0
            if get_run_full_infer():
                infer_full_universe_snapshot()


def main() -> None:
    trade_loop()


if __name__ == "__main__":
    main()

