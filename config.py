from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _require_key(obj: dict[str, Any], key: str, where: str) -> Any:
    if key not in obj:
        raise KeyError(f"{where} 缺少必填字段: {key}")
    return obj[key]


def _require_type(value: Any, expected: type | tuple[type, ...], name: str) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{name} 类型错误, 期望 {expected}, 实际 {type(value)}")


def _as_positive_int(value: Any, name: str) -> int:
    _require_type(value, int, name)
    if value <= 0:
        raise ValueError(f"{name} 必须为正整数")
    return value


def _as_non_negative_int(value: Any, name: str) -> int:
    _require_type(value, int, name)
    if value < 0:
        raise ValueError(f"{name} 必须为非负整数")
    return value


def _as_non_negative_float(value: Any, name: str) -> float:
    _require_type(value, (int, float), name)
    out = float(value)
    if out < 0:
        raise ValueError(f"{name} 必须为非负数")
    return out


def _as_time_str(value: Any, name: str) -> str:
    _require_type(value, str, name)
    text = value.strip()
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"{name} 时间格式错误，需为 HH:MM 或 HH:MM:SS")
    try:
        hh = int(parts[0])
        mm = int(parts[1])
        ss = int(parts[2]) if len(parts) == 3 else 0
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"{name} 时间格式错误: {value}") from exc
    if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
        raise ValueError(f"{name} 时间取值越界: {value}")
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _time_seconds(time_text: str) -> int:
    hh, mm, ss = [int(x) for x in time_text.split(":")]
    return hh * 3600 + mm * 60 + ss


def _apply_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = cfg.setdefault("paths", {})
    paths.setdefault("data_dir", "data")
    paths.setdefault("model_dir", "trained_model")
    paths.setdefault("log_dir", "logs")

    env = cfg.setdefault("env", {})
    env.setdefault("initial_cash", 100000.0)
    env.setdefault("buy_fee_rate", 0.0001)
    env.setdefault("sell_fee_rate", 0.0006)
    env.setdefault("lot_size", 100)
    env.setdefault("reward_type", "log_return")
    env.setdefault("reward_epsilon", 1e-8)

    train = cfg.setdefault("train", {})
    train.setdefault("seed", 42)
    train.setdefault("device", "cpu")
    train.setdefault("validate_every_rollouts", 100)
    train.setdefault("update_data_before_train", True)
    train.setdefault("update_data_before_infer", True)

    model = cfg.setdefault("model", {})
    feat = model.setdefault("features_extractor", {})
    feat.setdefault("architecture", "decoder_only")
    feat.setdefault("pooling", "last")
    feat.setdefault("dropout", 0.1)

    accel = model.setdefault("acceleration", {})
    accel.setdefault("use_sdpa", True)
    accel.setdefault("use_torch_compile", False)
    accel.setdefault("torch_compile_mode", "reduce-overhead")
    accel.setdefault("torch_compile_fullgraph", False)
    accel.setdefault("torch_compile_dynamic", False)

    save = cfg.setdefault("save", {})
    save.setdefault("best_model_name", "best_model")
    save.setdefault("best_log_file", "best_model_log.jsonl")
    save.setdefault("eval_csv_file", "eval_metrics.csv")

    trade = cfg.setdefault("trade", {})
    trade.setdefault("min_fee", 5.0)
    trade.setdefault("open_full_threshold", 0.7)
    trade.setdefault("flat_threshold", 0.3)
    trade.setdefault("open_candidate_threshold", 0.3)
    trade.setdefault("rebalance_diff_threshold", 0.1)
    trade.setdefault("min_trade_value", 100.0)
    trade.setdefault("bar_interval_minutes", 5)
    trade.setdefault("order_settle_initial_wait_seconds", 3)
    trade.setdefault("order_settle_max_wait_seconds", 10)
    trade.setdefault("in_bar_poll_sleep_seconds", 0.1)
    trade.setdefault("out_of_trading_sleep_seconds", 1.0)
    trade.setdefault("xiadan_path", r"C:\\同花顺软件\\同花顺\\xiadan.exe")
    trade.setdefault(
        "trading_sessions",
        [
            ["09:35:00", "11:30:00"],
            ["13:00:00", "15:00:00"],
        ],
    )

    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    _require_type(cfg, dict, "config")

    train_codes = _require_key(cfg, "train_codes", "config")
    infer_codes = _require_key(cfg, "infer_codes", "config")
    _require_type(train_codes, list, "train_codes")
    _require_type(infer_codes, list, "infer_codes")
    if not train_codes:
        raise ValueError("train_codes 不能为空")

    features = _require_key(cfg, "features", "config")
    _require_type(features, dict, "features")
    indicators = _require_key(features, "indicators", "features")
    _require_type(indicators, list, "features.indicators")
    if not indicators:
        raise ValueError("features.indicators 不能为空")

    sampler = _require_key(cfg, "sampler", "config")
    _require_type(sampler, dict, "sampler")
    rollout_steps = _as_positive_int(
        _require_key(sampler, "rollout_steps", "sampler"), "sampler.rollout_steps"
    )
    infer_window_bars = _as_positive_int(
        _require_key(sampler, "infer_window_bars", "sampler"), "sampler.infer_window_bars"
    )
    _as_non_negative_int(
        _require_key(sampler, "extra_warmup_bars", "sampler"), "sampler.extra_warmup_bars"
    )
    _as_non_negative_int(
        _require_key(sampler, "next_open_guard_bars", "sampler"),
        "sampler.next_open_guard_bars",
    )

    warmup_map = _require_key(sampler, "indicator_warmup_bars", "sampler")
    _require_type(warmup_map, dict, "sampler.indicator_warmup_bars")
    if not warmup_map:
        raise ValueError("sampler.indicator_warmup_bars 不能为空")
    for k, v in warmup_map.items():
        _require_type(k, str, "sampler.indicator_warmup_bars 的 key")
        _as_non_negative_int(v, f"sampler.indicator_warmup_bars.{k}")

    model = _require_key(cfg, "model", "config")
    _require_type(model, dict, "model")
    feat = _require_key(model, "features_extractor", "model")
    _require_type(feat, dict, "model.features_extractor")
    arch = _require_key(feat, "architecture", "model.features_extractor")
    if arch != "decoder_only":
        raise ValueError("目前仅支持 model.features_extractor.architecture=decoder_only")

    window_size = _as_positive_int(
        _require_key(feat, "window_size", "model.features_extractor"),
        "model.features_extractor.window_size",
    )
    _as_positive_int(_require_key(feat, "d_model", "model.features_extractor"), "d_model")
    _as_positive_int(_require_key(feat, "n_heads", "model.features_extractor"), "n_heads")
    _as_positive_int(_require_key(feat, "n_layers", "model.features_extractor"), "n_layers")
    _as_positive_int(_require_key(feat, "ffn_dim", "model.features_extractor"), "ffn_dim")

    if window_size != infer_window_bars:
        raise ValueError(
            "当前实现要求 model.features_extractor.window_size "
            "与 sampler.infer_window_bars 一致"
        )

    ppo = _require_key(cfg, "ppo", "config")
    _require_type(ppo, dict, "ppo")
    n_steps = _as_positive_int(_require_key(ppo, "n_steps", "ppo"), "ppo.n_steps")
    _as_positive_int(_require_key(ppo, "batch_size", "ppo"), "ppo.batch_size")
    _as_positive_int(_require_key(ppo, "n_epochs", "ppo"), "ppo.n_epochs")
    if n_steps != rollout_steps:
        raise ValueError("当前实现要求 ppo.n_steps 与 sampler.rollout_steps 一致")

    train = _require_key(cfg, "train", "config")
    _require_type(train, dict, "train")
    _as_positive_int(
        _require_key(train, "validate_every_rollouts", "train"),
        "train.validate_every_rollouts",
    )
    _as_positive_int(_require_key(train, "total_timesteps", "train"), "train.total_timesteps")

    env = _require_key(cfg, "env", "config")
    _require_type(env, dict, "env")
    reward_type = _require_key(env, "reward_type", "env")
    _require_type(reward_type, str, "env.reward_type")
    if reward_type not in {"log_return", "equity_delta"}:
        raise ValueError("env.reward_type 仅支持: log_return / equity_delta")
    reward_epsilon = _require_key(env, "reward_epsilon", "env")
    _require_type(reward_epsilon, (int, float), "env.reward_epsilon")
    if float(reward_epsilon) <= 0:
        raise ValueError("env.reward_epsilon 必须 > 0")

    trade = _require_key(cfg, "trade", "config")
    _require_type(trade, dict, "trade")
    for key in (
        "min_fee",
        "open_full_threshold",
        "flat_threshold",
        "open_candidate_threshold",
        "rebalance_diff_threshold",
        "min_trade_value",
        "order_settle_initial_wait_seconds",
        "order_settle_max_wait_seconds",
        "in_bar_poll_sleep_seconds",
        "out_of_trading_sleep_seconds",
    ):
        _require_type(_require_key(trade, key, "trade"), (int, float), f"trade.{key}")
    _as_positive_int(_require_key(trade, "bar_interval_minutes", "trade"), "trade.bar_interval_minutes")
    xiadan_path = _require_key(trade, "xiadan_path", "trade")
    _require_type(xiadan_path, str, "trade.xiadan_path")
    if not str(xiadan_path).strip():
        raise ValueError("trade.xiadan_path 不能为空")
    if float(trade["min_fee"]) < 0:
        raise ValueError("trade.min_fee 必须 >= 0")
    _as_non_negative_float(trade["min_trade_value"], "trade.min_trade_value")
    _as_non_negative_float(
        trade["order_settle_initial_wait_seconds"],
        "trade.order_settle_initial_wait_seconds",
    )
    _as_non_negative_float(
        trade["order_settle_max_wait_seconds"],
        "trade.order_settle_max_wait_seconds",
    )
    _as_non_negative_float(
        trade["in_bar_poll_sleep_seconds"],
        "trade.in_bar_poll_sleep_seconds",
    )
    _as_non_negative_float(
        trade["out_of_trading_sleep_seconds"],
        "trade.out_of_trading_sleep_seconds",
    )
    if float(trade["order_settle_max_wait_seconds"]) < float(trade["order_settle_initial_wait_seconds"]):
        raise ValueError("trade.order_settle_max_wait_seconds 必须 >= trade.order_settle_initial_wait_seconds")
    for key in ("open_full_threshold", "flat_threshold", "open_candidate_threshold", "rebalance_diff_threshold"):
        v = float(trade[key])
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"trade.{key} 必须在 [0, 1] 之间")

    sessions = _require_key(trade, "trading_sessions", "trade")
    _require_type(sessions, list, "trade.trading_sessions")
    if not sessions:
        raise ValueError("trade.trading_sessions 不能为空")
    for idx, session in enumerate(sessions):
        _require_type(session, (list, tuple), f"trade.trading_sessions[{idx}]")
        if len(session) != 2:
            raise ValueError(f"trade.trading_sessions[{idx}] 必须为 [start, end]")
        start = _as_time_str(session[0], f"trade.trading_sessions[{idx}][0]")
        end = _as_time_str(session[1], f"trade.trading_sessions[{idx}][1]")
        if _time_seconds(start) >= _time_seconds(end):
            raise ValueError(f"trade.trading_sessions[{idx}] 要求 start < end")

    save = _require_key(cfg, "save", "config")
    _require_type(save, dict, "save")
    _require_type(_require_key(save, "eval_csv_file", "save"), str, "save.eval_csv_file")


def load_config(path: str | Path = "config.json") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    _require_type(cfg, dict, "config")
    cfg = _apply_defaults(cfg)
    validate_config(cfg)
    return cfg

