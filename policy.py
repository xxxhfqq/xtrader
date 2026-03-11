from __future__ import annotations

from typing import Any

from extractor_tiny_transformer import TinyTransformerFeatureExtractor
from features import OHLCV_COLUMNS


ACCOUNT_FEATURE_DIM = 7


def build_market_feature_cols(indicators: list[str]) -> list[str]:
    return [*OHLCV_COLUMNS, *indicators]


def build_policy_kwargs(cfg: dict[str, Any], market_feature_dim: int) -> dict[str, Any]:
    feat_cfg = cfg["model"]["features_extractor"]
    accel_cfg = cfg["model"]["acceleration"]
    head_cfg = cfg["model"]["policy_head"]

    return {
        "features_extractor_class": TinyTransformerFeatureExtractor,
        "features_extractor_kwargs": {
            "window_size": int(feat_cfg["window_size"]),
            "market_feature_dim": int(market_feature_dim),
            "account_feature_dim": ACCOUNT_FEATURE_DIM,
            "d_model": int(feat_cfg["d_model"]),
            "n_heads": int(feat_cfg["n_heads"]),
            "n_layers": int(feat_cfg["n_layers"]),
            "ffn_dim": int(feat_cfg["ffn_dim"]),
            "dropout": float(feat_cfg["dropout"]),
            "pooling": str(feat_cfg.get("pooling", "last")),
            "use_sdpa": bool(accel_cfg.get("use_sdpa", True)),
        },
        "net_arch": {
            "pi": list(head_cfg["pi"]),
            "vf": list(head_cfg["vf"]),
        },
    }

