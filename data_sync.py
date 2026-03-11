from __future__ import annotations

from typing import Iterable


def refresh_symbol_data(
    symbols: Iterable[str],
    *,
    include_history: bool = True,
    verbose: bool = False,
) -> None:
    """
    统一刷新股票数据：
    1) 可选更新历史数据（baostock）
    2) 尝试更新复权比例
    3) 尝试补齐 infer 最新数据（akshare）
    """
    from x_fin.x_bs import update_data_from_bs
    from x_fin.x_ak import set_ratio, update_infer_data_from_ak

    uniq_symbols = list(dict.fromkeys(symbols))
    for symbol in uniq_symbols:
        if include_history:
            try:
                try:
                    update_data_from_bs(symbol, verbose=verbose)
                except TypeError:
                    update_data_from_bs(symbol)
            except Exception as exc:
                raise RuntimeError(f"更新历史数据失败: {symbol}, error={exc}") from exc

        try:
            set_ratio(symbol)
        except Exception as exc:
            raise RuntimeError(f"更新复权比例失败: {symbol}, error={exc}") from exc

        try:
            update_infer_data_from_ak(symbol)
        except Exception as exc:
            raise RuntimeError(f"更新 infer 数据失败: {symbol}, error={exc}") from exc

