"""把 SQLite 日线数据组装成 (T, N) 对齐面板。"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import database


class PanelError(ValueError):
    pass


def build_panel(
    codes: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    stats = database.stats()
    start = start or stats["start_date"]
    end = end or stats["end_date"]
    if not start or not end:
        raise PanelError("数据库暂无行情数据，请先同步数据")

    rows = database.get_bars_bulk(codes, start, end)
    if not rows:
        raise PanelError("所选区间/股票没有行情数据")

    dates = sorted({r["date"] for r in rows})
    if codes:
        present = {r["code"] for r in rows}
        code_list = [c for c in codes if c in present]
    else:
        code_list = sorted({r["code"] for r in rows})
    if not code_list:
        raise PanelError("所选股票在该区间内无数据")

    T, N = len(dates), len(code_list)
    date_idx = {d: i for i, d in enumerate(dates)}
    code_idx = {c: j for j, c in enumerate(code_list)}
    fields = ("open", "high", "low", "close", "volume", "amount")
    arrays = {f: np.full((T, N), np.nan, dtype=np.float64) for f in fields}
    tradable = np.zeros((T, N), dtype=bool)

    for r in rows:
        i = date_idx[r["date"]]
        j = code_idx[r["code"]]
        for f in fields:
            v = r.get(f)
            if v is not None:
                arrays[f][i, j] = float(v)
        c = r.get("close")
        if c is not None and np.isfinite(c):
            tradable[i, j] = True

    # 停牌/缺失价格前值填充，保证持仓收益连续
    for f in ("open", "high", "low", "close"):
        arr = arrays[f]
        for t in range(1, T):
            mask = ~np.isfinite(arr[t]) & np.isfinite(arr[t - 1])
            if mask.any():
                arr[t, mask] = arr[t - 1, mask]
    for f in ("volume", "amount"):
        arrays[f] = np.nan_to_num(arrays[f], nan=0.0)

    close = arrays["close"]
    prev = np.vstack([np.full((1, N), np.nan), close[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = close / prev - 1.0
    returns[~np.isfinite(returns)] = np.nan

    # 有效交易日：有成交且价格存在
    valid = tradable & (arrays["volume"] > 0)
    arrays["valid"] = valid
    arrays["tradable"] = tradable
    arrays["returns"] = returns
    arrays["vwap"] = np.where(
        arrays["volume"] > 0, arrays["amount"] / np.maximum(arrays["volume"], 1e-9), np.nan
    )
    arrays["dates"] = dates
    arrays["codes"] = code_list
    arrays["names"] = database.get_stock_names(code_list)
    return arrays
