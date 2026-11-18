"""指数序列构建：无真实指数数据时，用股票池合成宽基指数代理。"""
from __future__ import annotations

import numpy as np


def proxy_indices(closes: np.ndarray, dates: list[str],
                  volumes: np.ndarray | None = None,
                  n_large: int = 100) -> list[dict]:
    """closes/volumes: (T, N)。返回沪深300/中证500 代理指数。

    以日均成交额排序，前 n_large 只作为大盘成分，其余作为中小盘成分；
    等权合成，基点 1000。
    """
    if closes.size == 0 or len(dates) < 2:
        return []
    T, N = closes.shape
    if volumes is not None and volumes.size == closes.size:
        score = np.nanmean(volumes, axis=0)
    else:
        score = np.nanmean(closes, axis=0)
    score = np.nan_to_num(score, nan=0.0)
    order = np.argsort(-score)
    n_large = max(10, min(int(n_large), max(10, N)))
    groups = [
        (order[:n_large], "000300.SH", "沪深300·演示"),
        (order[n_large:], "000905.SH", "中证500·演示"),
    ]
    out = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for members, code, name in groups:
            if len(members) < 5:
                continue
            sub = closes[:, members]
            rets = sub[1:] / sub[:-1] - 1.0
            valid = np.isfinite(rets)
            counts = valid.sum(axis=1)
            mean = np.where(valid, rets, 0.0).sum(axis=1) / np.maximum(counts, 1)
            mean = np.where(counts >= 3, mean, 0.0)
            nav = 1000.0 * np.cumprod(1.0 + mean)
            rows = [(code, dates[0], 1000.0)]
            rows += [
                (code, dates[i + 1], round(float(nav[i]), 2))
                for i in range(len(nav))
            ]
            out.append({"code": code, "name": name, "rows": rows})
    return out
