"""合成行情数据生成器。

当 miniQMT(xtquant) 不可用时，用于本地演示与测试：生成带行业结构、
动量效应与波动聚集的日线数据，保证因子与回测流程开箱即用。
"""
from __future__ import annotations

import datetime as dt
from typing import Callable

import numpy as np

INDUSTRIES = [
    ("半导体", 0.00045, 0.026),
    ("新能源", 0.00040, 0.024),
    ("医药生物", 0.00025, 0.019),
    ("银行", 0.00012, 0.010),
    ("食品饮料", 0.00028, 0.016),
    ("国防军工", 0.00022, 0.023),
    ("计算机", 0.00035, 0.027),
    ("汽车", 0.00030, 0.020),
]

NAME_CHARS = list("华鑫瑞恒远达创联智数云科泰盛宏安信捷腾光电微芯药芯源海新能源")
NAME_SUFFIX = ["科技", "电子", "医药", "能源", "智造", "股份", "科技", "数据", "新材", "动力"]


def _trading_days(start: str, end: str) -> list[str]:
    d0 = dt.date.fromisoformat(start)
    d1 = dt.date.fromisoformat(end)
    holidays = set()
    for year in range(d0.year, d1.year + 1):
        for md in ((1, 1), (1, 2), (5, 1), (5, 2), (5, 3), (10, 1), (10, 2),
                   (10, 3), (10, 4), (10, 5)):
            holidays.add(dt.date(year, *md))
    days = []
    cur = d0
    while cur <= d1:
        if cur.weekday() < 5 and cur not in holidays:
            days.append(cur.isoformat())
        cur += dt.timedelta(days=1)
    return days


def _stock_names(n: int, rng: np.random.Generator) -> list[str]:
    seen: set[str] = set()
    names = []
    while len(names) < n:
        name = "".join(rng.choice(NAME_CHARS, size=2)) + NAME_SUFFIX[len(names) % len(NAME_SUFFIX)]
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def generate(
    start: str = "2023-01-03",
    end: str = "2026-11-17",
    n_stocks: int = 150,
    seed: int = 7,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[list[dict], list[tuple]]:
    rng = np.random.default_rng(seed)
    dates = _trading_days(start, end)
    names = _stock_names(n_stocks, rng)
    T = len(dates)

    # 市场因子：慢牛 + 两段回撤，波动聚集
    vol = np.abs(rng.normal(0.010, 0.003, T)) + 0.004
    mkt = rng.normal(0.00035, 1.0, T) * vol
    mkt[150:200] -= 0.0025
    mkt[520:585] -= 0.0022

    stocks: list[dict] = []
    bars: list[tuple] = []
    for i in range(n_stocks):
        code = f"{'600' if i % 2 == 0 else '000'}{100 + i:03d}"
        ind_name, ind_drift, ind_vol = INDUSTRIES[i % len(INDUSTRIES)]
        beta = rng.uniform(0.65, 1.35)
        idio_vol = ind_vol * rng.uniform(0.8, 1.4)
        drift = ind_drift * rng.uniform(0.2, 2.2)
        # 行业共同冲击 + 个股特质，带轻微动量
        ind_shock = rng.normal(0, ind_vol * 0.5, T)
        idio = rng.normal(0, idio_vol, T)
        ret = drift + beta * mkt + 0.35 * ind_shock + idio
        ret = np.clip(ret, -0.098, 0.098)
        ret = ret + 0.045 * np.concatenate([[0.0], ret[:-1]])

        close = 8.0 * np.exp(np.cumsum(ret))
        prev = np.concatenate([[close[0] / (1 + ret[0])], close[:-1]])
        open_ = prev * (1 + rng.normal(0, 0.006, T))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, T)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, T)))
        base_vol = rng.uniform(4e6, 6e7) * (1 + 0.6 * (i % 5))
        volume = base_vol * np.exp(rng.normal(0, 0.35, T)) * (1 + 6 * np.abs(ret))
        amount = volume * (open_ + close) / 2

        stocks.append({
            "code": code,
            "name": names[i],
            "industry": ind_name,
            "market": "SH" if code.startswith("6") else "SZ",
            "list_date": "2015-06-01",
        })
        for t in range(T):
            bars.append((
                code, dates[t],
                round(float(open_[t]), 2), round(float(high[t]), 2),
                round(float(low[t]), 2), round(float(close[t]), 2),
                round(float(volume[t]), 0), round(float(amount[t]), 0),
            ))
        if progress and (i % 15 == 0 or i == n_stocks - 1):
            progress((i + 1) / n_stocks, f"生成合成行情 {i + 1}/{n_stocks}")

    return stocks, bars
