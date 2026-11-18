"""内置因子库。direction=1 表示因子值越大越好，-1 表示越小越好。"""
from __future__ import annotations

BUILTIN_FACTORS: list[dict] = [
    {
        "id": "momentum_20",
        "name": "momentum_20",
        "label": "动量 20 日",
        "category": "动量/反转",
        "description": "过去 20 个交易日复权收益，捕捉中期趋势延续。",
        "expression": "close / ts_delay(close, 20) - 1",
        "direction": 1,
    },
    {
        "id": "momentum_60",
        "name": "momentum_60",
        "label": "动量 60 日",
        "category": "动量/反转",
        "description": "过去 60 个交易日收益，经典中期动量。",
        "expression": "close / ts_delay(close, 60) - 1",
        "direction": 1,
    },
    {
        "id": "reversal_5",
        "name": "reversal_5",
        "label": "短期反转 5 日",
        "category": "动量/反转",
        "description": "过去 5 日收益取负，捕捉短期超跌反弹。",
        "expression": "-(close / ts_delay(close, 5) - 1)",
        "direction": 1,
    },
    {
        "id": "volatility_20",
        "name": "volatility_20",
        "label": "低波动 20 日",
        "category": "波动率",
        "description": "20 日日收益波动率取负，低波动异象。",
        "expression": "-ts_std(returns, 20)",
        "direction": 1,
    },
    {
        "id": "bias_20",
        "name": "bias_20",
        "label": "均线乖离 20 日",
        "category": "技术指标",
        "description": "收盘价相对 20 日均线的偏离度。",
        "expression": "(close - ts_mean(close, 20)) / ts_mean(close, 20)",
        "direction": 1,
    },
    {
        "id": "volume_ratio",
        "name": "volume_ratio",
        "label": "量比（20 日）",
        "category": "量价",
        "description": "当日成交量相对 20 日均量的倍数，衡量放量程度。",
        "expression": "volume / ts_mean(volume, 20)",
        "direction": 1,
    },
    {
        "id": "turnover_stability",
        "name": "turnover_stability",
        "label": "成交额稳定性",
        "category": "量价",
        "description": "成交额相对长期均值的波动率取负，偏好稳定资金关注。",
        "expression": "-ts_std(volume / ts_mean(volume, 60), 20)",
        "direction": 1,
    },
    {
        "id": "amihud",
        "name": "amihud",
        "label": "非流动性 (Amihud)",
        "category": "流动性",
        "description": "单位成交额带来的价格冲击，数值越低流动性越好。",
        "expression": "ts_mean(abs(returns) / (amount / 1e8 + 1e-6), 20)",
        "direction": -1,
    },
    {
        "id": "price_position_60",
        "name": "price_position_60",
        "label": "价格分位 60 日",
        "category": "技术指标",
        "description": "收盘价在 60 日高低区间中的位置，接近 1 表示突破新高。",
        "expression": "(close - ts_min(low, 60)) / (ts_max(high, 60) - ts_min(low, 60) + 1e-9)",
        "direction": 1,
    },
    {
        "id": "rsi_14",
        "name": "rsi_14",
        "label": "RSI 14",
        "category": "技术指标",
        "description": "相对强弱指标，衡量近期上涨动能占比。",
        "expression": "100 - 100 / (1 + ts_mean(max(ts_delta(close, 1), 0), 14) / (ts_mean(abs(ts_delta(close, 1)), 14) + 1e-9))",
        "direction": 1,
    },
    {
        "id": "macd",
        "name": "macd",
        "label": "MACD 快慢线",
        "category": "技术指标",
        "description": "12 日均线相对 26 日均线的强弱，趋势跟踪。",
        "expression": "ts_mean(close, 12) / ts_mean(close, 26) - 1",
        "direction": 1,
    },
    {
        "id": "amplitude_20",
        "name": "amplitude_20",
        "label": "低振幅 20 日",
        "category": "波动率",
        "description": "日均振幅取负，偏好走势平稳的标的。",
        "expression": "-ts_mean((high - low) / ts_delay(close, 1), 20)",
        "direction": 1,
    },
    {
        "id": "volume_price_corr",
        "name": "volume_price_corr",
        "label": "量价相关 20 日",
        "category": "量价",
        "description": "成交量与收益的 20 日相关系数，衡量量价配合度。",
        "expression": "ts_corr(volume, returns, 20)",
        "direction": 1,
    },
    {
        "id": "cs_momentum_rank",
        "name": "cs_momentum_rank",
        "label": "截面动量排名",
        "category": "动量/反转",
        "description": "20 日收益在全市场中的分位排名，行业中性化前的标准动量。",
        "expression": "rank(close / ts_delay(close, 20) - 1)",
        "direction": 1,
    },
]


def all_factors() -> list[dict]:
    return [
        {**f, "is_custom": False}
        for f in BUILTIN_FACTORS
    ]


def find_factor(factor_id: str) -> dict | None:
    for f in BUILTIN_FACTORS:
        if f["id"] == factor_id or f["name"] == factor_id:
            return f
    return None
