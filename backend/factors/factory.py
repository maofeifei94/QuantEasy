"""因子工厂：批量生成候选因子并按 ICIR 排序，辅助因子挖掘。"""
from __future__ import annotations

CANDIDATES: list[dict] = []


def _add(name: str, label: str, category: str, expression: str) -> None:
    CANDIDATES.append({
        "name": name, "label": label, "category": category,
        "expression": expression,
    })


for _n in (5, 10, 20, 60):
    _add(f"fac_mom_{_n}", f"动量 {_n} 日", "动量/反转", f"close / ts_delay(close, {_n}) - 1")
    _add(f"fac_rev_{_n}", f"反转 {_n} 日", "动量/反转", f"-(close / ts_delay(close, {_n}) - 1)")
    _add(f"fac_vol_{_n}", f"低波动 {_n} 日", "波动率", f"-ts_std(returns, {_n})")
    _add(f"fac_bias_{_n}", f"乖离 {_n} 日", "技术指标",
         f"close / ts_mean(close, {_n}) - 1")
    _add(f"fac_vr_{_n}", f"量比 {_n} 日", "量价", f"volume / ts_mean(volume, {_n})")

_add("fac_rsi_14", "RSI 14", "技术指标",
     "100 - 100 / (1 + ts_mean(max(ts_delta(close, 1), 0), 14) / (ts_mean(abs(ts_delta(close, 1)), 14) + 1e-9))")
_add("fac_macd", "MACD 快慢线", "技术指标",
     "ts_mean(close, 12) / ts_mean(close, 26) - 1")
_add("fac_price_pos_20", "价格分位 20 日", "技术指标",
     "(close - ts_min(low, 20)) / (ts_max(high, 20) - ts_min(low, 20) + 1e-9)")
_add("fac_amp_20", "低振幅 20 日", "波动率",
     "-ts_mean((high - low) / ts_delay(close, 1), 20)")
_add("fac_skew_20", "收益偏度 20 日", "波动率", "ts_skew(returns, 20)")
_add("fac_kurt_20", "收益峰度 20 日", "波动率", "-ts_kurt(returns, 20)")
_add("fac_decay_mom_20", "衰减动量 20 日", "动量/反转",
     "close / decay_linear(close, 20) - 1")
_add("fac_tsrank_close_20", "价格时序分位 20 日", "技术指标", "ts_rank(close, 20)")
_add("fac_corr_vp_20", "量价相关 20 日", "量价", "ts_corr(volume, returns, 20)")
_add("fac_turnover_stab", "成交额稳定性", "量价",
     "-ts_std(volume / ts_mean(volume, 60), 20)")
_add("fac_amihud", "Amihud 非流动性", "流动性",
     "-(ts_mean(abs(returns) / (amount / 1e8 + 1e-6), 20))")
_add("fac_zscore_rev_5", "标准化反转 5 日", "动量/反转",
     "zscore(returns) * -1 - ts_zscore(returns, 5)")
