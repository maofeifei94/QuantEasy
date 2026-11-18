"""因子预处理：去极值、行业中性化、截面标准化。

风格对齐主流量化平台（米筐 / 聚宽 / Qlib）的标准多因子流程：
MAD 去极值 → 行业均值中性化 → 截面 z-score。
"""
from __future__ import annotations

import jax.numpy as jnp

from .ops import _as2d, cs_zscore


def winsorize_mad(x, k: float = 3.0):
    """截面 MAD 去极值：median ± k * 1.4826 * MAD。"""
    x = _as2d(x)
    valid = jnp.isfinite(x)
    cnt = jnp.sum(valid, axis=1, keepdims=True)
    pos = jnp.maximum((cnt.astype(jnp.int32) - 1) // 2, 0)
    s = jnp.sort(jnp.where(valid, x, jnp.inf), axis=1)
    med = jnp.take_along_axis(s, pos, axis=1)
    dev = jnp.where(valid, jnp.abs(x - med), jnp.inf)
    ds = jnp.sort(dev, axis=1)
    mad = jnp.take_along_axis(ds, pos, axis=1)
    scale = 1.4826 * mad
    lower = med - k * scale
    upper = med + k * scale
    clipped = jnp.clip(x, lower, upper)
    out = jnp.where(valid & (cnt >= 5) & (scale > 1e-12), clipped, jnp.where(valid, x, jnp.nan))
    return out


def neutralize_industry(x, industry_idx, n_industries: int):
    """按行业分组减去组内均值（对每行独立执行）。"""
    x = _as2d(x)
    valid = jnp.isfinite(x)
    filled = jnp.where(valid, x, 0.0)
    groups = jnp.arange(max(int(n_industries), 1))
    onehot = (industry_idx[None, :] == groups[:, None]).astype(jnp.float64)
    sums = filled @ onehot.T
    counts = valid.astype(jnp.float64) @ onehot.T
    means = sums / jnp.maximum(counts, 1.0)
    group_mean = jnp.take(means, industry_idx, axis=1)
    stock_counts = jnp.take(counts, industry_idx, axis=1)
    out = x - group_mean
    return jnp.where(valid & (stock_counts > 0), out, jnp.nan)


def apply_preprocess(x, panel: dict, options: dict | None,
                     force_standardize: bool = False):
    """options: {winsorize, neutralize, standardize, winsor_k}"""
    options = options or {}
    out = _as2d(x)
    if options.get("winsorize"):
        out = winsorize_mad(out, float(options.get("winsor_k", 3.0)))
    if options.get("neutralize") and panel.get("n_industries"):
        out = neutralize_industry(out, jnp.asarray(panel["industry_idx"]), panel["n_industries"])
    if options.get("standardize") or force_standardize:
        out = cs_zscore(out)
    return out
