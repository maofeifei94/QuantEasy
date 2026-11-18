"""基于 JAX 的时序/截面因子算子。

所有算子以 (T, N) 面板数组为输入（T=交易日，N=股票），NaN 视为缺失值。
滚动窗口使用 cumsum / sliding_window_view 实现，可在 GPU/TPU 上自动加速。
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def _as2d(x) -> jnp.ndarray:
    arr = jnp.asarray(x, dtype=jnp.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _roll_sum(x: jnp.ndarray, n: int) -> jnp.ndarray:
    if n <= 0:
        raise ValueError("窗口长度必须为正整数")
    if x.shape[0] < n:
        return jnp.full_like(x, jnp.nan)
    pad = jnp.concatenate([jnp.zeros((n - 1,) + x.shape[1:], dtype=x.dtype), x], axis=0)
    cs = jnp.cumsum(pad, axis=0)
    head = jnp.concatenate([jnp.zeros((1,) + x.shape[1:], dtype=x.dtype), cs[:-n]], axis=0)
    return cs[n - 1:] - head


def _masked(x, n: int, minp: int | None = None):
    x = _as2d(x)
    valid = jnp.isfinite(x)
    filled = jnp.where(valid, x, 0.0)
    cnt = _roll_sum(valid.astype(jnp.float64), n)
    minp = max(2, n // 2) if minp is None else minp
    ok = (cnt >= minp) & jnp.isfinite(cnt)
    return filled, valid, cnt, ok


def ts_sum(x, n: int):
    filled, _, cnt, ok = _masked(x, n)
    s = _roll_sum(filled, n)
    return jnp.where(ok, s, jnp.nan)


def ts_mean(x, n: int):
    filled, _, cnt, ok = _masked(x, n)
    s = _roll_sum(filled, n)
    return jnp.where(ok, s / jnp.maximum(cnt, 1.0), jnp.nan)


def ts_std(x, n: int):
    filled, _, cnt, ok = _masked(x, n)
    mean = _roll_sum(filled, n) / jnp.maximum(cnt, 1.0)
    ex2 = _roll_sum(filled * filled, n) / jnp.maximum(cnt, 1.0)
    var = jnp.maximum(ex2 - mean * mean, 0.0)
    return jnp.where(ok & (cnt >= 3), jnp.sqrt(var), jnp.nan)


def ts_min(x, n: int):
    x = _as2d(x)
    if x.shape[0] < n:
        return jnp.full_like(x, jnp.nan)
    win = jnp.lib.stride_tricks.sliding_window_view(x, n, axis=0)
    win = jnp.where(jnp.isfinite(win), win, jnp.inf)
    res = jnp.min(win, axis=-1)
    res = jnp.where(jnp.isfinite(res), res, jnp.nan)
    pad = jnp.full((n - 1,) + x.shape[1:], jnp.nan)
    return jnp.concatenate([pad, res], axis=0)


def ts_max(x, n: int):
    x = _as2d(x)
    if x.shape[0] < n:
        return jnp.full_like(x, jnp.nan)
    win = jnp.lib.stride_tricks.sliding_window_view(x, n, axis=0)
    win = jnp.where(jnp.isfinite(win), win, -jnp.inf)
    res = jnp.max(win, axis=-1)
    res = jnp.where(jnp.isfinite(res), res, jnp.nan)
    pad = jnp.full((n - 1,) + x.shape[1:], jnp.nan)
    return jnp.concatenate([pad, res], axis=0)


def ts_rank(x, n: int):
    """当前值在过去 n 个观测中的分位（0~1）。"""
    x = _as2d(x)
    if x.shape[0] < n:
        return jnp.full_like(x, jnp.nan)
    win = jnp.lib.stride_tricks.sliding_window_view(x, n, axis=0)
    last = win[..., -1:]
    valid = jnp.isfinite(win)
    cnt = jnp.sum(valid, axis=-1)
    le = jnp.sum(jnp.where(valid & (win <= last), 1.0, 0.0), axis=-1)
    res = jnp.where(cnt >= max(2, n // 2), le / jnp.maximum(cnt, 1), jnp.nan)
    pad = jnp.full((n - 1,) + x.shape[1:], jnp.nan)
    return jnp.concatenate([pad, res], axis=0)


def ts_delay(x, n: int):
    x = _as2d(x)
    n = int(n)
    if n <= 0:
        return x
    if n >= x.shape[0]:
        return jnp.full_like(x, jnp.nan)
    pad = jnp.full((n,) + x.shape[1:], jnp.nan)
    return jnp.concatenate([pad, x[:-n]], axis=0)


def ts_delta(x, n: int):
    return _as2d(x) - ts_delay(x, n)


def ts_corr(x, y, n: int):
    x, y = _as2d(x), _as2d(y)
    valid = jnp.isfinite(x) & jnp.isfinite(y)
    xv = jnp.where(valid, x, 0.0)
    yv = jnp.where(valid, y, 0.0)
    cnt = _roll_sum(valid.astype(jnp.float64), n)
    sx = _roll_sum(xv, n)
    sy = _roll_sum(yv, n)
    sxy = _roll_sum(xv * yv, n)
    sxx = _roll_sum(xv * xv, n)
    syy = _roll_sum(yv * yv, n)
    c = jnp.maximum(cnt, 1.0)
    cov = sxy / c - (sx / c) * (sy / c)
    vx = jnp.maximum(sxx / c - (sx / c) ** 2, 0.0)
    vy = jnp.maximum(syy / c - (sy / c) ** 2, 0.0)
    denom = jnp.sqrt(vx * vy)
    ok = (cnt >= max(3, n // 2)) & (denom > 1e-12)
    return jnp.where(ok, cov / jnp.where(ok, denom, 1.0), jnp.nan)


def cs_rank(x):
    """截面排序分位（0~1），忽略 NaN。"""
    x = _as2d(x)
    valid = jnp.isfinite(x)
    filled = jnp.where(valid, x, jnp.inf)
    order = jnp.argsort(filled, axis=1)
    ranks = jnp.argsort(order, axis=1).astype(jnp.float64)
    cnt = jnp.sum(valid, axis=1, keepdims=True).astype(jnp.float64)
    out = (ranks + 0.5) / jnp.maximum(cnt, 1.0)
    return jnp.where(valid & (cnt > 1), out, jnp.nan)


def cs_zscore(x):
    x = _as2d(x)
    valid = jnp.isfinite(x)
    cnt = jnp.maximum(jnp.sum(valid, axis=1, keepdims=True), 1)
    mean = jnp.where(valid, x, 0.0).sum(axis=1, keepdims=True) / cnt
    diff = jnp.where(valid, x - mean, 0.0)
    std = jnp.sqrt((diff * diff).sum(axis=1, keepdims=True) / cnt)
    return jnp.where(valid & (std > 1e-12), diff / jnp.where(std > 1e-12, std, 1.0), jnp.nan)


def safe_div(a, b):
    return a / jnp.where(jnp.abs(b) > 1e-12, b, jnp.nan)


NAMESPACE = {
    "ts_mean": ts_mean,
    "ts_std": ts_std,
    "ts_sum": ts_sum,
    "ts_min": ts_min,
    "ts_max": ts_max,
    "ts_rank": ts_rank,
    "ts_delay": ts_delay,
    "ts_delta": ts_delta,
    "ts_corr": ts_corr,
    "rank": cs_rank,
    "zscore": cs_zscore,
    "log": jnp.log,
    "abs": jnp.abs,
    "sign": jnp.sign,
    "sqrt": jnp.sqrt,
    "exp": jnp.exp,
    "tanh": jnp.tanh,
    "max": jnp.maximum,
    "min": jnp.minimum,
    "where": jnp.where,
    "clip": jnp.clip,
    "safe_div": safe_div,
}
