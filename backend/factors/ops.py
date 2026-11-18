"""基于 JAX 的时序/截面因子算子。

所有算子以 (T, N) 面板数组为输入（T=交易日，N=股票），NaN 视为缺失值。
滚动窗口使用 cumsum / sliding_window_view 实现，可在 GPU/TPU 上自动加速。
"""
from __future__ import annotations

import jax
import numpy as np

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


def _window(x: jnp.ndarray, n: int) -> jnp.ndarray:
    """构造形状 (T, N, n) 的滚动窗口，最后一维按时间升序（旧→新）。"""
    x = _as2d(x)
    return jnp.stack([ts_delay(x, n - 1 - k) for k in range(n)], axis=-1)


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
    win = _window(x, n)
    win = jnp.where(jnp.isfinite(win), win, jnp.inf)
    res = jnp.min(win, axis=-1)
    return jnp.where(jnp.isfinite(res), res, jnp.nan)


def ts_max(x, n: int):
    x = _as2d(x)
    if x.shape[0] < n:
        return jnp.full_like(x, jnp.nan)
    win = _window(x, n)
    win = jnp.where(jnp.isfinite(win), win, -jnp.inf)
    res = jnp.max(win, axis=-1)
    return jnp.where(jnp.isfinite(res), res, jnp.nan)


def ts_rank(x, n: int):
    """当前值在过去 n 个观测中的分位（0~1）。"""
    x = _as2d(x)
    if x.shape[0] < n:
        return jnp.full_like(x, jnp.nan)
    win = _window(x, n)
    last = win[..., -1:]
    valid = jnp.isfinite(win)
    cnt = jnp.sum(valid, axis=-1)
    le = jnp.sum(jnp.where(valid & (win <= last), 1.0, 0.0), axis=-1)
    return jnp.where(cnt >= max(2, n // 2), le / jnp.maximum(cnt, 1), jnp.nan)


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


def ts_zscore(x, n: int):
    mean = ts_mean(x, n)
    std = ts_std(x, n)
    return ( _as2d(x) - mean) / jnp.where(std > 1e-12, std, jnp.nan)


def _moments(x, n: int):
    filled, valid, cnt, ok = _masked(x, n)
    c = jnp.maximum(cnt, 1.0)
    m1 = _roll_sum(filled, n) / c
    m2 = _roll_sum(filled * filled, n) / c
    m3 = _roll_sum(filled * filled * filled, n) / c
    m4 = _roll_sum(filled * filled * filled * filled, n) / c
    return m1, m2, m3, m4, cnt, ok


def ts_skew(x, n: int):
    m1, m2, m3, _, cnt, ok = _moments(x, n)
    var = jnp.maximum(m2 - m1 * m1, 1e-12)
    std = jnp.sqrt(var)
    skew = (m3 - 3 * m1 * m2 + 2 * m1 ** 3) / (std ** 3)
    return jnp.where(ok & (cnt >= 5), skew, jnp.nan)


def ts_kurt(x, n: int):
    m1, m2, m3, m4, cnt, ok = _moments(x, n)
    var = jnp.maximum(m2 - m1 * m1, 1e-12)
    kurt = (m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1 ** 4) / (var ** 2) - 3.0
    return jnp.where(ok & (cnt >= 5), kurt, jnp.nan)


def ts_argmax(x, n: int):
    x = _as2d(x)
    if x.shape[0] < n:
        return jnp.full_like(x, jnp.nan)
    win = _window(x, n)
    filled = jnp.where(jnp.isfinite(win), win, -jnp.inf)
    pos = jnp.argmax(filled, axis=-1).astype(jnp.float64) / max(n - 1, 1)
    return jnp.where(jnp.any(jnp.isfinite(win), axis=-1), pos, jnp.nan)


def ts_argmin(x, n: int):
    x = _as2d(x)
    if x.shape[0] < n:
        return jnp.full_like(x, jnp.nan)
    win = _window(x, n)
    filled = jnp.where(jnp.isfinite(win), win, jnp.inf)
    pos = jnp.argmin(filled, axis=-1).astype(jnp.float64) / max(n - 1, 1)
    return jnp.where(jnp.any(jnp.isfinite(win), axis=-1), pos, jnp.nan)


def decay_linear(x, n: int):
    """线性衰减加权均值，越近的观测权重越大。"""
    x = _as2d(x)
    if x.shape[0] < n:
        return jnp.full_like(x, jnp.nan)
    win = _window(x, n)
    valid = jnp.isfinite(win)
    weights = jnp.arange(1, n + 1, dtype=jnp.float64)
    num = jnp.sum(jnp.where(valid, win, 0.0) * weights, axis=-1)
    den = jnp.sum(jnp.where(valid, weights, 0.0), axis=-1)
    return jnp.where(den > 0, num / jnp.maximum(den, 1e-12), jnp.nan)


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


def _group_ids(group, n: int) -> np.ndarray:
    """把行业标签数组规范为 0..k-1 的整数分组，长度不符时视为单组。"""
    arr = np.asarray(group).ravel()
    if arr.size != n:
        return np.zeros(n, dtype=np.int64)
    _, inverse = np.unique(arr.astype(np.int64), return_inverse=True)
    return inverse


def cs_winsorize(x, n: float = 3.0):
    """截面去极值：按 ±n 倍标准差截断。"""
    x = np.asarray(_as2d(x), dtype=np.float64)
    valid = np.isfinite(x)
    cnt = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    mean = np.where(valid, x, 0.0).sum(axis=1, keepdims=True) / cnt
    diff = np.where(valid, x - mean, 0.0)
    std = np.sqrt((diff * diff).sum(axis=1, keepdims=True) / cnt)
    k = max(float(n), 0.1)
    lo = mean - k * np.maximum(std, 1e-12)
    hi = mean + k * np.maximum(std, 1e-12)
    return np.where(valid, np.clip(x, lo, hi), np.nan)


def group_rank(x, group):
    """行业内截面排名分位（0~1），group 传入 industry 变量。"""
    x = np.asarray(_as2d(x), dtype=np.float64)
    T, N = x.shape
    gids = _group_ids(group, N)
    out = np.full((T, N), np.nan)
    rows = np.arange(T)[:, None]
    for gid in np.unique(gids):
        cols = np.where(gids == gid)[0]
        if cols.size == 0:
            continue
        sub = x[:, cols]
        valid = np.isfinite(sub)
        order = np.argsort(np.where(valid, sub, np.inf), axis=1)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[rows, order] = np.arange(cols.size, dtype=np.float64)
        cnt = valid.sum(axis=1, keepdims=True).astype(np.float64)
        out[:, cols] = np.where(valid, (ranks + 0.5) / np.maximum(cnt, 1.0), np.nan)
    return out


def group_neutralize(x, group):
    """行业内去均值（行业中性化）。"""
    x = np.asarray(_as2d(x), dtype=np.float64)
    T, N = x.shape
    gids = _group_ids(group, N)
    out = np.full((T, N), np.nan)
    for gid in np.unique(gids):
        cols = np.where(gids == gid)[0]
        if cols.size == 0:
            continue
        sub = x[:, cols]
        valid = np.isfinite(sub)
        cnt = np.maximum(valid.sum(axis=1, keepdims=True), 1)
        mean = np.where(valid, sub, 0.0).sum(axis=1, keepdims=True) / cnt
        out[:, cols] = np.where(valid, sub - mean, np.nan)
    return out


def group_zscore(x, group):
    """行业内标准化：(x - 行业均值) / 行业标准差。"""
    x = np.asarray(_as2d(x), dtype=np.float64)
    T, N = x.shape
    gids = _group_ids(group, N)
    out = np.full((T, N), np.nan)
    for gid in np.unique(gids):
        cols = np.where(gids == gid)[0]
        if cols.size == 0:
            continue
        sub = x[:, cols]
        valid = np.isfinite(sub)
        cnt = np.maximum(valid.sum(axis=1, keepdims=True), 1)
        mean = np.where(valid, sub, 0.0).sum(axis=1, keepdims=True) / cnt
        diff = np.where(valid, sub - mean, 0.0)
        std = np.sqrt((diff * diff).sum(axis=1, keepdims=True) / cnt)
        out[:, cols] = np.where(valid & (std > 1e-12),
                                diff / np.where(std > 1e-12, std, 1.0), np.nan)
    return out


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
    "ts_zscore": ts_zscore,
    "ts_skew": ts_skew,
    "ts_kurt": ts_kurt,
    "ts_argmax": ts_argmax,
    "ts_argmin": ts_argmin,
    "decay_linear": decay_linear,
    "rank": cs_rank,
    "zscore": cs_zscore,
    "cs_winsorize": cs_winsorize,
    "group_rank": group_rank,
    "group_neutralize": group_neutralize,
    "group_zscore": group_zscore,
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
