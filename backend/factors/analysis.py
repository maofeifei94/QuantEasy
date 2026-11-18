"""因子有效性分析：IC、IC 衰减、分位收益。"""
from __future__ import annotations

import numpy as np


def _forward_returns(close: np.ndarray, valid: np.ndarray, horizon: int) -> np.ndarray:
    T, N = close.shape
    fwd = np.full((T, N), np.nan, dtype=np.float64)
    if T <= horizon:
        return fwd
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = close[horizon:] / close[:-horizon] - 1.0
    mask = valid[horizon:] & valid[:-horizon] & np.isfinite(raw)
    fwd[:-horizon] = np.where(mask, raw, np.nan)
    return fwd


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    return ranks


def _row_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return np.nan
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom <= 1e-12:
        return np.nan
    return float((ra * rb).sum() / denom)


def daily_ic(score: np.ndarray, fwd: np.ndarray, min_stocks: int = 8):
    T = score.shape[0]
    values = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        s, f = score[t], fwd[t]
        mask = np.isfinite(s) & np.isfinite(f)
        if mask.sum() < min_stocks:
            continue
        values[t] = _row_corr(s[mask], f[mask])
    return values


def analyze(score: np.ndarray, close: np.ndarray, valid: np.ndarray,
            dates: list[str], codes: list[str], names: dict[str, str],
            horizon: int = 5) -> dict:
    fwd = _forward_returns(close, valid, horizon)
    ic_series = daily_ic(score, fwd)
    finite_ic = ic_series[np.isfinite(ic_series)]
    ic_mean = float(finite_ic.mean()) if finite_ic.size else 0.0
    ic_std = float(finite_ic.std(ddof=1)) if finite_ic.size > 1 else 0.0
    ic_ir = ic_mean / ic_std * np.sqrt(252 / max(horizon, 1)) if ic_std > 1e-12 else 0.0
    positive = float((finite_ic > 0).mean()) if finite_ic.size else 0.0

    ic_decay = []
    for h in (1, 2, 5, 10, 20):
        f = _forward_returns(close, valid, h)
        series = daily_ic(score, f)
        s = series[np.isfinite(series)]
        ic_decay.append({
            "horizon": h,
            "ic": round(float(s.mean()), 4) if s.size else 0.0,
        })

    # 分位组合：每个截面按因子值分 5 组，统计未来 horizon 日平均收益
    groups = [[] for _ in range(5)]
    for t in range(score.shape[0]):
        s, f = score[t], fwd[t]
        mask = np.isfinite(s) & np.isfinite(f)
        if mask.sum() < 10:
            continue
        s, f = s[mask], f[mask]
        ranks = _rank(s)
        q = np.minimum((ranks / len(ranks)) * 5, 4).astype(int)
        for gi in range(5):
            sel = f[q == gi]
            if sel.size:
                groups[gi].append(sel.mean())
    quantile_returns = []
    for gi in range(5):
        arr = np.asarray(groups[gi])
        mean_ret = float(arr.mean()) if arr.size else 0.0
        quantile_returns.append({
            "group": f"Q{gi + 1}",
            "return": round(mean_ret, 5),
            "annualized": round(mean_ret * 252 / horizon, 5),
        })

    # 单只样本股的因子序列（用于展示）
    counts = np.isfinite(score).sum(axis=0)
    sample_idx = int(np.argmax(counts)) if counts.size else 0
    series = score[:, sample_idx] if score.size else np.array([])
    finite_pos = np.where(np.isfinite(series))[0]
    if finite_pos.size:
        tail = finite_pos[-260:]
        sample = {
            "code": codes[sample_idx],
            "name": names.get(codes[sample_idx], codes[sample_idx]),
            "dates": [dates[i] for i in tail],
            "values": [round(float(series[i]), 5) for i in tail],
        }
    else:
        sample = {"code": "", "name": "", "dates": [], "values": []}

    total_valid = int(valid.sum())
    coverage = float(np.isfinite(score).sum() / total_valid) if total_valid else 0.0
    finite_score = score[np.isfinite(score)]
    ic_pairs = [
        {"date": dates[i], "value": round(float(ic_series[i]), 4)}
        for i in range(len(dates)) if np.isfinite(ic_series[i])
    ]
    if len(ic_pairs) > 400:
        step = max(1, len(ic_pairs) // 400)
        ic_pairs = ic_pairs[::step]

    return {
        "horizon": horizon,
        "ic": {
            "mean": round(ic_mean, 4),
            "std": round(ic_std, 4),
            "ir": round(float(ic_ir), 3),
            "positive_rate": round(positive, 3),
            "series": ic_pairs,
        },
        "ic_decay": ic_decay,
        "quantile_returns": quantile_returns,
        "sample_series": sample,
        "stats": {
            "coverage": round(coverage, 3),
            "mean": round(float(finite_score.mean()), 5) if finite_score.size else 0.0,
            "std": round(float(finite_score.std()), 5) if finite_score.size else 0.0,
        },
    }
