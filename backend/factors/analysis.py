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


def quick_ic(score: np.ndarray, close: np.ndarray, valid: np.ndarray,
             horizon: int = 5) -> dict:
    """轻量 IC 统计，供因子工厂批量筛选使用。"""
    fwd = _forward_returns(close, valid, horizon)
    series = daily_ic(score, fwd)
    vals = series[np.isfinite(series)]
    if vals.size == 0:
        return {"mean": 0.0, "ir": 0.0, "positive_rate": 0.0, "coverage": 0.0}
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    ir = mean / std * np.sqrt(252 / max(horizon, 1)) if std > 1e-12 else 0.0
    coverage = float(np.isfinite(score).sum() / max(valid.sum(), 1))
    return {
        "mean": round(mean, 4),
        "ir": round(float(ir), 3),
        "positive_rate": round(float((vals > 0).mean()), 3),
        "coverage": round(coverage, 3),
    }


def analyze(score: np.ndarray, close: np.ndarray, valid: np.ndarray,
            dates: list[str], codes: list[str], names: dict[str, str],
            horizon: int = 5, industries: dict[str, str] | None = None,
            n_groups: int = 5) -> dict:
    n_groups = max(2, min(int(n_groups or 5), 10))
    fwd = _forward_returns(close, valid, horizon)
    ic_series = daily_ic(score, fwd)
    finite_ic = ic_series[np.isfinite(ic_series)]
    ic_mean = float(finite_ic.mean()) if finite_ic.size else 0.0
    ic_std = float(finite_ic.std(ddof=1)) if finite_ic.size > 1 else 0.0
    ic_ir = ic_mean / ic_std * np.sqrt(252 / max(horizon, 1)) if ic_std > 1e-12 else 0.0
    ic_t = ic_mean / ic_std * np.sqrt(finite_ic.size) if ic_std > 1e-12 and finite_ic.size else 0.0
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

    # 分位组合：每个截面按因子值分 n_groups 组，统计未来 horizon 日平均收益
    groups = [[] for _ in range(n_groups)]
    for t in range(score.shape[0]):
        s, f = score[t], fwd[t]
        mask = np.isfinite(s) & np.isfinite(f)
        if mask.sum() < 10:
            continue
        s, f = s[mask], f[mask]
        ranks = _rank(s)
        q = np.minimum((ranks / len(ranks)) * n_groups, n_groups - 1).astype(int)
        for gi in range(n_groups):
            sel = f[q == gi]
            if sel.size:
                groups[gi].append(sel.mean())
    quantile_returns = []
    for gi in range(n_groups):
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

    year_values: dict[str, list[float]] = {}
    for i, d in enumerate(dates):
        if np.isfinite(ic_series[i]):
            year_values.setdefault(d[:4], []).append(float(ic_series[i]))
    ic_by_year = [
        {"year": y, "ic": round(float(np.mean(v)), 4), "days": len(v)}
        for y, v in sorted(year_values.items())
    ]

    month_values: dict[str, list[float]] = {}
    for i, d in enumerate(dates):
        if np.isfinite(ic_series[i]):
            month_values.setdefault(d[:7], []).append(float(ic_series[i]))
    ic_monthly = [
        {
            "year": int(k[:4]), "month": int(k[5:7]),
            "ic": round(float(np.mean(v)), 4), "days": len(v),
        }
        for k, v in sorted(month_values.items())
    ]

    industry_ic: list[dict] = []
    if industries:
        groups: dict[str, list[int]] = {}
        for j, code in enumerate(codes):
            groups.setdefault(industries.get(code, "未分类"), []).append(j)
        for ind, members in groups.items():
            if len(members) < 5:
                continue
            idx = np.asarray(members)
            series = daily_ic(
                score[:, idx], fwd[:, idx],
                min_stocks=max(5, min(10, len(members))),
            )
            vals = series[np.isfinite(series)]
            if vals.size:
                industry_ic.append({
                    "industry": ind,
                    "ic": round(float(vals.mean()), 4),
                    "days": int(vals.size),
                })
        industry_ic.sort(key=lambda x: -abs(x["ic"]))

    return {
        "horizon": horizon,
        "ic": {
            "mean": round(ic_mean, 4),
            "std": round(ic_std, 4),
            "ir": round(float(ic_ir), 3),
            "t_stat": round(float(ic_t), 2),
            "positive_rate": round(positive, 3),
            "series": ic_pairs,
        },
        "ic_by_year": ic_by_year,
        "ic_monthly": ic_monthly,
        "industry_ic": industry_ic,
        "ic_decay": ic_decay,
        "quantile_returns": quantile_returns,
        "quantile_nav": quantile_nav(score, close, valid, dates, n_groups),
        "n_groups": n_groups,
        "stability": factor_stability(score, valid),
        "sample_series": sample,
        "stats": {
            "coverage": round(coverage, 3),
            "mean": round(float(finite_score.mean()), 5) if finite_score.size else 0.0,
            "std": round(float(finite_score.std()), 5) if finite_score.size else 0.0,
        },
    }


def factor_stability(score: np.ndarray, valid: np.ndarray,
                     lags: tuple[int, ...] = (1, 5, 20)) -> list[dict]:
    """因子截面排名的自相关，衡量信号稳定性与隐式换手。"""
    T = score.shape[0]
    out = []
    for lag in lags:
        corrs = []
        for t in range(T - lag):
            mask = valid[t] & valid[t + lag] & np.isfinite(score[t]) & np.isfinite(score[t + lag])
            if mask.sum() < 10:
                continue
            c = _row_corr(score[t][mask], score[t + lag][mask])
            if np.isfinite(c):
                corrs.append(c)
        mean = float(np.mean(corrs)) if corrs else 0.0
        out.append({
            "lag": lag,
            "autocorr": round(mean, 3),
            "turnover": round(max(0.0, 1.0 - mean), 3),
        })
    return out


def quantile_nav(score: np.ndarray, close: np.ndarray, valid: np.ndarray,
                 dates: list[str], n_groups: int = 5) -> dict:
    """因子分层组合净值：每日按因子值分组等权持有，统计次日收益。"""
    T, N = score.shape
    if T < 2:
        return {"dates": [], "groups": []}
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = close[1:] / close[:-1] - 1.0
    group_rets = np.zeros((n_groups, T - 1), dtype=np.float64)
    for t in range(T - 1):
        mask = np.isfinite(score[t]) & valid[t] & np.isfinite(rets[t])
        if mask.sum() < 10:
            continue
        s = score[t][mask]
        r = rets[t][mask]
        ranks = _rank(s)
        q = np.minimum((ranks / len(ranks)) * n_groups, n_groups - 1).astype(int)
        for g in range(n_groups):
            sel = r[q == g]
            if sel.size:
                group_rets[g, t] = sel.mean()

    navs = np.cumprod(1.0 + group_rets, axis=1)
    navs = np.concatenate([np.ones((n_groups, 1)), navs], axis=1)
    ls_raw = navs[-1] / np.maximum(navs[0], 1e-9)
    ls_nav = ls_raw / ls_raw[0]
    step = max(1, T // 320)
    idx = list(range(0, T, step))
    if T - 1 not in idx:
        idx.append(T - 1)
    return {
        "dates": [dates[i] for i in idx],
        "groups": [
            {"group": f"Q{g + 1}", "nav": [round(float(navs[g, i]), 4) for i in idx]}
            for g in range(n_groups)
        ],
        "long_short": [round(float(ls_nav[i]), 4) for i in idx],
    }


def mean_factor_correlation(scores: list[np.ndarray], valid: np.ndarray,
                            min_stocks: int = 8) -> np.ndarray:
    """多因子两两之间的平均截面秩相关系数矩阵。"""
    k = len(scores)
    if k == 0:
        return np.zeros((0, 0))
    total = np.zeros((k, k), dtype=np.float64)
    days = 0
    T = scores[0].shape[0]
    for t in range(T):
        mask = valid[t].copy()
        for s in scores:
            mask &= np.isfinite(s[t])
        if mask.sum() < min_stocks:
            continue
        ranks = []
        for s in scores:
            r = _rank(s[t][mask])
            r = (r - r.mean()) / (r.std() + 1e-12)
            ranks.append(r)
        mat = np.corrcoef(np.vstack(ranks))
        if np.isfinite(mat).all():
            total += mat
            days += 1
    if days == 0:
        return np.full((k, k), np.nan)
    return total / days
