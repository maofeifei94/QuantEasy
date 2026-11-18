"""业务编排层：数据同步、因子预览、回测执行。"""
from __future__ import annotations

import datetime as dt
import threading
import traceback
from typing import Any

import numpy as np
from scipy.optimize import minimize

from . import database, jobs, panel
from .backtest import engine as bt_engine
from .datasources import miniqmt, synthetic
from .factors import analysis, builtin
from .factors import preprocess as pre_mod
from .factors.engine import FactorError, compute, validate
from .indices import proxy_indices


# --------------------------------------------------------------------------- #
# 数据同步
# --------------------------------------------------------------------------- #
def _persist_indices(items: list[dict]) -> None:
    if not items:
        return
    database.upsert_indices([{"code": i["code"], "name": i.get("name", i["code"])} for i in items])
    rows = [row for item in items for row in item.get("rows", [])]
    database.insert_index_bars(rows)


def ensure_indices() -> None:
    """为已有行情补齐宽基指数代理（启动或首次回测时调用）。"""
    if database.count_index_bars() > 0:
        return
    stats = database.stats()
    if not stats["stock_count"] or not stats["start_date"]:
        return
    data = panel.build_panel(None, stats["start_date"], stats["end_date"])
    items = proxy_indices(data["close"], data["dates"], data["volume"])
    _persist_indices(items)


def start_sync(payload: dict) -> str:
    job_id = jobs.create("sync")
    thread = threading.Thread(target=_sync_worker, args=(job_id, payload), daemon=True)
    thread.start()
    return job_id


def _sync_worker(job_id: str, payload: dict) -> None:
    try:
        jobs.cleanup()
        source = (payload.get("source") or "auto").lower()
        start = payload.get("start") or "2022-01-01"
        end = payload.get("end") or dt.date.today().isoformat()
        codes = payload.get("codes") or None
        limit = int(payload.get("limit") or 300)
        replace = payload.get("replace", True)

        def progress(p: float, msg: str) -> None:
            jobs.update(job_id, progress=p * 0.7, message=msg)

        warnings: list[str] = []
        data = None
        if source in ("auto", "miniqmt"):
            if miniqmt.is_available():
                try:
                    jobs.update(job_id, progress=0.02, message="正在通过 miniQMT 拉取行情")
                    data = miniqmt.sync(
                        codes=codes, start=start, end=end, limit=limit,
                        dividend_type=payload.get("dividend_type", "front"),
                        progress=progress,
                    )
                except Exception as exc:  # 数据源异常
                    if source == "miniqmt":
                        raise
                    warnings.append(f"miniQMT 拉取失败，已切换演示数据：{exc}")
            else:
                msg = ("当前环境未检测到 xtquant（miniQMT），已使用内置合成行情演示。"
                       "在安装 miniQMT 客户端的 Windows 机器上运行即可直连真实行情。")
                if source == "miniqmt":
                    raise RuntimeError(msg)
                warnings.append(msg)

        if data is None:
            jobs.update(job_id, progress=0.05, message="生成合成行情数据")
            today = dt.date.today()
            gen_start = payload.get("start") or (today - dt.timedelta(days=3 * 365)).isoformat()
            gen_end = payload.get("end") or today.isoformat()
            stocks, bars, indices = synthetic.generate(
                start=gen_start, end=gen_end,
                n_stocks=int(payload.get("demo_stocks") or 150),
                progress=progress,
            )
            data = {"stocks": stocks, "bars": bars, "indices": indices, "source": "synthetic"}

        jobs.update(job_id, progress=0.75, message="写入 SQLite 数据库")
        if replace:
            database.clear_market_data()
        database.upsert_stocks(data["stocks"])
        total = len(data["bars"])
        for i in range(0, total, 20000):
            database.insert_bars(data["bars"][i: i + 20000])
            jobs.update(job_id, progress=0.75 + 0.2 * min(1.0, (i + 20000) / max(total, 1)),
                        message=f"写入日线 {min(i + 20000, total)}/{total}")
        _persist_indices(data.get("indices") or [])

        stats = database.stats()
        database.set_meta("data_source", data["source"])
        database.set_meta("last_sync", {
            "time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": data["source"],
            "start": stats["start_date"],
            "end": stats["end_date"],
            "bars": stats["bar_count"],
        })
        stats["source"] = data["source"]
        jobs.update(
            job_id, status="done", progress=1.0,
            message=f"同步完成：{stats['stock_count']} 只股票 / {stats['bar_count']} 条日线",
            result={
                "stats": stats,
                "source": data["source"],
                "warnings": warnings,
            },
        )
    except Exception as exc:
        jobs.update(job_id, status="error", error=str(exc), message="同步失败")
        traceback.print_exc()


# --------------------------------------------------------------------------- #
# 因子解析
# --------------------------------------------------------------------------- #
def resolve_factor(payload: dict) -> dict:
    factor_id = (payload.get("factor_id") or "").strip()
    expression = (payload.get("expression") or "").strip()
    label, description = "", ""
    direction = 1
    is_custom = False

    if expression:
        label = payload.get("label") or "自定义因子"
        description = payload.get("description") or ""
        direction = int(payload.get("direction") or 1)
        is_custom = True
    elif factor_id:
        meta = builtin.find_factor(factor_id)
        if meta:
            expression = meta["expression"]
            label = meta["label"]
            description = meta["description"]
            direction = meta["direction"]
        else:
            custom = None
            for f in database.list_factors():
                if str(f["id"]) == factor_id or f["name"] == factor_id:
                    custom = f
                    break
            if custom is None:
                raise FactorError(f"因子不存在: {factor_id}")
            expression = custom["expression"]
            label = custom["name"]
            description = custom.get("description") or ""
            direction = custom.get("direction", 1)
            is_custom = True
    else:
        raise FactorError("请选择因子或填写自定义表达式")

    direction = int(payload.get("direction") or direction)
    if direction not in (1, -1):
        direction = 1
    validate(expression)
    return {
        "id": factor_id,
        "label": label,
        "description": description,
        "expression": expression,
        "direction": direction,
        "is_custom": is_custom,
    }


def _resolve_factor_specs(payload: dict) -> list[dict]:
    """支持单因子（factor_id/expression）与多因子组合（factors 列表）。"""
    items = payload.get("factors")
    specs: list[dict] = []
    if isinstance(items, list) and items:
        for item in items:
            if isinstance(item, str):
                item = {"factor_id": item}
            if not isinstance(item, dict):
                continue
            spec = resolve_factor(item)
            spec["weight"] = float(item.get("weight", 1) or 0)
            specs.append(spec)
    else:
        spec = resolve_factor(payload)
        spec["weight"] = 1.0
        specs.append(spec)
    if not specs:
        raise FactorError("请至少选择一个因子")
    total = sum(abs(s["weight"]) for s in specs)
    if total <= 1e-12:
        raise FactorError("因子权重之和不能为 0")
    for spec in specs:
        spec["weight"] = spec["weight"] / total
    return specs


STYLE_EXPRESSIONS = (
    "close / ts_delay(close, 20) - 1",
    "-ts_std(returns, 20)",
    "volume / ts_mean(volume, 20)",
)


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """截面回归残差化：y 对 x 逐日回归后取残差。"""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(x)
    cnt = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    ym = np.where(valid, y, 0.0).sum(axis=1, keepdims=True) / cnt
    xm = np.where(valid, x, 0.0).sum(axis=1, keepdims=True) / cnt
    yc = np.where(valid, y - ym, 0.0)
    xc = np.where(valid, x - xm, 0.0)
    var = (xc * xc).sum(axis=1, keepdims=True) / cnt
    cov = (xc * yc).sum(axis=1, keepdims=True) / cnt
    beta = np.where(var > 1e-12, cov / np.maximum(var, 1e-12), 0.0)
    return np.where(valid, y - beta * (x - xm), y)


def _orthogonalize(score: np.ndarray, data: dict) -> np.ndarray:
    """对动量/波动/量比风格因子做正交化。"""
    out = np.asarray(score, dtype=np.float64)
    for expr in STYLE_EXPRESSIONS:
        try:
            base = np.asarray(compute(expr, data))
            base = np.asarray(pre_mod.apply_preprocess(
                base, data, {"standardize": True}, force_standardize=True
            ))
        except Exception:
            continue
        out = _residualize(out, base)
    return out


def _prepare(payload: dict):
    codes = payload.get("codes") or None
    if isinstance(codes, str):
        codes = [c.strip() for c in codes.split(",") if c.strip()]
    if codes is not None and len(codes) == 0:
        codes = None
    start = payload.get("start") or None
    end = payload.get("end") or None
    data = panel.build_panel(codes, start, end)

    specs = _resolve_factor_specs(payload)
    options = payload.get("preprocess") or {}
    multi = len(specs) > 1

    components = []
    for spec in specs:
        raw = compute(spec["expression"], data)
        scored = pre_mod.apply_preprocess(
            raw * spec["direction"], data, options, force_standardize=multi
        )
        if options.get("orthogonalize"):
            scored = _orthogonalize(np.asarray(scored), data)
        components.append(np.asarray(scored))

    if multi:
        stack = np.stack(components)
        weights = np.asarray([s["weight"] for s in specs]).reshape(-1, 1, 1)
        numerator = np.nansum(stack * weights, axis=0)
        denominator = np.nansum(np.where(np.isfinite(stack), np.abs(weights), 0.0), axis=0)
        score = np.where(denominator > 1e-12, numerator / np.maximum(denominator, 1e-12), np.nan)
    else:
        score = components[0]

    factor = {
        "label": specs[0]["label"] if not multi else " + ".join(s["label"] for s in specs),
        "expression": specs[0]["expression"] if not multi else " | ".join(s["expression"] for s in specs),
        "direction": specs[0]["direction"] if not multi else 1,
        "is_multi": multi,
        "factors": [
            {
                "label": s["label"],
                "direction": s["direction"],
                "weight": round(float(s["weight"]), 4),
                "expression": s["expression"],
            }
            for s in specs
        ],
        "preprocess": {
            "winsorize": bool(options.get("winsorize")),
            "neutralize": bool(options.get("neutralize")),
            "standardize": bool(options.get("standardize")) or multi,
            "orthogonalize": bool(options.get("orthogonalize")),
        },
    }
    return data, factor, score


# --------------------------------------------------------------------------- #
# 股票池过滤
# --------------------------------------------------------------------------- #
def _apply_filters(data: dict, filters: dict | None) -> np.ndarray:
    filters = filters or {}
    valid = data["valid"].copy()
    codes = data["codes"]

    exclude_star = filters.get("exclude_star", True)
    exclude_bj = filters.get("exclude_bj", True)
    for j, code in enumerate(codes):
        base = code.split(".")[0]
        if exclude_star and base.startswith("688"):
            valid[:, j] = False
        if exclude_bj and (base.startswith("8") or base.startswith("4")):
            valid[:, j] = False

    min_days = int(filters.get("min_list_days") or 0)
    if min_days > 0:
        tradable = data["tradable"]
        for j in range(len(codes)):
            idx = np.where(tradable[:, j])[0]
            if idx.size == 0:
                valid[:, j] = False
            else:
                valid[: idx[0] + min_days, j] = False

    min_amount = float(filters.get("min_amount") or 0)
    if min_amount > 0:
        amount = data["amount"]
        n = 20
        cs = np.cumsum(np.vstack([np.zeros((1, amount.shape[1])), amount]), axis=0)
        roll = np.full_like(amount, np.nan)
        if amount.shape[0] >= n:
            roll[n - 1:] = (cs[n:] - cs[:-n]) / n
        valid &= np.nan_to_num(roll, nan=0.0) >= min_amount * 1e8

    if filters.get("exclude_limit_up"):
        rets = data["returns"]
        valid &= ~(np.isfinite(rets) & (rets >= 0.095))
    return valid


def _erc_weights(cov: np.ndarray, iters: int = 60) -> np.ndarray:
    """等风险贡献（Risk Parity）权重，迭代求解。"""
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])
    w = np.ones(n) / n
    for _ in range(iters):
        port = float(w @ cov @ w)
        if port <= 1e-16:
            break
        mrc = (cov @ w) / port
        rc = w * mrc
        target = 1.0 / n
        w = w * np.sqrt(np.maximum(target, 1e-12) / np.maximum(rc, 1e-12))
        w = np.maximum(w, 1e-8)
        w = w / w.sum()
    return w


def _risk_parity_targets(score: np.ndarray, valid: np.ndarray, returns: np.ndarray,
                         rebalance: np.ndarray, top_n: int, top_frac: float,
                         max_weight: float, window: int = 60) -> np.ndarray:
    """在每次调仓日按协方差计算等风险贡献权重。"""
    T, N = score.shape
    target = np.zeros((T, N), dtype=np.float64)
    rets = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    for t in np.where(rebalance)[0]:
        row = np.where(valid[t] & np.isfinite(score[t]), score[t], -np.inf)
        n_valid = int(np.isfinite(row).sum())
        if n_valid < 2:
            continue
        k = min(top_n, n_valid) if top_n > 0 else max(1, int(round(top_frac * n_valid)))
        sel = np.argsort(-row)[:k]
        start = max(0, t - window)
        sub = rets[start:t][:, sel]
        if sub.shape[0] >= 20 and len(sel) >= 2:
            cov = np.atleast_2d(np.cov(sub, rowvar=False))
            cov = 0.7 * cov + 0.3 * np.diag(np.diag(cov))
            w = _erc_weights(cov)
        else:
            w = np.ones(len(sel)) / len(sel)
        if max_weight > 0:
            cap = max(max_weight, 1.0 / len(sel))
            w = np.minimum(w, cap)
            w = w / max(w.sum(), 1e-12)
        target[t, sel] = w
    return target


def _rolling_vol(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """20 日滚动波动率，缺失值用截面中位数填充。"""
    T, N = returns.shape
    vol = np.full((T, N), np.nan, dtype=np.float64)
    for t in range(window - 1, T):
        w = returns[t - window + 1: t + 1]
        ok = np.isfinite(w)
        cnt = ok.sum(axis=0)
        mean = np.where(ok, w, 0.0).sum(axis=0) / np.maximum(cnt, 1)
        var = np.where(ok, (w - mean) ** 2, 0.0).sum(axis=0) / np.maximum(cnt, 1)
        v = np.sqrt(var)
        vol[t] = np.where(cnt >= 10, v, np.nan)
    for t in range(T):
        row = vol[t]
        if np.isfinite(row).any():
            vol[t] = np.where(np.isfinite(row), row, np.nanmedian(row))
        else:
            vol[t] = 0.02
    return np.maximum(vol, 1e-3)


# --------------------------------------------------------------------------- #
# 因子预览
# --------------------------------------------------------------------------- #
def factor_preview(payload: dict) -> dict:
    data, factor, score = _prepare(payload)
    valid = _apply_filters(data, payload.get("filters"))
    horizon = int(payload.get("horizon") or 5)
    result = analysis.analyze(
        score, data["close"], valid, data["dates"], data["codes"],
        data["names"], horizon=horizon, industries=data["industries"],
    )
    return {
        "factor": {
            "label": factor["label"],
            "expression": factor["expression"],
            "direction": factor["direction"],
            "factors": factor["factors"],
            "preprocess": factor["preprocess"],
        },
        "universe": {
            "stocks": len(data["codes"]),
            "start": data["dates"][0],
            "end": data["dates"][-1],
            "days": len(data["dates"]),
        },
        **result,
    }


def factor_correlation(payload: dict) -> dict:
    """多个因子之间的平均截面秩相关系数矩阵。"""
    factor_ids = payload.get("factor_ids") or []
    if len(factor_ids) < 2:
        raise ValueError("至少选择两个因子")
    if len(factor_ids) > 8:
        raise ValueError("最多分析 8 个因子")
    codes = payload.get("codes") or None
    data = panel.build_panel(codes, payload.get("start") or None, payload.get("end") or None)
    specs = [resolve_factor({"factor_id": fid}) for fid in factor_ids]
    scores = [
        np.asarray(compute(spec["expression"], data) * spec["direction"])
        for spec in specs
    ]
    matrix = analysis.mean_factor_correlation(scores, data["valid"])
    return {
        "labels": [s["label"] for s in specs],
        "matrix": [
            [round(float(v), 3) if np.isfinite(v) else None for v in row]
            for row in matrix
        ],
    }


def mine_factors(payload: dict) -> dict:
    """因子工厂：批量评估候选表达式，按 ICIR 排序。"""
    from .factors.factory import CANDIDATES

    codes = payload.get("codes") or None
    data = panel.build_panel(codes, payload.get("start") or None, payload.get("end") or None)
    valid = data["valid"]
    horizon = int(payload.get("horizon") or 5)
    limit = min(int(payload.get("limit") or 12), len(CANDIDATES))

    items = []
    for cand in CANDIDATES:
        try:
            score = np.asarray(compute(cand["expression"], data))
            stats = analysis.quick_ic(score, data["close"], valid, horizon)
        except Exception:
            continue
        if stats["coverage"] < 0.2:
            continue
        direction = 1 if stats["mean"] >= 0 else -1
        items.append({
            **cand,
            "direction": direction,
            "ic": round(abs(stats["mean"]), 4),
            "icir": round(abs(stats["ir"]), 3),
            "positive_rate": stats["positive_rate"],
            "coverage": stats["coverage"],
        })
    items.sort(key=lambda x: -x["icir"])
    return {
        "horizon": horizon,
        "universe": {
            "stocks": len(data["codes"]),
            "start": data["dates"][0],
            "end": data["dates"][-1],
            "days": len(data["dates"]),
        },
        "items": items[:limit],
    }


# --------------------------------------------------------------------------- #
# 回测
# --------------------------------------------------------------------------- #
def _rebalance_mask(score: np.ndarray, valid: np.ndarray, freq: int,
                    dates: list[str] | None = None, mode: str = "interval",
                    weekday: int = 0, month_day: int = 1) -> tuple[np.ndarray, int]:
    """生成调仓掩码：按交易日间隔 / 每周固定星期 / 每月第 N 个交易日。"""
    T, N = score.shape
    usable = (np.isfinite(score) & valid).sum(axis=1)
    min_stocks = max(2, min(10, N))
    eligible = np.where(usable >= min_stocks)[0]
    if eligible.size == 0:
        raise ValueError("因子在所选区间内没有足够的有效值，请检查表达式或放宽过滤条件")
    warmup = int(eligible[0])
    mask = np.zeros(T, dtype=bool)
    if mode == "weekly" and dates:
        weekday = max(0, min(int(weekday), 4))
        for i in range(warmup, T):
            if dt.date.fromisoformat(dates[i]).weekday() == weekday:
                mask[i] = True
        if not mask.any():
            mask[warmup::max(1, freq)] = True
    elif mode == "monthly" and dates:
        month_day = max(1, min(int(month_day), 10))
        counts: dict[str, int] = {}
        for i in range(warmup, T):
            key = dates[i][:7]
            counts[key] = counts.get(key, 0) + 1
            if counts[key] == month_day:
                mask[i] = True
    else:
        mask[warmup::max(1, freq)] = True
    return mask, warmup


def _rebalance_args(payload: dict) -> tuple[str, int, int]:
    """解析调仓方式参数：(mode, weekday, month_day)。"""
    mode = payload.get("rebalance_mode") or "interval"
    if mode not in ("interval", "weekly", "monthly"):
        mode = "interval"
    weekday = max(0, min(int(payload.get("rebalance_weekday") or 0), 4))
    month_day = max(1, min(int(payload.get("rebalance_month_day") or 1), 10))
    return mode, weekday, month_day


def _monthly_returns(dates: list[str], nav: np.ndarray) -> list[dict]:
    out: list[dict] = []
    prev_nav = 1.0
    for i in range(1, len(dates)):
        if dates[i][:7] != dates[i - 1][:7]:
            cur = float(nav[i])
            out.append({
                "year": int(dates[i - 1][:4]),
                "month": int(dates[i - 1][5:7]),
                "return": round(cur / prev_nav - 1, 4),
            })
            prev_nav = cur
    if dates:
        out.append({
            "year": int(dates[-1][:4]),
            "month": int(dates[-1][5:7]),
            "return": round(float(nav[-1]) / prev_nav - 1, 4),
        })
    return out


def _holdings(dates: list[str], weights: np.ndarray, rebalance: np.ndarray,
              names: dict[str, str], codes: list[str], limit: int = 6) -> list[dict]:
    idxs = [i for i in range(len(dates)) if rebalance[i] and np.abs(weights[i]).sum() > 1e-9]
    idxs = idxs[-limit:]
    out = []
    for i in idxs:
        w = weights[i]
        order = np.argsort(-np.abs(w))[:10]
        items = [
            {
                "code": codes[j],
                "name": names.get(codes[j], codes[j]),
                "weight": round(float(w[j]), 4),
            }
            for j in order if abs(w[j]) > 1e-6
        ]
        out.append({"date": dates[i], "items": items})
    return out


def _market_benchmark(dates: list[str], start: str, end: str) -> np.ndarray:
    """全市场等权基准净值（用于自选股票池时比较）。"""
    allp = panel.build_panel(None, start, end)
    all_dates = allp["dates"]
    if len(all_dates) < 2:
        return np.ones(len(dates))
    close, valid = allp["close"], allp["valid"]
    weights = valid.astype(np.float64)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = close[1:] / close[:-1] - 1.0
    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
    market_rets = np.sum(weights[:-1] * rets, axis=1)
    pos = {d: i for i, d in enumerate(all_dates)}
    aligned = []
    for d in dates[1:]:
        k = pos.get(d)
        aligned.append(float(market_rets[k - 1]) if k and k >= 1 else 0.0)
    return np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(aligned))])


def _custom_benchmark(dates: list[str], codes: list[str]) -> np.ndarray:
    """自选股票池等权基准净值（按给定代码列表）。"""
    if not codes:
        raise ValueError("自定义基准为空")
    sub = panel.build_panel(codes, dates[0], dates[-1])
    sub_dates = sub["dates"]
    if len(sub_dates) < 2:
        return np.ones(len(dates))
    close, valid = sub["close"], sub["valid"]
    weights = valid.astype(np.float64)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = close[1:] / close[:-1] - 1.0
    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
    bench_rets = np.sum(weights[:-1] * rets, axis=1)
    pos = {d: i for i, d in enumerate(sub_dates)}
    aligned = []
    for d in dates[1:]:
        k = pos.get(d)
        aligned.append(float(bench_rets[k - 1]) if k and k >= 1 else 0.0)
    return np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(aligned))])


def _market_timing(dates: list[str], start: str, end: str, window: int) -> np.ndarray:
    """大盘择时：全市场等权指数在均线上方时持仓，否则空仓。"""
    if window <= 1:
        return np.ones(len(dates))
    allp = panel.build_panel(None, start, end)
    ad = allp["dates"]
    if len(ad) < 2:
        return np.ones(len(dates))
    close, valid = allp["close"], allp["valid"]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = close[1:] / close[:-1] - 1.0
    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
    w = valid[:-1].astype(np.float64)
    mkt = np.sum(w * rets, axis=1) / np.maximum(w.sum(axis=1), 1.0)
    nav = np.concatenate([[1.0], np.cumprod(1.0 + mkt)])
    cs = np.cumsum(np.concatenate([[0.0], nav]))
    ma = np.full_like(nav, np.nan)
    if len(nav) >= window:
        ma[window - 1:] = (cs[window:] - cs[:-window]) / window
    sig = np.where(np.isnan(ma), 1.0, (nav > ma).astype(np.float64))
    pos = {d: i for i, d in enumerate(ad)}
    return np.array([sig[pos[d]] if d in pos else 1.0 for d in dates], dtype=np.float64)


def _attribution(weights: np.ndarray, close: np.ndarray, codes: list[str],
                 names: dict[str, str], industries: dict[str, str]) -> dict:
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = close[1:] / close[:-1] - 1.0
    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
    contrib = weights[:-1] * rets
    per_stock = np.nansum(contrib, axis=0)
    order = np.argsort(-np.abs(per_stock))[:12]
    stocks = [
        {
            "code": codes[j],
            "name": names.get(codes[j], codes[j]),
            "industry": industries.get(codes[j], "未分类"),
            "contribution": round(float(per_stock[j]), 4),
        }
        for j in order
    ]
    by_ind: dict[str, float] = {}
    for j, code in enumerate(codes):
        ind = industries.get(code, "未分类")
        by_ind[ind] = by_ind.get(ind, 0.0) + float(per_stock[j])
    industry_rows = sorted(
        ({"industry": k, "contribution": round(v, 4)} for k, v in by_ind.items()),
        key=lambda x: -abs(x["contribution"]),
    )[:10]
    return {"stocks": stocks, "industries": industry_rows}


def _trades(weights: np.ndarray, rebalance: np.ndarray, dates: list[str],
            codes: list[str], names: dict[str, str], limit: int = 6) -> list[dict]:
    idxs = np.where(rebalance)[0][-limit:]
    out = []
    for i in idxs:
        prev = weights[i - 1] if i > 0 else np.zeros_like(weights[i])
        delta = weights[i] - prev
        order = np.argsort(-np.abs(delta))[:8]
        items = [
            {
                "code": codes[j],
                "name": names.get(codes[j], codes[j]),
                "side": "买入" if delta[j] > 0 else "卖出",
                "change": round(float(delta[j]), 4),
            }
            for j in order if abs(delta[j]) > 0.004
        ]
        if items:
            out.append({"date": dates[i], "items": items})
    return out


def _segment_metrics(net: np.ndarray, nav: np.ndarray) -> dict:
    net = np.asarray(net, dtype=np.float64)
    nav = np.asarray(nav, dtype=np.float64)
    n = max(len(net), 1)
    if len(nav) < 2 or nav[0] <= 0:
        return {"annual_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}
    ann = float((nav[-1] / nav[0]) ** (252 / n) - 1)
    std = float(np.std(net))
    sharpe = float(np.mean(net) / std * np.sqrt(252)) if std > 1e-12 else 0.0
    peak = np.maximum.accumulate(nav)
    max_dd = float(np.min(nav / peak - 1))
    win = float(np.mean(net > 0)) if n else 0.0
    return {
        "annual_return": round(ann, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win, 3),
    }


def _brinson(weights: np.ndarray, close: np.ndarray, valid: np.ndarray,
             codes: list[str], industries: dict[str, str]) -> dict | None:
    """Brinson-Fachler 归因：超额收益拆分为行业配置、选股与交互效应。"""
    T, N = weights.shape
    if T < 2:
        return None
    groups: dict[str, list[int]] = {}
    for j, code in enumerate(codes):
        groups.setdefault(industries.get(code, "未分类"), []).append(j)
    if len(groups) < 2:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = close[1:] / close[:-1] - 1.0
    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
    acc = {k: {"allocation": 0.0, "selection": 0.0, "interaction": 0.0} for k in groups}
    for t in range(T - 1):
        v = valid[t]
        total = float(v.sum())
        if total <= 0:
            continue
        r = rets[t]
        for ind, members in groups.items():
            idx = np.asarray(members)
            mv = v[idx]
            cnt = float(mv.sum())
            if cnt <= 0:
                continue
            wb = cnt / total
            rb = float(r[idx][mv].mean())
            wj = weights[t, idx]
            wp = float(np.nansum(wj))
            rp = float(np.nansum(wj * r[idx]) / wp) if abs(wp) > 1e-9 else 0.0
            acc[ind]["allocation"] += (wp - wb) * rb
            acc[ind]["selection"] += wb * (rp - rb)
            acc[ind]["interaction"] += (wp - wb) * (rp - rb)
    rows = []
    for ind, a in acc.items():
        total_eff = a["allocation"] + a["selection"] + a["interaction"]
        rows.append({
            "industry": ind,
            "allocation": round(a["allocation"], 4),
            "selection": round(a["selection"], 4),
            "interaction": round(a["interaction"], 4),
            "total": round(total_eff, 4),
        })
    rows.sort(key=lambda x: -abs(x["total"]))
    totals = {k: round(sum(r[k] for r in rows), 4)
              for k in ("allocation", "selection", "interaction", "total")}
    return {"industries": rows, "totals": totals}


def _bootstrap(net: np.ndarray, samples: int = 400) -> dict | None:
    """日收益 Bootstrap 重采样，给出年化收益与夏普的 95% 置信区间。"""
    net = np.asarray(net, dtype=np.float64)
    n = len(net)
    if n < 60:
        return None
    rng = np.random.default_rng(42)
    pick = rng.integers(0, n, size=(samples, n))
    sim = net[pick]
    nav = np.prod(1.0 + sim, axis=1)
    ann = nav ** (252.0 / n) - 1.0
    std = sim.std(axis=1, ddof=1)
    sharpe = np.where(std > 1e-12, sim.mean(axis=1) / std * np.sqrt(252), 0.0)

    def ci(arr):
        return {
            "low": round(float(np.percentile(arr, 2.5)), 4),
            "mid": round(float(np.median(arr)), 4),
            "high": round(float(np.percentile(arr, 97.5)), 4),
        }
    return {"samples": samples, "annual_return": ci(ann), "sharpe": ci(sharpe)}


def _optimize_strategy_weights(matrix: np.ndarray, method: str) -> np.ndarray | None:
    """策略组合权重优化：最小方差 / 最大夏普 / 风险平价 / 等权。

    matrix 形状为 (策略数, 交易日数)。返回 None 表示沿用用户手填权重。
    """
    m = matrix.shape[0]
    if method in ("", "manual", None):
        return None
    if m < 2:
        return np.ones(m)
    if method == "equal":
        return np.full(m, 1.0 / m)
    cov = np.atleast_2d(np.cov(matrix))
    cov = 0.7 * cov + 0.3 * np.diag(np.diag(cov))
    mu = np.mean(matrix, axis=1) * 252.0
    x0 = np.full(m, 1.0 / m)
    bounds = [(0.0, 1.0)] * m
    cons = ({"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},)
    if method == "risk_parity":
        return _erc_weights(cov)
    if method == "min_var":
        res = minimize(lambda w: float(w @ cov @ w) * 1e4, x0, bounds=bounds,
                       constraints=cons, method="SLSQP",
                       options={"maxiter": 500, "ftol": 1e-12})
    elif method == "max_sharpe":
        def neg_sharpe(w):
            var = float(w @ cov @ w)
            if var <= 1e-18:
                return 1e6
            return -float(w @ mu) / np.sqrt(var)

        res = minimize(neg_sharpe, x0, bounds=bounds, constraints=cons, method="SLSQP")
    else:
        return None
    if not res.success or not np.all(np.isfinite(res.x)):
        return x0
    return np.maximum(res.x, 0.0) / max(float(np.sum(np.maximum(res.x, 0.0))), 1e-12)


def combine_portfolios(payload: dict) -> dict:
    """多策略组合：按权重合成已保存回测的收益曲线。"""
    items = payload.get("items") or []
    if not (2 <= len(items) <= 6):
        raise ValueError("请选择 2~6 个回测进行组合")
    records, raw_weights = [], []
    optimizer = str(payload.get("optimizer") or "manual")
    for item in items:
        bt = database.get_backtest(int(item.get("id")))
        if bt is None:
            raise ValueError(f"回测不存在: {item.get('id')}")
        w = float(item.get("weight", 1) or 0)
        if optimizer == "manual" and w <= 0:
            continue
        records.append(bt)
        raw_weights.append(w if w > 0 else 1.0)
    if len(records) < 2:
        raise ValueError("至少需要两个有效权重的回测")

    date_sets = [set(r["result"]["dates"]) for r in records]
    common = sorted(set.intersection(*date_sets))
    if len(common) < 30:
        raise ValueError("所选回测的日期区间重叠不足")
    ret_maps = []
    for r in records:
        res = r["result"]
        ret_maps.append({
            d: (res["daily_returns"][i - 1] if i > 0 else 0.0)
            for i, d in enumerate(res["dates"])
        })
    matrix = np.array([
        [ret_maps[k].get(d, 0.0) or 0.0 for d in common]
        for k in range(len(records))
    ])
    optimized = _optimize_strategy_weights(matrix, optimizer)
    if optimized is not None:
        weights = [float(x) for x in optimized]
    else:
        optimizer = "manual"
        total_w = sum(raw_weights)
        weights = [w / total_w for w in raw_weights]
    combined = np.average(matrix, axis=0, weights=weights)
    nav = np.concatenate([[1.0], np.cumprod(1.0 + combined)])
    metrics = _segment_metrics(combined, nav)
    metrics.update({
        "total_return": round(float(nav[-1] - 1), 4),
        "annual_volatility": round(float(np.std(combined) * np.sqrt(252)), 4),
    })
    corr = np.corrcoef(matrix)
    return {
        "dates": common,
        "optimizer": optimizer,
        "nav": [round(float(v), 4) for v in nav],
        "metrics": metrics,
        "items": [
            {
                "id": r["id"], "name": r["name"], "weight": round(float(w), 4),
                "annual_return": r["metrics"].get("annual_return"),
                "sharpe": r["metrics"].get("sharpe"),
            }
            for r, w in zip(records, weights)
        ],
        "correlation": {
            "labels": [r["name"] for r in records],
            "matrix": [[round(float(v), 3) for v in row] for row in corr],
        },
    }


def _index_benchmark(dates: list[str], code: str) -> np.ndarray:
    """从 index_bars 构建基准净值，缺失日按前值填充。"""
    rows = database.get_index_bars(code, dates[0], dates[-1])
    if len(rows) < 30:
        raise ValueError("指数数据不足")
    mapping = {r["date"]: float(r["close"]) for r in rows}
    vals: list[float | None] = []
    last = None
    for d in dates:
        if d in mapping:
            last = mapping[d]
        vals.append(last)
    if vals[0] is None:
        first = next((v for v in vals if v is not None), None)
        if first is None:
            raise ValueError("指数数据不足")
        vals = [first if v is None else v for v in vals]
    out = np.array(vals, dtype=np.float64)
    for i in range(1, len(out)):
        if not np.isfinite(out[i]):
            out[i] = out[i - 1]
    return out / out[0]


def _annual_returns(dates: list[str], nav: np.ndarray) -> list[dict]:
    out: list[dict] = []
    prev_nav = 1.0
    for i in range(1, len(dates)):
        if dates[i][:4] != dates[i - 1][:4]:
            out.append({
                "year": int(dates[i - 1][:4]),
                "return": round(float(nav[i - 1]) / prev_nav - 1, 4),
            })
            prev_nav = float(nav[i - 1])
    if dates:
        out.append({
            "year": int(dates[-1][:4]),
            "return": round(float(nav[-1]) / prev_nav - 1, 4),
        })
    return out


def _rolling_risk(net: np.ndarray, dates: list[str], window: int = 60) -> dict | None:
    """滚动 60 日夏普与年化波动率。"""
    net = np.asarray(net, dtype=np.float64)
    if len(net) < window:
        return None
    csum = np.cumsum(np.insert(net, 0, 0.0))
    csq = np.cumsum(np.insert(net * net, 0, 0.0))
    mean = (csum[window:] - csum[:-window]) / window
    mean_sq = (csq[window:] - csq[:-window]) / window
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)
    sharpe = np.where(std > 1e-12, mean / std * np.sqrt(252), 0.0)
    vol = std * np.sqrt(252)
    return {
        "window": window,
        "dates": dates[window:],
        "sharpe": [round(float(v), 3) for v in sharpe],
        "volatility": [round(float(v), 4) for v in vol],
    }


def _return_hist(net: np.ndarray, bins: int = 15) -> dict | None:
    net = np.asarray(net, dtype=np.float64)
    if net.size < 30:
        return None
    counts, edges = np.histogram(net * 100, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return {
        "labels": [f"{c:.2f}" for c in centers],
        "values": [int(c) for c in counts],
        "centers": [round(float(c), 2) for c in centers],
    }


def _extended_metrics(net: np.ndarray, bench_nav: np.ndarray,
                      monthly: list[dict], nav: np.ndarray | None = None) -> dict:
    out: dict[str, float] = {}
    if len(bench_nav) > 1 and len(net) > 1:
        bench_ret = bench_nav[1:] / bench_nav[:-1] - 1.0
        n = min(len(net), len(bench_ret))
        a, b = net[-n:], bench_ret[-n:]
        var_b = np.var(b, ddof=1) if n > 1 else 0.0
        beta = float(np.cov(a, b, ddof=1)[0, 1] / var_b) if var_b > 1e-12 else 0.0
        alpha = float((np.mean(a) - beta * np.mean(b)) * 252)
        active = a - b
        tracking = float(np.std(active, ddof=1) * np.sqrt(252)) if n > 1 else 0.0
        info_ratio = float(np.mean(active) / np.std(active, ddof=1) * np.sqrt(252)) \
            if n > 1 and np.std(active, ddof=1) > 1e-12 else 0.0
        downside = a[a < 0]
        downside_vol = float(np.std(downside, ddof=1) * np.sqrt(252)) if downside.size > 1 else 0.0
        sortino = float(np.mean(a) * np.sqrt(252) / np.std(downside, ddof=1)) \
            if downside.size > 1 and np.std(downside, ddof=1) > 1e-12 else 0.0
        up_mask = b > 0
        down_mask = b < 0
        up_capture = float(np.mean(a[up_mask]) / np.mean(b[up_mask])) \
            if up_mask.sum() > 2 and abs(np.mean(b[up_mask])) > 1e-12 else 0.0
        down_capture = float(np.mean(a[down_mask]) / np.mean(b[down_mask])) \
            if down_mask.sum() > 2 and abs(np.mean(b[down_mask])) > 1e-12 else 0.0
        var95 = float(np.percentile(a, 5))
        cvar95 = float(a[a <= var95].mean())
        out.update({
            "alpha": round(alpha, 4),
            "beta": round(beta, 3),
            "tracking_error": round(tracking, 4),
            "information_ratio": round(info_ratio, 3),
            "downside_volatility": round(downside_vol, 4),
            "sortino": round(sortino, 3),
            "up_capture": round(up_capture, 3),
            "down_capture": round(down_capture, 3),
            "var_95": round(var95, 4),
            "cvar_95": round(cvar95, 4),
        })
    if nav is not None and len(nav) > 1:
        peak = nav[0]
        peak_idx = 0
        max_days = 0
        for i, v in enumerate(nav):
            if v >= peak:
                peak = v
                peak_idx = i
            else:
                max_days = max(max_days, i - peak_idx)
        out["max_drawdown_days"] = int(max_days)
    if monthly:
        values = np.array([m["return"] for m in monthly], dtype=np.float64)
        out["monthly_win_rate"] = round(float(np.mean(values > 0)), 3)
        out["best_month"] = round(float(values.max()), 4)
        out["worst_month"] = round(float(values.min()), 4)
    return out


def _drawdown_episodes(dates: list[str], nav: np.ndarray, top: int = 5) -> dict | None:
    """识别回撤区间：峰值 → 谷底 → 修复，输出最深的前若干个区间。"""
    nav = np.asarray(nav, dtype=np.float64)
    T = len(nav)
    if T < 5:
        return None
    peak = nav[0]
    peak_i = 0
    episodes = []
    i = 1
    while i < T:
        if nav[i] >= peak:
            peak = nav[i]
            peak_i = i
            i += 1
            continue
        trough_i = i
        j = i
        while j < T and nav[j] < peak:
            if nav[j] < nav[trough_i]:
                trough_i = j
            j += 1
        recovered = j < T
        depth = float(nav[trough_i] / peak - 1.0)
        episodes.append({
            "peak_date": dates[peak_i],
            "trough_date": dates[trough_i],
            "recovery_date": dates[j] if recovered else None,
            "depth": round(depth, 4),
            "days": int(trough_i - peak_i),
            "recovery_days": int(j - trough_i) if recovered else None,
            "recovered": recovered,
        })
        i = j
        if j < T:
            peak = nav[j]
            peak_i = j
    if not episodes:
        return None
    episodes.sort(key=lambda e: e["depth"])
    recovered_days = [e["recovery_days"] for e in episodes if e["recovered"]]
    peak_all = np.maximum.accumulate(nav)
    current = float(nav[-1] / peak_all[-1] - 1.0)
    return {
        "items": episodes[:top],
        "count": len(episodes),
        "avg_recovery_days": round(float(np.mean(recovered_days)), 1) if recovered_days else None,
        "current_drawdown": round(current, 4),
        "recovered_ratio": round(len(recovered_days) / len(episodes), 3),
    }


def _style_exposure(weights: np.ndarray, data: dict, max_points: int = 160) -> dict | None:
    """组合在动量/反转/波动/流动性/Beta 上的截面标准化暴露时间序列。"""
    close = data["close"]
    returns = data["returns"]
    amount = data["amount"]
    T, N = close.shape
    if T < 25 or N < 5:
        return None

    def shifted(x, k):
        out = np.full_like(x, np.nan)
        if k <= 0:
            return x.copy()
        if T > k:
            out[k:] = x[:-k]
        return out

    mom20 = close / np.maximum(shifted(close, 20), 1e-9) - 1.0
    rev5 = -(close / np.maximum(shifted(close, 5), 1e-9) - 1.0)
    vol20 = _rolling_vol(returns, 20)
    liq20 = np.log1p(np.maximum(
        np.nanmean(np.stack([shifted(amount, k) for k in range(20)]), axis=0), 0.0
    ))
    market = np.nanmean(returns, axis=1)

    idxs = [t for t in range(0, T, 5) if np.abs(weights[t]).sum() > 1e-9]
    if len(idxs) > max_points:
        idxs = idxs[:: max(1, len(idxs) // max_points)]

    def zscore(row):
        ok = np.isfinite(row)
        if ok.sum() < 3:
            return None
        m = float(np.mean(row[ok]))
        s = float(np.std(row[ok]))
        return (row - m) / max(s, 1e-9)

    names = ["动量", "短期反转", "波动率", "流动性", "Beta"]
    series = {name: [] for name in names}
    out_dates = []
    for t in idxs:
        cols = {
            "动量": mom20[t],
            "短期反转": rev5[t],
            "波动率": vol20[t],
            "流动性": liq20[t],
        }
        if t >= 60:
            win = returns[t - 59: t + 1]
            mkt = market[t - 59: t + 1]
            okm = np.isfinite(mkt)
            if okm.sum() >= 30 and win.shape[0] == 60:
                x = mkt - np.nanmean(mkt)
                denom = float(np.sum(x ** 2))
                if denom > 1e-12:
                    wm = np.nanmean(win, axis=0)
                    cov = np.nansum((win - wm) * x[:, None], axis=0) / denom
                    cols["Beta"] = cov
        w_row = weights[t]
        gross = float(np.abs(w_row).sum())
        out_dates.append(data["dates"][t])
        for name in names:
            z = zscore(cols[name]) if name in cols else None
            if z is None:
                series[name].append(None)
            else:
                series[name].append(round(float(np.sum(w_row * np.nan_to_num(z)) / gross), 3))
    if not out_dates:
        return None
    return {
        "dates": out_dates,
        "series": [{"name": name, "values": series[name]} for name in names],
    }


def _industry_exposure(weights: np.ndarray, rebalance: np.ndarray,
                       codes: list[str], industries: dict[str, str]) -> list[dict]:
    idxs = [i for i in range(len(rebalance))
            if rebalance[i] and np.abs(weights[i]).sum() > 1e-9]
    if not idxs:
        return []
    groups: dict[str, list[int]] = {}
    for j, code in enumerate(codes):
        groups.setdefault(industries.get(code, "未分类"), []).append(j)
    out = []
    for name, members in groups.items():
        last = float(np.sum(weights[idxs[-1], members]))
        avg = float(np.mean([np.sum(weights[i, members]) for i in idxs]))
        if abs(last) < 1e-6 and abs(avg) < 1e-6:
            continue
        out.append({"industry": name, "last": round(last, 4), "avg": round(avg, 4)})
    return sorted(out, key=lambda x: -abs(x["last"]))


def _position_cap(payload: dict) -> float:
    """最大总仓位比例，0 或 100 表示满仓。"""
    pct = float(payload.get("max_position_pct") or 100.0)
    if pct <= 0:
        return 1.0
    return max(0.05, min(pct, 100.0)) / 100.0


def _trade_friction_args(payload: dict):
    """解析交易摩擦控制参数：最小持有天数、单次调仓换手上限。"""
    hold = max(0, min(int(payload.get("min_hold_days") or 0), 120))
    cap = max(0.0, min(float(payload.get("turnover_cap_pct") or 0), 100.0)) / 100.0
    return hold, cap


def _industry_cap_args(payload: dict, data: dict):
    """解析行业总权重上限参数，返回 (industry_idx, cap, n_industries)。"""
    cap = max(0.0, min(float(payload.get("industry_cap") or 0), 100.0)) / 100.0
    if cap <= 0 or "industry_idx" not in data:
        return None, 0.0, 0
    return data["industry_idx"], cap, int(data["n_industries"])


def run_backtest(payload: dict) -> dict:
    data, factor, score = _prepare(payload)
    dates, codes = data["dates"], data["codes"]
    close = data["close"]
    valid = _apply_filters(data, payload.get("filters"))

    freq = max(1, min(int(payload.get("rebalance_days") or 10), 120))
    reb_mode, reb_weekday, reb_month_day = _rebalance_args(payload)
    rebalance, warmup = _rebalance_mask(
        score, valid, freq, dates, reb_mode, reb_weekday, reb_month_day
    )
    top_n = int(payload.get("top_n") or 0)
    top_frac = float(payload.get("top_pct") or 0.2)
    commission_bps = float(payload.get("commission_bps") or 15)
    stamp_duty_bps = float(payload.get("stamp_duty_bps") or 0)
    slippage_bps = float(payload.get("slippage_bps") or 0)
    if "stamp_duty_bps" not in payload and "slippage_bps" not in payload:
        # 兼容旧配置：commission_bps 视为单边全包成本
        stamp_duty_bps = 0.0
        slippage_bps = 0.0
    commission = commission_bps / 10000.0
    stamp_duty = stamp_duty_bps / 10000.0
    slippage = slippage_bps / 10000.0
    long_short = bool(payload.get("long_short") or False)
    benchmark_type = payload.get("benchmark") or "universe"
    weighting = payload.get("weighting") or "equal"
    if weighting not in ("equal", "score", "inverse_vol", "risk_parity"):
        weighting = "equal"
    max_weight = float(payload.get("max_weight") or 0) / 100.0
    vol = _rolling_vol(data["returns"]) if weighting == "inverse_vol" else None
    weight_override = None
    use_override = False
    if weighting == "risk_parity" and not long_short:
        weight_override = _risk_parity_targets(
            score, valid, data["returns"], rebalance, top_n, top_frac, max_weight
        )
        use_override = True
    timing_window = int(payload.get("timing_window") or 0) if payload.get("market_timing") else 0
    timing = _market_timing(dates, dates[0], dates[-1], timing_window) if timing_window > 1 else None
    trade_constraints = bool(payload.get("trade_constraints"))
    stop_loss = max(0.0, min(float(payload.get("stop_loss_pct") or 0), 50)) / 100.0
    take_profit = max(0.0, min(float(payload.get("take_profit_pct") or 0), 200)) / 100.0
    trailing_stop = max(0.0, min(float(payload.get("trailing_stop_pct") or 0), 50)) / 100.0
    position_cap = _position_cap(payload)
    industry_idx, industry_cap, n_industries = _industry_cap_args(payload, data)
    min_hold_days, turnover_cap = _trade_friction_args(payload)
    buy_block = sell_block = None
    if trade_constraints:
        rets = data["returns"]
        volume = data["volume"]
        buy_block = (rets >= 0.095) | (volume <= 0)
        sell_block = (rets <= -0.095) | (volume <= 0)

    sim = bt_engine.run_backtest(
        score=score, close=close, valid=valid, rebalance=rebalance,
        top_frac=top_frac, top_n=top_n, long_short=long_short,
        commission=commission, stamp_duty=stamp_duty, slippage=slippage,
        weighting=weighting,
        max_weight=max_weight, vol=vol, timing=timing,
        buy_block=buy_block, sell_block=sell_block, constraints=trade_constraints,
        weight_override=weight_override, use_override=use_override,
        stop_loss=stop_loss, take_profit=take_profit, trailing_stop=trailing_stop,
        industry_idx=industry_idx, industry_cap=industry_cap, n_industries=n_industries,
        min_hold_days=min_hold_days, turnover_cap=turnover_cap, position_cap=position_cap,
    )
    bench = sim["benchmark"]
    benchmark_name = "股票池等权"
    if benchmark_type == "market":
        try:
            bench = _market_benchmark(dates, dates[0], dates[-1])
            sim["benchmark"] = bench
            benchmark_name = "全市场等权"
        except Exception:
            benchmark_type = "universe"
    elif benchmark_type.startswith("universe:"):
        try:
            uid = int(benchmark_type.split(":", 1)[1])
            found = next((u for u in database.list_universes() if u["id"] == uid), None)
            if not found:
                raise ValueError("自选池不存在")
            bench = _custom_benchmark(dates, found["codes"])
            sim["benchmark"] = bench
            benchmark_name = f"{found['name']} 等权"
        except Exception:
            benchmark_type = "universe"
            benchmark_name = "股票池等权"
    elif benchmark_type.startswith("index:"):
        code = benchmark_type.split(":", 1)[1]
        try:
            bench = _index_benchmark(dates, code)
            sim["benchmark"] = bench
            matched = next((i for i in database.list_indices() if i["code"] == code), None)
            benchmark_name = matched["name"] if matched else code
        except Exception:
            benchmark_type = "universe"
            benchmark_name = "股票池等权"

    metrics = sim["metrics"]
    bench_total = float(bench[-1] - 1)
    bench_annual = float((bench[-1]) ** (252 / max(len(bench) - 1, 1)) - 1)
    monthly = _monthly_returns(dates, sim["nav"])
    metrics.update({
        "benchmark_total_return": round(bench_total, 4),
        "benchmark_annual_return": round(bench_annual, 4),
        "excess_annual_return": round(metrics["annual_return"] - bench_annual, 4),
    })
    metrics.update(_extended_metrics(
        np.asarray(sim["returns"]), np.asarray(bench), monthly, np.asarray(sim["nav"])
    ))
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    metrics["trade_days"] = int(metrics["trade_days"])

    w = sim["weights"]
    metrics["rebalance_count"] = int((np.abs(w[rebalance]).sum(axis=1) > 1e-9).sum()) if rebalance.any() else 0
    metrics["avg_positions"] = int(np.mean(
        (np.abs(w[rebalance]) > 1e-6).sum(axis=1)
    )) if rebalance.any() else 0

    oos = None
    split = float(payload.get("oos_split") or 0)
    if 0.05 < split < 0.95:
        total = len(dates)
        k = max(30, min(total - 30, int(total * (1.0 - split))))
        net_arr = np.asarray(sim["returns"], dtype=np.float64)
        nav_arr = np.asarray(sim["nav"], dtype=np.float64)
        oos = {
            "split_date": dates[k],
            "in_sample": _segment_metrics(net_arr[:k - 1], nav_arr[:k] / nav_arr[0]),
            "out_of_sample": _segment_metrics(net_arr[k - 1:], nav_arr[k:] / nav_arr[k]),
        }

    attribution = _attribution(w, close, codes, data["names"], data["industries"])
    brinson = _brinson(w, close, valid, codes, data["industries"])
    bootstrap = _bootstrap(np.asarray(sim["returns"]))

    analysis_res = analysis.analyze(
        score, close, valid, dates, codes, data["names"], horizon=5,
        industries=data["industries"],
    )

    def r(arr, nd=5):
        return [round(float(x), nd) if np.isfinite(x) else None for x in arr]

    cfg_name = payload.get("name") or f"{factor['label']} 回测"
    result = {
        "dates": dates,
        "nav": r(sim["nav"], 4),
        "benchmark": r(bench, 4),
        "drawdown": r(sim["drawdown"], 4),
        "drawdowns": _drawdown_episodes(dates, np.asarray(sim["nav"])),
        "daily_returns": r(sim["returns"], 5),
        "turnover": r(sim["turnover"], 4),
        "metrics": metrics,
        "monthly": monthly,
        "annual_returns": _annual_returns(dates, np.asarray(sim["nav"])),
        "annual_returns_benchmark": _annual_returns(dates, np.asarray(bench)),
        "holdings": _holdings(dates, w, rebalance, data["names"], codes),
        "industry_exposure": _industry_exposure(w, rebalance, codes, data["industries"]),
        "style_exposure": _style_exposure(w, data),
        "attribution": attribution,
        "brinson": brinson,
        "bootstrap": bootstrap,
        "rolling_risk": _rolling_risk(np.asarray(sim["returns"]), dates),
        "return_hist": _return_hist(np.asarray(sim["returns"])),
        "trades": _trades(w, rebalance, dates, codes, data["names"]),
        "oos": oos,
        "factor": {
            "label": factor["label"],
            "expression": factor["expression"],
            "direction": factor["direction"],
            "factors": factor["factors"],
            "preprocess": factor["preprocess"],
            "ic": analysis_res["ic"],
            "ic_by_year": analysis_res["ic_by_year"],
            "ic_decay": analysis_res["ic_decay"],
            "industry_ic": analysis_res["industry_ic"],
            "quantile_returns": analysis_res["quantile_returns"],
        },
        "universe": {
            "stocks": len(codes),
            "start": dates[0],
            "end": dates[-1],
            "days": len(dates),
            "warmup": warmup,
        },
        "config": {
            "name": cfg_name,
            "codes": codes,
            "start": dates[0],
            "end": dates[-1],
            "rebalance_days": freq,
            "rebalance_mode": reb_mode,
            "rebalance_weekday": reb_weekday,
            "rebalance_month_day": reb_month_day,
            "top_n": top_n,
            "top_pct": top_frac,
            "commission_bps": commission_bps,
            "stamp_duty_bps": stamp_duty_bps,
            "slippage_bps": slippage_bps,
            "buy_cost_bps": round(commission_bps + slippage_bps, 2),
            "sell_cost_bps": round(commission_bps + stamp_duty_bps + slippage_bps, 2),
            "long_short": long_short,
            "benchmark": benchmark_type,
            "benchmark_name": benchmark_name,
            "weighting": weighting,
            "max_weight": max_weight * 100,
            "industry_cap": industry_cap * 100,
            "max_position_pct": position_cap * 100,
            "min_hold_days": min_hold_days,
            "turnover_cap_pct": turnover_cap * 100,
            "market_timing": timing_window > 1,
            "timing_window": timing_window,
            "trade_constraints": trade_constraints,
            "stop_loss_pct": stop_loss * 100,
            "take_profit_pct": take_profit * 100,
            "trailing_stop_pct": trailing_stop * 100,
            "oos_split": split,
            "filters": payload.get("filters") or {},
            "preprocess": factor["preprocess"],
            "factors": factor["factors"],
        },
    }
    bt_id = database.save_backtest(
        cfg_name,
        {**result["config"], "factor": result["factor"]},
        metrics, result,
    )
    result["id"] = bt_id
    result["saved"] = True
    return result


def optimize_backtest(payload: dict) -> dict:
    """参数扫描：调仓周期 × 持仓数量，返回绩效网格（不写入历史）。"""
    data, factor, score = _prepare(payload)
    valid = _apply_filters(data, payload.get("filters"))
    rebs = sorted({int(x) for x in (payload.get("rebalance_grid") or [5, 10, 21])
                   if 1 <= int(x) <= 120})[:4]
    tops = sorted({int(x) for x in (payload.get("top_grid") or [5, 10, 20])
                   if 1 <= int(x) <= 100})[:4]
    if not rebs or not tops:
        raise ValueError("参数扫描范围不能为空")
    commission_bps = float(payload.get("commission_bps") or 15)
    stamp_duty_bps = float(payload.get("stamp_duty_bps") or 0)
    slippage_bps = float(payload.get("slippage_bps") or 0)
    if "stamp_duty_bps" not in payload and "slippage_bps" not in payload:
        # 兼容旧配置：commission_bps 视为单边全包成本
        stamp_duty_bps = 0.0
        slippage_bps = 0.0
    commission = commission_bps / 10000.0
    stamp_duty = stamp_duty_bps / 10000.0
    slippage = slippage_bps / 10000.0
    long_short = bool(payload.get("long_short") or False)
    weighting = payload.get("weighting") or "equal"
    if weighting not in ("equal", "score", "inverse_vol", "risk_parity"):
        weighting = "equal"
    max_weight = float(payload.get("max_weight") or 0) / 100.0
    vol = _rolling_vol(data["returns"]) if weighting == "inverse_vol" else None
    timing_window = int(payload.get("timing_window") or 0) if payload.get("market_timing") else 0
    timing = _market_timing(data["dates"], data["dates"][0], data["dates"][-1],
                            timing_window) if timing_window > 1 else None
    trade_constraints = bool(payload.get("trade_constraints"))
    stop_loss = max(0.0, min(float(payload.get("stop_loss_pct") or 0), 50)) / 100.0
    take_profit = max(0.0, min(float(payload.get("take_profit_pct") or 0), 200)) / 100.0
    trailing_stop = max(0.0, min(float(payload.get("trailing_stop_pct") or 0), 50)) / 100.0
    position_cap = _position_cap(payload)
    industry_idx, industry_cap, n_industries = _industry_cap_args(payload, data)
    min_hold_days, turnover_cap = _trade_friction_args(payload)
    buy_block = sell_block = None
    if trade_constraints:
        rets = data["returns"]
        volume = data["volume"]
        buy_block = (rets >= 0.095) | (volume <= 0)
        sell_block = (rets <= -0.095) | (volume <= 0)

    cells = []
    for freq in rebs:
        rebalance, _ = _rebalance_mask(score, valid, freq)
        for top_n in tops:
            override = None
            use_override = False
            if weighting == "risk_parity" and not long_short:
                override = _risk_parity_targets(
                    score, valid, data["returns"], rebalance, top_n, 0.2, max_weight
                )
                use_override = True
            sim = bt_engine.run_backtest(
                score=score, close=data["close"], valid=valid, rebalance=rebalance,
                top_frac=0.2, top_n=top_n, long_short=long_short,
                commission=commission, stamp_duty=stamp_duty, slippage=slippage,
                weighting=weighting,
                max_weight=max_weight, vol=vol, timing=timing,
                buy_block=buy_block, sell_block=sell_block, constraints=trade_constraints,
                weight_override=override, use_override=use_override,
                stop_loss=stop_loss, take_profit=take_profit, trailing_stop=trailing_stop,
                industry_idx=industry_idx, industry_cap=industry_cap, n_industries=n_industries,
                min_hold_days=min_hold_days, turnover_cap=turnover_cap, position_cap=position_cap,
            )
            m = sim["metrics"]
            cells.append({
                "rebalance_days": freq,
                "top_n": top_n,
                "annual_return": round(m["annual_return"], 4),
                "total_return": round(m["total_return"], 4),
                "sharpe": round(m["sharpe"], 3),
                "max_drawdown": round(m["max_drawdown"], 4),
                "annual_turnover": round(m["annual_turnover"], 2),
            })
    best = max(cells, key=lambda c: c["sharpe"])
    return {
        "rebalance_days": rebs,
        "top_n": tops,
        "cells": cells,
        "best": best,
        "factor": {"label": factor["label"], "factors": factor["factors"]},
        "base": {
            "start": data["dates"][0],
            "end": data["dates"][-1],
            "stocks": len(data["codes"]),
            "commission_bps": commission * 10000,
            "long_short": long_short,
        },
    }


# --------------------------------------------------------------------------- #
# 最新持仓信号
# --------------------------------------------------------------------------- #
def latest_signal(payload: dict) -> dict:
    """按当前因子与过滤条件生成最新交易日的目标持仓。"""
    data, factor, score = _prepare(payload)
    valid = _apply_filters(data, payload.get("filters"))
    dates, codes = data["dates"], data["codes"]
    usable = (np.isfinite(score) & valid).sum(axis=1)
    min_stocks = max(2, min(10, len(codes)))
    idxs = np.where(usable >= min_stocks)[0]
    if idxs.size == 0:
        raise ValueError("当前因子没有足够的有效值，无法生成信号")
    t = int(idxs[-1])

    row = np.where(valid[t] & np.isfinite(score[t]), score[t], -np.inf)
    order = np.argsort(-row)
    n_valid = int(valid[t].sum())
    top_n = int(payload.get("top_n") or 0)
    if top_n > 0:
        k = min(top_n, n_valid)
    else:
        k = int(round(float(payload.get("top_pct") or 0.2) * n_valid))
    k = max(1, k)
    selected = order[:k]

    weighting = payload.get("weighting") or "equal"
    if weighting not in ("equal", "score", "inverse_vol", "risk_parity"):
        weighting = "equal"
    if weighting == "risk_parity":
        sub = np.nan_to_num(data["returns"][max(0, t - 60):t][:, selected], nan=0.0)
        if sub.shape[0] >= 20 and k >= 2:
            cov = np.atleast_2d(np.cov(sub, rowvar=False))
            cov = 0.7 * cov + 0.3 * np.diag(np.diag(cov))
            weights = _erc_weights(cov)
        else:
            weights = np.ones(k) / k
    else:
        if weighting == "score":
            raw = (k - np.arange(k)).astype(np.float64)
        elif weighting == "inverse_vol":
            vol = _rolling_vol(data["returns"])[t, selected]
            raw = 1.0 / np.maximum(vol, 1e-4)
        else:
            raw = np.ones(k, dtype=np.float64)
        weights = raw / raw.sum()
    max_weight = float(payload.get("max_weight") or 0) / 100.0
    if max_weight > 0:
        cap = max(max_weight, 1.0 / k)
        for _ in range(4):
            weights = np.minimum(weights, cap)
            weights = weights / max(weights.sum(), 1e-12)

    industry_cap = max(0.0, min(float(payload.get("industry_cap") or 0), 100.0)) / 100.0
    if industry_cap > 0:
        ind_of = [data["industries"].get(codes[j], "未分类") for j in selected]
        totals: dict[str, float] = {}
        for name, w in zip(ind_of, weights):
            totals[name] = totals.get(name, 0.0) + float(w)
        scale = np.array([min(1.0, industry_cap / totals[name]) if totals[name] > 1e-9 else 1.0
                          for name in ind_of])
        weights = weights * scale  # 超出部分留作现金

    position_cap = _position_cap(payload)
    if position_cap < 1.0:
        total = float(weights.sum())
        if total > position_cap:
            weights = weights * (position_cap / total)

    close = data["close"]
    items = []
    for rank, j in enumerate(selected):
        prev_close = close[t - 1, j] if t > 0 else np.nan
        change = (close[t, j] / prev_close - 1.0) if np.isfinite(prev_close) and prev_close else np.nan
        items.append({
            "code": codes[j],
            "name": data["names"].get(codes[j], codes[j]),
            "industry": data["industries"].get(codes[j], "未分类"),
            "weight": round(float(weights[rank]), 4),
            "score": round(float(score[t, j]), 4),
            "close": round(float(close[t, j]), 2) if np.isfinite(close[t, j]) else None,
            "change_pct": round(float(change) * 100, 2) if np.isfinite(change) else None,
        })
    items.sort(key=lambda x: -x["weight"])
    return {
        "as_of": dates[t],
        "items": items,
        "factor": {"label": factor["label"], "factors": factor["factors"],
                   "preprocess": factor["preprocess"]},
        "config": {
            "weighting": weighting,
            "top_n": k,
            "max_weight": max_weight * 100,
            "industry_cap": industry_cap * 100,
            "max_position_pct": position_cap * 100,
            "benchmark": payload.get("benchmark") or "universe",
        },
    }


def screen_stocks(payload: dict) -> dict:
    """多条件选股：按交易日对因子做排名/阈值过滤，条件支持 AND / OR 组合。"""
    codes = payload.get("codes") or None
    data = panel.build_panel(codes, payload.get("start") or None, payload.get("end") or None)
    valid = _apply_filters(data, payload.get("filters"))
    conds = payload.get("conditions") or []
    if not conds:
        raise ValueError("请至少添加一个选股条件")
    if len(conds) > 6:
        raise ValueError("最多支持 6 个选股条件")
    logic = payload.get("logic") or "and"
    if logic not in ("and", "or"):
        logic = "and"

    specs = []
    for c in conds:
        if not isinstance(c, dict):
            c = {"factor_id": c}
        spec = resolve_factor(c)
        raw = np.asarray(compute(spec["expression"], data) * spec["direction"], dtype=np.float64)
        mode = c.get("mode") or "rank_top"
        if mode not in ("rank_top", "rank_bottom", "gte", "lte"):
            mode = "rank_top"
        scope = c.get("scope") or "all"
        if scope not in ("all", "industry"):
            scope = "all"
        try:
            value = float(c.get("value"))
        except (TypeError, ValueError):
            raise ValueError(f"条件「{spec['label']}」的阈值无效")
        specs.append({"spec": spec, "raw": raw, "mode": mode, "value": value, "scope": scope})

    base_counts = (np.isfinite(specs[0]["raw"]) & valid).sum(axis=1)
    date = payload.get("date")
    if date:
        if date not in data["dates"]:
            raise ValueError("所选日期不在数据范围内")
        t = data["dates"].index(date)
        if base_counts[t] < 2:
            raise ValueError("所选日期有效数据不足")
    else:
        cand = np.where(base_counts >= max(2, min(10, len(data["codes"]))))[0]
        if cand.size == 0:
            raise ValueError("没有足够的有效数据")
        t = int(cand[-1])

    N = len(data["codes"])
    valid_t = valid[t]
    mask = valid_t.copy()
    passes: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    cond_detail = []
    for item in specs:
        row = item["raw"][t]
        base = (valid_t if logic == "or" else mask) & np.isfinite(row)
        if not base.any():
            raise ValueError(f"条件「{item['spec']['label']}」在所选日期没有有效值")
        mode, value = item["mode"], item["value"]
        if mode in ("rank_top", "rank_bottom"):
            pct = max(0.1, min(value, 100.0))
            key = np.where(base, -row, np.inf) if mode == "rank_top" else np.where(base, row, np.inf)
            ranks = np.full(N, np.inf)
            if item["scope"] == "industry":
                ind = np.asarray(data["industry_idx"])
                for g in np.unique(ind[base]):
                    pos = np.where(base & (ind == g))[0]
                    order = pos[np.argsort(key[pos])]
                    n = len(order)
                    if n:
                        ranks[order] = (np.arange(n) + 1.0) / n
            else:
                order = np.argsort(key)
                n = int(base.sum())
                ranks[order[:n]] = (np.arange(n) + 1.0) / n
            passed = base & (ranks <= pct / 100.0)
        else:
            passed = base & ((row >= value) if mode == "gte" else (row <= value))
        vals = row[base]
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        z = np.where(base, (row - mu) / max(sd, 1e-9), np.nan)
        prefer_high = mode in ("rank_top", "gte")
        zs.append(z if prefer_high else -np.asarray(z))
        rows.append(row)
        cond_detail.append({
            "label": item["spec"]["label"],
            "mode": mode,
            "value": value,
            "passed": int(passed.sum()),
        })
        passes.append(passed)
        if logic == "and":
            mask = passed

    if logic == "or":
        mask = np.zeros(N, dtype=bool)
        for p in passes:
            mask |= p

    idxs = np.where(mask)[0]
    close = data["close"]
    prev_close = close[t - 1] if t > 0 else np.full(N, np.nan)
    score = np.nansum(np.stack(zs), axis=0) if idxs.size else np.zeros(N)
    order = idxs[np.argsort(-score[idxs])] if idxs.size else np.array([], dtype=int)
    limit = max(1, min(int(payload.get("limit") or 50), 200))

    items = []
    for j in order[:limit]:
        change = (close[t, j] / prev_close[j] - 1.0) if np.isfinite(prev_close[j]) and prev_close[j] else np.nan
        items.append({
            "code": data["codes"][j],
            "name": data["names"].get(data["codes"][j], data["codes"][j]),
            "industry": data["industries"].get(data["codes"][j], "未分类"),
            "score": round(float(score[j]), 3),
            "close": round(float(close[t, j]), 2) if np.isfinite(close[t, j]) else None,
            "change_pct": round(float(change) * 100, 2) if np.isfinite(change) else None,
            "values": [
                round(float(row[j]), 4) if np.isfinite(row[j]) else None
                for row in rows
            ],
        })
    return {
        "as_of": data["dates"][t],
        "logic": logic,
        "count": int(mask.sum()),
        "returned": len(items),
        "universe_size": int(valid_t.sum()),
        "conditions": cond_detail,
        "labels": [item["spec"]["label"] for item in specs],
        "items": items,
    }


def batch_backtest(payload: dict) -> dict:
    """用同一套配置批量回测全部内置因子，输出排行榜。"""
    codes = payload.get("codes") or None
    data = panel.build_panel(codes, payload.get("start") or None, payload.get("end") or None)
    valid = _apply_filters(data, payload.get("filters"))
    options = payload.get("preprocess") or {}
    freq = max(1, min(int(payload.get("rebalance_days") or 10), 120))
    top_n = int(payload.get("top_n") or 0)
    top_frac = float(payload.get("top_pct") or 0.2)
    commission_bps = float(payload.get("commission_bps") or 15)
    stamp_duty_bps = float(payload.get("stamp_duty_bps") or 0)
    slippage_bps = float(payload.get("slippage_bps") or 0)
    if "stamp_duty_bps" not in payload and "slippage_bps" not in payload:
        stamp_duty_bps = slippage_bps = 0.0
    commission = commission_bps / 10000.0
    stamp_duty = stamp_duty_bps / 10000.0
    slippage = slippage_bps / 10000.0
    long_short = bool(payload.get("long_short") or False)
    weighting = payload.get("weighting") or "equal"
    if weighting not in ("equal", "score", "inverse_vol", "risk_parity"):
        weighting = "equal"
    max_weight = float(payload.get("max_weight") or 0) / 100.0
    vol = _rolling_vol(data["returns"]) if weighting == "inverse_vol" else None
    timing_window = int(payload.get("timing_window") or 0) if payload.get("market_timing") else 0
    timing = _market_timing(data["dates"], data["dates"][0], data["dates"][-1],
                            timing_window) if timing_window > 1 else None
    trade_constraints = bool(payload.get("trade_constraints"))
    stop_loss = max(0.0, min(float(payload.get("stop_loss_pct") or 0), 50)) / 100.0
    take_profit = max(0.0, min(float(payload.get("take_profit_pct") or 0), 200)) / 100.0
    trailing_stop = max(0.0, min(float(payload.get("trailing_stop_pct") or 0), 50)) / 100.0
    position_cap = _position_cap(payload)
    industry_idx, industry_cap, n_industries = _industry_cap_args(payload, data)
    min_hold_days, turnover_cap = _trade_friction_args(payload)
    reb_mode, reb_weekday, reb_month_day = _rebalance_args(payload)
    buy_block = sell_block = None
    if trade_constraints:
        rets = data["returns"]
        volume = data["volume"]
        buy_block = (rets >= 0.095) | (volume <= 0)
        sell_block = (rets <= -0.095) | (volume <= 0)

    items = []
    for meta in builtin.all_factors():
        try:
            raw = compute(meta["expression"], data)
            score = np.asarray(pre_mod.apply_preprocess(
                raw * meta["direction"], data, options, force_standardize=False
            ))
            rebalance, warmup = _rebalance_mask(
                score, valid, freq, data["dates"], reb_mode, reb_weekday, reb_month_day
            )
            override = None
            use_override = False
            if weighting == "risk_parity" and not long_short:
                override = _risk_parity_targets(
                    score, valid, data["returns"], rebalance, top_n, top_frac, max_weight
                )
                use_override = True
            sim = bt_engine.run_backtest(
                score=score, close=data["close"], valid=valid, rebalance=rebalance,
                top_frac=top_frac, top_n=top_n, long_short=long_short,
                commission=commission, stamp_duty=stamp_duty, slippage=slippage,
                weighting=weighting, max_weight=max_weight, vol=vol, timing=timing,
                buy_block=buy_block, sell_block=sell_block, constraints=trade_constraints,
                weight_override=override, use_override=use_override,
                stop_loss=stop_loss, take_profit=take_profit, trailing_stop=trailing_stop,
                industry_idx=industry_idx, industry_cap=industry_cap, n_industries=n_industries,
                min_hold_days=min_hold_days, turnover_cap=turnover_cap, position_cap=position_cap,
            )
            stats = analysis.quick_ic(score, data["close"], valid, 5)
        except Exception:
            continue
        m = sim["metrics"]
        items.append({
            "id": meta["id"],
            "label": meta["label"],
            "category": meta["category"],
            "annual_return": round(m["annual_return"], 4),
            "total_return": round(m["total_return"], 4),
            "sharpe": round(m["sharpe"], 3),
            "max_drawdown": round(m["max_drawdown"], 4),
            "annual_turnover": round(m["annual_turnover"], 2),
            "ic": stats["mean"],
            "icir": stats["ir"],
        })
    items.sort(key=lambda x: -x["sharpe"])
    return {
        "items": items,
        "config": {
            "start": data["dates"][0],
            "end": data["dates"][-1],
            "stocks": len(data["codes"]),
            "rebalance_days": freq,
            "top_n": top_n,
            "top_pct": top_frac,
            "weighting": weighting,
        },
    }


def cost_sensitivity(payload: dict) -> dict:
    """交易成本敏感性：不同单边综合成本下的绩效变化。"""
    data, factor, score = _prepare(payload)
    valid = _apply_filters(data, payload.get("filters"))
    freq = max(1, min(int(payload.get("rebalance_days") or 10), 120))
    reb_mode, reb_weekday, reb_month_day = _rebalance_args(payload)
    rebalance, _ = _rebalance_mask(
        score, valid, freq, data["dates"], reb_mode, reb_weekday, reb_month_day
    )
    top_n = int(payload.get("top_n") or 0)
    top_frac = float(payload.get("top_pct") or 0.2)
    long_short = bool(payload.get("long_short") or False)
    weighting = payload.get("weighting") or "equal"
    if weighting not in ("equal", "score", "inverse_vol"):
        weighting = "equal"
    max_weight = float(payload.get("max_weight") or 0) / 100.0
    vol = _rolling_vol(data["returns"]) if weighting == "inverse_vol" else None
    timing_window = int(payload.get("timing_window") or 0) if payload.get("market_timing") else 0
    timing = _market_timing(data["dates"], data["dates"][0], data["dates"][-1],
                            timing_window) if timing_window > 1 else None
    trade_constraints = bool(payload.get("trade_constraints"))
    buy_block = sell_block = None
    if trade_constraints:
        rets = data["returns"]
        volume = data["volume"]
        buy_block = (rets >= 0.095) | (volume <= 0)
        sell_block = (rets <= -0.095) | (volume <= 0)
    stop_loss = max(0.0, min(float(payload.get("stop_loss_pct") or 0), 50)) / 100.0
    take_profit = max(0.0, min(float(payload.get("take_profit_pct") or 0), 200)) / 100.0
    trailing_stop = max(0.0, min(float(payload.get("trailing_stop_pct") or 0), 50)) / 100.0
    position_cap = _position_cap(payload)
    industry_idx, industry_cap, n_industries = _industry_cap_args(payload, data)

    levels = [0, 5, 10, 15, 20, 30]
    items = []
    for bps in levels:
        sim = bt_engine.run_backtest(
            score=score, close=data["close"], valid=valid, rebalance=rebalance,
            top_frac=top_frac, top_n=top_n, long_short=long_short,
            commission=bps / 10000.0, stamp_duty=0.0, slippage=0.0,
            weighting=weighting, max_weight=max_weight, vol=vol, timing=timing,
            buy_block=buy_block, sell_block=sell_block, constraints=trade_constraints,
            stop_loss=stop_loss, take_profit=take_profit, trailing_stop=trailing_stop,
            industry_idx=industry_idx, industry_cap=industry_cap, n_industries=n_industries,
            min_hold_days=min_hold_days, turnover_cap=turnover_cap, position_cap=position_cap,
        )
        m = sim["metrics"]
        items.append({
            "cost_bps": bps,
            "annual_return": round(m["annual_return"], 4),
            "total_return": round(m["total_return"], 4),
            "sharpe": round(m["sharpe"], 3),
            "max_drawdown": round(m["max_drawdown"], 4),
        })
    current = float(payload.get("commission_bps") or 15)
    return {
        "items": items,
        "current_cost_bps": current,
        "factor": {"label": factor["label"]},
        "config": {"start": data["dates"][0], "end": data["dates"][-1],
                   "stocks": len(data["codes"]), "rebalance_days": freq},
    }


def walk_forward(payload: dict) -> dict:
    """滚动验证：每折在样本内选择最优参数，再应用于下一段样本外并拼接净值。"""
    data, factor, score = _prepare(payload)
    dates, close = data["dates"], data["close"]
    valid_all = _apply_filters(data, payload.get("filters"))
    T = len(dates)
    folds = max(2, min(int(payload.get("wf_folds") or 4), 6))
    seg = T // (folds + 1)
    if seg < 30:
        raise ValueError("样本区间过短，无法进行滚动验证")
    bounds = [i * seg for i in range(folds + 1)] + [T]
    top_grid = sorted({int(x) for x in (payload.get("wf_top_grid") or [5, 10, 20])
                       if 1 <= int(x) <= 100})[:4]
    reb_grid = sorted({int(x) for x in (payload.get("wf_reb_grid") or [5, 10, 21])
                       if 1 <= int(x) <= 60})[:4]
    if not top_grid or not reb_grid:
        raise ValueError("滚动验证参数范围不能为空")

    commission_bps = float(payload.get("commission_bps") or 15)
    stamp_duty_bps = float(payload.get("stamp_duty_bps") or 0)
    slippage_bps = float(payload.get("slippage_bps") or 0)
    if "stamp_duty_bps" not in payload and "slippage_bps" not in payload:
        stamp_duty_bps = slippage_bps = 0.0
    weighting = payload.get("weighting") or "equal"
    if weighting not in ("equal", "score", "inverse_vol"):
        weighting = "equal"
    max_weight = float(payload.get("max_weight") or 0) / 100.0
    industry_idx, industry_cap, n_industries = _industry_cap_args(payload, data)
    position_cap = _position_cap(payload)
    min_hold_days, turnover_cap = _trade_friction_args(payload)
    stop_loss = max(0.0, min(float(payload.get("stop_loss_pct") or 0), 50)) / 100.0
    take_profit = max(0.0, min(float(payload.get("take_profit_pct") or 0), 200)) / 100.0
    trailing_stop = max(0.0, min(float(payload.get("trailing_stop_pct") or 0), 50)) / 100.0
    trade_constraints = bool(payload.get("trade_constraints"))
    timing_window = int(payload.get("timing_window") or 0) if payload.get("market_timing") else 0
    timing = _market_timing(dates, dates[0], dates[-1], timing_window) if timing_window > 1 else None
    buy_block = sell_block = None
    if trade_constraints:
        rets = data["returns"]
        volume = data["volume"]
        buy_block = (rets >= 0.095) | (volume <= 0)
        sell_block = (rets <= -0.095) | (volume <= 0)

    def run_window(i0: int, i1: int, top_n: int, reb: int):
        sl = slice(i0, i1)
        rebalance, _ = _rebalance_mask(score[sl], valid_all[sl], reb)
        return bt_engine.run_backtest(
            score=score[sl], close=close[sl], valid=valid_all[sl], rebalance=rebalance,
            top_frac=0.2, top_n=top_n, long_short=False,
            commission=commission_bps / 10000.0,
            stamp_duty=stamp_duty_bps / 10000.0,
            slippage=slippage_bps / 10000.0,
            weighting=weighting, max_weight=max_weight,
            vol=_rolling_vol(data["returns"][sl]) if weighting == "inverse_vol" else None,
            timing=timing[sl] if timing is not None else None,
            buy_block=buy_block[sl] if buy_block is not None else None,
            sell_block=sell_block[sl] if sell_block is not None else None,
            constraints=trade_constraints,
            stop_loss=stop_loss, take_profit=take_profit, trailing_stop=trailing_stop,
            industry_idx=industry_idx, industry_cap=industry_cap, n_industries=n_industries,
            min_hold_days=min_hold_days, turnover_cap=turnover_cap, position_cap=position_cap,
        )

    fold_rows, stitched, stitched_dates = [], [], []
    for k in range(folds):
        i0, i1, i2 = bounds[k], bounds[k + 1], bounds[k + 2]
        best = None
        for top_n in top_grid:
            for reb in reb_grid:
                try:
                    sim = run_window(i0, i1, top_n, reb)
                except (ValueError, FloatingPointError):
                    continue
                m = sim["metrics"]
                score_key = float(m["sharpe"])
                if best is None or score_key > best["sharpe"]:
                    best = {"top_n": top_n, "reb": reb, "sharpe": score_key,
                            "annual_return": float(m["annual_return"])}
        if best is None:
            raise ValueError("样本内窗口太短，无法完成参数选择")
        sim_oos = run_window(i1, i2, best["top_n"], best["reb"])
        rets = np.asarray(sim_oos["returns"], dtype=np.float64)
        nav = np.asarray(sim_oos["nav"], dtype=np.float64)
        stitched.append(rets)
        stitched_dates.extend(dates[i1:i2] if k == 0 else dates[i1 + 1:i2])
        fold_rows.append({
            "fold": k + 1,
            "is_start": dates[i0], "is_end": dates[i1 - 1],
            "oos_start": dates[i1], "oos_end": dates[i2 - 1],
            "top_n": best["top_n"],
            "rebalance_days": best["reb"],
            "is_annual_return": round(best["annual_return"], 4),
            "is_sharpe": round(best["sharpe"], 3),
            "oos_return": round(float(nav[-1] - 1.0), 4),
            "oos_sharpe": sim_oos["metrics"]["sharpe"],
            "oos_max_drawdown": sim_oos["metrics"]["max_drawdown"],
        })

    net = np.concatenate(stitched) if stitched else np.array([0.0])
    nav = np.concatenate([[1.0], np.cumprod(1.0 + net)])
    metrics = _segment_metrics(net, nav)
    metrics.update({
        "total_return": round(float(nav[-1] - 1.0), 4),
        "annual_volatility": round(float(np.std(net) * np.sqrt(252)), 4),
    })
    return {
        "dates": stitched_dates,
        "nav": [round(float(v), 4) for v in nav],
        "metrics": metrics,
        "folds": fold_rows,
        "factor": {"label": factor["label"], "factors": factor["factors"]},
        "config": {
            "folds": folds,
            "top_grid": top_grid,
            "reb_grid": reb_grid,
            "start": dates[0],
            "end": dates[-1],
            "stocks": len(data["codes"]),
        },
    }


def get_factor_series(payload: dict) -> dict:
    """单只股票的因子/价格序列，用于前端叠加展示。"""
    data, factor, score = _prepare(payload)
    code = payload.get("code")
    if code not in data["codes"]:
        code = data["codes"][0]
    j = data["codes"].index(code)
    close = data["close"][:, j]
    vals = score[:, j]
    out_dates, out_price, out_factor = [], [], []
    for i in range(len(data["dates"])):
        if np.isfinite(close[i]) and np.isfinite(vals[i]):
            out_dates.append(data["dates"][i])
            out_price.append(round(float(close[i]), 3))
            out_factor.append(round(float(vals[i]), 5))
    return {
        "code": code,
        "name": data["names"].get(code, code),
        "dates": out_dates,
        "price": out_price,
        "factor": out_factor,
        "factor_meta": {
            "label": factor["label"],
            "expression": factor["expression"],
        },
    }


def stock_bars(code: str, start: str | None = None, end: str | None = None) -> dict:
    rows = database.get_bars_bulk([code], start or "1900-01-01", end or "2999-12-31")
    name = database.get_stock_names([code]).get(code, code)
    return {
        "code": code,
        "name": name,
        "dates": [r["date"] for r in rows],
        "open": [r["open"] for r in rows],
        "high": [r["high"] for r in rows],
        "low": [r["low"] for r in rows],
        "close": [r["close"] for r in rows],
        "volume": [r["volume"] for r in rows],
    }
