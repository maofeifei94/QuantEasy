"""业务编排层：数据同步、因子预览、回测执行。"""
from __future__ import annotations

import datetime as dt
import threading
import traceback
from typing import Any

import numpy as np

from . import database, jobs, panel
from .backtest import engine as bt_engine
from .datasources import miniqmt, synthetic
from .factors import analysis, builtin
from .factors.engine import FactorError, compute, validate


# --------------------------------------------------------------------------- #
# 数据同步
# --------------------------------------------------------------------------- #
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
            stocks, bars = synthetic.generate(
                start=gen_start, end=gen_end,
                n_stocks=int(payload.get("demo_stocks") or 150),
                progress=progress,
            )
            data = {"stocks": stocks, "bars": bars, "source": "synthetic"}

        jobs.update(job_id, progress=0.75, message="写入 SQLite 数据库")
        if replace:
            database.clear_market_data()
        database.upsert_stocks(data["stocks"])
        total = len(data["bars"])
        for i in range(0, total, 20000):
            database.insert_bars(data["bars"][i: i + 20000])
            jobs.update(job_id, progress=0.75 + 0.2 * min(1.0, (i + 20000) / max(total, 1)),
                        message=f"写入日线 {min(i + 20000, total)}/{total}")

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
    try:
        validate(expression)
    except FactorError:
        raise
    return {
        "id": factor_id,
        "label": label,
        "description": description,
        "expression": expression,
        "direction": direction,
        "is_custom": is_custom,
    }


def _prepare(payload: dict):
    codes = payload.get("codes") or None
    if isinstance(codes, str):
        codes = [c.strip() for c in codes.split(",") if c.strip()]
    if codes is not None and len(codes) == 0:
        codes = None
    start = payload.get("start") or None
    end = payload.get("end") or None
    data = panel.build_panel(codes, start, end)
    factor = resolve_factor(payload)
    raw = compute(factor["expression"], data)
    score = raw * factor["direction"]
    factor["raw"] = raw
    return data, factor, score


# --------------------------------------------------------------------------- #
# 因子预览
# --------------------------------------------------------------------------- #
def factor_preview(payload: dict) -> dict:
    data, factor, score = _prepare(payload)
    horizon = int(payload.get("horizon") or 5)
    result = analysis.analyze(
        score, data["close"], data["valid"], data["dates"], data["codes"],
        data["names"], horizon=horizon,
    )
    return {
        "factor": {
            "label": factor["label"],
            "expression": factor["expression"],
            "direction": factor["direction"],
        },
        "universe": {
            "stocks": len(data["codes"]),
            "start": data["dates"][0],
            "end": data["dates"][-1],
            "days": len(data["dates"]),
        },
        **result,
    }


# --------------------------------------------------------------------------- #
# 回测
# --------------------------------------------------------------------------- #
def _rebalance_mask(score: np.ndarray, freq: int) -> tuple[np.ndarray, int]:
    T, N = score.shape
    finite = np.isfinite(score)
    usable = finite.sum(axis=1)
    min_stocks = max(2, min(10, N))
    eligible = np.where(usable >= min_stocks)[0]
    if eligible.size == 0:
        raise ValueError("因子在所选区间内没有足够的有效值，请检查表达式或扩大股票池")
    warmup = int(eligible[0])
    mask = np.zeros(T, dtype=bool)
    mask[warmup::max(1, freq)] = True
    return mask, warmup


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


def run_backtest(payload: dict) -> dict:
    data, factor, score = _prepare(payload)
    dates, codes = data["dates"], data["codes"]
    close, valid = data["close"], data["valid"]

    freq = int(payload.get("rebalance_days") or 10)
    rebalance, warmup = _rebalance_mask(score, max(1, min(freq, 120)))
    top_n = int(payload.get("top_n") or 0)
    top_frac = float(payload.get("top_pct") or 0.2)
    commission = float(payload.get("commission_bps") or 15) / 10000.0
    long_short = bool(payload.get("long_short") or False)

    sim = bt_engine.run_backtest(
        score=score, close=close, valid=valid, rebalance=rebalance,
        top_frac=top_frac, top_n=top_n, long_short=long_short,
        commission=commission,
    )
    metrics = sim["metrics"]
    bench = sim["benchmark"]
    bench_total = float(bench[-1] - 1)
    bench_annual = float((bench[-1]) ** (252 / max(len(bench) - 1, 1)) - 1)
    metrics.update({
        "benchmark_total_return": round(bench_total, 4),
        "benchmark_annual_return": round(bench_annual, 4),
        "excess_annual_return": round(metrics["annual_return"] - bench_annual, 4),
    })
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    metrics["trade_days"] = int(metrics["trade_days"])

    # 持仓数量
    w = sim["weights"]
    pos_counts = (np.abs(w[rebalance]).sum(axis=1) > 1e-9)
    metrics["rebalance_count"] = int(pos_counts.sum())
    metrics["avg_positions"] = int(np.mean(
        (np.abs(w[rebalance]) > 1e-6).sum(axis=1)
    )) if rebalance.any() else 0

    analysis_res = analysis.analyze(
        score, close, valid, dates, codes, data["names"],
        horizon=5,
    )

    def r(arr, nd=5):
        return [round(float(x), nd) if np.isfinite(x) else None for x in arr]

    result = {
        "dates": dates,
        "nav": r(sim["nav"], 4),
        "benchmark": r(bench, 4),
        "drawdown": r(sim["drawdown"], 4),
        "daily_returns": r(sim["returns"], 5),
        "turnover": r(sim["turnover"], 4),
        "metrics": metrics,
        "monthly": _monthly_returns(dates, sim["nav"]),
        "holdings": _holdings(dates, w, rebalance, data["names"], codes),
        "factor": {
            "label": factor["label"],
            "expression": factor["expression"],
            "direction": factor["direction"],
            "ic": analysis_res["ic"],
            "ic_decay": analysis_res["ic_decay"],
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
            "name": payload.get("name") or f"{factor['label']} 回测",
            "codes": codes,
            "start": dates[0],
            "end": dates[-1],
            "rebalance_days": freq,
            "top_n": top_n,
            "top_pct": top_frac,
            "commission_bps": commission * 10000,
            "long_short": long_short,
        },
    }
    bt_id = database.save_backtest(
        result["config"]["name"],
        {**result["config"], "factor": result["factor"]},
        metrics, result,
    )
    result["id"] = bt_id
    result["saved"] = True
    return result


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
