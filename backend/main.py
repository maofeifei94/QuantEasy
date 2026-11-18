"""FastAPI 入口：REST API + 前端静态资源。"""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import database, jobs, services
from .datasources import miniqmt
from .factors.engine import FactorError

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="AlphaLab · A股量化分析平台", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.on_event("startup")
def _startup() -> None:
    database.init_db()
    if database.stats()["stock_count"] == 0:
        threading.Thread(target=_seed_demo, daemon=True).start()
    else:
        threading.Thread(target=services.ensure_indices, daemon=True).start()


def _seed_demo() -> None:
    try:
        job_id = services.start_sync({"source": "synthetic", "demo_stocks": 150, "replace": True})
        for _ in range(600):
            job = jobs.get(job_id)
            if job and job["status"] in ("done", "error"):
                break
            threading.Event().wait(0.5)
        database.set_meta("demo_seeded", True)
    except Exception:
        pass


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/status")
def status() -> dict:
    stats = database.stats()
    stats["miniqmt_available"] = miniqmt.is_available()
    stats["demo_seeded"] = database.get_meta("demo_seeded", False)
    return stats


@app.get("/api/datasource")
def datasource() -> dict:
    return {
        "miniqmt_available": miniqmt.is_available(),
        "miniqmt_hint": (
            "已检测到 xtquant，可直接从 miniQMT 同步真实行情。"
            if miniqmt.is_available()
            else "未检测到 xtquant。请在安装 miniQMT 客户端并登录的机器上运行，"
                 "或使用内置合成行情进行因子/回测研究。"
        ),
    }


@app.post("/api/data/sync")
def sync_data(payload: dict = Body(default={})) -> dict:
    job_id = services.start_sync(payload)
    return {"job_id": job_id}


@app.get("/api/data/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


@app.get("/api/stocks")
def stocks(q: str = "", limit: int = 500) -> dict:
    return {"items": database.list_stocks(q=q, limit=limit)}


@app.get("/api/stocks/{code}/bars")
def stock_bars(code: str, start: str | None = None, end: str | None = None) -> dict:
    data = services.stock_bars(code, start, end)
    if not data["dates"]:
        raise HTTPException(status_code=404, detail="未找到该股票行情")
    return data


@app.get("/api/indices")
def indices() -> dict:
    return {"items": database.list_indices()}


@app.get("/api/data/quality")
def data_quality() -> dict:
    return database.data_quality()


@app.get("/api/universes")
def universes() -> dict:
    return {"items": database.list_universes()}


@app.post("/api/universes")
def save_universe(payload: dict = Body(...)) -> dict:
    name = (payload.get("name") or "").strip()
    codes = payload.get("codes") or []
    if not name:
        raise HTTPException(status_code=400, detail="股票池名称不能为空")
    if not isinstance(codes, list) or not codes:
        raise HTTPException(status_code=400, detail="股票池不能为空")
    return database.save_universe(name, [str(c) for c in codes])


@app.delete("/api/universes/{universe_id}")
def delete_universe(universe_id: int) -> dict:
    database.delete_universe(universe_id)
    return {"ok": True}


@app.get("/api/factors")
def factors() -> dict:
    from .factors.builtin import all_factors
    custom = [{**f, "is_custom": True, "label": f["name"]} for f in database.list_factors()]
    return {"items": all_factors() + custom}


@app.post("/api/factors")
def create_factor(payload: dict = Body(...)) -> dict:
    name = (payload.get("name") or "").strip()
    expression = (payload.get("expression") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="因子名称不能为空")
    try:
        from .factors.engine import validate
        validate(expression)
    except FactorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if any(f["name"] == name for f in database.list_factors()):
        raise HTTPException(status_code=400, detail="因子名称已存在")
    return database.create_factor(
        name, expression, payload.get("description", ""),
        int(payload.get("direction", 1)),
    )


@app.delete("/api/factors/{factor_id}")
def delete_factor(factor_id: int) -> dict:
    database.delete_factor(factor_id)
    return {"ok": True}


@app.post("/api/factors/preview")
def factor_preview(payload: dict = Body(...)) -> dict:
    try:
        return services.factor_preview(payload)
    except (FactorError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"因子预览失败：{exc}") from exc


@app.post("/api/factors/series")
def factor_series(payload: dict = Body(...)) -> dict:
    try:
        return services.get_factor_series(payload)
    except (FactorError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/factors/correlation")
def factor_correlation(payload: dict = Body(...)) -> dict:
    try:
        return services.factor_correlation(payload)
    except (FactorError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/factors/mine")
def mine_factors(payload: dict = Body(...)) -> dict:
    try:
        return services.mine_factors(payload)
    except (FactorError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"因子挖掘失败：{exc}") from exc


@app.post("/api/signal")
def latest_signal(payload: dict = Body(...)) -> dict:
    try:
        return services.latest_signal(payload)
    except (FactorError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/portfolio/combine")
def combine_portfolios(payload: dict = Body(...)) -> dict:
    try:
        return services.combine_portfolios(payload)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/backtest/optimize")
def optimize_backtest(payload: dict = Body(...)) -> dict:
    try:
        return services.optimize_backtest(payload)
    except (FactorError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"参数扫描失败：{exc}") from exc


@app.post("/api/backtest/batch")
def batch_backtest(payload: dict = Body(...)) -> dict:
    try:
        return services.batch_backtest(payload)
    except (FactorError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"批量评测失败：{exc}") from exc


@app.post("/api/backtest")
def backtest(payload: dict = Body(...)) -> dict:
    try:
        return services.run_backtest(payload)
    except (FactorError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"回测失败：{exc}") from exc


@app.get("/api/backtests")
def backtests(limit: int = 30) -> dict:
    return {"items": database.list_backtests(limit=limit)}


@app.get("/api/backtests/{bt_id}")
def backtest_detail(bt_id: int) -> dict:
    item = database.get_backtest(bt_id)
    if item is None:
        raise HTTPException(status_code=404, detail="回测记录不存在")
    return item
