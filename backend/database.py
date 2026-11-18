"""SQLite 数据访问层（Python 内置 sqlite3）。"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "quant.db"

_write_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS stocks (
    code        TEXT PRIMARY KEY,
    name        TEXT,
    industry    TEXT,
    market      TEXT,
    list_date   TEXT,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS daily_bars (
    code    TEXT NOT NULL,
    date    TEXT NOT NULL,
    open    REAL,
    high    REAL,
    low     REAL,
    close   REAL,
    volume  REAL,
    amount  REAL,
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_bars_date ON daily_bars(date);

CREATE TABLE IF NOT EXISTS custom_factors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    expression  TEXT NOT NULL,
    description TEXT,
    direction   INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backtests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    config      TEXT NOT NULL,
    metrics     TEXT NOT NULL,
    result      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key    TEXT PRIMARY KEY,
    value  TEXT
);
"""


def get_conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=60, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def upsert_stocks(rows: Iterable[dict[str, Any]]) -> None:
    sql = """INSERT INTO stocks(code,name,industry,market,list_date,updated_at)
             VALUES(?,?,?,?,?,?)
             ON CONFLICT(code) DO UPDATE SET
                name=excluded.name, industry=excluded.industry,
                market=excluded.market, list_date=excluded.list_date,
                updated_at=excluded.updated_at"""
    now = _now()
    values = [
        (r["code"], r.get("name", ""), r.get("industry", ""), r.get("market", ""),
         r.get("list_date", ""), now)
        for r in rows
    ]
    if not values:
        return
    with _write_lock, get_conn() as conn:
        conn.executemany(sql, values)


def insert_bars(rows: Iterable[tuple]) -> int:
    """rows: (code, date, open, high, low, close, volume, amount)"""
    sql = """INSERT OR REPLACE INTO daily_bars
             (code,date,open,high,low,close,volume,amount) VALUES(?,?,?,?,?,?,?,?)"""
    rows = list(rows)
    if not rows:
        return 0
    with _write_lock, get_conn() as conn:
        conn.executemany(sql, rows)
    return len(rows)


def clear_market_data() -> None:
    with _write_lock, get_conn() as conn:
        conn.execute("DELETE FROM daily_bars")
        conn.execute("DELETE FROM stocks")


def set_meta(key: str, value: Any) -> None:
    with _write_lock, get_conn() as conn:
        conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, ensure_ascii=False)),
        )


def get_meta(key: str, default: Any = None) -> Any:
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except json.JSONDecodeError:
        return default


def stats() -> dict[str, Any]:
    with get_conn() as conn:
        stocks = conn.execute("SELECT COUNT(*) c FROM stocks").fetchone()["c"]
        bars = conn.execute("SELECT COUNT(*) c FROM daily_bars").fetchone()["c"]
        rng = conn.execute(
            "SELECT MIN(date) a, MAX(date) b FROM daily_bars"
        ).fetchone()
        factors = conn.execute("SELECT COUNT(*) c FROM custom_factors").fetchone()["c"]
        backtests = conn.execute("SELECT COUNT(*) c FROM backtests").fetchone()["c"]
    return {
        "stock_count": stocks,
        "bar_count": bars,
        "start_date": rng["a"],
        "end_date": rng["b"],
        "custom_factor_count": factors,
        "backtest_count": backtests,
        "source": get_meta("data_source", "none"),
        "last_sync": get_meta("last_sync", None),
    }


def list_stocks(q: str = "", limit: int = 500) -> list[dict[str, Any]]:
    with get_conn() as conn:
        if q:
            pattern = f"%{q}%"
            rows = conn.execute(
                """WITH ranked AS (
                       SELECT code,date,close,volume,amount,
                              ROW_NUMBER() OVER (PARTITION BY code ORDER BY date DESC) rn
                       FROM daily_bars)
                   SELECT s.code,s.name,s.industry,r.date,r.close,r.volume,r.amount,
                          (SELECT close FROM ranked r2 WHERE r2.code=s.code AND r2.rn=2) AS prev_close
                   FROM stocks s LEFT JOIN ranked r ON r.code=s.code AND r.rn=1
                   WHERE s.code LIKE ? OR s.name LIKE ?
                   ORDER BY s.code LIMIT ?""",
                (pattern, pattern, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """WITH ranked AS (
                       SELECT code,date,close,volume,amount,
                              ROW_NUMBER() OVER (PARTITION BY code ORDER BY date DESC) rn
                       FROM daily_bars)
                   SELECT s.code,s.name,s.industry,r.date,r.close,r.volume,r.amount,
                          (SELECT close FROM ranked r2 WHERE r2.code=s.code AND r2.rn=2) AS prev_close
                   FROM stocks s LEFT JOIN ranked r ON r.code=s.code AND r.rn=1
                   ORDER BY s.code LIMIT ?""",
                (limit,),
            ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        prev = d.get("prev_close")
        d["change_pct"] = (
            (d["close"] / prev - 1) * 100 if prev and d.get("close") else None
        )
        out.append(d)
    return out


def get_bars_bulk(
    codes: list[str] | None, start: str, end: str
) -> list[dict[str, Any]]:
    with get_conn() as conn:
        if codes:
            placeholders = ",".join("?" for _ in codes)
            rows = conn.execute(
                f"""SELECT code,date,open,high,low,close,volume,amount FROM daily_bars
                    WHERE date BETWEEN ? AND ? AND code IN ({placeholders})
                    ORDER BY date, code""",
                (start, end, *codes),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT code,date,open,high,low,close,volume,amount FROM daily_bars
                   WHERE date BETWEEN ? AND ?
                   ORDER BY date, code""",
                (start, end),
            ).fetchall()
    return [dict(r) for r in rows]


def get_stock_names(codes: list[str]) -> dict[str, str]:
    if not codes:
        return {}
    placeholders = ",".join("?" for _ in codes)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT code,name FROM stocks WHERE code IN ({placeholders})", codes
        ).fetchall()
    return {r["code"]: r["name"] for r in rows}


def get_stock_meta(codes: list[str]) -> dict[str, dict[str, str]]:
    """返回 code -> {name, industry} 映射。"""
    if not codes:
        return {}
    placeholders = ",".join("?" for _ in codes)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT code,name,industry FROM stocks WHERE code IN ({placeholders})", codes
        ).fetchall()
    return {
        r["code"]: {"name": r["name"] or r["code"], "industry": r["industry"] or "未分类"}
        for r in rows
    }


def list_factors() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM custom_factors ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def create_factor(name: str, expression: str, description: str, direction: int) -> dict:
    with _write_lock, get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO custom_factors(name,expression,description,direction,created_at)
               VALUES(?,?,?,?,?)""",
            (name, expression, description, direction, _now()),
        )
        factor_id = cur.lastrowid
    return {"id": factor_id, "name": name, "expression": expression,
            "description": description, "direction": direction}


def delete_factor(factor_id: int) -> None:
    with _write_lock, get_conn() as conn:
        conn.execute("DELETE FROM custom_factors WHERE id=?", (factor_id,))


def save_backtest(name: str, config: dict, metrics: dict, result: dict) -> int:
    with _write_lock, get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO backtests(name,config,metrics,result,created_at) VALUES(?,?,?,?,?)",
            (name, json.dumps(config, ensure_ascii=False),
             json.dumps(metrics, ensure_ascii=False),
             json.dumps(result, ensure_ascii=False), _now()),
        )
        return cur.lastrowid


def list_backtests(limit: int = 50) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id,name,config,metrics,created_at FROM backtests
               ORDER BY id DESC LIMIT ?""", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["config"] = json.loads(d["config"])
        d["metrics"] = json.loads(d["metrics"])
        out.append(d)
    return out


def get_backtest(bt_id: int) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM backtests WHERE id=?", (bt_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["config"] = json.loads(d["config"])
    d["metrics"] = json.loads(d["metrics"])
    d["result"] = json.loads(d["result"])
    return d


def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
