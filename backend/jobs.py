"""轻量后台任务管理（数据同步等耗时操作）。"""
from __future__ import annotations

import threading
import time
import uuid

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def create(kind: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "running",
            "progress": 0.0,
            "message": "任务已创建",
            "result": None,
            "error": None,
            "created_at": time.time(),
        }
    return job_id


def update(job_id: str, *, progress: float | None = None, message: str | None = None,
           status: str | None = None, result=None, error: str | None = None) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        if progress is not None:
            job["progress"] = max(0.0, min(1.0, progress))
        if message is not None:
            job["message"] = message
        if status is not None:
            job["status"] = status
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error


def get(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def cleanup(max_age: float = 3600) -> None:
    now = time.time()
    with _lock:
        for key in [k for k, v in _jobs.items() if now - v["created_at"] > max_age]:
            _jobs.pop(key, None)
