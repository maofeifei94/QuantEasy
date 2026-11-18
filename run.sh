#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
exec python3 -m uvicorn backend.main:app --host 0.0.0.0 --port "$PORT"
