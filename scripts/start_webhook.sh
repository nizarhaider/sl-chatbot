#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "$0")" >/dev/null && pwd)/.."
exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8080 --env-file .env
