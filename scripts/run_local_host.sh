#!/usr/bin/env bash
set -euo pipefail

exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --env-file .env
