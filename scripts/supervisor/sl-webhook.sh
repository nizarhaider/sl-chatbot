#!/usr/bin/env bash
set -euo pipefail

cd /workspace/sl-chatbot
mkdir -p run_logs
exec > >(tee -a run_logs/webhook.log) 2>&1
exec .venv/bin/uvicorn app.main:app \
  --host 0.0.0.0 --port 8081 --env-file .env
