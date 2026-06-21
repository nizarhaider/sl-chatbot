#!/usr/bin/env bash
set -euo pipefail

export REALTIME_TTS_DEVICE="${REALTIME_TTS_DEVICE:-mps}"
export REALTIME_TTS_DTYPE="${REALTIME_TTS_DTYPE:-float16}"
export REALTIME_TTS_NUM_STEPS="${REALTIME_TTS_NUM_STEPS:-22,22}"
export REALTIME_TTS_SPEED="${REALTIME_TTS_SPEED:-1.0}"
export GEMMA_PREWARM="${GEMMA_PREWARM:-false}"

exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --env-file .env
