#!/usr/bin/env bash
set -euo pipefail

# We rely on python-dotenv / uvicorn to load .env natively,
# but we export a few explicit fallbacks here if they aren't in .env

export GEMINI_STT_MODEL="${GEMINI_STT_MODEL:-gemini-2.5-flash-lite}"
export GEMINI_LLM_MODEL="${GEMINI_LLM_MODEL:-gemini-2.5-flash-lite}"
export REALTIME_TTS_DEVICE="${REALTIME_TTS_DEVICE:-mps}"
export REALTIME_TTS_DTYPE="${REALTIME_TTS_DTYPE:-float16}"
export REALTIME_TTS_NUM_STEPS="${REALTIME_TTS_NUM_STEPS:-12,12}"

exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --env-file .env
