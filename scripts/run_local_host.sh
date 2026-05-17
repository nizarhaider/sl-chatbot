#!/usr/bin/env bash
set -euo pipefail

# We rely on python-dotenv / uvicorn to load .env natively,
# but we export a few explicit fallbacks here if they aren't in .env

export VOICE_OUTPUT_PROVIDER="${VOICE_OUTPUT_PROVIDER:-omnivoice_local}"
export OMNIVOICE_LOCAL_DEVICE="${OMNIVOICE_LOCAL_DEVICE:-mps}"
export OMNIVOICE_NUM_STEP="${OMNIVOICE_NUM_STEP:-8}"
export OMNIVOICE_SPEED="${OMNIVOICE_SPEED:-1.18}"
export VOICE_TTS_MAX_CHARS="${VOICE_TTS_MAX_CHARS:-160}"
export TTS_CACHE_SIZE="${TTS_CACHE_SIZE:-64}"

exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --env-file .env
