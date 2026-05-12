#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "${line#\#}" != "$line" ]] && continue
    [[ "$line" != *=* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    if [[ -z "${!key+x}" ]]; then
      export "$key=$value"
    fi
  done < .env
fi

if [[ -z "${GOOGLE_API_KEY:-}" && -n "${GEMINI_API_KEY:-}" ]]; then
  export GOOGLE_API_KEY="${GEMINI_API_KEY}"
fi

export VOICE_OUTPUT_PROVIDER="${VOICE_OUTPUT_PROVIDER:-omnivoice_local}"
export OMNIVOICE_LOCAL_DEVICE="${OMNIVOICE_LOCAL_DEVICE:-mps}"
export OMNIVOICE_NUM_STEP="${OMNIVOICE_NUM_STEP:-8}"
export OMNIVOICE_SPEED="${OMNIVOICE_SPEED:-1.18}"
export VOICE_TTS_MAX_CHARS="${VOICE_TTS_MAX_CHARS:-160}"
export TTS_CACHE_SIZE="${TTS_CACHE_SIZE:-64}"

exec uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
