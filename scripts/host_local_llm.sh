#!/usr/bin/env bash
# Serve the production Gemma 4B GGUF locally on macOS with Metal acceleration.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_REPO="google/gemma-4-E4B-it-qat-q4_0-gguf"
MODEL_FILE="gemma-4-E4B_q4_0-it.gguf"
MODEL_DIR="${MODEL_DIR:-${HOME}/Library/Caches/serendibai/models/gemma-4-E4B-it-qat-q4_0}"
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/${MODEL_FILE}}"
LLM_PORT="${LLM_PORT:-8000}"
LLM_HOST="${LLM_HOST:-127.0.0.1}"

log() { printf '▶ %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

if [ -n "${LLAMA_SERVER:-}" ]; then
  test -x "${LLAMA_SERVER}" || fail "LLAMA_SERVER is not executable: ${LLAMA_SERVER}"
elif command -v llama-server >/dev/null 2>&1; then
  LLAMA_SERVER="$(command -v llama-server)"
else
  command -v brew >/dev/null 2>&1 || fail "Install Homebrew or set LLAMA_SERVER to a llama-server binary."
  log "Installing the prebuilt llama.cpp Homebrew bottle..."
  brew install llama.cpp
  LLAMA_SERVER="$(command -v llama-server)"
fi

if [ ! -s "${MODEL_PATH}" ]; then
  mkdir -p "${MODEL_DIR}"
  if command -v aws >/dev/null 2>&1 && aws sts get-caller-identity >/dev/null 2>&1; then
    log "Downloading Gemma 4B from the SerendibAI S3 cache..."
    aws s3 cp "s3://serendibai-models/runtime-cache/${MODEL_FILE}" "${MODEL_PATH}" --no-progress
  else
    HF_TOKEN="${HF_TOKEN:-$(sed -n 's/^HF_TOKEN=//p' "${ROOT_DIR}/.env" 2>/dev/null | head -n 1)}"
    test -n "${HF_TOKEN}" || fail "S3 access is unavailable and HF_TOKEN is not set."
    log "Downloading Gemma 4B from Hugging Face..."
    curl --fail --location --retry 3 --continue-at - \
      -H "Authorization: Bearer ${HF_TOKEN}" \
      "https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}" \
      --output "${MODEL_PATH}"
  fi
fi

log "Serving Gemma 4B at http://${LLM_HOST}:${LLM_PORT}/v1"
exec "${LLAMA_SERVER}" \
  --model "${MODEL_PATH}" \
  --alias "${MODEL_REPO}" \
  --n-gpu-layers 99 \
  --ctx-size 4096 \
  --flash-attn on \
  --jinja \
  --host "${LLM_HOST}" \
  --port "${LLM_PORT}"
