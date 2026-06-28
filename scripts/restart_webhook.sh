#!/usr/bin/env bash

set -euo pipefail

APP_PORT="${APP_PORT:-8081}"
SESSION_NAME="${SESSION_NAME:-sl-webhook}"
REMOTE_DIR="${REMOTE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="${REMOTE_DIR}/run_logs"
LOG_FILE="${LOG_DIR}/webhook.log"
ENV_FILE="${ENV_FILE:-.env}"
WAIT_SECONDS="${WAIT_SECONDS:-180}"

log() { echo "▶ $*"; }

cd "${REMOTE_DIR}"
mkdir -p "${LOG_DIR}"

if [ ! -x ".venv/bin/uvicorn" ]; then
  echo "ERROR: .venv/bin/uvicorn not found. Run this from a synced repo with dependencies installed."
  exit 1
fi

log "Stopping existing ${SESSION_NAME} tmux session..."
tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true

log "Stopping stray uvicorn processes for app.main:app..."
pkill -f 'uvicorn app.main:app' 2>/dev/null || true

log "Starting webhook on port ${APP_PORT}..."
tmux new-session -d -s "${SESSION_NAME}" \
  "cd '${REMOTE_DIR}' && \
   .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port '${APP_PORT}' --env-file '${ENV_FILE}' \
   > '${LOG_FILE}' 2>&1"

log "Waiting for startup..."
for ((attempt = 1; attempt <= WAIT_SECONDS; attempt++)); do
  if curl -fsS "http://127.0.0.1:${APP_PORT}/" >/dev/null 2>&1; then
    echo ""
    log "Webhook is healthy on http://127.0.0.1:${APP_PORT}/"
    echo ""
    tmux ls 2>/dev/null || true
    echo ""
    tail -n 60 "${LOG_FILE}" || true
    exit 0
  fi

  if (( attempt == 1 || attempt % 10 == 0 )); then
    echo "Waiting for port ${APP_PORT}... attempt ${attempt}/${WAIT_SECONDS}"
    if [ -f "${LOG_FILE}" ]; then
      tail -n 12 "${LOG_FILE}" | sed 's/^/    /'
    fi
  fi
  sleep 1
done

echo ""
echo "ERROR: webhook did not become healthy within ${WAIT_SECONDS}s."
echo ""
echo "tmux sessions:"
tmux ls 2>/dev/null || true
echo ""
echo "latest log:"
tail -n 160 "${LOG_FILE}" 2>/dev/null || true
exit 1
