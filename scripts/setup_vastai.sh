#!/usr/bin/env bash
# =============================================================================
# setup_vastai.sh — One-shot setup for a fresh Vast.ai GPU box
# Uses a temporary Cloudflare trycloudflare.com tunnel by default.
# =============================================================================

set -euo pipefail

SSH_PORT="${1:?Usage: $0 <SSH_PORT> <HOST_IP>}"
HOST_IP="${2:?Usage: $0 <SSH_PORT> <HOST_IP>}"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/vastai_ssh_file}"
REMOTE="root@${HOST_IP}"
REMOTE_DIR="/workspace/sl-chatbot"
APP_PORT="${APP_PORT:-8081}"

PUBLIC_WEBHOOK_URL="${PUBLIC_WEBHOOK_URL:-}"
USE_TEMP_TUNNEL="${USE_TEMP_TUNNEL:-true}"
CLOUDFLARED_TUNNEL_TOKEN="${CLOUDFLARED_TUNNEL_TOKEN:-}"

SSH="ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE}"
SCP="scp -P ${SSH_PORT} -i ${SSH_KEY}"

log() { echo "▶ $*"; }

log "Checking machine..."
$SSH "uname -a && nvidia-smi --query-gpu=name --format=csv,noheader && which uv git python3"

log "Preparing remote repo..."
$SSH "
  if [ ! -d ${REMOTE_DIR}/.git ]; then
    mkdir -p /workspace
    git clone https://github.com/nizarhaider/sl-chatbot.git ${REMOTE_DIR}
  else
    echo 'Repo already exists, skipping clone'
  fi
  mkdir -p ${REMOTE_DIR}/run_logs
"

log ".env sync..."
if [ -f .env ]; then
  $SCP .env ${REMOTE}:${REMOTE_DIR}/
else
  echo 'WARNING: .env not found locally; skipping .env copy.'
fi

log "Installing system packages..."
$SSH "apt-get update -qq && apt-get install -y portaudio19-dev curl gnupg tmux"

log "Installing cloudflared..."
$SSH "curl -L -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && \
  dpkg -i /tmp/cloudflared.deb || apt-get install -f -y && \
  rm -f /tmp/cloudflared.deb"

if [ -n "${CLOUDFLARED_TUNNEL_TOKEN}" ]; then
  log "Installing cloudflared connector service..."
  $SSH "cloudflared service install '${CLOUDFLARED_TUNNEL_TOKEN}' && service cloudflared restart && service cloudflared status"
fi

log "Running uv sync..."
$SSH "cd ${REMOTE_DIR} && uv sync"

log "Compile-checking Python modules..."
$SSH "cd ${REMOTE_DIR} && .venv/bin/python -m py_compile \
  app/main.py \
  app/webhooks/whatsapp.py \
  app/services/webrtc.py \
  app/services/whatsapp_api.py \
  app/voice_agent/agent.py \
  app/voice_agent/gemini_turn_pipeline.py && echo 'COMPILE OK'"

log "Starting webhook in tmux..."
$SSH "tmux kill-session -t sl-webhook 2>/dev/null || true; \
  tmux new-session -d -s sl-webhook \
  'cd ${REMOTE_DIR} && .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT} --env-file .env'"

log "Waiting for server to boot..."
sleep 10

log "Health check..."
$SSH "
  attempt=0
  until [ \$attempt -ge 5 ]; do
    if ss -ltnp | grep ${APP_PORT} >/dev/null 2>&1; then
      break
    fi
    attempt=\$((attempt + 1))
    echo 'Waiting for port ${APP_PORT} to open... attempt' \$attempt
    sleep 2
  done

  if ! ss -ltnp | grep ${APP_PORT} >/dev/null 2>&1; then
    echo 'WARNING: port not listening yet'
    tmux ls || true
    tail -n 40 ${REMOTE_DIR}/run_logs/webhook.log || true
    exit 1
  fi

  curl -sS http://127.0.0.1:${APP_PORT}/ && echo ''
  curl -sS 'http://127.0.0.1:${APP_PORT}/webhook?hub.mode=subscribe&hub.verify_token=my_secure_verify_token_123&hub.challenge=12345'
  echo ''
"

if [ -z "${PUBLIC_WEBHOOK_URL}" ] && [ "${USE_TEMP_TUNNEL}" = "true" ]; then
  log "Starting temporary Cloudflare tunnel..."

  $SSH "
    tmux kill-session -t sl-tunnel 2>/dev/null || true
    rm -f /tmp/cloudflared.log
    tmux new-session -d -s sl-tunnel \
      'cloudflared tunnel --url http://localhost:${APP_PORT} > /tmp/cloudflared.log 2>&1'
  "

  sleep 10

  TMP_TUNNEL_URL="$(
    $SSH "grep -o 'https://[-a-zA-Z0-9]*\.trycloudflare.com' /tmp/cloudflared.log | head -n1" || true
  )"

  if [ -z "${TMP_TUNNEL_URL}" ]; then
    echo "ERROR: Failed to obtain temporary Cloudflare tunnel URL."
    echo "Remote tunnel log:"
    $SSH "cat /tmp/cloudflared.log || true"
    exit 1
  fi

  PUBLIC_WEBHOOK_URL="${TMP_TUNNEL_URL}/webhook"
  log "Temporary webhook URL: ${PUBLIC_WEBHOOK_URL}"
fi

if [ -z "${PUBLIC_WEBHOOK_URL}" ]; then
  echo "ERROR: PUBLIC_WEBHOOK_URL is empty."
  echo "Set PUBLIC_WEBHOOK_URL or enable USE_TEMP_TUNNEL=true."
  exit 1
fi

log "Checking public webhook URL..."
public_response="$(curl -sS -m 15 "${PUBLIC_WEBHOOK_URL}?hub.mode=subscribe&hub.verify_token=my_secure_verify_token_123&hub.challenge=12345" || true)"

if [ "${public_response}" = "12345" ]; then
  log "Public webhook URL verified: ${PUBLIC_WEBHOOK_URL}"
else
  echo "WARNING: public webhook URL did not return expected challenge."
  echo "URL: ${PUBLIC_WEBHOOK_URL}"
  echo "Response: ${public_response}"
  exit 1
fi

log "✅ Setup complete! Webhook running on ${HOST_IP}:${APP_PORT}"
log ""
log "Webhook URL:"
log "  ${PUBLIC_WEBHOOK_URL}"
log ""
log "Useful commands:"
log "  Attach to webhook:  ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} -t 'tmux attach -t sl-webhook'"
log "  Attach to tunnel:   ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} -t 'tmux attach -t sl-tunnel'"
log "  Watch tunnel log:   ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'tail -f /tmp/cloudflared.log'"
log "  Watch logs:         ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'tail -f ${REMOTE_DIR}/run_logs/webhook.log'"
log "  Watch important:    ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'tail -f ${REMOTE_DIR}/run_logs/important.log'"