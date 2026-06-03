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
NGROK_AUTH_TOKEN="${NGROK_AUTH_TOKEN:-}"

PUBLIC_WEBHOOK_URL="${PUBLIC_WEBHOOK_URL:-}"
USE_TEMP_TUNNEL="${USE_TEMP_TUNNEL:-true}"

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

log "Installing ngrok..."
# Download and install ngrok (stable) on remote host if not present
if [ -z "${NGROK_AUTH_TOKEN}" ]; then
  echo 'WARNING: NGROK_AUTH_TOKEN not set; ngrok will run in demo mode (limited tunnel time).'
fi
$SSH "if ! command -v ngrok >/dev/null 2>&1; then \
    curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo 'deb https://ngrok-agent.s3.amazonaws.com bookworm main' \
  | tee /etc/apt/sources.list.d/ngrok.list \
  && apt-get update -qq \
  && apt-get install -y ngrok; \
  else echo 'ngrok already installed'; fi"

if [ -n "${NGROK_AUTH_TOKEN}" ]; then
  log "Configuring ngrok auth token on remote..."
  $SSH "ngrok config add-authtoken '${NGROK_AUTH_TOKEN}'"
fi

log "Setting up ngrok tunnel..."
# Create local ngrok.yml with substituted APP_PORT
LOCAL_NGROK_YML="/tmp/ngrok_${APP_PORT}.yml"
sed "s/APP_PORT_PLACEHOLDER/${APP_PORT}/g" ngrok.yml > "${LOCAL_NGROK_YML}"
$SCP "${LOCAL_NGROK_YML}" ${REMOTE}:${REMOTE_DIR}/ngrok.yml
rm -f "${LOCAL_NGROK_YML}"

log "Starting ngrok in tmux..."
$SSH "
  tmux kill-session -t sl-ngrok 2>/dev/null || true
  tmux new-session -d -s sl-ngrok \
    'cd ${REMOTE_DIR} && ngrok start --config ./ngrok.yml > /tmp/ngrok.log 2>&1'
"

sleep 3

log "Running uv sync..."
$SSH "cd ${REMOTE_DIR} && uv sync"

log "Ensuring OpenAI dependency is installed..."
$SSH "cd ${REMOTE_DIR} && .venv/bin/python - <<'PY' || uv add openai
import openai
PY"

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
  mkdir -p ${REMOTE_DIR}/run_logs; \
  tmux new-session -d -s sl-webhook \
  'cd ${REMOTE_DIR} && .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT} --env-file .env > run_logs/webhook.log 2>&1'"

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
    echo 'ERROR: port ${APP_PORT} is not listening.'
    echo ''
    echo 'tmux sessions:'
    tmux ls || true
    echo ''
    echo 'sl-webhook pane:'
    tmux capture-pane -t sl-webhook -p 2>/dev/null | tail -n 80 || true
    echo ''
    echo 'webhook.log:'
    tail -n 120 ${REMOTE_DIR}/run_logs/webhook.log || true
    exit 1
  fi

  curl -sS http://127.0.0.1:${APP_PORT}/ && echo ''
  curl -sS 'http://127.0.0.1:${APP_PORT}/webhook?hub.mode=subscribe&hub.verify_token=my_secure_verify_token_123&hub.challenge=12345'
  echo ''
"

if [ -z "${PUBLIC_WEBHOOK_URL}" ] && [ "${USE_TEMP_TUNNEL}" = "true" ]; then
  log "Retrieving ngrok tunnel URL from service..."

  sleep 2

  TMP_TUNNEL_URL="$(
    $SSH "curl -sS http://127.0.0.1:4040/api/tunnels | python3 -c 'import sys,json;obj=json.load(sys.stdin);print(obj.get(\"tunnels\")[0].get(\"public_url\"))' || true"
  )"

  if [ -z "${TMP_TUNNEL_URL}" ]; then
    echo "ERROR: Failed to obtain ngrok tunnel URL from service."
    echo "Checking ngrok service status:"
    $SSH "ngrok service status || echo 'ngrok service may not be running'"
    echo ""
    echo "Try manually:"
    echo "  ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'ngrok service status'"
    echo "  ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'curl http://127.0.0.1:4040/api/tunnels | jq'"
    exit 1
  fi

  # ngrok returns https://something.ngrok.io — use that plus /webhook
  PUBLIC_WEBHOOK_URL="${TMP_TUNNEL_URL}/webhook"
  log "Temporary webhook URL: ${PUBLIC_WEBHOOK_URL}"
fi

if [ -z "${PUBLIC_WEBHOOK_URL}" ]; then
  echo "ERROR: PUBLIC_WEBHOOK_URL is empty."
  echo "Set PUBLIC_WEBHOOK_URL or enable USE_TEMP_TUNNEL=true."
  exit 1
fi

log "Waiting for WhatsApp webhook verification..."
log "Use this callback URL in WhatsApp:"
log "  ${PUBLIC_WEBHOOK_URL}"
log ""

while true; do
  public_response="$(curl -sS -m 15 "${PUBLIC_WEBHOOK_URL}?hub.mode=subscribe&hub.verify_token=my_secure_verify_token_123&hub.challenge=12345" || true)"

  if [ "${public_response}" = "12345" ]; then
    log "✅ WhatsApp webhook verification is working: ${PUBLIC_WEBHOOK_URL}"
    break
  fi

  echo "Waiting for verification to work... response: ${public_response:-<empty>}"
  sleep 5
done

log "✅ Setup complete! Webhook running on ${HOST_IP}:${APP_PORT}"
log ""
log "Webhook URL:"
log "  ${PUBLIC_WEBHOOK_URL}"
log ""
log "Useful commands:"
log "  Attach to webhook:  ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} -t 'tmux attach -t sl-webhook'"
log "  Attach to ngrok:    ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} -t 'tmux attach -t sl-ngrok'"
log "  Get ngrok URL:      ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'curl http://127.0.0.1:4040/api/tunnels | jq .tunnels[0].public_url'"
log "  Watch logs:         ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'tail -f ${REMOTE_DIR}/run_logs/webhook.log'"
log "  Watch important:    ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'tail -f ${REMOTE_DIR}/run_logs/important.log'"