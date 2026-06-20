#!/usr/bin/env bash
# =============================================================================
# setup_vastai_whisper_medium_si_merged.sh — Setup a Vast.ai instance using the
# SPEAK-ASR/whisper-medium-si-merged ASR model and local Gemma 4 12B Q4.
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
REMOTE_BRANCH="${REMOTE_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"

WHISPER_MODEL_OVERRIDE="${WHISPER_MODEL_OVERRIDE:-SPEAK-ASR/whisper-medium-si-merged}"
WHISPER_DEVICE_OVERRIDE="${WHISPER_DEVICE_OVERRIDE:-cuda}"
GEMMA_MODEL_REPO_OVERRIDE="${GEMMA_MODEL_REPO_OVERRIDE:-google/gemma-4-12B-it-qat-q4_0-gguf}"
GEMMA_N_GPU_LAYERS_OVERRIDE="${GEMMA_N_GPU_LAYERS_OVERRIDE:--1}"
GEMMA_CONTEXT_TOKENS_OVERRIDE="${GEMMA_CONTEXT_TOKENS_OVERRIDE:-4096}"

SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE}"
SCP="scp -P ${SSH_PORT} -i ${SSH_KEY}"

log() { echo "▶ $*"; }

log "Checking machine..."
$SSH "uname -a && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && which uv git python3"

log "Preparing remote repo on branch ${REMOTE_BRANCH}..."
$SSH "
  if [ ! -d ${REMOTE_DIR}/.git ]; then
    mkdir -p /workspace
    git clone https://github.com/nizarhaider/sl-chatbot.git ${REMOTE_DIR}
  fi
  cd ${REMOTE_DIR}
  git fetch origin ${REMOTE_BRANCH} || true
  if git show-ref --verify --quiet refs/heads/${REMOTE_BRANCH}; then
    git switch ${REMOTE_BRANCH}
    git reset --hard origin/${REMOTE_BRANCH}
  elif git ls-remote --exit-code --heads origin ${REMOTE_BRANCH} >/dev/null 2>&1; then
    git switch -c ${REMOTE_BRANCH} origin/${REMOTE_BRANCH}
  else
    echo 'ERROR: branch ${REMOTE_BRANCH} not found in remote origin'
    exit 1
  fi
  mkdir -p ${REMOTE_DIR}/run_logs
"

log ".env sync..."
if [ -f .env ]; then
  $SCP .env "${REMOTE}:${REMOTE_DIR}/"
else
  echo "WARNING: .env not found locally; skipping .env copy."
fi

if [ -f .env ]; then
  # shellcheck source=/dev/null
  set -a
  source .env
  set +a
fi

log "Installing system packages..."
$SSH "apt-get update -qq && apt-get install -y portaudio19-dev curl gnupg tmux build-essential cmake"

log "Running uv sync with CUDA llama.cpp build..."
$SSH "cd ${REMOTE_DIR} && \
  CMAKE_ARGS='-DGGML_CUDA=on' FORCE_CMAKE=1 \
  uv sync --no-binary-package llama-cpp-python --reinstall-package llama-cpp-python"

log "Compile-checking Python modules..."
$SSH "cd ${REMOTE_DIR} && .venv/bin/python -m py_compile \
  app/main.py \
  app/webhooks/whatsapp.py \
  app/services/webrtc.py \
  app/services/whatsapp_api.py \
  app/voice_agent/agent.py \
  app/voice_agent/gemini_turn_pipeline.py && echo 'COMPILE OK'"

log "Installing ngrok..."
if [ -z "${NGROK_AUTH_TOKEN}" ]; then
  echo "WARNING: NGROK_AUTH_TOKEN not set; ngrok will run in demo mode if allowed."
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

log "Starting ngrok in tmux..."
$SSH "
  if tmux has-session -t sl-ngrok 2>/dev/null; then
    if curl -sS http://127.0.0.1:4040/api/tunnels >/dev/null 2>&1; then
      echo 'ngrok is already running and healthy; skipping startup'
      exit 0
    fi
    echo 'Found existing sl-ngrok session but ngrok API is unavailable; restarting'
    tmux kill-session -t sl-ngrok 2>/dev/null || true
  fi
  tmux new-session -d -s sl-ngrok \
    'cd ${REMOTE_DIR} && ngrok http ${APP_PORT} --log=stdout > /tmp/ngrok.log 2>&1'
"

sleep 3

log "Starting webhook in tmux..."
$SSH "tmux kill-session -t sl-webhook 2>/dev/null || true; \
  mkdir -p ${REMOTE_DIR}/run_logs; \
  tmux new-session -d -s sl-webhook \
  'cd ${REMOTE_DIR} && \
   export WHISPER_MODEL=${WHISPER_MODEL_OVERRIDE} && \
   export WHISPER_DEVICE=${WHISPER_DEVICE_OVERRIDE} && \
   export GEMMA_MODEL_REPO=${GEMMA_MODEL_REPO_OVERRIDE} && \
   export GEMMA_N_GPU_LAYERS=${GEMMA_N_GPU_LAYERS_OVERRIDE} && \
   export GEMMA_CONTEXT_TOKENS=${GEMMA_CONTEXT_TOKENS_OVERRIDE} && \
   export GEMMA_PREWARM=true && \
   .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT} --env-file .env \
   > run_logs/webhook.log 2>&1'"

log "Waiting for server to boot..."
$SSH "
  attempt=0
  until [ \$attempt -ge 180 ]; do
    if ss -ltnp | grep ${APP_PORT} >/dev/null 2>&1; then
      break
    fi
    attempt=\$((attempt + 1))
    echo 'Waiting for port ${APP_PORT} to open... attempt' \$attempt '(model download/prewarm may still be running)'
    if [ -f ${REMOTE_DIR}/run_logs/webhook.log ]; then
      echo '--- latest webhook startup log ---'
      tail -n 12 ${REMOTE_DIR}/run_logs/webhook.log | sed 's/^/    /'
    fi
    sleep 2
  done

  if ! ss -ltnp | grep ${APP_PORT} >/dev/null 2>&1; then
    echo 'ERROR: port ${APP_PORT} is not listening.'
    echo ''
    echo 'tmux sessions:'
    tmux ls || true
    echo ''
    echo 'sl-webhook pane:'
    tmux capture-pane -t sl-webhook -p 2>/dev/null | tail -n 100 || true
    echo ''
    echo 'webhook.log:'
    tail -n 160 ${REMOTE_DIR}/run_logs/webhook.log || true
    exit 1
  fi

  curl -sS http://127.0.0.1:${APP_PORT}/ && echo ''
  curl -sS 'http://127.0.0.1:${APP_PORT}/webhook?hub.mode=subscribe&hub.verify_token=my_secure_verify_token_123&hub.challenge=12345'
  echo ''
"

if [ -z "${PUBLIC_WEBHOOK_URL}" ] && [ "${USE_TEMP_TUNNEL}" = "true" ]; then
  log "Retrieving ngrok tunnel URL..."

  TMP_TUNNEL_URL=""
  for i in 1 2 3 4 5; do
    TMP_TUNNEL_URL="$($SSH "curl -sS http://127.0.0.1:4040/api/tunnels | python3 -c 'import sys,json;obj=json.load(sys.stdin);t=obj.get(\"tunnels\") or [];print(t[0].get(\"public_url\", \"\") if t else \"\")' || true")"
    if [ -n "${TMP_TUNNEL_URL}" ]; then
      break
    fi
    echo "Waiting for ngrok API to become available... attempt ${i}"
    sleep 2
  done

  if [ -z "${TMP_TUNNEL_URL}" ]; then
    echo "ERROR: Failed to obtain ngrok tunnel URL."
    echo "Check ngrok status on remote host:"
    echo "  ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'ps aux | grep ngrok'"
    echo "  ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'cat /tmp/ngrok.log'"
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

log "Waiting for WhatsApp webhook verification..."
log "Use this callback URL in WhatsApp:"
log "  ${PUBLIC_WEBHOOK_URL}"
log ""

while true; do
  public_response="$(curl -sS -m 15 "${PUBLIC_WEBHOOK_URL}?hub.mode=subscribe&hub.verify_token=my_secure_verify_token_123&hub.challenge=12345" || true)"

  if [ "${public_response}" = "12345" ]; then
    log "WhatsApp webhook verification is working: ${PUBLIC_WEBHOOK_URL}"
    break
  fi

  echo "Waiting for verification to work... response: ${public_response:-<empty>}"
  sleep 5
done

log "Setup complete. Webhook running on ${HOST_IP}:${APP_PORT}"
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
