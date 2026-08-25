#!/usr/bin/env bash
# =============================================================================
# setup_vastai.sh — Setup a Vast.ai instance using the
# SPEAK-ASR/whisper-medium-si-merged ASR model and local Gemma 4 E4B Q4.
# =============================================================================

set -euo pipefail

SSH_PORT="${1:?Usage: $0 <SSH_PORT> <HOST_IP>}"
HOST_IP="${2:?Usage: $0 <SSH_PORT> <HOST_IP>}"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/vastai_ssh_file}"
REMOTE="root@${HOST_IP}"
REMOTE_DIR="/workspace/sl-chatbot"
APP_PORT="${APP_PORT:-8081}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MODEL="${VLLM_MODEL:-google/gemma-4-E4B-it}"
VLLM_STARTUP_TIMEOUT_ATTEMPTS="${VLLM_STARTUP_TIMEOUT_ATTEMPTS:-900}"
APP_STARTUP_TIMEOUT_ATTEMPTS="${APP_STARTUP_TIMEOUT_ATTEMPTS:-450}"
PUBLIC_VERIFY_TIMEOUT_ATTEMPTS="${PUBLIC_VERIFY_TIMEOUT_ATTEMPTS:-3}"
HF_RUNTIME_CACHE_BUCKET="${HF_RUNTIME_CACHE_BUCKET:-2broke2code/serendibai-vllm-cuda-cache}"

PUBLIC_WEBHOOK_URL="https://whatsapp.serendibai.lk/webhook"
REMOTE_BRANCH="${REMOTE_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"

SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE}"
SCP="scp -o StrictHostKeyChecking=accept-new -P ${SSH_PORT} -i ${SSH_KEY}"

log() { echo "▶ $*"; }

RUNTIME_CACHE_KEY="vllm-cuda128-$(sha256sum uv.lock | awk '{print $1}')"

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
ENV_SYNC_FILE="$(mktemp)"
cleanup_env_sync() { rm -f "${ENV_SYNC_FILE}"; }
trap cleanup_env_sync EXIT

if [ -f .env ]; then
  cp .env "${ENV_SYNC_FILE}"
else
  : > "${ENV_SYNC_FILE}"
  echo "WARNING: .env not found locally; syncing only exported runtime variables."
fi

VERIFY_TOKEN="${VERIFY_TOKEN:-$(sed -n 's/^VERIFY_TOKEN=//p' "${ENV_SYNC_FILE}" | head -n 1)}"
if [ -z "${VERIFY_TOKEN}" ]; then
  echo "ERROR: VERIFY_TOKEN is not set in .env or the environment."
  exit 1
fi

CLOUDFLARED_TUNNEL_TOKEN="${CLOUDFLARED_TUNNEL_TOKEN:-$(sed -n 's/^CLOUDFLARED_TUNNEL_TOKEN=//p' "${ENV_SYNC_FILE}" | head -n 1)}"
if [ -z "${CLOUDFLARED_TUNNEL_TOKEN}" ]; then
  echo "ERROR: CLOUDFLARED_TUNNEL_TOKEN is not set in .env or the environment."
  exit 1
fi

CLOUDFLARED_TUNNEL_TOKEN="${CLOUDFLARED_TUNNEL_TOKEN}" python3 - "${ENV_SYNC_FILE}" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text().splitlines() if path.exists() else []
key = "CLOUDFLARED_TUNNEL_TOKEN"
value = os.environ[key]
output = [f"{key}={value}" if line.startswith(key + "=") else line for line in lines]
if not any(line.startswith(key + "=") for line in lines):
    output.append(f"{key}={value}")
path.write_text("\n".join(output) + "\n")
PY

# Credentials kept in ~/.zshrc are not necessarily exported into a child bash
# process. Pull the deployment key from the login zsh environment when needed,
# then merge it into the remote env file without ever printing its value.
if [ -z "${PINECONE_API_KEY:-}" ]; then
  PINECONE_API_KEY="$(zsh -lic 'printf "%s" "${PINECONE_API_KEY:-}"' 2>/dev/null || true)"
fi
if [ -n "${PINECONE_API_KEY:-}" ]; then
  PINECONE_API_KEY="${PINECONE_API_KEY}" python3 - "${ENV_SYNC_FILE}" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text().splitlines() if path.exists() else []
key = "PINECONE_API_KEY"
value = os.environ[key]
updated = False
output = []
for line in lines:
    if line.startswith(key + "="):
        output.append(f"{key}={value}")
        updated = True
    else:
        output.append(line)
if not updated:
    output.append(f"{key}={value}")
path.write_text("\n".join(output) + "\n")
PY
  log "PINECONE_API_KEY found in the zsh environment; including it in the remote .env."
else
  echo "WARNING: PINECONE_API_KEY is not available in .env or the zsh environment."
fi

$SCP "${ENV_SYNC_FILE}" "${REMOTE}:${REMOTE_DIR}/.env"

log "Waiting for base image package setup..."
$SSH "
  for attempt in \$(seq 1 60); do
    if ! fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
      exit 0
    fi
    echo 'Waiting for package manager lock... attempt' \$attempt
    sleep 5
  done
  echo 'ERROR: package manager lock did not clear in five minutes.' >&2
  exit 1
"

log "Installing system packages..."
$SSH "apt-get update -qq && apt-get install -y portaudio19-dev curl gnupg tmux"

log "Running cached, locked uv sync with prebuilt CUDA dependencies..."
$SSH "
  cd ${REMOTE_DIR}
  if [ -n \"\${HF_TOKEN:-}\" ]; then
    cache_dir=\$(env -u UV_NO_CACHE uv cache dir)
    mkdir -p \"\${cache_dir}\"
    env -u UV_NO_CACHE uvx --from huggingface_hub hf buckets sync \
      hf://buckets/${HF_RUNTIME_CACHE_BUCKET}/${RUNTIME_CACHE_KEY} \
      \"\${cache_dir}\" --ignore-times --quiet || true
  fi
"
$SSH "cd ${REMOTE_DIR} && env -u UV_NO_CACHE uv sync --frozen"

log "Saving reusable CUDA and vLLM packages to Hugging Face..."
$SSH "
  cd ${REMOTE_DIR}
  if [ -n \"\${HF_TOKEN:-}\" ]; then
    env -u UV_NO_CACHE uvx --from huggingface_hub hf buckets sync \
      \"\$(env -u UV_NO_CACHE uv cache dir)\" \
      hf://buckets/${HF_RUNTIME_CACHE_BUCKET}/${RUNTIME_CACHE_KEY} \
      --ignore-times --quiet
  else
    echo 'HF_TOKEN is unavailable; skipping runtime cache upload.'
  fi
"

log "Compile-checking Python modules..."
$SSH "cd ${REMOTE_DIR} && find app -name '*.py' -print0 | xargs -0 .venv/bin/python -m py_compile && echo 'COMPILE OK'"

log "Starting vLLM and the permanent Cloudflare tunnel..."
$SSH "
  mkdir -p ${REMOTE_DIR}/run_logs
  if ! pgrep -f '.venv/bin/vllm serve' >/dev/null 2>&1; then
    nohup sh -c 'cd ${REMOTE_DIR} && set -a && . .env && set +a && \
      export LD_LIBRARY_PATH=\$(find .venv/lib -type d -name lib -printf %p: 2>/dev/null)\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH} && \
      exec .venv/bin/vllm serve ${VLLM_MODEL} --host 127.0.0.1 --port ${VLLM_PORT} \
      --dtype float16 --max-model-len 2048 --gpu-memory-utilization 0.55 \
      --enable-auto-tool-choice --tool-call-parser gemma4' \
      > run_logs/vllm.log 2>&1 < /dev/null &
  fi
  if ! pgrep -f 'cloudflared tunnel run' >/dev/null 2>&1; then
    nohup sh -c 'cd ${REMOTE_DIR} && set -a && . .env && set +a && \
      TUNNEL_TOKEN=\"\$CLOUDFLARED_TUNNEL_TOKEN\" exec /opt/instance-tools/bin/cloudflared tunnel run' \
      > run_logs/cloudflared.log 2>&1 < /dev/null &
  fi
"

log "Waiting for vLLM to become ready..."
$SSH "
  for attempt in \$(seq 1 ${VLLM_STARTUP_TIMEOUT_ATTEMPTS}); do
    if curl -fsS http://127.0.0.1:${VLLM_PORT}/v1/models >/dev/null; then
      exit 0
    fi
    sleep 2
  done
  tail -n 120 ${REMOTE_DIR}/run_logs/vllm.log || true
  exit 1
"

log "Starting webhook..."
$SSH "
  if ss -ltnp | grep ${APP_PORT} >/dev/null 2>&1; then
    echo 'Webhook is already listening; keeping the existing process.'
  elif pgrep -f 'uvicorn app.main:app' >/dev/null 2>&1; then
    echo 'Webhook startup is already in progress; keeping the existing process.'
  else
    mkdir -p ${REMOTE_DIR}/run_logs
    nohup sh -c \
      'cd ${REMOTE_DIR} && \
       exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT} --env-file .env' \
       > run_logs/webhook.log 2>&1 < /dev/null &
  fi
"

log "Waiting for server to boot..."
log "Allowing up to $((APP_STARTUP_TIMEOUT_ATTEMPTS * 2 / 60)) minutes for model prewarm..."
$SSH "
  attempt=0
  ready=false
  until [ \$attempt -ge ${APP_STARTUP_TIMEOUT_ATTEMPTS} ]; do
    if ss -ltnp | grep ${APP_PORT} >/dev/null 2>&1 \
      && curl -fsS http://127.0.0.1:${APP_PORT}/ \
      | grep -q ready; then
      ready=true
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

  if [ "\$ready" != 'true' ]; then
    echo 'ERROR: server did not become ready on port ${APP_PORT}.'
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
  curl -sS --get 'http://127.0.0.1:${APP_PORT}/webhook' \
    --data-urlencode 'hub.mode=subscribe' \
    --data-urlencode "hub.verify_token=${VERIFY_TOKEN}" \
    --data-urlencode 'hub.challenge=12345'
  echo ''
"

log "Waiting for WhatsApp webhook verification..."
log "Use this callback URL in WhatsApp:"
log "  ${PUBLIC_WEBHOOK_URL}"
log ""

VERIFICATION_OK=false
for attempt in $(seq 1 "${PUBLIC_VERIFY_TIMEOUT_ATTEMPTS}"); do
  public_response="$(curl -4 -sS -m 15 --get "${PUBLIC_WEBHOOK_URL}" \
    --data-urlencode 'hub.mode=subscribe' \
    --data-urlencode "hub.verify_token=${VERIFY_TOKEN}" \
    --data-urlencode 'hub.challenge=12345' || true)"

  if [ "${public_response}" = "12345" ]; then
    log "WhatsApp webhook verification is working: ${PUBLIC_WEBHOOK_URL}"
    VERIFICATION_OK=true
    break
  fi

  echo "Waiting for verification to work... attempt ${attempt}/${PUBLIC_VERIFY_TIMEOUT_ATTEMPTS}; response: ${public_response:-<empty>}"
  sleep 5
done

if [ "${VERIFICATION_OK}" != "true" ]; then
  echo "ERROR: Webhook verification did not succeed within $((PUBLIC_VERIFY_TIMEOUT_ATTEMPTS * 5 / 60)) minutes."
  exit 1
fi

log "Setup complete. Webhook running on ${HOST_IP}:${APP_PORT}"
log ""
log "Webhook URL:"
log "  ${PUBLIC_WEBHOOK_URL}"
log ""
log "Useful commands:"
log "  Attach to webhook:  ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} -t 'tmux attach -t sl-webhook'"
log "  Watch logs:         ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'tail -f ${REMOTE_DIR}/run_logs/webhook.log'"
log "  Watch important:    ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'tail -f ${REMOTE_DIR}/run_logs/important.log'"
