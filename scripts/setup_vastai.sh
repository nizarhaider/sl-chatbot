#!/usr/bin/env bash
# =============================================================================
# setup_vastai.sh  —  One-shot setup for a fresh Vast.ai GPU box
#
# Usage (run from your LOCAL repo root):
#   ./scripts/setup_vastai.sh <SSH_PORT> <HOST_IP>
#
# Example:
#   ./scripts/setup_vastai.sh 42609 143.55.45.86
#
# What it does:
#   1. Checks machine basics (GPU, uv, git)
#   2. Clones the repo on the remote if missing
#   3. Copies .env to the remote host
#   4. Installs portaudio19-dev
#   5. Runs uv sync
#   6. Compile-checks all Python modules
#   7. Starts the webhook in a tmux session
#   8. Runs a local health check
# =============================================================================

set -euo pipefail

SSH_PORT="${1:?Usage: $0 <SSH_PORT> <HOST_IP>}"
HOST_IP="${2:?Usage: $0 <SSH_PORT> <HOST_IP>}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/vastai_ssh_file}"
REMOTE="root@${HOST_IP}"
REMOTE_DIR="/workspace/sl-chatbot"
APP_PORT=8081

SSH="ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE}"
SCP="scp -P ${SSH_PORT} -i ${SSH_KEY}"

log() { echo "▶ $*"; }

# ── 1. Machine check ──────────────────────────────────────────────────────────
log "Checking machine..."
$SSH "uname -a && nvidia-smi --query-gpu=name --format=csv,noheader && which uv git python3"

# ── 2. Clone or sync repo ────────────────────────────────────────────────────
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

# ── 3. Sync local environment ─────────────────────────────────────────────────
log ".env sync..."
if [ -f .env ]; then
  $SCP .env ${REMOTE}:${REMOTE_DIR}/
else
  echo 'WARNING: .env not found locally; skipping .env copy.'
fi

log "Files synced."

# ── 4. System packages ────────────────────────────────────────────────────────
log "Installing portaudio19-dev..."
$SSH "apt-get update -qq && apt-get install -y portaudio19-dev"

# ── 5. Python deps ────────────────────────────────────────────────────────────
log "Running uv sync (this takes a few minutes)..."
$SSH "cd ${REMOTE_DIR} && uv sync"

# ── 6. Compile check ──────────────────────────────────────────────────────────
log "Compile-checking Python modules..."
$SSH "cd ${REMOTE_DIR} && .venv/bin/python -m py_compile \
  app/main.py \
  app/webhooks/whatsapp.py \
  app/services/webrtc.py \
  app/services/whatsapp_api.py \
  app/voice_agent/agent.py \
  app/voice_agent/gemini_turn_pipeline.py && echo 'COMPILE OK'"

# ── 7. Start webhook ──────────────────────────────────────────────────────────
log "Starting webhook in tmux..."
$SSH "tmux new-session -d -s sl-webhook 'cd ${REMOTE_DIR} && .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT} --env-file .env'"

# ── 8. Health check ───────────────────────────────────────────────────────────
log "Waiting for server to boot..."
sleep 10

log "Health check..."
$SSH "
  ss -ltnp | grep ${APP_PORT} || echo 'WARNING: port not listening yet'
  curl -sS http://127.0.0.1:${APP_PORT}/ && echo ''
  curl -sS 'http://127.0.0.1:${APP_PORT}/webhook?hub.mode=subscribe&hub.verify_token=my_secure_verify_token_123&hub.challenge=12345'
  echo ''
"

log "✅ Setup complete! Webhook running on ${HOST_IP}:${APP_PORT}"
log ""
log "Useful commands:"
log "  Attach to webhook:  ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} -t 'tmux attach -t sl-webhook'"
log "  Watch logs:         ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'tail -f ${REMOTE_DIR}/run_logs/webhook.log'"
log "  Watch important:    ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'tail -f ${REMOTE_DIR}/run_logs/important.log'"
