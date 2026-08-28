#!/usr/bin/env bash
# =============================================================================
# setup_vastai.sh — Setup a Vast.ai instance using the
# SPEAK-ASR/whisper-medium-si-merged ASR model and local Gemma 4 E4B QAT GGUF.
# =============================================================================

set -euo pipefail

SSH_PORT="${1:?Usage: $0 <SSH_PORT> <HOST_IP>}"
HOST_IP="${2:?Usage: $0 <SSH_PORT> <HOST_IP>}"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/vastai_ssh_file}"
REMOTE="root@${HOST_IP}"
REMOTE_DIR="/workspace/sl-chatbot"
APP_PORT="${APP_PORT:-8081}"
LLM_PORT="${LLM_PORT:-8000}"
LLM_MODEL="${LLM_MODEL:-google/gemma-4-E4B-it-qat-q4_0-gguf}"
LLM_MODEL_REPO="google/gemma-4-E4B-it-qat-q4_0-gguf"
LLM_MODEL_FILE="gemma-4-E4B_q4_0-it.gguf"
LLM_MODEL_DIR="/workspace/models/gemma-4-E4B-it-qat-q4_0"
LLAMA_CPP_TAG="${LLAMA_CPP_TAG:-b10675}"
LLAMA_CPP_DIR="/workspace/llama.cpp"
LLM_STARTUP_TIMEOUT_ATTEMPTS="${LLM_STARTUP_TIMEOUT_ATTEMPTS:-300}"
APP_STARTUP_TIMEOUT_ATTEMPTS="${APP_STARTUP_TIMEOUT_ATTEMPTS:-150}"
PUBLIC_VERIFY_TIMEOUT_ATTEMPTS="${PUBLIC_VERIFY_TIMEOUT_ATTEMPTS:-3}"
UV_SYNC_TIMEOUT_SECONDS="${UV_SYNC_TIMEOUT_SECONDS:-900}"

PUBLIC_WEBHOOK_URL="https://whatsapp.serendibai.lk/webhook"
REMOTE_BRANCH="${REMOTE_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"

SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE}"
SCP="scp -o StrictHostKeyChecking=accept-new -P ${SSH_PORT} -i ${SSH_KEY}"

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
ENV_SYNC_FILE="$(mktemp)"
S3_CACHE_FILE="$(mktemp)"
cleanup_env_sync() { rm -f "${ENV_SYNC_FILE}" "${S3_CACHE_FILE}"; }
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

if command -v aws >/dev/null 2>&1 && aws sts get-caller-identity >/dev/null 2>&1; then
  log "Creating temporary S3 model-cache links..."
  uv run --quiet --no-project --with boto3 --with 'botocore[crt]' \
    python - "${LLM_MODEL_FILE}" > "${S3_CACHE_FILE}" <<'PY'
import shlex
import sys

import boto3

bucket = "serendibai-models"
key = f"runtime-cache/{sys.argv[1]}"
client = boto3.client("s3", region_name="ap-southeast-1")
for variable, operation in (
    ("S3_CACHE_GET_URL", "get_object"),
    ("S3_CACHE_PUT_URL", "put_object"),
):
    url = client.generate_presigned_url(
        operation,
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=21600,
    )
    print(f"{variable}={shlex.quote(url)}")
PY
  $SCP "${S3_CACHE_FILE}" "${REMOTE}:/tmp/sl-chatbot-s3-cache.env"
else
  : > "${S3_CACHE_FILE}"
  log "S3 credentials unavailable; using Hugging Face directly."
fi

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

log "Installing minimal system packages..."
$SSH "apt-get update -qq && apt-get install -y --no-install-recommends build-essential cmake ninja-build portaudio19-dev curl"

log "Installing Python runtime, compiling llama-server, and downloading Gemma in parallel..."
$SSH "
  set -euo pipefail
  cd ${REMOTE_DIR}
  timeout ${UV_SYNC_TIMEOUT_SECONDS}s env -u UV_NO_CACHE uv sync --frozen --no-dev &
  uv_pid=\$!

  (
    if [ ! -x ${LLAMA_CPP_DIR}/build/bin/llama-server ]; then
      rm -rf ${LLAMA_CPP_DIR}
      git clone --depth 1 --branch ${LLAMA_CPP_TAG} https://github.com/ggml-org/llama.cpp.git ${LLAMA_CPP_DIR}
      cmake -S ${LLAMA_CPP_DIR} -B ${LLAMA_CPP_DIR}/build -G Ninja \
        -DGGML_CUDA=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
      cmake --build ${LLAMA_CPP_DIR}/build --target llama-server --parallel \"\$(nproc)\"
    fi
  ) &
  llama_pid=\$!

  (
    mkdir -p ${LLM_MODEL_DIR}
    model_path=${LLM_MODEL_DIR}/${LLM_MODEL_FILE}
    cache_hit=false
    if [ -s /tmp/sl-chatbot-s3-cache.env ]; then
      . /tmp/sl-chatbot-s3-cache.env
      if curl --fail --location --retry 2 --continue-at - \
        \"\$S3_CACHE_GET_URL\" --output \"\$model_path\"; then
        cache_hit=true
      fi
    fi
    if [ \"\$cache_hit\" != true ]; then
      hf_token=\$(sed -n 's/^HF_TOKEN=//p' .env | head -n 1)
      test -n \"\$hf_token\"
      curl --fail --location --retry 3 --continue-at - \
        -H \"Authorization: Bearer \$hf_token\" \
        https://huggingface.co/${LLM_MODEL_REPO}/resolve/main/${LLM_MODEL_FILE} \
        --output \"\$model_path\"
      if [ -n \"\${S3_CACHE_PUT_URL:-}\" ]; then
        curl --fail --silent --show-error --request PUT \
          --upload-file \"\$model_path\" \"\$S3_CACHE_PUT_URL\" \
          && echo 'Seeded the S3 Gemma cache.' \
          || echo 'WARNING: Could not seed the S3 Gemma cache.'
      fi
    fi
    rm -f /tmp/sl-chatbot-s3-cache.env
  ) &
  model_pid=\$!

  wait \$uv_pid
  wait \$llama_pid
  wait \$model_pid
"

log "Pre-downloading Whisper and OmniVoice concurrently into the shared Hugging Face cache..."
$SSH "
  set -euo pipefail
  cd ${REMOTE_DIR}
  hf_token=\$(sed -n 's/^HF_TOKEN=//p' .env | head -n 1)
  test -n \"\$hf_token\"
  HF_TOKEN=\"\$hf_token\" .venv/bin/hf download SPEAK-ASR/whisper-medium-si-merged &
  asr_pid=\$!
  HF_TOKEN=\"\$hf_token\" .venv/bin/hf download 2broke2code/serendib-omnivoice-finetuned-v2 &
  tts_pid=\$!
  wait \$asr_pid
  wait \$tts_pid
"

log "Compile-checking Python modules..."
$SSH "cd ${REMOTE_DIR} && find app -name '*.py' -print0 | xargs -0 .venv/bin/python -m py_compile && echo 'COMPILE OK'"

log "Starting local Gemma and the permanent Cloudflare tunnel..."
$SSH "
  cd ${REMOTE_DIR}
  mkdir -p run_logs
  install -m 755 scripts/supervisor/sl-llm.sh /opt/supervisor-scripts/sl-llm.sh
  install -m 755 scripts/supervisor/sl-cloudflared.sh /opt/supervisor-scripts/sl-cloudflared.sh
  install -m 644 scripts/supervisor/sl-llm.conf /etc/supervisor/conf.d/sl-llm.conf
  install -m 644 scripts/supervisor/sl-cloudflared.conf /etc/supervisor/conf.d/sl-cloudflared.conf
  supervisorctl reread
  supervisorctl update
  supervisorctl restart sl-llm
  supervisorctl restart sl-cloudflared
"

log "Waiting for local Gemma to become ready..."
$SSH "
  for attempt in \$(seq 1 ${LLM_STARTUP_TIMEOUT_ATTEMPTS}); do
    if curl -fsS http://127.0.0.1:${LLM_PORT}/v1/models >/dev/null; then
      exit 0
    fi
    sleep 2
  done
  tail -n 120 ${REMOTE_DIR}/run_logs/llm.log || true
  exit 1
"
log "Starting webhook..."
$SSH "
  cd ${REMOTE_DIR}
  install -m 755 scripts/supervisor/sl-webhook.sh /opt/supervisor-scripts/sl-webhook.sh
  install -m 644 scripts/supervisor/sl-webhook.conf /etc/supervisor/conf.d/sl-webhook.conf
  supervisorctl reread
  supervisorctl update
  supervisorctl restart sl-webhook
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
    echo 'Supervisor status:'
    supervisorctl status sl-llm sl-webhook sl-cloudflared || true
    echo ''
    echo 'sl-webhook supervisor log:'
    supervisorctl tail -100 sl-webhook stderr || true
    echo ''
    supervisorctl tail -160 sl-webhook stdout || true
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
log "  Service status:     ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'supervisorctl status sl-llm sl-webhook sl-cloudflared'"
log "  Watch logs:         ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'tail -f ${REMOTE_DIR}/run_logs/webhook.log'"
log "  Watch important:    ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'tail -f ${REMOTE_DIR}/run_logs/important.log'"
