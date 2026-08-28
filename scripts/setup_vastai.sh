#!/usr/bin/env bash
# Rent and configure the lean Vast.ai voice runtime. With SSH_PORT and HOST_IP
# arguments it configures that existing host; without arguments it rents one.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [ "$#" -eq 0 ]; then
DISK_GB="${DISK_GB:-50}"
MIN_GPU_RAM_GB="${MIN_GPU_RAM_GB:-16}"
MIN_CPU_CORES="${MIN_CPU_CORES:-8}"
MIN_INTERNET_DOWN_MBIT="${MIN_INTERNET_DOWN_MBIT:-500}"
MIN_CUDA_VERSION="${MIN_CUDA_VERSION:-12.8}"
REMOTE_BRANCH="${REMOTE_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/vastai_ssh_file}"
TEMPLATE_HASH="${TEMPLATE_HASH:-247f2f26d31d533719c1fc4c9b5cbf93}"
INSTANCE_LABEL="${INSTANCE_LABEL:-serendibai-whatsapp}"
DRY_RUN="${DRY_RUN:-false}"
STARTUP_TIMEOUT_ATTEMPTS="${STARTUP_TIMEOUT_ATTEMPTS:-60}"
MAX_INSTANCE_ATTEMPTS="${MAX_INSTANCE_ATTEMPTS:-3}"

log() { printf '▶ %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

command -v uvx >/dev/null 2>&1 || fail "uvx is required: https://docs.astral.sh/uv/"
test -f .env || fail "${ROOT_DIR}/.env is required"
test -f "${SSH_KEY}" || fail "SSH key not found: ${SSH_KEY}"
[[ "${MIN_GPU_RAM_GB}" =~ ^[0-9]+$ ]] || fail "MIN_GPU_RAM_GB must be numeric"
[ "${MIN_GPU_RAM_GB}" -ge 16 ] || fail "Voice runtime deployments require at least 16 GB VRAM"
[[ "${MIN_CPU_CORES}" =~ ^[0-9]+$ ]] || fail "MIN_CPU_CORES must be numeric"
[[ "${MIN_INTERNET_DOWN_MBIT}" =~ ^[0-9]+$ ]] || fail "MIN_INTERNET_DOWN_MBIT must be numeric"

PYTHON="${ROOT_DIR}/.venv/bin/python"
test -x "${PYTHON}" || fail "Run 'uv sync' locally once before deploying"

VASTAI_API_KEY="$(${PYTHON} -c \
  'from dotenv import dotenv_values; print(dotenv_values(".env").get("VASTAI_API_KEY", ""))')"
test -n "${VASTAI_API_KEY}" || fail "VASTAI_API_KEY is missing from .env"

VASTAI=(uvx --from vastai vastai --api-key "${VASTAI_API_KEY}" --raw)
QUERY="num_gpus=1 gpu_ram>=${MIN_GPU_RAM_GB} cpu_cores_effective>=${MIN_CPU_CORES} cpu_arch=amd64 disk_space>=${DISK_GB} cuda_vers>=${MIN_CUDA_VERSION} direct_port_count>=1"

log "Finding the cheapest verified on-demand GPU with at least ${MIN_GPU_RAM_GB} GB VRAM, ${MIN_CPU_CORES} effective CPU cores, and ${MIN_INTERNET_DOWN_MBIT} Mbps ingress..."
ATTEMPTED_OFFER_IDS=""

select_offer() {
  "${VASTAI[@]}" search offers "${QUERY}" \
    --storage "${DISK_GB}" --order dph --limit 200 \
  | EXCLUDED_OFFER_IDS="${ATTEMPTED_OFFER_IDS}" MIN_INTERNET_DOWN_MBIT="${MIN_INTERNET_DOWN_MBIT}" "${PYTHON}" -c '
import json
import os
import re
import sys

offers = json.load(sys.stdin)
allowed = re.compile(r"^RTX (?:30|40)\d{2}(?:S| Super| Ti)?$", re.IGNORECASE)
excluded = {value for value in os.environ.get("EXCLUDED_OFFER_IDS", "").split(",") if value}
minimum_down = float(os.environ["MIN_INTERNET_DOWN_MBIT"])
eligible = [
    offer for offer in offers
    if str(offer.get("id", "")) not in excluded
    and allowed.search(str(offer.get("gpu_name", "")))
    and float(offer.get("inet_down") or 0) >= minimum_down
]
if not eligible:
    raise SystemExit("No untried eligible Vast.ai offer is currently available")
offer = min(eligible, key=lambda row: float(row.get("dph_total", "inf")))
fields = (
    offer["id"],
    offer["gpu_name"],
    int(offer["gpu_ram"]),
    float(offer["dph_total"]),
    offer.get("geolocation", "unknown"),
    float(offer.get("reliability", 0)),
)
print("\t".join(map(str, fields)))
'
}

if [ "${DRY_RUN}" = "true" ]; then
  OFFER="$(select_offer)"
  IFS=$'\t' read -r OFFER_ID GPU_NAME GPU_RAM HOURLY_PRICE LOCATION RELIABILITY <<<"${OFFER}"
  log "Selected offer ${OFFER_ID}: ${GPU_NAME}, ${GPU_RAM} MiB VRAM, \$${HOURLY_PRICE}/hour including ${DISK_GB} GB storage, ${LOCATION}, reliability ${RELIABILITY}"
  log "Dry run complete; no instance was created."
  exit 0
fi

EXISTING_CONNECTION="$(${VASTAI[@]} show instances | "${PYTHON}" -c '
import json
import sys

rows = json.load(sys.stdin)
if isinstance(rows, dict):
    rows = rows.get("instances", [])
rows = [
    row for row in rows
    if row.get("label") == "'"${INSTANCE_LABEL}"'"
    and row.get("actual_status") == "running"
    and float(row.get("gpu_ram") or 0) >= '"${MIN_GPU_RAM_GB}"' * 1024
]
rows.sort(key=lambda row: float(row.get("start_date") or 0), reverse=True)
for row in rows:
    mappings = (row.get("ports") or {}).get("22/tcp") or []
    if mappings and row.get("public_ipaddr"):
        print("\t".join(str(value or "") for value in (
            row.get("id"), row.get("public_ipaddr"), mappings[0].get("HostPort", ""))))
        break
')"

destroy_instance() {
  local id="$1"
  if [ -n "${id:-}" ]; then
    log "Destroying failed instance ${id}..."
    "${VASTAI[@]}" destroy instance "${id}" --yes >/dev/null 2>&1 || true
  fi
}

for instance_attempt in $(seq 1 "${MAX_INSTANCE_ATTEMPTS}"); do
  INSTANCE_ID=""
  SSH_HOST=""
  SSH_PORT=""

  if [ "${instance_attempt}" -eq 1 ] && [ -n "${EXISTING_CONNECTION}" ]; then
    IFS=$'\t' read -r INSTANCE_ID SSH_HOST SSH_PORT <<<"${EXISTING_CONNECTION}"
    log "Reusing existing running instance ${INSTANCE_ID} at ${SSH_HOST}:${SSH_PORT}."
  else
    OFFER="$(select_offer)"
    IFS=$'\t' read -r OFFER_ID GPU_NAME GPU_RAM HOURLY_PRICE LOCATION RELIABILITY <<<"${OFFER}"
    ATTEMPTED_OFFER_IDS="${ATTEMPTED_OFFER_IDS:+${ATTEMPTED_OFFER_IDS},}${OFFER_ID}"
    log "Selected offer ${OFFER_ID}: ${GPU_NAME}, ${GPU_RAM} MiB VRAM, \$${HOURLY_PRICE}/hour including ${DISK_GB} GB storage, ${LOCATION}, reliability ${RELIABILITY}"
    log "Creating Vast.ai instance (attempt ${instance_attempt}/${MAX_INSTANCE_ATTEMPTS})..."
    if ! CREATE_RESULT="$(${VASTAI[@]} create instance "${OFFER_ID}" \
      --template_hash "${TEMPLATE_HASH}" \
      --disk "${DISK_GB}" \
      --label "${INSTANCE_LABEL}" \
      --ssh --direct --cancel-unavail 2>&1)"; then
      if [[ "${CREATE_RESULT}" == *"lacks credit"* ]]; then
        fail "Vast.ai account lacks credit; top up the account before deploying."
      fi
      log "Offer ${OFFER_ID} became unavailable; trying another offer."
      continue
    fi
    if [[ "${CREATE_RESULT}" == *"lacks credit"* ]]; then
      fail "Vast.ai account lacks credit; top up the account before deploying."
    fi
    if ! INSTANCE_ID="$(printf '%s' "${CREATE_RESULT}" | "${PYTHON}" -c \
      'import json,sys; print(json.load(sys.stdin).get("new_contract", ""))')"; then
      log "Offer ${OFFER_ID} did not return an instance ID; trying another offer."
      continue
    fi
    if [ -z "${INSTANCE_ID}" ]; then
      log "Offer ${OFFER_ID} became unavailable; trying another offer."
      continue
    fi
    log "Created instance ${INSTANCE_ID}."
  fi

  log "Waiting for instance ${INSTANCE_ID} and SSH endpoint (max five minutes)..."
  SSH_READY=false
  for attempt in $(seq 1 "${STARTUP_TIMEOUT_ATTEMPTS}"); do
  INSTANCE="$("${VASTAI[@]}" show instance "${INSTANCE_ID}")"
  CONNECTION="$(printf '%s' "${INSTANCE}" | "${PYTHON}" -c '
import json
import sys

instance = json.load(sys.stdin)
if isinstance(instance, list):
    instance = instance[0] if instance else {}
ssh_mappings = (instance.get("ports") or {}).get("22/tcp") or []
ssh_port = ssh_mappings[0].get("HostPort", "") if ssh_mappings else ""
print("\t".join(str(value or "") for value in (
    instance.get("actual_status"),
    instance.get("public_ipaddr"),
    ssh_port,
)))
')"
  IFS=$'\t' read -r INSTANCE_STATUS SSH_HOST SSH_PORT <<<"${CONNECTION}"
  if [ "${INSTANCE_STATUS}" = "running" ] && [ -n "${SSH_HOST}" ] && [ -n "${SSH_PORT}" ]; then
    if ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new \
      -i "${SSH_KEY}" -p "${SSH_PORT}" "root@${SSH_HOST}" true 2>/dev/null; then
      SSH_READY=true
      break
    fi
  fi
  if [ "${attempt}" -eq "${STARTUP_TIMEOUT_ATTEMPTS}" ]; then
    log "Instance ${INSTANCE_ID} did not become SSH-ready within five minutes."
    break
  fi
  if [ $((attempt % 12)) -eq 0 ]; then
    log "Still waiting for instance ${INSTANCE_ID} SSH (attempt ${attempt}/${STARTUP_TIMEOUT_ATTEMPTS}); the next retry is bounded."
  fi
  sleep 5
done

  if [ "${SSH_READY}" != "true" ]; then
    destroy_instance "${INSTANCE_ID}"
    continue
  fi

  log "Deploying branch ${REMOTE_BRANCH} to instance ${INSTANCE_ID} without a post-SSH deadline..."
  if env \
    REMOTE_BRANCH="${REMOTE_BRANCH}" SSH_KEY="${SSH_KEY}" \
    "${ROOT_DIR}/scripts/setup_vastai.sh" "${SSH_PORT}" "${SSH_HOST}"; then
    log "Deployment complete."
    log "Instance ID: ${INSTANCE_ID}"
    log "Destroy instance ${INSTANCE_ID} in Vast.ai as soon as the call is finished."
    exit 0
  fi

  log "Setup failed; terminating instance ${INSTANCE_ID}."
  destroy_instance "${INSTANCE_ID}"
  if [ "${instance_attempt}" -lt "${MAX_INSTANCE_ATTEMPTS}" ]; then
    log "Trying a different offer..."
    continue
  fi
done

fail "All ${MAX_INSTANCE_ATTEMPTS} instance attempts failed; no server was left running."
fi

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
LLAMA_VERSION="${LLAMA_VERSION:-b10612}"

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
  attempt=0
  while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
    attempt=\$((attempt + 1))
    if [ \$((attempt % 12)) -eq 0 ]; then
      echo 'Still waiting for the package manager lock... attempt' \$attempt
    fi
    sleep 5
  done
"

log "Installing minimal system packages..."
$SSH "apt-get update -qq && apt-get install -y --no-install-recommends portaudio19-dev curl"

log "Installing Python runtime and prebuilt llama.cpp while downloading Gemma in parallel..."
$SSH "
  set -euo pipefail
  cd ${REMOTE_DIR}
  env -u UV_NO_CACHE uv sync --frozen --no-dev &
  uv_pid=\$!

  (
    if [ ! -x /root/.local/bin/llama ]; then
      curl --fail --location --retry 3 https://llama.app/install.sh \
        | env LLAMA_VERSION=${LLAMA_VERSION} sh
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
  install -d -m 755 /opt/supervisor-scripts
  printf '%s\\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'cd /workspace/sl-chatbot' \
    'mkdir -p run_logs' \
    'exec >>run_logs/llm.log 2>&1' \
    'exec /root/.local/bin/llama serve --model /workspace/models/gemma-4-E4B-it-qat-q4_0/gemma-4-E4B_q4_0-it.gguf --alias google/gemma-4-E4B-it-qat-q4_0-gguf --n-gpu-layers 99 --ctx-size 4096 --flash-attn on --jinja --host 127.0.0.1 --port 8000' \
    > /opt/supervisor-scripts/sl-llm.sh
  printf '%s\\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'cd /workspace/sl-chatbot' \
    'mkdir -p run_logs' \
    'exec >>run_logs/cloudflared.log 2>&1' \
    'set -a; . ./.env; set +a' \
    'exec /opt/instance-tools/bin/cloudflared tunnel run' \
    > /opt/supervisor-scripts/sl-cloudflared.sh
  chmod 755 /opt/supervisor-scripts/sl-llm.sh /opt/supervisor-scripts/sl-cloudflared.sh
  printf '%s\\n' \
    '[program:sl-llm]' \
    'command=/opt/supervisor-scripts/sl-llm.sh' \
    'autostart=true' \
    'autorestart=unexpected' \
    'startsecs=2' \
    'stdout_logfile=/dev/stdout' \
    'stdout_logfile_maxbytes=0' \
    'redirect_stderr=true' \
    > /etc/supervisor/conf.d/sl-llm.conf
  printf '%s\\n' \
    '[program:sl-cloudflared]' \
    'command=/opt/supervisor-scripts/sl-cloudflared.sh' \
    'autostart=true' \
    'autorestart=unexpected' \
    'startsecs=2' \
    'stdout_logfile=/dev/stdout' \
    'stdout_logfile_maxbytes=0' \
    'redirect_stderr=true' \
    > /etc/supervisor/conf.d/sl-cloudflared.conf
  supervisorctl reread
  supervisorctl update
  supervisorctl restart sl-llm
  supervisorctl restart sl-cloudflared
"

log "Waiting for local Gemma to become ready..."
$SSH "
  attempt=0
  until curl -fsS http://127.0.0.1:${LLM_PORT}/v1/models >/dev/null; do
    attempt=\$((attempt + 1))
    if [ \$((attempt % 15)) -eq 0 ]; then
      echo 'Still waiting for local Gemma... attempt' \$attempt
      tail -n 20 ${REMOTE_DIR}/run_logs/llm.log || true
    fi
    sleep 2
  done
"
log "Starting webhook..."
$SSH "
  cd ${REMOTE_DIR}
  printf '%s\\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'cd /workspace/sl-chatbot' \
    'mkdir -p run_logs' \
    'exec >>run_logs/webhook.log 2>&1' \
    'exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8081 --env-file .env' \
    > /opt/supervisor-scripts/sl-webhook.sh
  chmod 755 /opt/supervisor-scripts/sl-webhook.sh
  printf '%s\\n' \
    '[program:sl-webhook]' \
    'command=/opt/supervisor-scripts/sl-webhook.sh' \
    'autostart=true' \
    'autorestart=unexpected' \
    'startsecs=2' \
    'stdout_logfile=/dev/stdout' \
    'stdout_logfile_maxbytes=0' \
    'redirect_stderr=true' \
    > /etc/supervisor/conf.d/sl-webhook.conf
  supervisorctl reread
  supervisorctl update
  supervisorctl restart sl-webhook
"

log "Waiting for server to boot..."
log "Waiting without a deadline for model prewarm..."
$SSH "
  attempt=0
  until ss -ltnp | grep ${APP_PORT} >/dev/null 2>&1 \
    && curl -fsS http://127.0.0.1:${APP_PORT}/ | grep -q ready; do
    attempt=\$((attempt + 1))
    if [ \$((attempt % 15)) -eq 0 ]; then
      echo 'Still waiting for port ${APP_PORT}... attempt' \$attempt '(model download/prewarm may still be running)'
      if [ -f ${REMOTE_DIR}/run_logs/webhook.log ]; then
        echo '--- latest webhook startup log ---'
        tail -n 12 ${REMOTE_DIR}/run_logs/webhook.log | sed 's/^/    /'
      fi
    fi
    sleep 2
  done

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

attempt=0
while true; do
  public_response="$(curl -4 -sS --get "${PUBLIC_WEBHOOK_URL}" \
    --data-urlencode 'hub.mode=subscribe' \
    --data-urlencode "hub.verify_token=${VERIFY_TOKEN}" \
    --data-urlencode 'hub.challenge=12345' || true)"

  if [ "${public_response}" = "12345" ]; then
    log "WhatsApp webhook verification is working: ${PUBLIC_WEBHOOK_URL}"
    break
  fi

  attempt=$((attempt + 1))
  echo "Waiting for verification to work... attempt ${attempt}; response: ${public_response:-<empty>}"
  sleep 5
done

log "Setup complete. Webhook running on ${HOST_IP}:${APP_PORT}"
log ""
log "Webhook URL:"
log "  ${PUBLIC_WEBHOOK_URL}"
log ""
log "Useful commands:"
log "  Service status:     ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'supervisorctl status sl-llm sl-webhook sl-cloudflared'"
log "  Watch logs:         ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'tail -f ${REMOTE_DIR}/run_logs/webhook.log'"
log "  Watch important:    ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'tail -f ${REMOTE_DIR}/run_logs/important.log'"
