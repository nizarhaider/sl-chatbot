#!/usr/bin/env bash
# Rent and configure the lean Vast.ai voice runtime. With SSH_PORT and HOST_IP
# arguments it configures that existing host; without arguments it rents one.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [ "$#" -eq 0 ]; then
DISK_GB="${DISK_GB:-50}"
MIN_GPU_RAM_GB="${MIN_GPU_RAM_GB:-24}"
MIN_CPU_CORES="${MIN_CPU_CORES:-8}"
MIN_INTERNET_DOWN_MBIT="${MIN_INTERNET_DOWN_MBIT:-700}"
MIN_CUDA_VERSION="${MIN_CUDA_VERSION:-12.8}"
REMOTE_BRANCH="${REMOTE_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/vastai_ssh_file}"
TEMPLATE_HASH="${TEMPLATE_HASH:-247f2f26d31d533719c1fc4c9b5cbf93}"
INSTANCE_LABEL="${INSTANCE_LABEL:-serendibai-whatsapp}"
DRY_RUN="${DRY_RUN:-false}"
STARTUP_TIMEOUT_ATTEMPTS="${STARTUP_TIMEOUT_ATTEMPTS:-24}"
MAX_INSTANCE_ATTEMPTS="${MAX_INSTANCE_ATTEMPTS:-3}"

log() { printf '▶ %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

command -v uvx >/dev/null 2>&1 || fail "uvx is required: https://docs.astral.sh/uv/"
test -f .env || fail "${ROOT_DIR}/.env is required"
test -f "${SSH_KEY}" || fail "SSH key not found: ${SSH_KEY}"
[[ "${MIN_GPU_RAM_GB}" =~ ^[0-9]+$ ]] || fail "MIN_GPU_RAM_GB must be numeric"
[ "${MIN_GPU_RAM_GB}" -ge 24 ] || fail "Gemma Q4 voice runtime deployments require at least 24 GB VRAM"
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
# The voice runtime CUDA build supports only the Ampere/Ada consumer cards
# selected below. Older Tesla cards (for example, P40) cannot run its kernels.
allowed = re.compile(r"\bRTX\s+(?:30|40)\d{2}\b", re.IGNORECASE)
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

EXISTING_INSTANCE_ID="$(${VASTAI[@]} show instances | "${PYTHON}" -c '
import json
import sys

rows = json.load(sys.stdin)
if isinstance(rows, dict):
    rows = rows.get("instances", [])
active = [row for row in rows if row.get("actual_status") in {"running", "loading", "creating"}]
if len(active) > 1:
    raise SystemExit("Refusing to deploy while more than one Vast.ai instance is active")
if active:
    row = active[0]
    if row.get("label") != "'"${INSTANCE_LABEL}"'":
        raise SystemExit("Refusing to deploy while another Vast.ai instance is active")
    if float(row.get("gpu_ram") or 0) < '"${MIN_GPU_RAM_GB}"' * 1024:
        raise SystemExit("The active Vast.ai instance does not have enough VRAM")
    print(row.get("id", ""))
')"

destroy_unready_instance() {
  local id="$1"
  log "Destroying instance ${id}: it did not become SSH-ready within two minutes."
  "${VASTAI[@]}" destroy instance "${id}" --yes >/dev/null
}

for instance_attempt in $(seq 1 "${MAX_INSTANCE_ATTEMPTS}"); do
  INSTANCE_ID=""
  SSH_HOST=""
  SSH_PORT=""

  if [ "${instance_attempt}" -eq 1 ] && [ -n "${EXISTING_INSTANCE_ID}" ]; then
    INSTANCE_ID="${EXISTING_INSTANCE_ID}"
    log "Reusing existing running instance ${INSTANCE_ID}; waiting for its SSH endpoint."
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

  log "Waiting for instance ${INSTANCE_ID} and SSH endpoint (max two minutes)..."
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
    log "Instance ${INSTANCE_ID} did not become SSH-ready within two minutes."
    break
  fi
  if [ $((attempt % 12)) -eq 0 ]; then
    log "Still waiting for instance ${INSTANCE_ID} SSH (attempt ${attempt}/${STARTUP_TIMEOUT_ATTEMPTS}); the next retry is bounded."
  fi
  sleep 5
done

  if [ "${SSH_READY}" != "true" ]; then
    destroy_unready_instance "${INSTANCE_ID}"
    fail "Instance ${INSTANCE_ID} was not SSH-ready within two minutes and was destroyed."
  fi

  log "Deploying branch ${REMOTE_BRANCH} to instance ${INSTANCE_ID} without a post-SSH deadline..."
  if env \
    REMOTE_BRANCH="${REMOTE_BRANCH}" SSH_KEY="${SSH_KEY}" \
    "${ROOT_DIR}/scripts/setup_vastai.sh" "${SSH_PORT}" "${SSH_HOST}"; then
    log "Deployment complete."
    log "Instance ID: ${INSTANCE_ID}"
    exit 0
  fi

  setup_status="$?"
  if [ "${setup_status}" -eq 42 ]; then
    destroy_unready_instance "${INSTANCE_ID}"
    fail "Instance ${INSTANCE_ID} failed the early Cloudflare tunnel check and was destroyed."
  fi

  fail "Setup failed on instance ${INSTANCE_ID}; it was left running for inspection."
done

fail "All ${MAX_INSTANCE_ATTEMPTS} instance attempts failed."
fi

#!/usr/bin/env bash
# =============================================================================
# setup_vastai.sh — Setup a Vast.ai instance using the
# SPEAK-ASR/whisper-medium-si-merged ASR model and Unsloth's Q4 Gemma 4 E4B GGUF.
# =============================================================================

set -euo pipefail

SSH_PORT="${1:?Usage: $0 <SSH_PORT> <HOST_IP>}"
HOST_IP="${2:?Usage: $0 <SSH_PORT> <HOST_IP>}"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/vastai_ssh_file}"
REMOTE="root@${HOST_IP}"
REMOTE_DIR="/workspace/sl-chatbot"
APP_PORT="${APP_PORT:-8081}"
LLM_PORT="${LLM_PORT:-8000}"
LLM_MODEL="unsloth/gemma-4-E4B-it-GGUF"
LLM_MODEL_FILE="gemma-4-E4B-it-UD-Q4_K_XL.gguf"
LLM_MODEL_PATH="/workspace/models/${LLM_MODEL_FILE}"

PUBLIC_WEBHOOK_URL="https://whatsapp.serendibai.lk/webhook"
REMOTE_BRANCH="${REMOTE_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"

SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE}"
SCP="scp -o StrictHostKeyChecking=accept-new -P ${SSH_PORT} -i ${SSH_KEY}"

log() { echo "▶ $*"; }

log "Checking machine..."
GPU_NAME="$($SSH 'nvidia-smi --query-gpu=name --format=csv,noheader')"
if ! printf '%s\n' "${GPU_NAME}" | grep -Eqi '(^| )RTX (30|40)[0-9][0-9]($| )'; then
  echo "ERROR: Unsupported GPU '${GPU_NAME}'. The voice runtime requires an RTX 30- or 40-series GPU."
  exit 1
fi
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

log "Checking Cloudflare tunnel connectivity before installing dependencies..."
if ! $SSH "
  set -euo pipefail
  cd ${REMOTE_DIR}
  set -a
  . ./.env
  set +a
  tunnel_log=\$(mktemp)
  /opt/instance-tools/bin/cloudflared tunnel --protocol http2 --edge-ip-version 4 run \
    --token \"\$CLOUDFLARED_TUNNEL_TOKEN\" >\"\$tunnel_log\" 2>&1 &
  tunnel_pid=\$!
  for attempt in \$(seq 1 12); do
    if grep -q 'Registered tunnel connection' \"\$tunnel_log\"; then
      kill \"\$tunnel_pid\" 2>/dev/null || true
      wait \"\$tunnel_pid\" 2>/dev/null || true
      rm -f \"\$tunnel_log\"
      exit 0
    fi
    if ! kill -0 \"\$tunnel_pid\" 2>/dev/null; then
      cat \"\$tunnel_log\" >&2
      rm -f \"\$tunnel_log\"
      exit 1
    fi
    sleep 5
  done
  cat \"\$tunnel_log\" >&2
  kill \"\$tunnel_pid\" 2>/dev/null || true
  wait \"\$tunnel_pid\" 2>/dev/null || true
  rm -f \"\$tunnel_log\"
  exit 1
"; then
  echo "ERROR: Cloudflare tunnel could not register on this host; refusing to install dependencies."
  exit 42
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
$SSH "apt-get update -qq && apt-get install -y --no-install-recommends portaudio19-dev curl ffmpeg"

log "Reusing the image Python/Torch runtime and installing app packages in parallel..."
$SSH "
  set -euo pipefail
  cd ${REMOTE_DIR}
  # The vastai/pytorch image already provides Torch and Torchaudio in /venv/main.
  # Keep the project launcher paths compatible without creating a duplicate CUDA venv.
  if [ -e .venv ] && [ ! -L .venv ]; then rm -rf .venv; fi
  if [ ! -e .venv ]; then ln -s /venv/main .venv; fi
  uv pip install --python /venv/main/bin/python \
    'fastapi>=0.129.0' 'uvicorn>=0.41.0' 'aiortc>=1.9.0' 'httpx>=0.27.0' \
    'numpy>=1.26.0' 'realtimetts[omnivoice]>=0.7.1' 'python-dotenv>=1.2.1' \
    'psycopg[binary]>=3.2' 'transformers>=5.3.0,<6' 'pinecone>=9.1.0' \
    'num2words>=0.5.14' 'boto3>=1.42.0' &
  uv_pid=\$!
  (
    # Build current llama.cpp with CUDA support; the image's cached binary may
    # predate Gemma 4 fixes. The model weights are downloaded separately below.
    if [ ! -x /workspace/llama.cpp/build/bin/llama-server ]; then
      rm -rf /workspace/llama.cpp
      git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /workspace/llama.cpp
      cmake -S /workspace/llama.cpp -B /workspace/llama.cpp/build \
        -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES='86;89' \
        -DLLAMA_CURL=OFF
      cmake --build /workspace/llama.cpp/build --config Release --target llama-server -j 8
    fi
    /workspace/llama.cpp/build/bin/llama-server --version
  ) &
  llama_pid=\$!
  wait \$uv_pid
  wait \$llama_pid
"

log "Pre-downloading Q4 Gemma, Whisper, and OmniVoice with Hugging Face Xet..."
$SSH "
  set -euo pipefail
  cd ${REMOTE_DIR}
  hf_token=\$(sed -n 's/^HF_TOKEN=//p' .env | head -n 1)
  test -n \"\$hf_token\"
  HF_TOKEN=\"\$hf_token\" HF_XET_HIGH_PERFORMANCE=1 \\
    .venv/bin/hf download ${LLM_MODEL} --include ${LLM_MODEL_FILE} --local-dir /workspace/models &
  llm_pid=\$!
  HF_TOKEN=\"\$hf_token\" HF_XET_HIGH_PERFORMANCE=1 \\
    .venv/bin/hf download SPEAK-ASR/whisper-medium-si-merged &
  asr_pid=\$!
  HF_TOKEN=\"\$hf_token\" HF_XET_HIGH_PERFORMANCE=1 \\
    .venv/bin/hf download 2broke2code/serendib-omnivoice-finetuned-v2 &
  tts_pid=\$!
  wait \$llm_pid
  wait \$asr_pid
  wait \$tts_pid
"

log "Compile-checking Python modules..."
$SSH "cd ${REMOTE_DIR} && find app -name '*.py' -print0 | xargs -0 .venv/bin/python -m py_compile && echo 'COMPILE OK'"

log "Starting the local Q4 Gemma server and permanent Cloudflare tunnel..."
$SSH "
  cd ${REMOTE_DIR}
  # Expose the in-memory live-call monitor through an allocated normal port.
  # Caddy keeps its standard token authentication in front of the dashboard.
  if ! grep -q '^  Voice Dashboard:' /etc/portal.yaml; then
    cat >> /etc/portal.yaml <<'YAML'
  Voice Dashboard:
    hostname: localhost
    external_port: 10100
    internal_port: 8081
    open_path: /dashboard
    name: Voice Dashboard
YAML
    supervisorctl restart caddy >/dev/null
  fi
  mkdir -p run_logs
  install -d -m 755 /opt/supervisor-scripts
  printf '%s\\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'cd /workspace/sl-chatbot' \
    'mkdir -p run_logs' \
    'exec >>run_logs/llm.log 2>&1' \
    'export LLAMA_ARG_CHAT_TEMPLATE_KWARGS=\"{\\\"enable_thinking\\\":false}\"' \
    'exec /workspace/llama.cpp/build/bin/llama-server --model ${LLM_MODEL_PATH} --alias ${LLM_MODEL} --n-gpu-layers 99 --ctx-size 4096 --parallel 1 --batch-size 32 --ubatch-size 32 --flash-attn off --temperature 1.0 --top-p 0.95 --top-k 64 --jinja --host 127.0.0.1 --port ${LLM_PORT}' \
    > /opt/supervisor-scripts/sl-llm.sh
  printf '%s\\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'cd /workspace/sl-chatbot' \
    'mkdir -p run_logs' \
    'exec >>run_logs/cloudflared.log 2>&1' \
    'set -a; . ./.env; set +a' \
    'exec /opt/instance-tools/bin/cloudflared tunnel --protocol http2 --edge-ip-version 4 run --token "\$CLOUDFLARED_TUNNEL_TOKEN"' \
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
  supervisorctl restart sl-llm sl-cloudflared
"
log "Waiting for local Gemma to become ready..."
$SSH "until curl -fsS http://127.0.0.1:${LLM_PORT}/v1/models >/dev/null; do sleep 2; done"
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
    && curl -fsS http://127.0.0.1:${APP_PORT}/ | grep -q 'WhatsApp Webhook Server is running'; do
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
log "  Service status:     ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'supervisorctl status sl-webhook sl-cloudflared'"
log "  Watch logs:         ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'tail -f ${REMOTE_DIR}/run_logs/webhook.log'"
log "  Watch important:    ssh -i ${SSH_KEY} -p ${SSH_PORT} ${REMOTE} 'tail -f ${REMOTE_DIR}/run_logs/important.log'"
