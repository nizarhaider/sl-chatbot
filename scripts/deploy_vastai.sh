#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DISK_GB="${DISK_GB:-80}"
MIN_GPU_RAM_GB="${MIN_GPU_RAM_GB:-32}"
MIN_CUDA_VERSION="${MIN_CUDA_VERSION:-13.0}"
REMOTE_BRANCH="${REMOTE_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/vastai_ssh_file}"
TEMPLATE_HASH="${TEMPLATE_HASH:-247f2f26d31d533719c1fc4c9b5cbf93}"
INSTANCE_LABEL="${INSTANCE_LABEL:-serendibai-whatsapp}"
DRY_RUN="${DRY_RUN:-false}"
STARTUP_TIMEOUT_ATTEMPTS="${STARTUP_TIMEOUT_ATTEMPTS:-60}"
SETUP_TIMEOUT_SECONDS="${SETUP_TIMEOUT_SECONDS:-1500}"
MAX_INSTANCE_ATTEMPTS="${MAX_INSTANCE_ATTEMPTS:-3}"
# Once SSH is available, let the host complete its build and model prewarm.

log() { printf '▶ %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

command -v uvx >/dev/null 2>&1 || fail "uvx is required: https://docs.astral.sh/uv/"
test -f .env || fail "${ROOT_DIR}/.env is required"
test -f "${SSH_KEY}" || fail "SSH key not found: ${SSH_KEY}"
[[ "${MIN_GPU_RAM_GB}" =~ ^[0-9]+$ ]] || fail "MIN_GPU_RAM_GB must be numeric"
[ "${MIN_GPU_RAM_GB}" -ge 24 ] || fail "Voice runtime deployments require at least 24 GB VRAM"

PYTHON="${ROOT_DIR}/.venv/bin/python"
test -x "${PYTHON}" || fail "Run 'uv sync' locally once before deploying"

VASTAI_API_KEY="$(${PYTHON} -c \
  'from dotenv import dotenv_values; print(dotenv_values(".env").get("VASTAI_API_KEY", ""))')"
test -n "${VASTAI_API_KEY}" || fail "VASTAI_API_KEY is missing from .env"

VASTAI=(uvx --from vastai vastai --api-key "${VASTAI_API_KEY}" --raw)
QUERY="num_gpus=1 gpu_ram>=${MIN_GPU_RAM_GB} cpu_arch=amd64 disk_space>=${DISK_GB} cuda_vers>=${MIN_CUDA_VERSION} direct_port_count>=1"

log "Finding the cheapest verified on-demand GPU with at least ${MIN_GPU_RAM_GB} GB VRAM..."
ATTEMPTED_OFFER_IDS=""

select_offer() {
  "${VASTAI[@]}" search offers "${QUERY}" \
    --storage "${DISK_GB}" --order dph --limit 200 \
  | EXCLUDED_OFFER_IDS="${ATTEMPTED_OFFER_IDS}" "${PYTHON}" -c '
import json
import os
import re
import sys

offers = json.load(sys.stdin)
allowed = re.compile(r"^RTX (?:40|50)\d{2}(?:S| Super| Ti)?$", re.IGNORECASE)
excluded = {value for value in os.environ.get("EXCLUDED_OFFER_IDS", "").split(",") if value}
eligible = [
    offer for offer in offers
    if str(offer.get("id", "")) not in excluded
    and allowed.search(str(offer.get("gpu_name", "")))
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
    CREATE_RESULT="$(${VASTAI[@]} create instance "${OFFER_ID}" \
      --template_hash "${TEMPLATE_HASH}" \
      --disk "${DISK_GB}" \
      --label "${INSTANCE_LABEL}" \
      --ssh --direct --cancel-unavail)"
    INSTANCE_ID="$(printf '%s' "${CREATE_RESULT}" | "${PYTHON}" -c \
      'import json,sys; print(json.load(sys.stdin).get("new_contract", ""))')"
    test -n "${INSTANCE_ID}" || fail "Vast.ai did not return a new instance ID"
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

  log "Deploying branch ${REMOTE_BRANCH} to instance ${INSTANCE_ID} (max $((SETUP_TIMEOUT_SECONDS / 60)) minutes after SSH is ready)..."
  env \
    REMOTE_BRANCH="${REMOTE_BRANCH}" SSH_KEY="${SSH_KEY}" \
    "${ROOT_DIR}/scripts/setup_vastai.sh" "${SSH_PORT}" "${SSH_HOST}" &
  SETUP_PID=$!
  SETUP_STATUS=1
  for elapsed in $(seq 0 5 "${SETUP_TIMEOUT_SECONDS}"); do
    if ! kill -0 "${SETUP_PID}" 2>/dev/null; then
      wait "${SETUP_PID}" && SETUP_STATUS=0 || SETUP_STATUS=$?
      break
    fi
    sleep 5
  done

  if kill -0 "${SETUP_PID}" 2>/dev/null; then
    log "Setup exceeded the ${SETUP_TIMEOUT_SECONDS}-second startup budget."
    kill "${SETUP_PID}" 2>/dev/null || true
    wait "${SETUP_PID}" 2>/dev/null || true
  elif [ "${SETUP_STATUS}" -eq 0 ]; then
    log "Deployment complete."
    log "Instance ID: ${INSTANCE_ID}"
    log "Destroy instance ${INSTANCE_ID} in Vast.ai as soon as the call is finished."
    exit 0
  fi

  log "Setup failed or exceeded the startup budget; terminating instance ${INSTANCE_ID}."
  destroy_instance "${INSTANCE_ID}"
  if [ "${instance_attempt}" -lt "${MAX_INSTANCE_ATTEMPTS}" ]; then
    log "Trying a different offer..."
    continue
  fi
done

fail "All ${MAX_INSTANCE_ATTEMPTS} instance attempts failed; no server was left running."
