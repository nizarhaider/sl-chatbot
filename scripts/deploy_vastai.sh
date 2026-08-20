#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DISK_GB="${DISK_GB:-80}"
MIN_GPU_RAM_GB="${MIN_GPU_RAM_GB:-32}"
REMOTE_BRANCH="${REMOTE_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/vastai_ssh_file}"
TEMPLATE_HASH="${TEMPLATE_HASH:-18e97fc6703dea11057cee364a8eaa8c}"
INSTANCE_LABEL="${INSTANCE_LABEL:-serendibai-whatsapp}"
DRY_RUN="${DRY_RUN:-false}"
STARTUP_TIMEOUT_ATTEMPTS="${STARTUP_TIMEOUT_ATTEMPTS:-60}"
MAX_INSTANCE_ATTEMPTS="${MAX_INSTANCE_ATTEMPTS:-3}"
SETUP_TIMEOUT_SECONDS="${SETUP_TIMEOUT_SECONDS:-1200}"

log() { printf '▶ %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

command -v uvx >/dev/null 2>&1 || fail "uvx is required: https://docs.astral.sh/uv/"
if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_BIN="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_BIN="gtimeout"
else
  fail "GNU timeout is required (install coreutils: brew install coreutils)"
fi
test -f .env || fail "${ROOT_DIR}/.env is required"
test -f "${SSH_KEY}" || fail "SSH key not found: ${SSH_KEY}"
[[ "${MIN_GPU_RAM_GB}" =~ ^[0-9]+$ ]] || fail "MIN_GPU_RAM_GB must be numeric"
[ "${MIN_GPU_RAM_GB}" -ge 32 ] || fail "Gemma 4 26B deployments require at least 32 GB VRAM"

PYTHON="${ROOT_DIR}/.venv/bin/python"
test -x "${PYTHON}" || fail "Run 'uv sync' locally once before deploying"

VASTAI_API_KEY="$(${PYTHON} -c \
  'from dotenv import dotenv_values; print(dotenv_values(".env").get("VASTAI_API_KEY", ""))')"
test -n "${VASTAI_API_KEY}" || fail "VASTAI_API_KEY is missing from .env"

VASTAI=(uvx --from vastai vastai --api-key "${VASTAI_API_KEY}" --raw)
QUERY="num_gpus=1 gpu_ram>=${MIN_GPU_RAM_GB} cpu_arch=amd64 disk_space>=${DISK_GB} cuda_vers>=12.8 direct_port_count>=1"

log "Finding the cheapest verified on-demand GPU with at least ${MIN_GPU_RAM_GB} GB VRAM..."
OFFER="$("${VASTAI[@]}" search offers "${QUERY}" \
  --storage "${DISK_GB}" --order dph --limit 200 \
  | "${PYTHON}" -c '
import json
import re
import sys

offers = json.load(sys.stdin)
allowed = re.compile(r"^RTX (?:30|40|50)\d{2}(?:S| Super| Ti)?$", re.IGNORECASE)
eligible = [offer for offer in offers if allowed.search(str(offer.get("gpu_name", "")))]
if not eligible:
    raise SystemExit("No eligible Vast.ai offer is currently available")
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
')"

IFS=$'\t' read -r OFFER_ID GPU_NAME GPU_RAM HOURLY_PRICE LOCATION RELIABILITY <<<"${OFFER}"
log "Selected offer ${OFFER_ID}: ${GPU_NAME}, ${GPU_RAM} MiB VRAM, \$${HOURLY_PRICE}/hour including ${DISK_GB} GB storage, ${LOCATION}, reliability ${RELIABILITY}"

if [ "${DRY_RUN}" = "true" ]; then
  log "Dry run complete; no instance was created."
  exit 0
fi

EXISTING_CONNECTION="$(${VASTAI[@]} show instances | "${PYTHON}" -c '
import json
import sys

rows = json.load(sys.stdin)
if isinstance(rows, dict):
    rows = rows.get("instances", [])
rows = [row for row in rows if row.get("label") == "'"${INSTANCE_LABEL}"'" and row.get("actual_status") == "running"]
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

  log "Deploying branch ${REMOTE_BRANCH} to instance ${INSTANCE_ID} (hard limit ${SETUP_TIMEOUT_SECONDS}s)..."
  if "${TIMEOUT_BIN}" --foreground "${SETUP_TIMEOUT_SECONDS}" env \
    REMOTE_BRANCH="${REMOTE_BRANCH}" SSH_KEY="${SSH_KEY}" \
    "${ROOT_DIR}/scripts/setup_vastai.sh" "${SSH_PORT}" "${SSH_HOST}"; then
    log "Deployment complete."
    log "Instance ID: ${INSTANCE_ID}"
    log "Destroy instance ${INSTANCE_ID} in Vast.ai as soon as the call is finished."
    exit 0
  fi

  log "Setup failed or exceeded ${SETUP_TIMEOUT_SECONDS}s on instance ${INSTANCE_ID}."
  destroy_instance "${INSTANCE_ID}"
done

fail "All ${MAX_INSTANCE_ATTEMPTS} instance attempts failed; no server was left running."
