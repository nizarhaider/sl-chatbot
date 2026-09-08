#!/usr/bin/env bash
# Rent and configure a Gemini Live-only Vast instance. With SSH_PORT and HOST_IP
# arguments it configures that existing host; without arguments it rents one.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DISK_GB="${DISK_GB:-20}"
MIN_GPU_RAM_GB="${MIN_GPU_RAM_GB:-8}"
MIN_CPU_CORES="${MIN_CPU_CORES:-2}"
MIN_INTERNET_DOWN_MBIT="${MIN_INTERNET_DOWN_MBIT:-200}"
REMOTE_BRANCH="${REMOTE_BRANCH:-$(git branch --show-current)}"
INSTANCE_LABEL="${INSTANCE_LABEL:-serendibai-gemini-live}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/vastai_ssh_file}"
TEMPLATE_HASH="${TEMPLATE_HASH:-247f2f26d31d533719c1fc4c9b5cbf93}"

log() { printf '▶ %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

test -f .env || fail ".env is required"
test -x .venv/bin/python || fail "Run 'uv sync' locally before deploying"
test -f "${SSH_KEY}" || fail "SSH key not found: ${SSH_KEY}"
command -v uvx >/dev/null || fail "uvx is required"

PYTHON="${ROOT_DIR}/.venv/bin/python"
VASTAI_API_KEY="$(${PYTHON} -c 'from dotenv import dotenv_values; print(dotenv_values(".env").get("VASTAI_API_KEY", ""))')"
test -n "${VASTAI_API_KEY}" || fail "VASTAI_API_KEY is missing from .env"
VASTAI=(uvx --from vastai vastai --api-key "${VASTAI_API_KEY}" --raw)

configure_host() {
  local ssh_port="$1" host_ip="$2" env_file
  env_file="$(mktemp)"
  cp .env "${env_file}"
  local pinecone_key
  pinecone_key="$(zsh -lic 'printf %s "${PINECONE_API_KEY:-}"' 2>/dev/null || true)"
  if [ -n "${pinecone_key}" ]; then
    PINECONE_VALUE="${pinecone_key}" "${PYTHON}" - "${env_file}" <<'PY'
from os import environ
from pathlib import Path

path = Path(__import__('sys').argv[1])
value = environ['PINECONE_VALUE']
lines = [line for line in path.read_text().splitlines() if not line.startswith('PINECONE_API_KEY=')]
lines.append(f'PINECONE_API_KEY={value}')
path.write_text('\n'.join(lines) + '\n')
PY
  fi
  trap 'rm -f "${env_file}"' RETURN

  log "Uploading the Gemini Live runtime to ${host_ip}:${ssh_port}"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "${SSH_KEY}" -p "${ssh_port}" "root@${host_ip}" \
    "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git curl ffmpeg"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "${SSH_KEY}" -p "${ssh_port}" "root@${host_ip}" \
    "if [ -d /workspace/sl_chatbot/.git ]; then cd /workspace/sl_chatbot && git fetch origin && git checkout '${REMOTE_BRANCH}' && git reset --hard 'origin/${REMOTE_BRANCH}'; else git clone --branch '${REMOTE_BRANCH}' '$(git remote get-url origin)' /workspace/sl_chatbot; fi"
  scp -q -o StrictHostKeyChecking=accept-new -i "${SSH_KEY}" -P "${ssh_port}" "${env_file}" "root@${host_ip}:/workspace/sl_chatbot/.env"

  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "${SSH_KEY}" -p "${ssh_port}" "root@${host_ip}" 'bash -s' <<'REMOTE'
set -euo pipefail
cd /workspace/sl_chatbot
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:${PATH}"
uv sync --no-dev
install -d /var/log/serendibai
cat >/usr/local/bin/serendibai-webhook <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /workspace/sl_chatbot
exec /workspace/sl_chatbot/.venv/bin/uvicorn app.api.app:create_app --host 0.0.0.0 --port 8081
EOF
chmod +x /usr/local/bin/serendibai-webhook
cat >/usr/local/bin/serendibai-cloudflared <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /workspace/sl_chatbot
TOKEN="$(/workspace/sl_chatbot/.venv/bin/python -c 'from dotenv import dotenv_values; print(dotenv_values(".env").get("CLOUDFLARED_TUNNEL_TOKEN", ""))')"
test -n "$TOKEN"
exec cloudflared tunnel --no-autoupdate run --token "$TOKEN"
EOF
chmod +x /usr/local/bin/serendibai-cloudflared
cat >/etc/supervisor/conf.d/serendibai.conf <<'EOF'
[program:serendibai-webhook]
command=/usr/local/bin/serendibai-webhook
directory=/workspace/sl_chatbot
autostart=true
autorestart=true
stderr_logfile=/var/log/serendibai/webhook.err.log
stdout_logfile=/var/log/serendibai/webhook.out.log

[program:serendibai-cloudflared]
command=/usr/local/bin/serendibai-cloudflared
directory=/workspace/sl_chatbot
autostart=true
autorestart=true
stderr_logfile=/var/log/serendibai/cloudflared.err.log
stdout_logfile=/var/log/serendibai/cloudflared.out.log
EOF
supervisorctl reread
supervisorctl update
supervisorctl restart serendibai-webhook serendibai-cloudflared
REMOTE

  log "Waiting for Gemini Live prewarm"
  for _ in $(seq 1 24); do
    status="$(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "${SSH_KEY}" -p "${ssh_port}" "root@${host_ip}" "curl -fsS http://127.0.0.1:8081/" || true)"
    if [[ "${status}" == *'"status":"ready"'* ]]; then
      log "Gemini Live runtime is ready"
      return
    fi
    sleep 5
  done
  fail "Gemini Live did not become ready; inspect /var/log/serendibai/webhook.err.log on the instance"
}

if [ "$#" -eq 2 ]; then
  configure_host "$1" "$2"
  exit 0
fi
[ "$#" -eq 0 ] || fail "Usage: $0 [SSH_PORT HOST_IP]"

QUERY="num_gpus=1 gpu_ram>=${MIN_GPU_RAM_GB} cpu_cores_effective>=${MIN_CPU_CORES} cpu_arch=amd64 disk_space>=${DISK_GB} direct_port_count>=1"
log "Finding a verified Vast offer for the Gemini Live bridge"
OFFER="$(${VASTAI[@]} search offers "${QUERY}" --storage "${DISK_GB}" --order dph --limit 200 | MIN_DOWN="${MIN_INTERNET_DOWN_MBIT}" "${PYTHON}" -c '
import json, os, sys
offers = [o for o in json.load(sys.stdin) if float(o.get("inet_down") or 0) >= float(os.environ["MIN_DOWN"])]
if not offers: raise SystemExit("No eligible Vast offer is currently available")
offer = min(offers, key=lambda o: float(o.get("dph_total") or "inf"))
print(offer["id"])
')"
log "Creating Vast instance from offer ${OFFER}"
CREATED="$(${VASTAI[@]} create instance "${OFFER}" --template_hash "${TEMPLATE_HASH}" --disk "${DISK_GB}" --label "${INSTANCE_LABEL}" --ssh --direct --cancel-unavail)"
INSTANCE_ID="$(printf '%s' "${CREATED}" | "${PYTHON}" -c 'import json,sys; print(json.load(sys.stdin).get("new_contract", ""))')"
test -n "${INSTANCE_ID}" || fail "Vast did not return an instance ID"

for _ in $(seq 1 30); do
  connection="$(${VASTAI[@]} show instance "${INSTANCE_ID}" | "${PYTHON}" -c '
import json,sys
x=json.load(sys.stdin); x=x[0] if isinstance(x,list) else x
m=(x.get("ports") or {}).get("22/tcp") or []
print("\t".join([str(x.get("public_ipaddr") or ""), str(m[0].get("HostPort") if m else "")]))
')"
  IFS=$'\t' read -r host_ip ssh_port <<<"${connection}"
  if [ -n "${host_ip}" ] && [ -n "${ssh_port}" ] && ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -i "${SSH_KEY}" -p "${ssh_port}" "root@${host_ip}" true 2>/dev/null; then
    configure_host "${ssh_port}" "${host_ip}"
    log "Instance ${INSTANCE_ID} is ready"
    exit 0
  fi
  sleep 5
done
fail "Instance ${INSTANCE_ID} did not become SSH-ready"
