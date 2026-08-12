#!/usr/bin/env bash
# Rent, deploy, inspect, or destroy the WhatsApp GPU server.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/vastai_ssh_file}"
DISK_GB="${DISK_GB:-40}"
APP_PORT="${APP_PORT:-8081}"
TEMPLATE_HASH="${TEMPLATE_HASH:-18e97fc6703dea11057cee364a8eaa8c}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-ghcr.io/nizarhaider/sl-chatbot-runtime:main}"
SETUP_LIMIT_SECONDS="${SETUP_LIMIT_SECONDS:-300}"
cd "$ROOT"

log() { printf '▶ %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

load_vast() {
  test -x .venv/bin/python || fail "Run: uv sync"
  test -f .env || fail "$ROOT/.env is missing"
  VASTAI_API_KEY="$(.venv/bin/python -c \
    'from dotenv import dotenv_values; print(dotenv_values(".env").get("VASTAI_API_KEY", ""))')"
  test -n "$VASTAI_API_KEY" || fail "VASTAI_API_KEY is missing from .env"
  VAST=(uvx --from vastai vastai --api-key "$VASTAI_API_KEY" --raw)
}

setup() {
  local host="${1:?Usage: $0 setup HOST SSH_PORT}" port="${2:?Usage: $0 setup HOST SSH_PORT}"
  local remote="root@$host" public_url runtime_env setup_started elapsed
  local -a connection=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=2 -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" -p "$port")
  test -f .env || fail "$ROOT/.env is missing"
  test -f "$SSH_KEY" || fail "SSH key not found: $SSH_KEY"
  setup_started=$SECONDS
  ssh "${connection[@]}" "$remote" \
    'test -f /opt/serendibai/runtime-ready && test -x /opt/serendibai/venv/bin/python' || \
    fail "The instance does not contain the prebuilt SerendibAI runtime"

  log "Deploying the committed runtime to $host"
  git archive --format=tar HEAD app pyproject.toml uv.lock README.md |
    ssh "${connection[@]}" "$remote" \
      'rm -rf /workspace/sl-chatbot && mkdir -p /workspace/sl-chatbot/run_logs && tar -xf - -C /workspace/sl-chatbot'
  runtime_env="$(mktemp)"
  trap "rm -f '$runtime_env'" EXIT
  .venv/bin/python - <<'PY' >"$runtime_env"
import shlex
from dotenv import dotenv_values

values = dotenv_values(".env")
for key in (
    "HF_TOKEN", "NGROK_AUTH_TOKEN", "VERIFY_TOKEN", "WHATSAPP_ACCESS_TOKEN",
    "WHATSAPP_TOKEN", "PHONE_NUMBER_ID", "DATABASE_URL",
):
    if value := values.get(key):
        print(f"{key}={shlex.quote(value)}")
PY
  chmod 600 "$runtime_env"
  scp -q -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" -P "$port" \
    "$runtime_env" "$remote:/workspace/sl-chatbot/.env"

  ssh "${connection[@]}" "$remote" "APP_PORT=$APP_PORT bash -s" <<'REMOTE'
set -euo pipefail
test -f /opt/serendibai/runtime-ready
test -x /opt/serendibai/venv/bin/python
test -x /usr/local/bin/ngrok
cd /workspace/sl-chatbot
/opt/serendibai/venv/bin/python -m compileall -q app
set -a; . ./.env; set +a
test -n "${NGROK_AUTH_TOKEN:-}"
ngrok config add-authtoken "$NGROK_AUTH_TOKEN" >/dev/null
cat >/etc/supervisor/conf.d/serendibai.conf <<EOF
[program:sl-webhook]
command=/opt/serendibai/venv/bin/dotenv -f .env run -- /opt/serendibai/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port $APP_PORT
directory=/workspace/sl-chatbot
environment=HF_HOME="/opt/serendibai/hf-cache"
autostart=true
autorestart=true
startsecs=15
stopsignal=TERM
stdout_logfile=/workspace/sl-chatbot/run_logs/server.log
redirect_stderr=true

[program:sl-ngrok]
command=/usr/local/bin/ngrok http $APP_PORT --log=stdout
directory=/workspace/sl-chatbot
autostart=true
autorestart=true
startsecs=5
stopsignal=TERM
stdout_logfile=/workspace/sl-chatbot/run_logs/ngrok.log
redirect_stderr=true
EOF
supervisorctl reread >/dev/null
supervisorctl update >/dev/null
supervisorctl restart sl-webhook sl-ngrok >/dev/null
REMOTE

  log "Waiting for model prewarm"
  while true; do
    if ssh "${connection[@]}" "$remote" "curl -fsS http://127.0.0.1:$APP_PORT/ >/dev/null" 2>/dev/null; then
      public_url="$(ssh "${connection[@]}" "$remote" \
        "curl -fsS http://127.0.0.1:4040/api/tunnels | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"tunnels\"][0][\"public_url\"])'")"
      elapsed=$((SECONDS - setup_started))
      test "$elapsed" -le "$SETUP_LIMIT_SECONDS" || fail "Server setup took ${elapsed}s; limit is ${SETUP_LIMIT_SECONDS}s"
      log "Server setup completed in ${elapsed}s"
      log "Webhook URL: $public_url/webhook"
      log "Logs: ssh -i $SSH_KEY -p $port $remote 'tail -f /workspace/sl-chatbot/run_logs/server.log'"
      return
    fi
    elapsed=$((SECONDS - setup_started))
    if [ "$elapsed" -ge "$SETUP_LIMIT_SECONDS" ]; then
      ssh "${connection[@]}" "$remote" 'tail -n 120 /workspace/sl-chatbot/run_logs/server.log' || true
      fail "Server did not become healthy within ${SETUP_LIMIT_SECONDS}s"
    fi
    sleep 2
  done
}

rent() {
  load_vast
  local query offer result instance_id state host port registry_token existing_ids old_id
  query="num_gpus=1 gpu_ram>=16 cpu_arch=amd64 disk_space>=${DISK_GB} cuda_vers>=12.8 direct_port_count>=1 reliability>=0.98 verified=true"
  log "Selecting the cheapest RTX 4070/30-series offer"
  offer="$("${VAST[@]}" search offers "$query" --storage "$DISK_GB" --order dph --limit 200 |
    .venv/bin/python -c '
import json,re,sys
rows=[r for r in json.load(sys.stdin) if re.search(r"^RTX (?:4070|30[0-9]{2})", str(r.get("gpu_name", "")), re.I)]
if not rows: raise SystemExit("No eligible offer is available")
r=min(rows, key=lambda x: float(x["dph_total"]))
print(r["id"], r["gpu_name"].replace(" ", "_"), r["dph_total"])
')"
  read -r offer_id gpu price <<<"$offer"
  log "Offer $offer_id: ${gpu//_/ } at \$$price/hour"
  [ "${DRY_RUN:-false}" = true ] && return

  registry_token="${GH_TOKEN:-}"
  test -n "$registry_token" || fail "GH_TOKEN is required to pull the private runtime image"
  existing_ids="$("${VAST[@]}" show instances | .venv/bin/python -c '
import json,sys
print(" ".join(str(row["id"]) for row in json.load(sys.stdin) if row.get("label") == "serendibai-whatsapp"))
')"
  result="$("${VAST[@]}" create instance "$offer_id" --template_hash "$TEMPLATE_HASH" \
    --image "$RUNTIME_IMAGE" --login "-u nizarhaider -p $registry_token ghcr.io" \
    --disk "$DISK_GB" --label serendibai-whatsapp --ssh --direct --cancel-unavail)"
  instance_id="$(printf '%s' "$result" | .venv/bin/python -c \
    'import json,sys; print(json.load(sys.stdin).get("new_contract", ""))')"
  test -n "$instance_id" || fail "Vast.ai did not return an instance ID"
  log "Created billable instance $instance_id"
  for _ in $(seq 1 180); do
    state="$("${VAST[@]}" show instance "$instance_id")"
    read -r host port <<<"$(printf '%s' "$state" | .venv/bin/python -c '
import json,sys
x=json.load(sys.stdin); x=x[0] if isinstance(x,list) else x
print(x.get("public_ipaddr", ""), ((x.get("ports") or {}).get("22/tcp") or [{}])[0].get("HostPort", ""))
')"
    if [ -n "$host" ] && [ -n "$port" ] && ssh -o BatchMode=yes -o ConnectTimeout=8 \
      -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" -p "$port" "root@$host" true 2>/dev/null; then
      if ! (setup "$host" "$port"); then
        "${VAST[@]}" destroy instance "$instance_id" -y || true
        fail "Fresh instance $instance_id failed setup and was destroyed"
      fi
      for old_id in $existing_ids; do
        log "Destroying replaced production instance $old_id"
        "${VAST[@]}" destroy instance "$old_id" -y
      done
      log "Instance ID: $instance_id"
      log "Destroy later with: $0 destroy $instance_id"
      return
    fi
    sleep 5
  done
  "${VAST[@]}" destroy instance "$instance_id" -y || true
  fail "Instance $instance_id was not SSH-ready; inspect or destroy it"
}

destroy() {
  local id="${1:?Usage: $0 destroy INSTANCE_ID}" answer
  load_vast
  read -r -p "Destroy Vast.ai instance $id? [y/N] " answer
  [[ "$answer" = [yY] ]] && "${VAST[@]}" destroy instance "$id" -y
}

case "${1:-}" in
  rent) rent ;;
  setup) setup "${2:-}" "${3:-}" ;;
  list) load_vast; "${VAST[@]}" show instances ;;
  destroy) destroy "${2:-}" ;;
  *) echo "Usage: $0 {rent|setup HOST SSH_PORT|list|destroy INSTANCE_ID}"; exit 2 ;;
esac
