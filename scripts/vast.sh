#!/usr/bin/env bash
# Rent, configure, inspect, or destroy the WhatsApp GPU server.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/vastai_ssh_file}"
DISK_GB="${DISK_GB:-40}"
APP_PORT="${APP_PORT:-8081}"
TEMPLATE_HASH="${TEMPLATE_HASH:-18e97fc6703dea11057cee364a8eaa8c}"

log() { printf '▶ %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

load_vast() {
  test -x "$ROOT/.venv/bin/python" || fail "Run: uv sync"
  test -f "$ROOT/.env" || fail "$ROOT/.env is missing"
  VASTAI_API_KEY="$($ROOT/.venv/bin/python -c \
    'from dotenv import dotenv_values; print(dotenv_values(".env").get("VASTAI_API_KEY", ""))')"
  test -n "$VASTAI_API_KEY" || fail "VASTAI_API_KEY is missing from .env"
  VAST=(uvx --from vastai vastai --api-key "$VASTAI_API_KEY" --raw)
}

rent() {
  load_vast
  local query offer result instance_id host port state
  query="num_gpus=1 gpu_ram>=16 cpu_arch=amd64 disk_space>=${DISK_GB} cuda_vers>=12.8 direct_port_count>=1 reliability>=0.98"
  log "Selecting the cheapest verified RTX 4070/30-series offer..."
  offer="$("${VAST[@]}" search offers "$query" --storage "$DISK_GB" --order dph --limit 200 |
    "$ROOT/.venv/bin/python" -c '
import json,re,sys
rows=json.load(sys.stdin)
allowed=re.compile(r"^RTX (?:4070|30[0-9]{2})", re.I)
rows=[row for row in rows if allowed.search(str(row.get("gpu_name", "")))]
if not rows: raise SystemExit("No eligible offer is available")
row=min(rows, key=lambda item: float(item["dph_total"]))
print("\t".join(map(str, (row["id"], row["gpu_name"], row["dph_total"]))))
')"
  IFS=$'\t' read -r offer_id gpu price <<<"$offer"
  log "Offer $offer_id: $gpu at \$$price/hour"
  if [ "${DRY_RUN:-false}" = true ]; then return; fi

  result="$("${VAST[@]}" create instance "$offer_id" --template_hash "$TEMPLATE_HASH" \
    --disk "$DISK_GB" --label serendibai-whatsapp --ssh --direct --cancel-unavail)"
  instance_id="$(printf '%s' "$result" | "$ROOT/.venv/bin/python" -c \
    'import json,sys; print(json.load(sys.stdin).get("new_contract", ""))')"
  test -n "$instance_id" || fail "Vast.ai did not return an instance ID"
  log "Created instance $instance_id; it is billable until destroyed"

  for _ in $(seq 1 180); do
    state="$("${VAST[@]}" show instance "$instance_id")"
    read -r host port <<<"$(printf '%s' "$state" | "$ROOT/.venv/bin/python" -c '
import json,sys
x=json.load(sys.stdin); x=x[0] if isinstance(x,list) else x
p=((x.get("ports") or {}).get("22/tcp") or [{}])[0].get("HostPort", "")
print(x.get("public_ipaddr", ""), p)
')"
    if [ -n "$host" ] && [ -n "$port" ] && ssh -o BatchMode=yes -o ConnectTimeout=8 \
      -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" -p "$port" "root@$host" true 2>/dev/null; then
      setup "$host" "$port"
      log "Instance ID: $instance_id"
      log "Destroy later with: $0 destroy $instance_id"
      return
    fi
    sleep 5
  done
  fail "Instance $instance_id was not SSH-ready; inspect or destroy it manually"
}

setup() {
  local host="${1:?Usage: $0 setup HOST SSH_PORT}" port="${2:?Usage: $0 setup HOST SSH_PORT}"
  local remote="root@$host" ssh_transport public_url tunnel_command
  local -a ssh_args
  ssh_transport="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i $SSH_KEY -p $port"
  ssh_args=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" -p "$port")
  test -f "$ROOT/.env" || fail "$ROOT/.env is missing"
  test -f "$SSH_KEY" || fail "SSH key not found: $SSH_KEY"
  [[ "$APP_PORT" =~ ^[0-9]+$ ]] || fail "APP_PORT must be numeric"

  log "Syncing the lean runtime to $host..."
  ssh "${ssh_args[@]}" "$remote" 'mkdir -p /workspace/sl-chatbot/run_logs'
  rsync -a --delete -e "$ssh_transport" \
    --exclude .git --exclude .venv --exclude .env --exclude run_logs \
    "$ROOT/" "$remote:/workspace/sl-chatbot/"
  scp -q -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" -P "$port" \
    "$ROOT/.env" "$remote:/workspace/sl-chatbot/.env"

  log "Installing the server profile and ngrok..."
  ssh "${ssh_args[@]}" "$remote" "APP_PORT=$APP_PORT bash -s" <<'REMOTE'
set -euo pipefail
apt-get update -qq
apt-get install -y -qq curl gnupg tmux
if ! command -v ngrok >/dev/null; then
  curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
  echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" > /etc/apt/sources.list.d/ngrok.list
  apt-get update -qq
  apt-get install -y -qq ngrok
fi
cd /workspace/sl-chatbot
uv sync --extra server
.venv/bin/python -m compileall -q app scripts/infer.py
set -a
. ./.env
set +a
test -n "${NGROK_AUTH_TOKEN:-}"
ngrok config add-authtoken "$NGROK_AUTH_TOKEN" >/dev/null
tmux kill-session -t sl-webhook 2>/dev/null || true
tmux kill-session -t sl-ngrok 2>/dev/null || true
tmux new-session -d -s sl-webhook "cd /workspace/sl-chatbot && .venv/bin/dotenv -f .env run -- .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port $APP_PORT > run_logs/server.log 2>&1"
tmux new-session -d -s sl-ngrok "ngrok http $APP_PORT --log=stdout > /workspace/sl-chatbot/run_logs/ngrok.log 2>&1"
REMOTE

  log "Waiting for model prewarm and health check..."
  for attempt in $(seq 1 300); do
    if ssh "${ssh_args[@]}" "$remote" "curl -fsS http://127.0.0.1:$APP_PORT/ >/dev/null" 2>/dev/null; then break; fi
    if [ "$attempt" -eq 300 ]; then
      ssh "${ssh_args[@]}" "$remote" 'tail -n 120 /workspace/sl-chatbot/run_logs/server.log' || true
      fail "Server did not become healthy"
    fi
    sleep 2
  done
  tunnel_command='python3 -c "import json,urllib.request; print(json.load(urllib.request.urlopen(\"http://127.0.0.1:4040/api/tunnels\"))[\"tunnels\"][0][\"public_url\"])"'
  public_url="$(ssh "${ssh_args[@]}" "$remote" "$tunnel_command")"
  log "Webhook URL: $public_url/webhook"
  log "Logs: ssh -i $SSH_KEY -p $port $remote 'tail -f /workspace/sl-chatbot/run_logs/server.log'"
}

destroy() {
  local instance_id="${1:?Usage: $0 destroy INSTANCE_ID}"
  load_vast
  read -r -p "Destroy Vast.ai instance $instance_id? [y/N] " answer
  [ "$answer" = y ] || [ "$answer" = Y ] || exit 0
  printf 'y\n' | "${VAST[@]}" destroy instance "$instance_id"
}

list_instances() {
  load_vast
  "${VAST[@]}" show instances
}

case "${1:-}" in
  rent) rent ;;
  setup) setup "${2:-}" "${3:-}" ;;
  destroy) destroy "${2:-}" ;;
  list) list_instances ;;
  *) echo "Usage: $0 {rent|setup HOST SSH_PORT|list|destroy INSTANCE_ID}"; exit 2 ;;
esac
