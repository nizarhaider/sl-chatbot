#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${1:-}" != "--env" || $# -ne 2 ]]; then
  echo "Usage: ./deploy.sh --env local|remote" >&2
  exit 2
fi

case "$2" in
  local)
    command -v uv >/dev/null || { echo "Install uv first." >&2; exit 1; }
    command -v cloudflared >/dev/null || { echo "Install cloudflared first." >&2; exit 1; }
    command -v ffmpeg >/dev/null || { echo "Install ffmpeg first." >&2; exit 1; }
    [[ -f .env ]] || { echo "voice-agent/.env is missing." >&2; exit 1; }
    uv sync --no-dev
    exec uv run --no-sync python - <<'PY'
import hashlib
import os
import runpy
import secrets

import psycopg
from dotenv import load_dotenv

load_dotenv(".env")
with psycopg.connect(os.environ["DATABASE_URL"]) as db:
    agent = db.execute(
        "select id from portal_agents where phone_number_id=%s and status<>'archived' order by updated_at desc limit 1",
        (os.environ["PHONE_NUMBER_ID"],),
    ).fetchone()
    if agent is None:
        raise SystemExit("No portal agent matches PHONE_NUMBER_ID.")
    token = secrets.token_urlsafe(32)
    db.execute(
        "update portal_agents set runtime_token_hash=%s, status='warming_up', heartbeat_at=null, telemetry='{}' where id=%s",
        (hashlib.sha256(token.encode()).hexdigest(), agent[0]),
    )

os.environ["PORTAL_URL"] = os.environ.get("PORTAL_URL", "https://portal.serendibai.lk")
os.environ["PORTAL_RUNTIME_TOKEN"] = token
print("Starting local voice server for portal agent", agent[0], flush=True)
runpy.run_module("app.portal_runtime", run_name="__main__")
PY
    ;;
  remote)
    echo "Remote deployment is not configured yet." >&2
    exit 1
    ;;
  *)
    echo "Usage: ./deploy.sh --env local|remote" >&2
    exit 2
    ;;
esac
