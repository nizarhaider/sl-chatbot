#!/usr/bin/env bash
set -euo pipefail

cd /workspace/sl-chatbot
mkdir -p run_logs
exec > >(tee -a run_logs/cloudflared.log) 2>&1
set -a
. ./.env
set +a
TUNNEL_TOKEN="${CLOUDFLARED_TUNNEL_TOKEN:?missing tunnel token}" \
  exec /opt/instance-tools/bin/cloudflared tunnel run
