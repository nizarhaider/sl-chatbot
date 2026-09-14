#!/usr/bin/env bash
set -euo pipefail
cd /workspace/voice-agent
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:${PATH}"
uv sync --no-dev
if ! command -v cloudflared >/dev/null; then
  curl -LsSf -o /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /usr/local/bin/cloudflared
fi
exec .venv/bin/python -m app.portal_runtime
