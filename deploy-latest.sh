#!/usr/bin/env bash
set -euo pipefail

source_dir=/opt/serendibai/voice-agent-source
target_dir=/opt/serendibai/voice-agent
revision_file=/opt/serendibai/.voice-agent-revision

if [[ ! -d "$source_dir/.git" ]]; then
  git clone --quiet --depth 1 https://github.com/nizarhaider/sl-chatbot.git "$source_dir"
fi

git -C "$source_dir" fetch --quiet --depth 1 origin main
revision=$(git -C "$source_dir" rev-parse FETCH_HEAD)
deployed_revision=$(cat "$revision_file" 2>/dev/null || true)

if [[ "$revision" == "$deployed_revision" ]]; then
  exit 0
fi

git -C "$source_dir" reset --quiet --hard "$revision"
rsync -a --delete --exclude=.env --exclude=.venv --exclude=.pytest_cache --exclude=.git "$source_dir/" "$target_dir/"
systemctl restart voice-agent.service
printf '%s\n' "$revision" > "$revision_file"
