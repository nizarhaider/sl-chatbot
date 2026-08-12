# syntax=docker/dockerfile:1.7

ARG TARGETPLATFORM=linux/amd64
FROM --platform=$TARGETPLATFORM vastai/pytorch@sha256:df0b400305def92eae2de53a2653138f1dbc880435cd7c5684cc593d4c585f12

ENV HF_HOME=/opt/serendibai/hf-cache \
    UV_PROJECT_ENVIRONMENT=/opt/serendibai/venv

RUN apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        -o Dpkg::Options::="--force-confold" --no-install-recommends curl supervisor \
    && curl -fsSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz \
        | tar -xz -C /usr/local/bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/serendibai/build
COPY pyproject.toml uv.lock ./
RUN uv sync --extra server --frozen --no-install-project

COPY app/__init__.py app/config.py ./app/
RUN --mount=type=secret,id=hf_token,required=true \
    HF_TOKEN="$(cat /run/secrets/hf_token)" /opt/serendibai/venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download

from app.config import (
    ASR_MODEL,
    ASR_REVISION,
    LLM_FILENAME,
    LLM_REPO,
    LLM_REVISION,
    TTS_DATASET,
    TTS_DATASET_REVISION,
    TTS_MODEL,
    TTS_REFERENCE_FILE,
    TTS_REVISION,
)

snapshot_download(repo_id=ASR_MODEL, revision=ASR_REVISION)
hf_hub_download(repo_id=LLM_REPO, filename=LLM_FILENAME, revision=LLM_REVISION)
snapshot_download(repo_id=TTS_MODEL, revision=TTS_REVISION)
hf_hub_download(
    repo_id=TTS_DATASET,
    repo_type="dataset",
    filename=TTS_REFERENCE_FILE,
    revision=TTS_DATASET_REVISION,
)
PY

RUN touch /opt/serendibai/runtime-ready
