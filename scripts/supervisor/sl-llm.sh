#!/usr/bin/env bash
set -euo pipefail

cd /workspace/sl-chatbot
mkdir -p run_logs
exec > >(tee -a run_logs/llm.log) 2>&1
exec /root/.local/bin/llama serve \
  --model /workspace/models/gemma-4-E4B-it-qat-q4_0/gemma-4-E4B_q4_0-it.gguf \
  --alias google/gemma-4-E4B-it-qat-q4_0-gguf \
  --n-gpu-layers 99 --ctx-size 4096 --flash-attn on --jinja \
  --host 127.0.0.1 --port 8000
