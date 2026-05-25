# Agent Handoff

This repo is deployed on a Vast.ai instance and receives WhatsApp webhook/call events through Cloudflare Tunnel.

## Remote Access

```bash
ssh -i ~/.ssh/vastai_ssh_file -p 31250 root@1.208.108.242
cd /workspace/sl-chatbot
```

Public webhook:

```text
https://webhook.hervestudio.lk/webhook
```

Verify token currently used in tests:

```text
my_secure_verify_token_123
```

## Current Runtime Shape

The voice call stack is intentionally single-path:

```text
WhatsApp Cloud webhook -> FastAPI webhook
WhatsApp WebRTC audio -> Gemini STT -> Gemini LLM -> RealtimeTTS OmniVoice -> WhatsApp WebRTC output track
```

RealtimeSTT, Gemini Live audio, Gemma audio, Edge TTS, and the old generic TTS service have been removed from the repo.

## Important Runtime Env

```bash
GEMINI_STT_MODEL=gemini-2.5-flash-lite
GEMINI_LLM_MODEL=gemini-2.5-flash-lite
REALTIME_TTS_DEVICE=cuda:0
REALTIME_TTS_DTYPE=float16
REALTIME_TTS_NUM_STEPS=12,12
REALTIME_TTS_REF_AUDIO=app/voices/sample_si_lk.mp3
REALTIME_TTS_REF_LANGUAGE=si
TURN_GREETING_PROTECTION_MAX_SECONDS=1.5
IMPORTANT_LOG_PATH=run_logs/important.log
```

## Important Logs

Use one clean log for call debugging:

```bash
tail -f run_logs/important.log
```

This log is intentionally filtered. It should include call lifecycle, WebRTC state, STT transcript, Gemini response, TTS completion, interruption, and warnings/errors.

## Service Commands

Check live processes:

```bash
ps -p "$(cat run_logs/webhook.pid)" -o pid,stat,etime,%cpu,%mem,cmd
ps -p "$(cat run_logs/tunnel.pid)" -o pid,stat,etime,%cpu,%mem,cmd
ss -ltnp | egrep "(8090)" || true
```

Restart webhook:

```bash
cd /workspace/sl-chatbot
kill "$(cat run_logs/webhook.pid)" 2>/dev/null || true
nohup .venv/bin/dotenv -f .env run -- env \
  GEMINI_STT_MODEL=gemini-2.5-flash-lite \
  GEMINI_LLM_MODEL=gemini-2.5-flash-lite \
  REALTIME_TTS_DEVICE=cuda:0 \
  REALTIME_TTS_DTYPE=float16 \
  REALTIME_TTS_NUM_STEPS=12,12 \
  TURN_GREETING_PROTECTION_MAX_SECONDS=1.5 \
  IMPORTANT_LOG_PATH=run_logs/important.log \
  .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8090 --env-file .env \
  > run_logs/webhook.log 2>&1 & echo $! > run_logs/webhook.pid
```

Cloudflare named tunnel is already running from `run_logs/cloudflared.token`:

```bash
tail -f run_logs/tunnel.log
```

## Files That Must Stay In Sync

Important source files currently synced between local and remote:

```text
app/main.py
app/webhooks/whatsapp.py
app/services/webrtc.py
app/services/whatsapp_api.py
app/voice_agent/agent.py
app/voice_agent/gemini_turn_pipeline.py
scripts/run_local_host.sh
pyproject.toml
uv.lock
README.md
AGENTS.md
```

Do not sync runtime-only files or bulky local assets unless explicitly requested:

```text
.env
.venv/
run_logs/
*.log
*.pid
training_custom_tts/datasets/
tts_debug_latest/
debug_*.wav
official_*.wav
realtimetts_*_smoke.wav
app/voices/ unless intentionally changing production reference voices
```

