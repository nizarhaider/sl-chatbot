# Agent Handoff

This repo runs a WhatsApp voice bot on a Vast.ai GPU machine.

The current voice path is:

```text
WhatsApp Cloud webhook -> FastAPI /webhook
WhatsApp Calling SDP offer -> aiortc peer connection
Inbound WhatsApp audio -> local VAD -> local Whisper STT
Whisper transcript -> local Qwen 4B Q4 via llama.cpp
Qwen text -> RealtimeTTS OmniVoice
OmniVoice PCM -> outbound aiortc audio track -> WhatsApp call
```

The assistant brain is local Qwen. Do not add hosted LLM API calls to the voice path. Incoming WhatsApp text messages are ignored; this repository is voice-call-only.

## Current Runtime Shape

Important files:

```text
app/main.py
app/api/app.py
app/api/logging.py
app/dashboard/router.py
app/dashboard/state.py
app/integrations/whatsapp/webhook.py
app/integrations/whatsapp/webrtc.py
app/integrations/whatsapp/client.py
app/voice/agent.py
app/voice/audio_track.py
app/voice/turn_pipeline.py
app/voice/asr.py
app/voice/llm.py
app/voice/tts.py
app/voice/config.py
app/voices/chandeera-female-sample.wav
scripts/run_local_host.sh
scripts/setup_vastai.sh
pyproject.toml
uv.lock
README.md
AGENTS.md
```

Do not sync these unless explicitly needed:

```text
.env
.venv/
run_logs/
*.log
*.pid
tts_debug_latest/
debug_*.wav
official_*.wav
realtimetts_*_smoke.wav
app/voices/ unless intentionally changing the production reference voice
```

## Voice Code Walkthrough

### `app/main.py`

Stable ASGI entrypoint that exposes `app = create_app()`.

### `app/api/app.py`

Creates the FastAPI app, mounts the WhatsApp webhook router, exposes `GET /`, and prewarms local voice models during lifespan startup.

Startup prewarm loads Qwen and OmniVoice. Startup fails if prewarm fails so a bad GPU/model setup is caught immediately.

### `app/api/logging.py`

Configures global logging and the filtered rotating important-call log.

### `app/dashboard/router.py`, `app/dashboard/state.py`

Serves `GET /dashboard` and `GET /dashboard/calls`. The dashboard records active and ended call metadata plus caller/assistant transcript events. Recent sessions are persisted to `run_logs/call_sessions.json`.

### `app/integrations/whatsapp/webhook.py`

Public webhook surface.

Routes:

- `GET /webhook`: Meta webhook verification
- `POST /webhook`: WhatsApp statuses and calls

On a WhatsApp call `connect` event with an SDP offer, it starts `webrtc_service.handle_offer(call_id, sdp_offer, caller_phone)` in the background. On `terminate`, it closes and removes the peer connection.

### `app/integrations/whatsapp/webrtc.py`

WhatsApp call SDP and aiortc bridge.

The service creates the peer connection, adds a `RealtimeAudioTrack` for outbound audio, accepts inbound WhatsApp audio, and starts `voice_agent.process_audio(call_id, phone, track, output_track)`.

### `app/integrations/whatsapp/client.py`

Minimal WhatsApp Graph API client for call actions: `pre_accept` and `accept`.

### `app/voice/agent.py`

Owns active call tasks, playback interruption counters, and delegates turn handling to `LocalQwenTurnPipeline`.

### `app/voice/audio_track.py`

Outbound aiortc audio track that buffers TTS PCM and emits 48 kHz stereo frames.

### `app/voice/turn_pipeline.py`

Core voice turn engine.

What it does:

1. Plays a multilingual greeting through OmniVoice.
2. Drops inbound frames briefly during greeting protection.
3. Uses local RMS VAD to detect caller turns.
4. Transcribes completed 16 kHz PCM turns with local Whisper.
5. Sends transcript text to local Qwen 4B Q4 via `llama-cpp-python`.
6. Sends Qwen response text to OmniVoice.
7. Streams synthesized PCM into `RealtimeAudioTrack`.

### `app/voice/asr.py`, `app/voice/llm.py`, `app/voice/tts.py`, `app/voice/config.py`

Focused wrappers and hardcoded settings for Whisper, Qwen, OmniVoice, prompts, and turn-control constants.

## Environment Variables That Matter

Only secrets and deployment credentials should be environment variables. Voice model settings, TTS settings, prompts, and turn-control values are hardcoded in `app/voice/config.py`; do not add environment variables for them unless explicitly requested.

Core runtime:

```bash
VERIFY_TOKEN=my_secure_verify_token_123
WHATSAPP_ACCESS_TOKEN=...
PHONE_NUMBER_ID=...
```

## Setup Commands For A New Vast.ai Box

Basic machine check:

```bash
ssh -i ~/.ssh/vastai_ssh_file -p <PORT> root@<HOST>
uname -a
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
which uv
which git
which python3
```

One-shot setup from local repo:

```bash
REMOTE_BRANCH=<branch-name> ./scripts/setup_vastai.sh <PORT> <HOST>
```

Manual dependency sync on the remote host:

```bash
cd /workspace/sl-chatbot
CMAKE_ARGS='-DGGML_CUDA=on' FORCE_CMAKE=1 \
  uv sync --no-binary-package llama-cpp-python --reinstall-package llama-cpp-python
```

Compile-check Python files:

```bash
cd /workspace/sl-chatbot
find app -name '*.py' -print0 | xargs -0 .venv/bin/python -m py_compile
```

## How To Run The Webhook Reliably

Use `tmux`.

```bash
cd /workspace/sl-chatbot
tmux kill-session -t sl-webhook 2>/dev/null || true
pkill -f 'uvicorn app.main:app' || true
tmux new-session -d -s sl-webhook \
  "cd /workspace/sl-chatbot && \
   .venv/bin/dotenv -f .env run -- \
   .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8090 --env-file .env \
   2>&1 | tee run_logs/webhook.log"
tmux ls
```

Verify local health:

```bash
cd /workspace/sl-chatbot
ss -ltnp | egrep '(8090|8081)' || true
curl -sS http://127.0.0.1:8090/
curl -sS 'http://127.0.0.1:8090/webhook?hub.mode=subscribe&hub.verify_token=my_secure_verify_token_123&hub.challenge=12345'
tail -n 80 run_logs/webhook.log
tail -n 80 run_logs/important.log
```

## Logs

Main webhook log:

```bash
tail -f /workspace/sl-chatbot/run_logs/webhook.log
```

Important filtered log:

```bash
tail -f /workspace/sl-chatbot/run_logs/important.log
```

Use these for webhook verification, call events, SDP handling, inbound audio track start, connection state changes, transcript timings, Qwen response timings, TTS completion, and interruption behavior.

## Healthy Call Path

1. Meta delivers `POST /webhook`.
2. `app.integrations.whatsapp.webhook` logs `Received call event: connect`.
3. `app.integrations.whatsapp.webrtc` logs SDP handling and an inbound audio track.
4. `LocalQwenTurnPipeline.run()` plays the greeting.
5. Caller speaks.
6. Logs show `Turn VAD: Speech started`.
7. Logs show `Turn VAD: Speech ended`.
8. Logs show `Turn transcript`.
9. Logs show `Turn response`.
10. Logs show `RealtimeTTS complete`.

If step 1 never happens, the issue is public routing rather than the app code.
