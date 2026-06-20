# Agent Handoff

This repo runs a WhatsApp voice bot on a Vast.ai GPU machine.

The current voice path is:

```text
WhatsApp Cloud webhook -> FastAPI /webhook
WhatsApp Calling SDP offer -> aiortc peer connection
Inbound WhatsApp audio -> local VAD -> local Whisper STT
Whisper transcript -> local Gemma 4 12B Q4 via llama.cpp
Gemma text -> RealtimeTTS OmniVoice
OmniVoice PCM -> outbound aiortc audio track -> WhatsApp call
```

The assistant brain is local Gemma. Do not add hosted LLM API calls to the voice path.

## Current Runtime Shape

Important files:

```text
app/main.py
app/webhooks/whatsapp.py
app/services/webrtc.py
app/services/whatsapp_api.py
app/voice_agent/agent.py
app/voice_agent/local_gemma_turn_pipeline.py
scripts/run_local_host.sh
scripts/setup_vastai.sh
scripts/setup_vastai_whisper_medium_si_merged.sh
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
training_custom_tts/datasets/
tts_debug_latest/
debug_*.wav
official_*.wav
realtimetts_*_smoke.wav
app/voices/ unless intentionally changing the production reference voice
```

## Voice Code Walkthrough

### `app/main.py`

FastAPI entrypoint.

What it does:

- Loads `.env`
- Configures global logging
- Adds filtered rotating logs for important call events
- Prewarms local voice models during lifespan startup
- Mounts the WhatsApp webhook router
- Exposes `GET /` for a simple health check

Startup prewarm loads Gemma when `GEMMA_PREWARM=true` and loads OmniVoice when `REALTIME_TTS_PREWARM=true`. Startup fails if prewarm fails so a bad GPU/model setup is caught immediately.

### `app/webhooks/whatsapp.py`

Public webhook surface.

Routes:

- `GET /webhook`: Meta webhook verification
- `POST /webhook`: WhatsApp events, including calls
- `POST /send-message`: internal text send endpoint

On a WhatsApp call `connect` event with an SDP offer, it starts `webrtc_service.handle_offer(call_id, sdp_offer, caller_phone)` in the background. On `terminate`, it closes and removes the peer connection.

### `app/services/webrtc.py`

WhatsApp call SDP and aiortc bridge.

The service creates the peer connection, adds a `RealtimeAudioTrack` for outbound audio, accepts inbound WhatsApp audio, and starts `voice_agent.process_audio(call_id, phone, track, output_track)`.

### `app/voice_agent/agent.py`

Bridge between aiortc tracks and the local voice turn pipeline.

Key classes:

- `RealtimeAudioTrack`: outbound aiortc audio track that buffers TTS PCM and emits 48 kHz stereo frames.
- `VoiceAgent`: owns active call tasks, playback interruption counters, and delegates turn handling to `LocalGemmaTurnPipeline`.

### `app/voice_agent/local_gemma_turn_pipeline.py`

Core voice turn engine.

What it does:

1. Plays a multilingual greeting through OmniVoice.
2. Drops inbound frames briefly during greeting protection.
3. Uses local RMS VAD to detect caller turns.
4. Transcribes completed 16 kHz PCM turns with local Whisper.
5. Sends transcript text to local Gemma 4 12B Q4 via `llama-cpp-python`.
6. Sends Gemma response text to OmniVoice.
7. Streams synthesized PCM into `RealtimeAudioTrack`.

Key model settings:

```bash
ASR model: SPEAK-ASR/whisper-medium-si-merged
WHISPER_DEVICE=cuda
GEMMA_MODEL_REPO=google/gemma-4-12B-it-qat-q4_0-gguf
GEMMA_MODEL_PATH=
GEMMA_MODEL_FILENAME=
GEMMA_MODEL_DIR=
GEMMA_N_GPU_LAYERS=-1
GEMMA_CONTEXT_TOKENS=4096
GEMMA_BATCH_TOKENS=512
GEMMA_THREADS=8
GEMMA_TEMPERATURE=0.2
GEMMA_MAX_OUTPUT_TOKENS=160
GEMMA_PREWARM=true
```

`GEMMA_N_GPU_LAYERS=-1` asks llama.cpp to place all supported layers in VRAM. If `GEMMA_MODEL_PATH` is not set, the code downloads the Q4 GGUF from Hugging Face.

## Environment Variables That Matter

Core runtime:

```bash
VERIFY_TOKEN=my_secure_verify_token_123
WHATSAPP_ACCESS_TOKEN=...
PHONE_NUMBER_ID=...
GRAPH_API_VERSION=v22.0
```

Turn control:

```bash
TURN_INPUT_CHUNK_MS=40
TURN_SILENCE_THRESHOLD=1000
TURN_END_SILENCE_CHUNKS=50
TURN_GREETING_DELAY_SECONDS=1.2
TURN_GREETING_PROTECTION_MAX_SECONDS=1.5
TURN_MIN_AUDIO_MS=500
```

OmniVoice:

```bash
REALTIME_TTS_DEVICE=cuda:0
REALTIME_TTS_DTYPE=float16
REALTIME_TTS_NUM_STEPS=12,12
REALTIME_TTS_REF_AUDIO=app/voices/sample_si_lk.mp3
REALTIME_TTS_REF_TEXT=...
REALTIME_TTS_REF_LANGUAGE=si
REALTIME_TTS_PREWARM=true
REALTIME_TTS_DEBUG=false
```

Logging:

```bash
IMPORTANT_LOG_PATH=run_logs/important.log
IMPORTANT_LOG_MAX_BYTES=1048576
IMPORTANT_LOG_BACKUPS=3
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
REMOTE_BRANCH=<branch-name> ./scripts/setup_vastai_whisper_medium_si_merged.sh <PORT> <HOST>
```

The setup script:

- Clones or updates `/workspace/sl-chatbot`
- Checks out `REMOTE_BRANCH`
- Copies local `.env`
- Installs system packages
- Builds `llama-cpp-python` with CUDA using `CMAKE_ARGS='-DGGML_CUDA=on'`
- Starts ngrok in `tmux`
- Starts webhook in `tmux`
- Waits for local health checks and public webhook verification

Manual dependency sync on the remote host:

```bash
cd /workspace/sl-chatbot
CMAKE_ARGS='-DGGML_CUDA=on' FORCE_CMAKE=1 \
  uv sync --no-binary-package llama-cpp-python --reinstall-package llama-cpp-python
```

Compile-check important Python files:

```bash
cd /workspace/sl-chatbot
.venv/bin/python -m py_compile \
  app/main.py \
  app/webhooks/whatsapp.py \
  app/services/webrtc.py \
  app/services/whatsapp_api.py \
  app/voice_agent/agent.py \
  app/voice_agent/local_gemma_turn_pipeline.py
```

## How To Run The Webhook Reliably

Use `tmux`.

```bash
cd /workspace/sl-chatbot
tmux kill-session -t sl-webhook 2>/dev/null || true
pkill -f 'uvicorn app.main:app' || true
tmux new-session -d -s sl-webhook \
  "cd /workspace/sl-chatbot && \
   .venv/bin/dotenv -f .env run -- env \
   WHISPER_DEVICE=cuda \
   GEMMA_N_GPU_LAYERS=-1 \
   GEMMA_PREWARM=true \
   REALTIME_TTS_DEVICE=cuda:0 \
   REALTIME_TTS_DTYPE=float16 \
   REALTIME_TTS_NUM_STEPS=12,12 \
   IMPORTANT_LOG_PATH=run_logs/important.log \
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

Use these for:

- webhook verification
- call event receipt
- SDP handling
- inbound audio track start
- connection state changes
- transcript timings
- Gemma response timings
- TTS completion
- interruption behavior

## Healthy Call Path

1. Meta delivers `POST /webhook`.
2. `app.webhooks.whatsapp` logs `Received call event: connect`.
3. `app.services.webrtc` logs SDP handling and an inbound audio track.
4. `LocalGemmaTurnPipeline.run()` plays the greeting.
5. Caller speaks.
6. Logs show `Turn VAD: Speech started`.
7. Logs show `Turn VAD: Speech ended`.
8. Logs show `Turn transcript`.
9. Logs show `Turn response`.
10. Logs show `RealtimeTTS complete`.

If step 1 never happens, the issue is public routing rather than the app code.
