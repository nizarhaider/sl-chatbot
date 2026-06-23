# WhatsApp Voice Bot

FastAPI service for WhatsApp Cloud API webhooks and WhatsApp Calling.

The voice stack is intentionally local:

```text
WhatsApp Cloud webhook -> FastAPI webhook
WhatsApp WebRTC audio -> local Whisper STT -> local Gemma 4 12B Q4 -> RealtimeTTS OmniVoice -> WhatsApp WebRTC audio
```

## Local Development

```bash
uv sync
bash scripts/run_local_host.sh
```

The app listens on `http://localhost:8000` by default.

## Webhook Endpoints

- `GET /webhook`: Meta webhook verification.
- `POST /webhook`: WhatsApp status and call events.
- `GET /`: health check.

Incoming text messages are ignored. This repository is voice-call-only.

## Runtime Layout

- [app/main.py](/Users/nizar/Documents/Projects/sl_chatbot/app/main.py:1): stable ASGI entrypoint.
- [app/api/app.py](/Users/nizar/Documents/Projects/sl_chatbot/app/api/app.py:1): FastAPI app creation and model prewarm.
- [app/integrations/whatsapp/webhook.py](/Users/nizar/Documents/Projects/sl_chatbot/app/integrations/whatsapp/webhook.py:1): webhook verification and call-event dispatch.
- [app/integrations/whatsapp/webrtc.py](/Users/nizar/Documents/Projects/sl_chatbot/app/integrations/whatsapp/webrtc.py:1): WhatsApp SDP/WebRTC handling.
- [app/voice/turn_pipeline.py](/Users/nizar/Documents/Projects/sl_chatbot/app/voice/turn_pipeline.py:1): local turn loop, VAD, ASR, Gemma, and TTS orchestration.

The pipeline uses simple RMS-based VAD, transcribes completed caller turns with Whisper, generates responses with local Gemma through `llama-cpp-python`, and streams OmniVoice PCM back into the WhatsApp WebRTC output track.

## Environment

Required:

- `WHATSAPP_ACCESS_TOKEN` or `WHATSAPP_TOKEN`
- `PHONE_NUMBER_ID`
- `VERIFY_TOKEN`

Voice:

- `GEMMA_MODEL_PATH`: optional local `.gguf` file path. If omitted, the app downloads from Hugging Face.
- `GEMMA_MODEL_REPO`: defaults to `google/gemma-4-12B-it-qat-q4_0-gguf`.
- `GEMMA_MODEL_FILENAME`: optional exact `.gguf` filename inside `GEMMA_MODEL_REPO`.
- `GEMMA_MODEL_DIR`: optional local download directory.
- `GEMMA_N_GPU_LAYERS`: defaults to `-1`, which asks llama.cpp to load all supported layers into VRAM.
- `GEMMA_CONTEXT_TOKENS`: defaults to `4096`.
- `GEMMA_MAX_OUTPUT_TOKENS`: defaults to `160`.
- `GEMMA_PREWARM`: defaults to `true`; startup fails if model prewarm fails.
- `WHISPER_MODEL`: defaults to `SPEAK-ASR/whisper-medium-si-merged`.
- `WHISPER_DEVICE`: defaults to `cuda`.
- `REALTIME_TTS_REF_AUDIO`: defaults to `app/voices/chandeera-female-sample.wav`.
- `REALTIME_TTS_REF_TEXT`: reference text for OmniVoice cloning.
- `REALTIME_TTS_REF_LANGUAGE`: defaults to `si`.
- `REALTIME_TTS_DEVICE`: defaults to `cuda:0`; use `mps` on Apple Silicon.
- `REALTIME_TTS_DTYPE`: defaults to `float16`.
- `REALTIME_TTS_NUM_STEPS`: defaults to `12,12`.
- `REALTIME_TTS_PREWARM`: defaults to `true`.
- `TURN_INPUT_CHUNK_MS`, `TURN_SILENCE_THRESHOLD`, `TURN_END_SILENCE_CHUNKS`: local VAD tuning.
- `TURN_GREETING_DELAY_SECONDS`, `TURN_GREETING_PROTECTION_MAX_SECONDS`: greeting timing controls.
- `IMPORTANT_LOG_PATH`: defaults to `run_logs/important.log`.

## Vast.ai Setup

```bash
REMOTE_BRANCH=<branch-name> ./scripts/setup_vastai.sh <PORT> <HOST>
```

The setup script clones or updates `/workspace/sl-chatbot`, syncs `.env`, builds `llama-cpp-python` with CUDA, starts ngrok and the webhook in `tmux`, and waits for local and public webhook verification.

Manual dependency sync on the remote host:

```bash
cd /workspace/sl-chatbot
CMAKE_ARGS='-DGGML_CUDA=on' FORCE_CMAKE=1 \
  uv sync --no-binary-package llama-cpp-python --reinstall-package llama-cpp-python
```

## Tests

```bash
uv run pytest -q
find app -name '*.py' -print0 | xargs -0 python3 -m py_compile
```
