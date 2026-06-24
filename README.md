# WhatsApp Voice Bot

FastAPI service for WhatsApp Cloud API webhooks and WhatsApp Calling.

The voice stack is intentionally local:

```text
WhatsApp Cloud webhook -> FastAPI webhook
WhatsApp WebRTC audio -> local Whisper STT -> local Qwen 4B Q4 -> RealtimeTTS OmniVoice -> WhatsApp WebRTC audio
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
- `GET /dashboard`: call sessions dashboard with live and ended transcripts.
- `GET /dashboard/calls`: call sessions dashboard JSON.
- `GET /`: health check.

Incoming text messages are ignored. This repository is voice-call-only.

## Runtime Layout

- [app/main.py](/Users/nizar/Documents/Projects/sl_chatbot/app/main.py:1): stable ASGI entrypoint.
- [app/api/app.py](/Users/nizar/Documents/Projects/sl_chatbot/app/api/app.py:1): FastAPI app creation and model prewarm.
- [app/integrations/whatsapp/webhook.py](/Users/nizar/Documents/Projects/sl_chatbot/app/integrations/whatsapp/webhook.py:1): webhook verification and call-event dispatch.
- [app/integrations/whatsapp/webrtc.py](/Users/nizar/Documents/Projects/sl_chatbot/app/integrations/whatsapp/webrtc.py:1): WhatsApp SDP/WebRTC handling.
- [app/voice/turn_pipeline.py](/Users/nizar/Documents/Projects/sl_chatbot/app/voice/turn_pipeline.py:1): local turn loop, VAD, ASR, Qwen, and TTS orchestration.

The pipeline uses simple RMS-based VAD, transcribes completed caller turns with Whisper, generates responses with local Qwen through `llama-cpp-python`, and streams OmniVoice PCM back into the WhatsApp WebRTC output track.

## Environment

Required:

- `WHATSAPP_ACCESS_TOKEN` or `WHATSAPP_TOKEN`
- `PHONE_NUMBER_ID`
- `VERIFY_TOKEN`

Voice settings are hardcoded in [app/voice/config.py](/Users/nizar/Documents/Projects/sl_chatbot/app/voice/config.py:1). Keep secrets and deployment credentials in `.env`; do not add environment variables for model tuning unless there is a strong operational reason.

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
