# WhatsApp Voice Bot

FastAPI service for WhatsApp Cloud API webhooks and WhatsApp calling.

The voice stack is intentionally narrow:

```text
WhatsApp Cloud webhook -> FastAPI webhook
WhatsApp WebRTC audio -> Gemini STT -> Gemini LLM -> RealtimeTTS OmniVoice -> WhatsApp WebRTC audio
```

## Local Development

```bash
uv sync
bash scripts/run_local_host.sh
```

The app listens on `http://localhost:8000` by default.

## Webhook Endpoints

- `GET /webhook`: Meta webhook verification.
- `POST /webhook`: WhatsApp text, status, and call events.
- `POST /send-message`: internal outbound WhatsApp message endpoint protected by `x-api-key`.

## Voice Flow

Main files:

- [app/main.py](/Users/nizar/Documents/Projects/sl_chatbot/app/main.py:1): FastAPI app and RealtimeTTS prewarm.
- [app/webhooks/whatsapp.py](/Users/nizar/Documents/Projects/sl_chatbot/app/webhooks/whatsapp.py:1): webhook parsing and dispatch.
- [app/services/webrtc.py](/Users/nizar/Documents/Projects/sl_chatbot/app/services/webrtc.py:1): WhatsApp SDP/WebRTC handling.
- [app/voice_agent/agent.py](/Users/nizar/Documents/Projects/sl_chatbot/app/voice_agent/agent.py:1): call task lifecycle and outbound audio track.
- [app/voice_agent/gemini_turn_pipeline.py](/Users/nizar/Documents/Projects/sl_chatbot/app/voice_agent/gemini_turn_pipeline.py:1): Gemini STT, Gemini LLM, VAD, and RealtimeTTS OmniVoice playback.

The pipeline uses simple RMS-based voice activity detection, sends each completed caller utterance to Gemini for transcription, sends the transcript history to the LLM, and streams OmniVoice audio chunks back into the WhatsApp WebRTC output track.

## Text Chat Flow

Incoming WhatsApp text messages still use [app/chat_agent](/Users/nizar/Documents/Projects/sl_chatbot/app/chat_agent). That path can answer store questions, search [data/products.xlsx](/Users/nizar/Documents/Projects/sl_chatbot/data/products.xlsx), create pending orders, confirm orders, append rows to [data/orders.xlsx](/Users/nizar/Documents/Projects/sl_chatbot/data/orders.xlsx), and notify a manager.

## Environment

Required:

- `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- `WHATSAPP_ACCESS_TOKEN` or `WHATSAPP_TOKEN`
- `PHONE_NUMBER_ID`
- `VERIFY_TOKEN`

Voice:

- `GEMINI_STT_MODEL`: defaults to `gemini-2.5-flash-lite`.
- `GEMINI_LLM_MODEL`: defaults to `gemini-2.5-flash-lite`.
- `REALTIME_TTS_REF_AUDIO`: defaults to `app/voices/sample_si_lk.mp3`.
- `REALTIME_TTS_REF_TEXT`: reference text for OmniVoice cloning.
- `REALTIME_TTS_REF_LANGUAGE`: defaults to `si`.
- `REALTIME_TTS_DEVICE`: defaults to `cuda:0`; use `mps` on Apple Silicon.
- `REALTIME_TTS_DTYPE`: defaults to `float16`.
- `REALTIME_TTS_NUM_STEPS`: defaults to `12,12`.
- `REALTIME_TTS_PREWARM`: defaults to `true`.
- `IMPORTANT_LOG_PATH`: defaults to `run_logs/important.log`.

Text commerce:

- `CHAT_AGENT_MODEL`
- `CHAT_AGENT_BUSINESS_NAME`
- `PRODUCT_CATALOG_PATH`
- `LOCAL_ORDERS_PATH`
- `MANAGER_WHATSAPP_NUMBER`
- optional Google Docs/Sheets service-account settings.

## Tests

```bash
uv run pytest -q
```

