# WhatsApp Voice Bot

FastAPI service for WhatsApp Cloud API webhooks and WhatsApp calling.

The voice stack is intentionally narrow:

```text
WhatsApp Cloud webhook -> FastAPI webhook
WhatsApp WebRTC audio -> Gemini Live session -> text response -> RealtimeTTS OmniVoice -> WhatsApp WebRTC audio
```

## Local Development

```bash
uv sync
bash scripts/run_local_host.sh
```

The app listens on `http://localhost:8000` by default.

## Ngrok temporary tunnel

When you need a temporary public URL for WhatsApp webhook verification (for example during remote setup), the `scripts/setup_vastai.sh` helper can install and start an `ngrok` service on the remote host and print the public `https://*.ngrok.io` callback URL. Set `USE_TEMP_TUNNEL=true` (default) to enable this behavior.

**Setup:**
1. Add your ngrok auth token to `.env`:
   ```bash
   NGROK_AUTH_TOKEN=your_ngrok_token_here
   ```
2. Run the setup script as usual (it will install ngrok, configure the auth token, set up the service, and print the public URL).

**How it works:**
- The setup script generates `ngrok.yml` from the template in the repo, substituting your `APP_PORT`
- It copies the config to the remote host and installs the ngrok service
- The ngrok service runs as a persistent daemon, forwarding traffic to your webhook server
- Use the printed URL plus `/webhook` as the webhook callback in the WhatsApp dashboard for verification

**Useful commands:**
```bash
# Check ngrok service status
ssh -i ~/.ssh/vastai_ssh_file -p <PORT> root@<HOST> 'ngrok service status'

# Get the current public URL
ssh -i ~/.ssh/vastai_ssh_file -p <PORT> root@<HOST> 'curl http://127.0.0.1:4040/api/tunnels | jq .tunnels[0].public_url'

# Stop the service (if needed)
ssh -i ~/.ssh/vastai_ssh_file -p <PORT> root@<HOST> 'ngrok service stop'

# Start it again
ssh -i ~/.ssh/vastai_ssh_file -p <PORT> root@<HOST> 'ngrok service start'
```

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
- [app/voice_agent/gemini_turn_pipeline.py](/Users/nizar/Documents/Projects/sl_chatbot/app/voice_agent/gemini_turn_pipeline.py:1): Gemini Live session, local VAD, and RealtimeTTS OmniVoice playback.

The pipeline uses simple RMS-based voice activity detection locally, streams caller audio into a single Gemini Live session with explicit activity boundaries, receives text responses from the Live session, and streams OmniVoice audio chunks back into the WhatsApp WebRTC output track.

## Text Chat Flow

Incoming WhatsApp text messages still use [app/chat_agent](/Users/nizar/Documents/Projects/sl_chatbot/app/chat_agent). That path can answer store questions, search [data/products.xlsx](/Users/nizar/Documents/Projects/sl_chatbot/data/products.xlsx), create pending orders, confirm orders, append rows to [data/orders.xlsx](/Users/nizar/Documents/Projects/sl_chatbot/data/orders.xlsx), and notify a manager.

## Environment

Required:

- `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- `WHATSAPP_ACCESS_TOKEN` or `WHATSAPP_TOKEN`
- `PHONE_NUMBER_ID`
- `VERIFY_TOKEN`

Voice:

- `GEMINI_LIVE_MODEL`: defaults to `gemini-live-2.5-flash-preview`.
- `GEMINI_LIVE_API_VERSION`: defaults to `v1beta`.
- `REALTIME_TTS_REF_AUDIO`: defaults to `app/voices/sample_si_lk.mp3`.
- `REALTIME_TTS_REF_TEXT`: reference text for OmniVoice cloning.
- `REALTIME_TTS_REF_LANGUAGE`: defaults to `si`.
- `REALTIME_TTS_DEVICE`: defaults to `cuda:0`; use `mps` on Apple Silicon.
- `REALTIME_TTS_DTYPE`: defaults to `float16`.
- `REALTIME_TTS_NUM_STEPS`: defaults to `12,12`.
- `REALTIME_TTS_PREWARM`: defaults to `true`.
- `TURN_INPUT_CHUNK_MS`, `TURN_SILENCE_THRESHOLD`, `TURN_END_SILENCE_CHUNKS`: local VAD tuning.
- `TURN_GREETING_DELAY_SECONDS`, `TURN_GREETING_PROTECTION_MAX_SECONDS`: greeting timing controls.
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
