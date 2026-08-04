# WhatsApp Voice Bot

Voice-only WhatsApp assistant powered by local models on a GPU host.

Incoming WhatsApp text messages are intentionally ignored. The live call path uses local Whisper, local Gemma through `llama-cpp-python`, and the SerendibAI OmniVoice V2 fine-tune through RealtimeTTS. Do not add hosted LLM calls to the voice path.

## Architecture

```mermaid
flowchart LR
    WA[WhatsApp Cloud] -->|webhook events| API[FastAPI /webhook]
    API -->|SDP offer| RTC[aiortc peer connection]
    RTC -->|inbound audio| VAD[Local RMS VAD]
    VAD -->|completed turn PCM| ASR[Local Whisper STT]
    ASR -->|transcript| LLM[Local Gemma 4 E4B Q4]
    LLM -->|response text| TTS[RealtimeTTS OmniVoice]
    TTS -->|48 kHz stereo PCM| RTC
    RTC -->|outbound audio| WA
```

```mermaid
sequenceDiagram
    participant WA as WhatsApp Cloud
    participant Webhook as FastAPI /webhook
    participant RTC as aiortc
    participant Pipeline as LocalGemmaTurnPipeline
    participant ASR as Whisper
    participant LLM as Gemma
    participant TTS as OmniVoice

    WA->>Webhook: call connect event with SDP offer
    Webhook->>RTC: create peer connection
    RTC-->>WA: pre_accept and accept with SDP answer
    RTC->>Pipeline: inbound audio track
    Pipeline->>TTS: synthesize greeting
    TTS-->>RTC: greeting PCM
    Pipeline->>Pipeline: VAD detects caller turn
    Pipeline->>ASR: transcribe 16 kHz PCM
    ASR-->>Pipeline: caller transcript
    Pipeline->>LLM: generate response or tool call
    LLM-->>Pipeline: optional property search or booking tool call
    Pipeline->>Pipeline: execute tool against Neon
    Pipeline->>LLM: tool result
    LLM-->>Pipeline: response text
    Pipeline->>TTS: synthesize response
    TTS-->>RTC: response PCM
    RTC-->>WA: outbound call audio
```

## Main Components

| Path | Purpose |
| --- | --- |
| [`app/main.py`](app/main.py) | ASGI entrypoint. |
| [`app/api/app.py`](app/api/app.py) | FastAPI app factory and startup model prewarm. |
| [`app/integrations/whatsapp/webhook.py`](app/integrations/whatsapp/webhook.py) | Meta verification, status events, and call event dispatch. |
| [`app/integrations/whatsapp/webrtc.py`](app/integrations/whatsapp/webrtc.py) | WhatsApp SDP handling and aiortc bridge. |
| [`app/voice/agent.py`](app/voice/agent.py) | Active call task ownership and interruption counters. |
| [`app/voice/turn_pipeline.py`](app/voice/turn_pipeline.py) | Greeting, VAD, ASR, LLM, TTS, and streaming turn loop. |
| [`app/voice/tools.py`](app/voice/tools.py) | Neon property search and viewing-appointment tools. |
| [`app/voice/config.py`](app/voice/config.py) | Voice model settings, prompts, and turn-control constants. |
| [`app/dashboard/router.py`](app/dashboard/router.py) | Browser dashboard for active and recent call sessions. |

## Runtime Endpoints

| Endpoint | Description |
| --- | --- |
| `GET /` | Health check. |
| `GET /webhook` | Meta webhook verification. |
| `POST /webhook` | WhatsApp statuses and call events. |
| `GET /dashboard` | Call sessions dashboard. |
| `GET /dashboard/calls` | Call sessions JSON. |

## Environment

Required secrets and deployment credentials:

```bash
VERIFY_TOKEN=my_secure_verify_token_123
WHATSAPP_ACCESS_TOKEN=...
PHONE_NUMBER_ID=...
DATABASE_URL=postgresql://...
```

`DATABASE_URL` should point to the same Neon database used by the hosted
dashboard so call status and transcript updates remain available after the GPU
instance is terminated.

At startup, the voice runtime creates `real_estate_properties` and
`property_appointments` when needed and seeds the initial Homelands inventory
for the customer mapped to `PHONE_NUMBER_ID`. Gemma reads property facts only
through the Neon-backed tools and stores confirmed viewing appointments with
the call ID and caller phone number.

Keep voice model paths, prompts, TTS settings, and turn-control values in [`app/voice/config.py`](app/voice/config.py). Use `.env` only for secrets and deployment credentials.

## Local Development

```bash
uv sync
bash scripts/run_local_host.sh
```

The local server listens on `http://localhost:8000`.

Useful checks:

```bash
curl -sS http://127.0.0.1:8000/
curl -sS 'http://127.0.0.1:8000/webhook?hub.mode=subscribe&hub.verify_token=my_secure_verify_token_123&hub.challenge=12345'
```

## Vast.ai Deployment

The production profile uses at least 16 GB of GPU VRAM. Whisper, OmniVoice, and
all 42 Gemma layers stay on CUDA. Inference mode, SDPA/flash attention, and a
smaller llama.cpp batch keep the exact model weights within the available VRAM.

See [`docs/hosting_cost_report.md`](docs/hosting_cost_report.md) for the
measured memory, latency, disk usage, and current provider comparison.

One-command rental and setup from this repo:

```bash
./scripts/deploy_vastai.sh
```

The deployer selects the cheapest verified, on-demand, single-GPU RTX 4070 or
30-series offer with at least 16 GB VRAM. It rents a 40 GB disk,
waits for SSH, then runs the setup and health checks. Preview the current choice
without renting anything with `DRY_RUN=true ./scripts/deploy_vastai.sh`.

To set up an instance that has already been rented:

```bash
REMOTE_BRANCH=<branch-name> ./scripts/setup_vastai.sh <SSH_PORT> <HOST_IP>
```

The setup script prepares `/workspace/sl-chatbot`, syncs `.env` when present, builds `llama-cpp-python` with CUDA, starts ngrok, starts the webhook in `tmux`, and waits for webhook verification.

Manual dependency sync on the remote host:

```bash
cd /workspace/sl-chatbot
uv sync
```

Run the webhook manually:

```bash
cd /workspace/sl-chatbot
tmux kill-session -t sl-webhook 2>/dev/null || true
tmux new-session -d -s sl-webhook \
  "cd /workspace/sl-chatbot && \
   .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8090 --env-file .env \
   2>&1 | tee run_logs/webhook.log"
```

## Verification

Compile-check Python files:

```bash
find app -name '*.py' -print0 | xargs -0 python3 -m py_compile
```

Run tests:

```bash
uv run pytest -q
```

Healthy call logs should show:

```text
Received call event: connect
Inbound audio track received
Turn VAD: Speech started
Turn VAD: Speech ended
Turn transcript
Turn response
RealtimeTTS complete
```

## Logs

Main webhook log:

```bash
tail -f /workspace/sl-chatbot/run_logs/webhook.log
```

Important filtered call log:

```bash
tail -f /workspace/sl-chatbot/run_logs/important.log
```
