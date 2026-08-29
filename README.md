# WhatsApp Voice Bot

Voice-only WhatsApp assistant powered by local models on a GPU host.

Incoming WhatsApp text messages are intentionally ignored. The live call path uses local Whisper, Unsloth Gemma 4 E4B UD-Q4_K_XL through llama.cpp, and the SerendibAI OmniVoice V2 fine-tune through RealtimeTTS. Do not add hosted LLM calls to the voice path.

## Architecture

```mermaid
flowchart LR
    WA[WhatsApp Cloud] -->|webhook events| API[FastAPI /webhook]
    API -->|SDP offer| RTC[aiortc peer connection]
    RTC -->|inbound audio| VAD[Local RMS VAD]
    VAD -->|completed turn PCM| ASR[Local Whisper STT]
    ASR -->|transcript| LLM[Unsloth Gemma 4 E4B UD-Q4_K_XL]
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
    Pipeline->>Pipeline: search properties in Pinecone
    Pipeline->>Pipeline: write confirmed appointments to Neon
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
| [`app/voice/tools.py`](app/voice/tools.py) | Pinecone property search and Neon appointment tools. |
| [`app/voice/pinecone_store.py`](app/voice/pinecone_store.py) | Pinecone property indexing and semantic retrieval. |
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
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=homelands-properties
```

`DATABASE_URL` should point to the same Neon database used by the hosted
dashboard so call status and transcript updates remain available after the GPU
instance is terminated.

At startup, the voice runtime creates `real_estate_properties` and
`property_appointments` when needed, seeds the initial Homelands inventory for
the customer mapped to `PHONE_NUMBER_ID`, and synchronizes active property
records into the Pinecone namespace for that customer. Gemma reads property
facts only through Pinecone-backed retrieval. Confirmed viewing appointments
remain transactional records in Neon with the call ID and caller phone number.

Keep voice model paths, prompts, TTS settings, and turn-control values in [`app/voice/config.py`](app/voice/config.py). Use `.env` only for secrets and deployment credentials.

## Gemma inference in Google Colab

Choose one of the Colab notebooks below, select a GPU runtime, run the cells in
order, and provide a Hugging Face token with access to the gated Gemma
repository through Colab Secrets (recommended) or the hidden runtime prompt.
Each final cell prints a temporary Gradio URL.

- [`docs/gemma_q4_colab_gradio.ipynb`](docs/gemma_q4_colab_gradio.ipynb): the
  production Q4_0 GGUF via CUDA llama.cpp; recommended for a T4 or other 16 GB GPU.
- [`docs/gemma_colab_gradio.ipynb`](docs/gemma_colab_gradio.ipynb): the official
  BF16 Transformers `text-generation` pipeline; requires a larger GPU.

To run the complete voice application separately:

```bash
uv sync
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8081 --env-file .env
```

The Gemma server listens on `http://localhost:8000`; the voice application
listens on `http://localhost:8081`.

Useful checks:

```bash
curl -sS http://127.0.0.1:8081/
curl -sS 'http://127.0.0.1:8081/webhook?hub.mode=subscribe&hub.verify_token=my_secure_verify_token_123&hub.challenge=12345'
```

## Vast.ai Deployment

The voice runtime deployer defaults to 16 GB of GPU VRAM. The deployer selects
the cheapest compatible RTX 30/40-series listing that meets this floor. Use a 48 GB
GPU for training, conversion, and the first full-stack validation.

See [`docs/hosting_cost_report.md`](docs/hosting_cost_report.md) for the
measured memory, latency, disk usage, and current provider comparison.

One-command rental and setup from this repo:

```bash
./scripts/setup_vastai.sh
```

The deployer selects the cheapest compatible verified, on-demand, single-GPU
offer with at least 16 GB VRAM and rents a 50 GB disk,
waits up to two minutes for SSH, then runs the setup and health checks. It destroys
a host automatically only when it cannot become SSH-ready in that two-minute
window; every later failure leaves the instance available for manual inspection.
After SSH is ready there is no elapsed-time limit. The locked Python
runtime, prebuilt llama.cpp, and the Q4 Gemma download concurrently; Whisper and
OmniVoice then warm the shared Hugging Face cache
before model prewarm.
The deployer prefers listings with at least 500 Mbps advertised ingress; actual
installation progress remains the acceptance signal rather than a synthetic speed
test. The deployer tries at most three distinct offers. Preview the current choice
without renting anything with `DRY_RUN=true ./scripts/setup_vastai.sh`. The
`MIN_GPU_RAM_GB` override cannot be set below the 16 GB hard floor.
The selected host must support CUDA 12.8 or newer.

To set up an instance that has already been rented:

```bash
REMOTE_BRANCH=<branch-name> ./scripts/setup_vastai.sh <SSH_PORT> <HOST_IP>
```

The setup script prepares `/workspace/sl-chatbot`, syncs `.env`, installs only the locked runtime dependencies, downloads Unsloth's Q4 Gemma GGUF with a prebuilt llama.cpp binary, starts the supervised services, and verifies the permanent webhook URL.

Manual dependency sync on the remote host:

```bash
cd /workspace/sl-chatbot
uv sync
```

Restart and inspect the supervised runtime:

```bash
cd /workspace/sl-chatbot
supervisorctl restart sl-webhook sl-cloudflared
supervisorctl status sl-webhook sl-cloudflared
```

## Verification

### LLM-only Sinhala quality evaluation

This evaluation hosts only local Gemma on a GPU. It uses an in-memory property
search and appointment service with the production tool contract, then sends the
full scripted conversation and tool traces to a separate judge model. It does not
start Whisper, OmniVoice, WebRTC, Pinecone, or Neon.

On a GPU host with the runtime environment installed:

```bash
GEMINI_API_KEY=... .venv/bin/python tests/llm_quality.py --report llm-quality-report.json
```

The stages cover Sinhala selection and continuity, property search, follow-up
context, broad-request clarification, narrowed search, incomplete booking details,
complete appointment booking, and confirmation. Every stage receives independent
language, continuity, clarification, tool, safety, booking, and quality scores.

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
