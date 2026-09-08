# SerendibAI Gemini Live voice runtime

This service answers WhatsApp calls with Gemini Live. Gemini handles speech
recognition, multilingual conversation, turn-taking, and speech synthesis. The
service only bridges WebRTC audio, executes property tools, archives calls, and
exposes the transcript dashboard.

## Runtime path

```text
WhatsApp Call -> aiortc -> Gemini Live -> aiortc -> WhatsApp Call
                              |
                        Property tools
                    (Neon, Pinecone, WhatsApp)
```

The agent starts by asking the caller to say English, Sinhala, or Tamil. Gemini
then listens and responds in the selected language using its native audio voice.
There is no local ASR, LLM, TTS, CUDA model download, or prerecorded greeting.

## Configuration

Create `.env` with the WhatsApp, Gemini, data, and tunnel settings used by the
runtime. The required Gemini variable is `GEMINI_API_KEY`. Property search and
booking require `DATABASE_URL`, `PINECONE_API_KEY`, and `PHONE_NUMBER_ID`.

## Local development

```bash
uv sync --all-groups
uv run uvicorn app.api.app:create_app --host 0.0.0.0 --port 8081
uv run pytest
```

The health endpoint returns `warming_up`, `ready`, or `error`. Startup opens a
Gemini Live connection and initializes the property tool service before marking
the runtime ready.

## Vast deployment

From the repository root, run:

```bash
./scripts/setup_vastai.sh
```

The script rents a Vast instance with at least 8 GB VRAM and 2 CPU cores,
deploys the current branch, configures the webhook and Cloudflare tunnel, and
verifies the service. Gemini performs inference remotely, so the instance does
not run ASR, LLM, or TTS locally and does not download open-source speech or
language models.

Live call transcripts are available at
https://dashboard.serendibai.lk/dashboard/calls.
