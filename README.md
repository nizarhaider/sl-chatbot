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
./deploy.sh --env local
uv run pytest
```

The health endpoint returns `warming_up`, `ready`, or `error`. Startup opens a
Gemini Live connection and initializes the property tool service before marking
the runtime ready.

## Deployment

From the repository root, run:

```bash
./deploy.sh --env local
```

The local mode connects the saved WhatsApp agent to the portal through a
temporary Cloudflare tunnel. Keep the Mac awake and the process running to
receive calls. `--env remote` is reserved for a future server provider.

Live call transcripts are available at
https://portal.serendibai.lk/dashboard.
