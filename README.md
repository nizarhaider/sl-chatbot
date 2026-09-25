# SerendibAI Gemini Live voice runtime

This service answers WhatsApp calls with Gemini Live. Gemini handles speech
recognition, multilingual conversation, turn-taking, and speech synthesis. The
service only bridges WebRTC audio, executes portal tools, archives calls, and
exposes the transcript dashboard.

## Runtime path

```text
WhatsApp Call -> aiortc -> Gemini Live -> aiortc -> WhatsApp Call
                              |
                         Portal tools
                    (knowledge, orders, booking)
```

The agent starts by asking the caller to say English, Sinhala, or Tamil. Gemini
then listens and responds in the selected language using its native audio voice.
There is no local ASR, LLM, TTS, CUDA model download, or prerecorded greeting.

## Configuration

Create `.env` with the WhatsApp, Gemini, database, AWS, and tunnel settings used
by the runtime. `GEMINI_API_KEY`, `DATABASE_URL`, and `PHONE_NUMBER_ID` are
required. The local deployment script creates a scoped portal token.

## Local development

```bash
./deploy.sh --env local
uv run --with pytest python -m pytest -q
```

The health endpoint returns `warming_up`, `ready`, or `error`. Startup opens a
Gemini Live connection and checks the portal tool service before marking
the runtime ready.

## Deployment

From the repository root, run:

```bash
./deploy.sh --env local
```

The local mode connects the saved WhatsApp agent to the portal through a
Cloudflare tunnel. Keep the Mac awake and the process running to
receive calls. `--env remote` is reserved for a future server provider.

The one live test downloads a private speech clip from the recordings bucket,
sends it to Gemini Live, and requires a spoken response. Call recordings are
archived to the same private bucket.

Live call transcripts are available at
https://portal.serendibai.lk/dashboard.
