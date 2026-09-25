# SerendibAI Gemini Live voice runtime

This service answers WhatsApp calls with Gemini Live. Gemini handles speech
recognition, multilingual conversation, turn-taking, and speech synthesis. The
service bridges WebRTC audio, runs database-backed tools, and archives calls.

## Runtime path

```text
WhatsApp Call -> aiortc -> Gemini Live -> aiortc -> WhatsApp Call
                              |
                          Neon tools
                    (knowledge, orders, booking)
```

The agent starts by asking the caller to say English, Sinhala, or Tamil. Gemini
then listens and responds in the selected language using its native audio voice.
There is no local ASR, LLM, TTS, CUDA model download, or prerecorded greeting.

`app/whatsapp.py` serves the webhook and WebRTC. `app/gemini.py` handles Live
audio. `app/utility/helper.py` handles call state, recording, and local runtime
startup. `app/utility/tools.py` handles Neon queries and the Meta API client.

## Configuration

Create `.env` with the WhatsApp, Gemini, database, AWS, and tunnel settings used
by the runtime. `GEMINI_API_KEY`, `DATABASE_URL`, and `PHONE_NUMBER_ID` are
required. The runtime reads its agent settings and stores call reports directly
in Neon. Meta should send calls to `https://whatsapp.serendibai.lk/webhook`.

## Local development

```bash
./deploy.sh --env local
uv run --with pytest python -m pytest -q
```

The health endpoint returns `warming_up`, `ready`, or `error`. Startup opens a
Gemini Live connection and checks Neon before marking
the runtime ready.

## Deployment

From the repository root, run:

```bash
./deploy.sh --env local
```

The local mode serves the WhatsApp webhook through the Cloudflare tunnel.
Keep the Mac awake and the process running to
receive calls. `--env remote` is reserved for a future server provider.

The one live test downloads a private speech clip from the recordings bucket,
sends it to Gemini Live, and requires a spoken response. Call recordings are
archived to the same private bucket.

Call transcripts and event history are stored in Neon.
