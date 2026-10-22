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
audio. `app/utility/helper.py` handles call state, recording, and runtime
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
Keep the Mac awake and the process running to receive calls.

### AWS server

The remote runtime runs on an Ubuntu ARM64 EC2 `t4g.small` in Mumbai
(`ap-south-1`), instance `i-0706865fb7080934b`, with an encrypted 8 GiB gp3 disk.
CPU credits use Standard mode
to avoid surplus credit charges. SSH is restricted to the operator's IP;
UDP ports 32768–60999 carry WebRTC audio. The webhook uses the existing
Cloudflare Tunnel at `https://whatsapp.serendibai.lk/webhook`.

Install Python 3.12, uv, ffmpeg, and cloudflared, then place this repository in
`/opt/serendibai/voice-agent`, owned by `ubuntu`. Copy the runtime's required
settings into `.env` with permissions `600`. Use the EC2 recording-storage
role instead of AWS access keys.

```bash
sudo cp voice-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now voice-agent
sudo systemctl status voice-agent
sudo journalctl -u voice-agent -n 50 --no-pager
```

Run `sudo systemctl restart voice-agent` after uploading a new version.
The service starts on boot and restarts after failures. Stop other copies
of this runtime before connecting the production tunnel.

Set `VOICE_AGENT_ID` in `.env` to pin a server to one customer's agent. Startup
checks that this agent owns the configured `PHONE_NUMBER_ID` and derives its
customer ID from Neon. Customer feature limits are checked on every tool execution.

Transcripts: https://portal.serendibai.lk/dashboard

### Inbound case demos

The portal's `/cases` page selects Support, Sales, Operations, Marketing or Talent.
Apply the portal repository's `db/cases.sql` migration before starting this version.
The caller confirms the generated `CASE` message from their WhatsApp number, then
calls the same chat within ten minutes. The selection is used once; calls without
a verified selection use the configured runtime agent. Demo calls are capped at
five minutes. Each demo has its own agent and tenant for tools, transcripts and
recordings; the existing Talent 360 configuration is unchanged.

Website demo calls require `PORTAL_DEMO_ENABLED=1` and
`PORTAL_URL=https://portal.serendibai.lk` on the runtime, plus the same
`DEMO_SIGNING_SECRET` on the runtime and portal. Outbound audio starts after
WhatsApp reports that the recipient answered. Demo calls are capped at three minutes.

The one live test downloads a private speech clip from the recordings bucket,
sends it to Gemini Live, and requires a spoken response. Call recordings are
archived to the same private bucket.

Call transcripts and event history are stored in Neon.
