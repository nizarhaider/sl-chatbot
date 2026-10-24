# SerendibAI Pipecat voice runtime

WhatsApp calls use Pipecat 1.14.0 SmallWebRTC and Gemini 3.8 Live. Gemini handles
speech understanding, server-side turn detection and speech generation. The
application has no local speech models and sends no explicit language locale.

## Editable settings

Edit `app/prompts.py` for shared instructions and greetings. Edit `app/config.py`
for model, default voice, locale, audio rates, limits and other runtime settings.
Per-agent business instructions and selected voices remain in Neon/the portal,
and override the shared defaults. Keep credentials in `.env`. Restart the runtime
after changing either Python settings file.

## Four runtime modules

- `app/server.py`: FastAPI webhooks, incoming WhatsApp calls and outbound case callbacks.
- `app/agent.py`: standard Pipecat pipeline, Gemini settings, greetings and tool hooks.
- `app/store.py`: tenant-scoped call reports, private stereo recordings and heartbeat.
- `app/tools.py`: business tools, permissions, Neon queries and Meta requests.

The pipeline follows the [Pipecat WhatsApp example](https://github.com/pipecat-ai/pipecat-examples/tree/main/whatsapp):
transport input -> user context -> Gemini Live -> transport output -> audio buffer
-> assistant context. Pipecat handles audio conversion, pacing, interruptions,
function-call responses and Gemini reconnection. A small outbound signaling adapter
uses SmallWebRTCConnection's public peer connection. Raw Gemini usage metadata is
preserved by one service override because the portal displays provider-specific fields.

The greeting asks for English, Sinhala or Tamil. Showcase agents use Leda and
Sri Lankan accent instructions; Talent360 retains its own settings. Timed local
speech detection has been removed; Gemini controls turn detection and interruptions.
The prompt retains the instruction to check in when a caller goes silent, but there
is no guaranteed five-second check-in with server-only VAD.

## Configuration and operation

Keep credentials in `.env`: `GEMINI_API_KEY`, `DATABASE_URL`, `PHONE_NUMBER_ID`,
`WHATSAPP_ACCESS_TOKEN`, `VERIFY_TOKEN`, and AWS recording access. Set
`WHATSAPP_APP_SECRET` to verify Meta webhook signatures. Never print credentials.
`VOICE_AGENT_ID` pins the runtime to an agent owning the configured phone number;
startup derives its tenant and concurrency limit from Neon.

`/` reports warming_up, ready or error. Meta calls `/webhook`. The temporary
portal `/cases` page submits signed `/cases/call` requests with
`PORTAL_DEMO_ENABLED=1` and the same `DEMO_SIGNING_SECRET` in the runtime and portal.
Existing permission starts the selected callback immediately; otherwise the approved
`serendibai_demo_call` template requests permission, and acceptance starts it.
Callbacks are limited to five minutes. Ordinary inbound calls use the host agent.
The older generic demo API and text-based CASE selection have been removed.
Historical database tables and records remain untouched.

`uv sync --locked` installs dependencies. `uv run pytest` runs checks, including a
private S3 recording through Pipecat/Gemini without dialing a real number.
`./deploy.sh --env local` starts on port 8081 and may connect the production tunnel;
do not start it alongside the hosted service. `--env remote` is used by
`voice-agent.service` on the existing Mumbai EC2 host. Production requires uv,
Python 3.12, ffmpeg and cloudflared. Upload source, run `uv sync --locked`, and
restart the service only during the agreed cutover. Remove deleted application
files from the deployment directory as part of that cutover.

Pushes to `main` deploy this directory to EC2 and restart `voice-agent.service`.

Shutdown stops calls, flushes recordings and drains database writes before closing
HTTP clients. Recordings remain MP3 in private S3, caller left and agent right.

## Deferred live validation

This restructure is not a hosted runtime deployment. Before cutover acceptance:

1. Call Talent360 inbound; confirm its original greeting and selected language.
2. Request one showcase callback, testing existing permission and permission acceptance.
3. Confirm Leda, all three language choices and silence behavior.
4. Interrupt a long response and check that queued speech stops promptly.
5. Complete a ticket or appointment; inspect the persisted tenant record.
6. Open the transcript and recording; check both speakers, timing and usage fields.

Transcripts: https://portal.serendibai.lk/dashboard
