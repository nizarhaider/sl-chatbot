# Agent Handoff

This repo runs a WhatsApp voice bot on a Vast.ai GPU machine.

The current voice path is:

```text
WhatsApp Cloud webhook -> FastAPI /webhook
WhatsApp Calling SDP offer -> aiortc peer connection
Inbound WhatsApp audio -> local VAD -> Gemini Live (audio in, text out)
Gemini text -> RealtimeTTS OmniVoice
OmniVoice PCM -> outbound aiortc audio track -> WhatsApp call
```

This file is the operator handoff for bringing the system up, debugging it, and understanding which code does what.

## Current Reality

The old host in earlier notes is stale.

The current machine I most recently set up is:

```bash
ssh -i ~/.ssh/vastai_ssh_file -p 64420 root@178.232.64.66
cd /workspace/sl-chatbot
```

Public webhook target remains:

```text
https://webhook.hervestudio.lk/webhook
```

Meta webhook verify token used in local checks:

```text
my_secure_verify_token_123
```

Current state on that machine at handoff time:

- App repo exists at `/workspace/sl-chatbot`
- Python deps are installed in `/workspace/sl-chatbot/.venv`
- Webhook server is running in detached `tmux` session `sl-webhook`
- It serves locally on `0.0.0.0:8090`
- A Cloudflare tunnel process is also running on the machine
- The public hostname still returns Cloudflare `503`, so the box is healthy but public routing is still blocked at Cloudflare config level

## What Is Working And What Is Not

Working:

- Local FastAPI root health check
- Local webhook verification endpoint
- OmniVoice model prewarm
- aiortc/voice code is present and imports cleanly
- Gemini Live turn pipeline code is wired into the call flow

Not working yet:

- `https://webhook.hervestudio.lk/webhook` is still returning `503`
- That means Meta cannot currently deliver real call webhooks to this box
- This is not an app-server failure anymore; it is a Cloudflare tunnel or public hostname routing problem

## Repo Layout That Matters

Important files:

```text
app/main.py
app/webhooks/whatsapp.py
app/services/webrtc.py
app/services/whatsapp_api.py
app/voice_agent/agent.py
app/voice_agent/gemini_turn_pipeline.py
scripts/run_local_host.sh
pyproject.toml
uv.lock
README.md
AGENTS.md
```

Do not sync these unless explicitly needed:

```text
.env
.venv/
run_logs/
*.log
*.pid
training_custom_tts/datasets/
tts_debug_latest/
debug_*.wav
official_*.wav
realtimetts_*_smoke.wav
app/voices/ unless intentionally changing the production reference voice
```

## Code Walkthrough

### `app/main.py`

This is the FastAPI entrypoint.

What it does:

- Loads `.env`
- Configures global logging
- Adds a filtered rotating log file for important call events
- Prewarms OmniVoice on startup
- Mounts the WhatsApp webhook router
- Exposes `GET /` for a simple health check

Important behavior:

- `ImportantEventFilter` only writes important lines into `run_logs/important.log`
- `prewarm_tts()` imports `voice_agent` and calls `voice_agent.prewarm_tts()`
- Startup uses FastAPI lifespan so TTS prewarm runs before the service is considered ready

### `app/webhooks/whatsapp.py`

This is the public webhook surface.

Routes:

- `GET /webhook`
  - Used by Meta webhook verification
  - Expects `hub.mode`, `hub.verify_token`, `hub.challenge`
  - Returns challenge text if token matches
- `POST /webhook`
  - Receives WhatsApp events
  - Handles text messages, status updates, and call events
- `POST /send-message`
  - Internal backend endpoint for sending WhatsApp text messages

Call flow in `POST /webhook`:

- Iterates through `entry -> changes -> value`
- Looks for `calls`
- On `event == "connect"` with `session.sdp_type == "offer"`:
  - logs the call
  - extracts the SDP offer
  - starts `webrtc_service.handle_offer(call_id, sdp_offer, caller_phone)` in the background
- On `event == "terminate"`:
  - closes and removes the peer connection immediately

Text flow:

- If a WhatsApp text message arrives, `handle_text_message()` is started in the background
- If `app.chat_agent` is not installed, the code still sends a fallback text reply:
  - `"Text chat is not configured on this deployment. Please call us to continue."`
- This fallback is only for text chat, not the voice path

### `app/services/webrtc.py`

This is the WhatsApp call SDP and aiortc bridge.

What `WebRTCService.handle_offer()` does:

1. Closes any existing peer connection for the same `call_id`
2. Creates a new `RTCPeerConnection`
3. Registers connection state change cleanup
4. Creates one outbound `RealtimeAudioTrack`
5. Adds that outbound audio track to the peer connection
6. When an inbound audio track arrives:
   - logs the event
   - starts `voice_agent.process_audio(call_id, phone, track, output_track)`
7. Applies the inbound SDP offer from WhatsApp
8. Creates an SDP answer
9. Refines the answer SDP with `_refine_sdp()`
10. Sends `pre_accept` via the WhatsApp Calling API
11. Sends `accept` via the WhatsApp Calling API

Why `_refine_sdp()` exists:

- WhatsApp calling is picky about SDP formatting
- The code normalizes items like:
  - fingerprint case
  - `a=mid`
  - `a=setup`
  - `a=group:BUNDLE`
  - origin address
- It also strips noisy SDP lines that are unnecessary or unwanted for this path

### `app/services/whatsapp_api.py`

This is the Graph API client.

Functions:

- `send_call_action(call_id, action, session=None)`
  - Sends `pre_accept` and `accept` to WhatsApp `/calls`
- `send_message(to, text)`
  - Sends a plain WhatsApp text message
- `send_image(to, image_url, caption="")`
  - Sends an image message

Important env used here:

- `WHATSAPP_ACCESS_TOKEN` or `WHATSAPP_TOKEN`
- `PHONE_NUMBER_ID`
- `GRAPH_API_VERSION` default is `v22.0`

### `app/voice_agent/agent.py`

This is the bridge between aiortc tracks and the Gemini/OmniVoice turn pipeline.

Key classes:

- `RealtimeAudioTrack`
  - Custom aiortc audio output track
  - Buffers PCM chunks from TTS
  - Resamples mono PCM to 48k stereo frames for outbound WebRTC
  - Emits silent frames if no audio is queued
- `VoiceAgent`
  - Owns active call tasks
  - Owns playback generation counters for interruption
  - Delegates turn handling to `GeminiTurnPipeline`

Important runtime behavior:

- `process_audio()` ensures only one active task per `call_id`
- `_interrupt_playback()` increments generation id and clears the output buffer
- `_prepare_tts_text()` normalizes whitespace and trims punctuation before TTS

### `app/voice_agent/gemini_turn_pipeline.py`

This is the core voice turn engine.

What it does:

1. Delays slightly before greeting
2. Plays a multilingual greeting through OmniVoice
3. Protects that greeting by discarding inbound audio briefly
4. Opens a Gemini Live session with text-only response modality
5. Uses local VAD to decide when the caller starts and stops speaking
6. Streams 16 kHz PCM chunks into Gemini Live
7. Waits for Gemini transcription and response text
8. Sends response text into OmniVoice
9. Streams synthesized PCM back into the outbound audio track

Important design decision:

- This is not separate STT and LLM anymore
- It uses one Gemini Live session for audio input and text output
- TTS is still local OmniVoice, not Gemini audio output

Key config values:

```bash
GEMINI_LIVE_MODEL=gemini-live-2.5-flash-preview
GEMINI_LIVE_API_VERSION=v1beta
TURN_INPUT_CHUNK_MS=40
TURN_SILENCE_THRESHOLD=1000
TURN_END_SILENCE_CHUNKS=50
TURN_GREETING_DELAY_SECONDS=1.2
TURN_GREETING_PROTECTION_MAX_SECONDS=1.5
```

Why the model is `gemini-live-2.5-flash-preview`:

- `gemini-live-2.5-flash-native-audio` was wrong for this architecture because we need text output into OmniVoice
- `gemini-3.1-flash-live-preview` returned connect-time `1011` internal errors in this setup
- `gemini-live-2.5-flash-preview` is the current text-capable live model used here

Key methods:

- `run()`
  - plays greeting
  - discards input during greeting protection
  - opens live session
  - enters `_run_live_session()`
- `_live_config()`
  - uses `response_modalities=[TEXT]`
  - uses the system prompt
  - disables automatic activity detection because local VAD controls turn boundaries
- `_run_live_session()`
  - resamples inbound audio to mono 16 kHz
  - computes RMS
  - when voice starts:
    - interrupts current playback
    - sends `ActivityStart`
  - while voice continues:
    - streams audio chunks into Gemini Live
  - when silence persists long enough:
    - sends `ActivityEnd`
    - handles the completed turn
- `_handle_live_turn()`
  - reads input transcription and response text from Gemini
  - logs transcript and response timing
  - synthesizes response through OmniVoice
- `_speak()`
  - routes TTS audio chunks into `RealtimeAudioTrack`
  - stops playback if a newer generation interrupts
- `_discard_input_audio()`
  - drops input frames for a short protected greeting window

### OmniVoice TTS Behavior

`RealtimeOmniVoiceTTS` wraps RealtimeTTS OmniVoice.

Important details:

- Prewarm loads the model once at startup
- `speak()` collects PCM chunks and forwards them to the outbound track
- The code subclasses OmniVoice engine behavior so generated audio is placed directly on a queue
- Output sample rate is taken from the engine and then up-converted by the outbound track if needed

## Environment Variables That Matter

Core runtime:

```bash
VERIFY_TOKEN=my_secure_verify_token_123
GOOGLE_API_KEY=...
WHATSAPP_ACCESS_TOKEN=...   # or WHATSAPP_TOKEN
PHONE_NUMBER_ID=...
GRAPH_API_VERSION=v22.0
```

Gemini Live:

```bash
GEMINI_LIVE_MODEL=gemini-live-2.5-flash-preview
GEMINI_LIVE_API_VERSION=v1beta
GEMINI_THINKING_LEVEL=
```

Turn control:

```bash
TURN_INPUT_CHUNK_MS=40
TURN_SILENCE_THRESHOLD=1000
TURN_END_SILENCE_CHUNKS=50
TURN_GREETING_DELAY_SECONDS=1.2
TURN_GREETING_PROTECTION_MAX_SECONDS=1.5
```

OmniVoice:

```bash
REALTIME_TTS_DEVICE=cuda:0
REALTIME_TTS_DTYPE=float16
REALTIME_TTS_NUM_STEPS=12,12
REALTIME_TTS_REF_AUDIO=app/voices/sample_si_lk.mp3
REALTIME_TTS_REF_TEXT=...
REALTIME_TTS_REF_LANGUAGE=si
REALTIME_TTS_PREWARM=true
REALTIME_TTS_DEBUG=false
```

Logging:

```bash
IMPORTANT_LOG_PATH=run_logs/important.log
IMPORTANT_LOG_MAX_BYTES=1048576
IMPORTANT_LOG_BACKUPS=3
```

## Setup Commands For A New Vast.ai Box

### 1. Basic machine check

```bash
ssh -i ~/.ssh/vastai_ssh_file -p <PORT> root@<HOST>
uname -a
nvidia-smi --query-gpu=name --format=csv,noheader
which uv
which git
which python3
```

### 2. Clone the repo

```bash
mkdir -p /workspace
git clone https://github.com/nizarhaider/sl-chatbot.git /workspace/sl-chatbot
cd /workspace/sl-chatbot
mkdir -p app/services scripts run_logs
```

### 3. Sync the current local working files

Run this from the local repo:

```bash
scp -P <PORT> -i ~/.ssh/vastai_ssh_file .env root@<HOST>:/workspace/sl-chatbot/.env
scp -P <PORT> -i ~/.ssh/vastai_ssh_file app/main.py root@<HOST>:/workspace/sl-chatbot/app/main.py
scp -P <PORT> -i ~/.ssh/vastai_ssh_file app/webhooks/whatsapp.py root@<HOST>:/workspace/sl-chatbot/app/webhooks/whatsapp.py
scp -P <PORT> -i ~/.ssh/vastai_ssh_file app/voice_agent/agent.py app/voice_agent/gemini_turn_pipeline.py root@<HOST>:/workspace/sl-chatbot/app/voice_agent/
scp -P <PORT> -i ~/.ssh/vastai_ssh_file app/services/webrtc.py app/services/whatsapp_api.py root@<HOST>:/workspace/sl-chatbot/app/services/
scp -P <PORT> -i ~/.ssh/vastai_ssh_file scripts/run_local_host.sh root@<HOST>:/workspace/sl-chatbot/scripts/
scp -P <PORT> -i ~/.ssh/vastai_ssh_file pyproject.toml uv.lock README.md AGENTS.md root@<HOST>:/workspace/sl-chatbot/
```

Do not accidentally scp `app/main.py` into repo root. It must land at `app/main.py`.

### 4. Install system packages and Python deps

```bash
ssh -i ~/.ssh/vastai_ssh_file -p <PORT> root@<HOST>
cd /workspace/sl-chatbot
apt-get update
apt-get install -y portaudio19-dev
uv sync
```

Why `portaudio19-dev` is needed:

- `pyaudio` build fails without it during `uv sync`

### 5. Compile-check important Python files

```bash
cd /workspace/sl-chatbot
.venv/bin/python -m py_compile \
  app/main.py \
  app/webhooks/whatsapp.py \
  app/services/webrtc.py \
  app/services/whatsapp_api.py \
  app/voice_agent/agent.py \
  app/voice_agent/gemini_turn_pipeline.py
```

## How To Run The Webhook Reliably

`nohup` and one-shot SSH backgrounding proved flaky on this host.

Use `tmux`.

### Start webhook in `tmux`

```bash
ssh -i ~/.ssh/vastai_ssh_file -p 64420 root@178.232.64.66
cd /workspace/sl-chatbot
tmux kill-session -t sl-webhook 2>/dev/null || true
pkill -f 'uvicorn app.main:app' || true
tmux new-session -d -s sl-webhook \
  "cd /workspace/sl-chatbot && \
   .venv/bin/dotenv -f .env run -- env \
   GEMINI_LIVE_MODEL=gemini-live-2.5-flash-preview \
   GEMINI_LIVE_API_VERSION=v1beta \
   REALTIME_TTS_DEVICE=cuda:0 \
   REALTIME_TTS_DTYPE=float16 \
   REALTIME_TTS_NUM_STEPS=12,12 \
   TURN_GREETING_PROTECTION_MAX_SECONDS=1.5 \
   IMPORTANT_LOG_PATH=run_logs/important.log \
   .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8090 --env-file .env \
   2>&1 | tee run_logs/webhook.log"
tmux ls
```

### Verify local app health

```bash
cd /workspace/sl-chatbot
ss -ltnp | egrep '(8090)' || true
curl -sS http://127.0.0.1:8090/
curl -sS 'http://127.0.0.1:8090/webhook?hub.mode=subscribe&hub.verify_token=my_secure_verify_token_123&hub.challenge=12345'
tail -n 80 run_logs/webhook.log
tail -n 80 run_logs/important.log
```

Expected:

- root endpoint returns JSON status
- verify endpoint returns `12345`
- `important.log` should contain `WEBHOOK_VERIFIED` after the verification curl

### Attach to the running webhook session

```bash
tmux attach -t sl-webhook
```

Detach with:

```bash
Ctrl-b d
```

## Cloudflare Tunnel Notes

The Cloudflare token we used is for tunnel:

```text
Name: sl-chatbot-webhook
Tunnel ID: 8fa9031f-0388-4a68-b0db-42ce2b80f2a9
Type: cloudflared
```

### Token-managed service install

On the box:

```bash
/opt/instance-tools/bin/cloudflared service install '<TOKEN>'
service cloudflared restart
service cloudflared status
pgrep -af cloudflared
```

What this does:

- installs a token-managed connector
- the route/origin mapping is controlled by Cloudflare remotely
- local `cloudflared.yml` is not the source of truth in this mode

Observed process:

```text
/opt/portal-aio/tunnel_manager/cloudflared --pidfile /var/run/cloudflared.pid --autoupdate-freq 24h0m0s tunnel run --token ...
```

### Explicit debug tunnel

I also started a manual debug connector in `tmux`:

```bash
tmux kill-session -t sl-tunnel 2>/dev/null || true
tmux new-session -d -s sl-tunnel \
  "/opt/portal-aio/tunnel_manager/cloudflared tunnel \
   --loglevel debug \
   --logfile /workspace/sl-chatbot/run_logs/tunnel-debug.log \
   --url http://localhost:8090 \
   run --token '<TOKEN>'"
tmux attach -t sl-tunnel
```

This proved:

- the connector itself can register to Cloudflare
- even with explicit `--url http://localhost:8090`, the public hostname still returned `503`
- therefore the remaining problem is not local origin reachability
- the remaining problem is the Cloudflare-side route/public hostname/origin mapping

### Public checks

From local machine:

```bash
curl -I -m 15 https://webhook.hervestudio.lk/webhook
```

Status meaning in this project:

- `200` or `400` from app path:
  - public routing works
- `503` from Cloudflare:
  - tunnel/hostname/origin mapping still broken
- `530` from Cloudflare:
  - tunnel/edge setup is even less complete

## Important Logs And What They Mean

### Main webhook log

```bash
tail -f /workspace/sl-chatbot/run_logs/webhook.log
```

Use this for:

- startup progress
- OmniVoice model downloads and prewarm
- general FastAPI logs
- full stack traces

### Filtered important log

```bash
tail -f /workspace/sl-chatbot/run_logs/important.log
```

Use this for:

- webhook verification
- call event receipt
- SDP handling
- inbound audio track start
- connection state changes
- transcript timings
- Gemini response timings
- TTS completion
- interruption behavior

### Tunnel debug log

```bash
tail -f /workspace/sl-chatbot/run_logs/tunnel-debug.log
```

Use this for:

- Cloudflare connector registration
- tunnel connection state
- debugging the explicit `sl-tunnel` session

## What A Healthy Call Path Should Look Like

Once Cloudflare is fixed, a successful inbound call should produce roughly this chain:

1. Meta delivers `POST /webhook`
2. `app.webhooks.whatsapp` logs:
   - `Received call event: connect ...`
3. `app.services.webrtc` logs:
   - `Processing SDP Offer`
   - `Incoming audio SDP`
   - `Answer audio SDP`
4. aiortc emits inbound audio track:
   - `Received audio track from WhatsApp ...`
5. `GeminiTurnPipeline.run()` plays greeting
6. `important.log` shows greeting timings and protected prompt discard
7. Caller speaks
8. `Turn VAD: Speech started`
9. Later `Turn VAD: Speech ended`
10. `Turn transcript for ...`
11. `Turn response for ...`
12. `RealtimeTTS complete ...`

If step 1 never happens:

- the issue is public routing, not the app code

## Current Known Problems

### 1. Cloudflare public route still broken

Symptoms:

- `curl -I https://webhook.hervestudio.lk/webhook` returns `503`
- local app is healthy
- both token-managed and explicit debug tunnel sessions can run

Conclusion:

- Cloudflare dashboard route/origin configuration still needs to be fixed

### 2. `nohup` launch attempts were unreliable on this host

Symptoms:

- background `uvicorn` sometimes exited or got cleaned up with SSH session behavior
- PID file handling via one-line SSH commands was also error-prone

Conclusion:

- use `tmux` for this machine

### 3. ALSA/JACK warnings are noisy but not the root problem

Symptoms:

- startup logs contain lots of ALSA/JACK messages

Conclusion:

- they do not prevent OmniVoice from loading
- they are not the reason public calls are failing

## Exact Commands I Used On The Current Box

SSH:

```bash
ssh -i ~/.ssh/vastai_ssh_file -p 64420 root@178.232.64.66
cd /workspace/sl-chatbot
```

Check local server:

```bash
ss -ltnp | egrep '(8090)' || true
curl -sS http://127.0.0.1:8090/
curl -sS 'http://127.0.0.1:8090/webhook?hub.mode=subscribe&hub.verify_token=my_secure_verify_token_123&hub.challenge=12345'
```

List sessions:

```bash
tmux ls
```

Attach webhook:

```bash
tmux attach -t sl-webhook
```

Attach tunnel:

```bash
tmux attach -t sl-tunnel
```

Check public route:

```bash
curl -I -m 15 https://webhook.hervestudio.lk/webhook
```

Check Cloudflare processes:

```bash
pgrep -af cloudflared
```

Check webhook logs:

```bash
tail -n 80 run_logs/webhook.log
tail -n 80 run_logs/important.log
```

## Next Step If You Pick This Up

The code side is in place.

The next meaningful step is not another local code change. It is to fix the Cloudflare-side public hostname routing so that:

```text
https://webhook.hervestudio.lk/webhook
```

actually forwards to this tunnel and this origin.

Once that returns an app-origin response instead of Cloudflare `503`, place a fresh WhatsApp call and watch:

```bash
tail -f /workspace/sl-chatbot/run_logs/important.log
```

If call events still do not appear after the public route is fixed, only then move back into app-level debugging.
