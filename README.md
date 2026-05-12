# WhatsApp Webhook Server

This is a fast, lightweight, and modern FastAPI application designed to act as a webhook receiver for WhatsApp Business API. It handles token verification and incoming message/status updates.

It uses `uv` for lightning-fast package management and dependency resolution.

## Local Development
Requires Docker.

```bash
docker compose up --build
```

The server will be available at `http://localhost:8000`.

## Local Mac Hosting With OmniVoice
This mode is intended for temporary hosting from an Apple Silicon Mac, where the FastAPI app runs directly on the host so OmniVoice can use `mps`.

1. Copy `.env.local.example` to `.env` and fill in the WhatsApp and Google credentials.
2. Install the local-only TTS dependencies into the project environment:

```bash
uv pip install "torch>=2.11.0" omnivoice soundfile
```

3. Start the app on the host:

```bash
bash scripts/run_local_host.sh
```

4. Start the HTTPS reverse proxy:

```bash
docker compose -f docker-compose.local-proxy.yml up -d
```

The local proxy in [Caddyfile.local](/Users/nizar/Documents/Projects/sl_chatbot/Caddyfile.local:1) forwards `webhook.hervestudio.lk` to `host.docker.internal:8000`.

Important:
- Your public DNS must point to your home or office public IP, not the laptop's private LAN IP.
- Your router must forward inbound `80` and `443` to this Mac.
- Docker Desktop must be running for the local Caddy proxy.

## EC2 Deployment
To deploy this on an AWS Ubuntu EC2 instance, you can use the following script as your **User Data** when launching the instance. It will automatically install Docker, pull this repository, and spin up the server on port 80.

```bash
#!/bin/bash
# 1. Update packages and install git
apt-get update -y
apt-get install -y git curl

# 2. Install Docker using the convenience script
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Enable and start Docker service
systemctl enable docker
systemctl start docker
usermod -aG docker ubuntu

# 3. Clone and Run Application
cd /home/ubuntu
git clone https://github.com/nizarhaider/sl-chatbot.git

# Change directory and set permissions
cd sl-chatbot
chown -R ubuntu:ubuntu /home/ubuntu/sl-chatbot

# Build the docker container and spin it up detached (-d)
docker compose up -d --build
```

Make sure your EC2 Security Group allows inbound HTTP traffic on port 80 (0.0.0.0/0).

## Tests

```bash
uv run pytest -q
```

## Environment Variables
- `VERIFY_TOKEN`: The token you provide to the Meta Dashboard during webhook setup. Defaults to `my_secure_verify_token_123`.
- `WHATSAPP_ACCESS_TOKEN`: Meta WhatsApp Cloud API access token. `WHATSAPP_TOKEN` is also accepted for compatibility.
- `PHONE_NUMBER_ID`: WhatsApp phone number ID used for outbound messages and call actions.
- `GRAPH_API_VERSION`: Optional Meta Graph API version. Defaults to `v22.0`.
- `CHAT_AGENT_MODEL`: Text chatbot model. Defaults to `gemini-2.5-flash-lite`.
- `CHAT_AGENT_BUSINESS_NAME`: Business name used by the WhatsApp text chatbot. Defaults to `SLT Mobitel`.
- `CHAT_AGENT_BUSINESS_DESCRIPTION`: Short business context used by the WhatsApp text chatbot.
- `CHAT_AGENT_ESCALATION_MESSAGE`: Where the chatbot should send users when it cannot answer a business-specific question. Defaults to `visit slt.lk or call 1212`.
- `CHAT_AGENT_SYSTEM_PROMPT`: Optional full override for the chatbot system prompt.
- `CHAT_AGENT_MAX_OUTPUT_TOKENS`: Optional chatbot response cap. Defaults to `150`.
- `CHAT_AGENT_TEMPERATURE`: Optional chatbot temperature. Defaults to `0.8`.
- `VOICE_OUTPUT_PROVIDER`: `gemini_live`, `omnivoice_remote`, or `omnivoice_local`. `omnivoice` is treated as `omnivoice_remote` for compatibility.
- `VOICE_PIPELINE_MODE`: `live` or `gemini_turn`. Defaults to `live`.
- `OMNIVOICE_REMOTE_BASE_URL`: Optional. Defaults to `https://k2-fsa-omnivoice.hf.space`.
- `OMNIVOICE_REMOTE_API_NAME`: Optional. Defaults to `_design_fn`.
- `OMNIVOICE_REMOTE_TIMEOUT_SECONDS`: Optional. Defaults to `120`.
- `OMNIVOICE_NUM_STEP`: Optional. Defaults to `16`.
- `OMNIVOICE_GUIDANCE_SCALE`: Optional. Defaults to `2.0`.
- `OMNIVOICE_SPEED`: Optional. Defaults to `1.12`.
- `OMNIVOICE_LOCAL_MODEL`: Optional. Defaults to `k2-fsa/OmniVoice`.
- `OMNIVOICE_LOCAL_DEVICE`: Optional. Defaults to auto-detected `mps`, `cuda:0`, or `cpu`.
- `OMNIVOICE_LOCAL_DTYPE`: Optional. Defaults to `float16`.
- `OMNIVOICE_ENGLISH_ACCENT`: Optional. Defaults to `Indian Accent / 印度口音`.
- `OMNIVOICE_GENDER`, `OMNIVOICE_AGE`, `OMNIVOICE_PITCH`, `OMNIVOICE_STYLE`, `OMNIVOICE_CHINESE_DIALECT`: Optional remote voice-design controls.

## Voice Pipeline
The voice call pipeline now supports these modes:

- `gemini_live`: existing Gemini native-audio pipeline, where STT, reasoning, and TTS all stay inside Gemini Live.
- `omnivoice`: Gemini Live still handles live audio input and reasoning, but it returns text only. The server then synthesizes outbound audio through the official public OmniVoice Hugging Face Space.
- `omnivoice_local`: Gemini Live still handles live audio input and reasoning, but the server synthesizes outbound audio locally through OmniVoice running on the host machine.
This remote OmniVoice mode keeps the EC2 service lightweight enough for small instances, but it adds a dependency on the Hugging Face-hosted OmniVoice Space for TTS availability and latency.

## Agent Layout
Text chat and voice calls are separated:

- [app/chat_agent](/Users/nizar/Documents/Projects/sl_chatbot/app/chat_agent): WhatsApp text chatbot logic.
- [app/voice_agent](/Users/nizar/Documents/Projects/sl_chatbot/app/voice_agent): WhatsApp voice call pipeline, WebRTC audio handling, Gemini Live integration, and TTS.
