# WhatsApp Webhook Server

This is a fast, lightweight, and modern FastAPI application designed to act as a webhook receiver for WhatsApp Business API. It handles token verification and incoming message/status updates.

It uses `uv` for lightning-fast package management and dependency resolution.

## Local Development
Requires Docker.

```bash
docker compose up --build
```

The server will be available at `http://localhost:8000`.

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

## Environment Variables
- `VERIFY_TOKEN`: The token you provide to the Meta Dashboard during webhook setup. Defaults to `my_secure_verify_token_123`.
- `VOICE_OUTPUT_PROVIDER`: `gemini_live` or `omnivoice`. Defaults to `gemini_live`.
- `OMNIVOICE_REMOTE_BASE_URL`: Optional. Defaults to `https://k2-fsa-omnivoice.hf.space`.
- `OMNIVOICE_REMOTE_API_NAME`: Optional. Defaults to `_design_fn`.
- `OMNIVOICE_REMOTE_TIMEOUT_SECONDS`: Optional. Defaults to `120`.
- `OMNIVOICE_NUM_STEP`: Optional. Defaults to `16`.
- `OMNIVOICE_GUIDANCE_SCALE`: Optional. Defaults to `2.0`.
- `OMNIVOICE_SPEED`: Optional. Defaults to `1.12`.
- `OMNIVOICE_ENGLISH_ACCENT`: Optional. Defaults to `Indian Accent / 印度口音`.
- `OMNIVOICE_GENDER`, `OMNIVOICE_AGE`, `OMNIVOICE_PITCH`, `OMNIVOICE_STYLE`, `OMNIVOICE_CHINESE_DIALECT`: Optional remote voice-design controls.

## Voice Pipeline
The voice call pipeline now supports two output modes:

- `gemini_live`: existing Gemini native-audio pipeline, where STT, reasoning, and TTS all stay inside Gemini Live.
- `omnivoice`: Gemini Live still handles live audio input and reasoning, but it returns text only. The server then synthesizes outbound audio through the official public OmniVoice Hugging Face Space.

This remote OmniVoice mode keeps the EC2 service lightweight enough for small instances, but it adds a dependency on the Hugging Face-hosted OmniVoice Space for TTS availability and latency.
