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
- `OMNIVOICE_MODEL_ID`: Optional. Defaults to `k2-fsa/OmniVoice`.
- `OMNIVOICE_DEVICE`: Optional. Defaults to `cpu`.
- `OMNIVOICE_DTYPE`: Optional. Defaults to `float32`.
- `OMNIVOICE_NUM_STEP`: Optional. Defaults to `16`.
- `OMNIVOICE_SPEED`: Optional. Defaults to `1.12`.
- `OMNIVOICE_INSTRUCT_EN`, `OMNIVOICE_INSTRUCT_SI`, `OMNIVOICE_INSTRUCT_TA`: Optional voice-design prompts per language when using OmniVoice.
- `OMNIVOICE_REF_AUDIO_EN`, `OMNIVOICE_REF_AUDIO_SI`, `OMNIVOICE_REF_AUDIO_TA`: Optional reference audio paths for voice cloning.
- `OMNIVOICE_REF_TEXT_EN`, `OMNIVOICE_REF_TEXT_SI`, `OMNIVOICE_REF_TEXT_TA`: Optional reference transcripts to avoid auto-transcription overhead during cloning.

## Voice Pipeline
The voice call pipeline now supports two output modes:

- `gemini_live`: existing Gemini native-audio pipeline, where STT, reasoning, and TTS all stay inside Gemini Live.
- `omnivoice`: Gemini Live still handles live audio input and reasoning, but it returns text only. The server then synthesizes the outbound audio locally with OmniVoice.

`omnivoice` gives you tighter control over voice quality and voice design, but it is much heavier operationally than the Gemini-native path. On small CPU-only instances, model load time and synthesis latency can dominate the call experience.
