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
