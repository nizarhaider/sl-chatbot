# SerendibAI WhatsApp Voice Bot

A local-GPU WhatsApp voice agent with Google ADK orchestration:

`WhatsApp WebRTC -> Sinhala Whisper -> ADK + Gemma -> OmniVoice V5 -> WebRTC`

Neon owns calls, transcripts, properties, and appointments. Hugging Face owns
datasets and model weights. This repository contains only runtime code, small
training entrypoints, CI, and this runbook.

## Code map

| Path | Purpose |
| --- | --- |
| `app/config.py` | Model pins, prompt, voice, and timing settings |
| `app/models.py` | ASR, LLM, and TTS adapters |
| `app/agent.py` | Google ADK sessions, local Gemma adapter, and function tools |
| `app/pipeline.py` | Voice turn, interruption, and playback flow |
| `app/database.py` | Neon calls, transcripts, property search, and booking |
| `app/whatsapp.py` | Meta webhook and WebRTC signaling |
| `scripts/vast.sh` | Rent, deploy, inspect, or destroy production |
| `scripts/finetune*.{sh,py}` | Reproduce voice or LLM fine-tuning |
| `scripts/infer.py` | Local voice-only Gradio UI |
| `tests/gpu_integration.py` | Staged production quality gate |

Edit `app/config.py` first for behavior changes. Google ADK owns conversational
history, agent state, and function dispatch for each active call. Do not add
intent keyword routes or canned property logic; Gemma chooses ADK tools and the
existing database service validates their arguments and effects.

## Setup

Install [uv](https://docs.astral.sh/uv/) and create an uncommitted `.env`:

```dotenv
HF_TOKEN=...
VASTAI_API_KEY=...
NGROK_AUTH_TOKEN=...
VERIFY_TOKEN=...
WHATSAPP_ACCESS_TOKEN=...
PHONE_NUMBER_ID=...
DATABASE_URL=postgresql://...
```

```bash
uv sync
uv run python -m compileall -q app scripts tests
```

## Deploy the WhatsApp server

```bash
./scripts/vast.sh rent
```

This rents a fast, reliable 16 GB+ Vast GPU from the standard PyTorch template,
sends only committed app files and required runtime credentials, downloads the
locked prebuilt Linux wheels and pinned Hugging Face model snapshots, then starts
supervised FastAPI/ngrok processes and prints the webhook URL. No custom image is
built or pulled. Download provisioning is reported separately; the subsequent
server setup and model-loading phase is limited to five minutes. After the new
server is healthy, any older `serendibai-whatsapp` instance is destroyed.

```bash
DRY_RUN=true ./scripts/vast.sh rent        # show the selected offer
./scripts/vast.sh list                     # inspect instances
./scripts/vast.sh setup HOST SSH_PORT      # redeploy an existing instance
./scripts/vast.sh destroy INSTANCE_ID      # stop billing
```

Remote logs are at `/workspace/sl-chatbot/run_logs/server.log`. `uv` keeps its
download cache at `/workspace/.cache/uv`; `--no-build` makes deployment fail
instead of compiling a missing wheel. Model downloads are cached separately at
`/workspace/.cache/huggingface` for retries on the same instance.

## Fine-tune

OmniVoice V5 reproduction on a CUDA host:

```bash
HF_TOKEN=... bash scripts/finetune.sh
RUN_NAME=serendib-v6 STEPS=1500 bash scripts/finetune.sh
```

The voice script downloads only the revision-pinned cloud manifest, reports
training audio minutes before training, and reports wall-clock seconds after
training. Keep its output in cloud storage, not Git.

Gemma 4 E4B LoRA reproduction from the private, versioned cloud dataset:

```bash
HF_TOKEN=... uv run scripts/finetune_llm.py
```

Use `--dataset` or `--output` only when intentionally creating a new version.
Dataset preparation was a one-off pipeline and is not duplicated in production
source; curate and version new data in Hugging Face first.

## Run voice inference locally

This loads only OmniVoice—no WhatsApp, ASR, LLM, or Neon:

```bash
uv sync --extra infer
uv run --extra infer python scripts/infer.py
```

Open <http://127.0.0.1:7860>. Add `--host 0.0.0.0` only for intentional LAN
access.

## Quality gate

Every pull request to `main` runs the required `GPU integration tests` check.
It rents one temporary 16 GB+ Vast GPU and verifies:

1. webhook verification;
2. Sinhala ASR against pinned audio;
3. ADK-managed English, Sinhala, and Tamil Gemma responses with a Gemini judge;
4. ADK session memory plus model-selected property search and booking function calls;
5. audible, faster-than-real-time OmniVoice output in all three languages;
6. a representative load below 16,384 MiB VRAM.

The workflow uploads its JSON report and audio samples, then always destroys the
test instance. It sends only a read-only Hugging Face token to the GPU. Required
GitHub secrets are `VAST_AI_API_KEY`, `HF_TOKEN`, and `GEMINI_API_KEY`.

The same workflow runs a cost guard at `00:00, 02:00, ... UTC` and destroys every
Vast instance on the account. This intentionally includes production and any CI
instance active at that moment.

## Data ownership

- Add and edit listings in Neon/dashboard. Server startup never seeds or
  overwrites inventory.
- Calls and transcripts are durable in Neon; Google ADK keeps each active call's
  session events and tool state in memory for the life of that call.
- Model weights, datasets, checkpoints, and generated audio stay in cloud
  storage.
- Secrets stay in `.env` or GitHub Secrets and are never committed.

## Major changes

- **2026-08-12:** Moved local Gemma orchestration to Google ADK. ADK now owns
  per-call session memory, model/tool events, and dispatch of the existing Neon
  property search and appointment functions.
- **2026-08-12:** Added a two-hour GitHub cost guard that destroys every Vast
  instance. Rotated multilingual progress phrases, delayed non-tool filler,
  preserved tool results across turns, retried malformed tool calls, and tightened
  the model contract for caller-selected appointment dates and confirmed bookings.
- **2026-08-12:** Removed boot-time property schema/seeding after confirming the
  16 production listings in Neon. Reduced the LLM trainer to its reproducible
  cloud-dataset job, merged the CI judge into the staged test runner, restricted
  deployments to committed runtime files/credentials, and condensed this
  runbook.
- **2026-08-12:** Promoted the fine-tuned Gemma 4 E4B model; added multilingual
  progress speech, neutral voice cadence, Sinhala number/place pronunciation,
  generic model-owned tool calling, and the temporary-GPU quality gate.
- **2026-08-11:** Reduced the repository to the voice runtime and archived large
  legacy assets at `s3://serendibai-models/sl-chatbot-legacy/2026-08-11/`.
- **2026-08-02:** Added Neon transcript storage, property tools, and one-command
  Vast deployment.
