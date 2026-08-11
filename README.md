# SerendibAI WhatsApp Voice Bot

A small, voice-only WhatsApp runtime. Calls stay local to the GPU host:

`WhatsApp WebRTC -> VAD -> Sinhala Whisper -> Gemma -> OmniVoice V5 -> WebRTC`

Property search, bookings, call status, and transcripts are stored in Neon. Model weights and the exact V5 reference clip are downloaded from pinned Hugging Face revisions. No dataset, checkpoint, generated audio, or dashboard bundle belongs in this repository.

## Files

| File | Change it when… |
| --- | --- |
| `app/config.py` | changing model pins, prompts, voice settings, or turn timing |
| `app/models.py` | changing Whisper, Gemma, or OmniVoice loading/inference |
| `app/pipeline.py` | changing the conversational turn flow |
| `app/audio.py` | changing VAD or WebRTC audio conversion |
| `app/database.py` | changing Neon call records or property tools |
| `app/whatsapp.py` | changing webhook, Graph API, or WebRTC behavior |
| `app/main.py` | changing startup, shutdown, logging, or health checks |
| `scripts/vast.sh` | renting/configuring the webhook GPU |
| `scripts/finetune.sh` | reproducing an OmniVoice fine-tune |
| `scripts/infer.py` | running voice-only Gradio inference locally |

The runtime configuration is intentionally centralized in `app/config.py`. Secrets go only in `.env`.

## First setup

Install [uv](https://docs.astral.sh/uv/) and create `.env`:

```dotenv
HF_TOKEN=...
VASTAI_API_KEY=...
NGROK_AUTH_TOKEN=...
VERIFY_TOKEN=...
WHATSAPP_ACCESS_TOKEN=...
PHONE_NUMBER_ID=...
DATABASE_URL=postgresql://...
```

Never commit `.env`. For tests and lightweight editing:

```bash
uv sync
uv run pytest -q
```

## 1. Vast.ai WhatsApp webserver

The server needs one verified RTX 4070/30-series GPU with at least 16 GB VRAM and a direct SSH port. From your Mac:

```bash
uv sync
chmod +x scripts/vast.sh
./scripts/vast.sh rent
```

The command selects the cheapest matching offer, rents it, syncs this repository plus `.env`, installs the `server` dependency profile, starts FastAPI and ngrok in `tmux`, waits for model prewarming, and prints the `/webhook` URL. Preview the offer without renting:

```bash
DRY_RUN=true ./scripts/vast.sh rent
```

Configure an instance you already rented, inspect instances, or destroy one:

```bash
./scripts/vast.sh setup HOST SSH_PORT
./scripts/vast.sh list
./scripts/vast.sh destroy INSTANCE_ID
```

The instance remains billable until `destroy` succeeds. Server logs live only at `/workspace/sl-chatbot/run_logs/server.log`; durable call records remain in Neon.

## 2. Fine-tune OmniVoice

Run this on a fresh CUDA Vast.ai instance. The script pins the OmniVoice source and V5 dataset revisions, downloads only rows listed in `train.jsonl` (never the holdouts), runs official tokenization and training, and leaves output under `/workspace`:

```bash
cd /workspace/sl-chatbot
export HF_TOKEN=...  # omit this if the instance already supplies it
bash scripts/finetune.sh
```

Optional run settings:

```bash
RUN_NAME=serendib-v6 STEPS=1500 bash scripts/finetune.sh
```

Use a new `RUN_NAME` for every run. The script reports `training_audio_minutes` before training and `Training wall-clock seconds` afterward; these are different measurements. Review the output, then upload the chosen final model directory to Hugging Face rather than committing it here:

```bash
hf upload YOUR_ACCOUNT/YOUR_MODEL /workspace/YOUR_RUN_OUTPUT
```

## 3. Local voice-model inference only

This path loads OmniVoice V5 and the pinned reference clip—no WhatsApp, Whisper, Gemma, or Neon. CUDA, Apple Silicon MPS, and CPU are selected automatically:

```bash
uv sync --extra infer
uv run --extra infer python scripts/infer.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). Models are cached by Hugging Face outside the repository. Use `--host 0.0.0.0 --port 7860` only when you deliberately want LAN access.

## Manual runtime checks

```bash
uv sync --extra server
uv run --extra server uvicorn app.main:app --host 127.0.0.1 --port 8081
curl -fsS http://127.0.0.1:8081/
curl -fsS 'http://127.0.0.1:8081/webhook?hub.mode=subscribe&hub.verify_token=YOUR_VERIFY_TOKEN&hub.challenge=12345'
```

Archived legacy comparisons, bundled voices, and the old local MLflow database are recoverable from `s3://serendibai-models/sl-chatbot-legacy/2026-08-11/`.
