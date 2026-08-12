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
| `scripts/finetune_llm.py` | preparing data or reproducing the Gemma fine-tune |
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

Never commit `.env`. For lightweight editing:

```bash
uv sync
uv run python -m compileall -q app scripts tests
```

## Production quality gate

Every pull request into `main` rents the cheapest compatible Vast.ai RTX 30/40-series GPU with at least 16 GB VRAM, production-grade advertised network/disk performance, and at least 99% reliability. It copies the repository once, then makes a separate SSH call for each integration-test stage. A stage prints status `200` when it passes; any other result stops the quality gate. Dependency setup has a ten-minute ceiling so a poor host cannot consume the whole job timeout:

1. rent and connect to the cheapest compatible verified GPU offer with at least 16 GB VRAM, using the pinned virtual-environment CUDA runtime rather than the host image's incidental toolkit version;
2. the FastAPI webhook starts and completes Meta's verification handshake;
3. the pinned Sinhala ASR transcribes a fixed, revision-pinned call-center recording;
4. the production Gemma wrapper verifies CUDA offload, then answers call-center requests in English, Sinhala, and Tamil with its real system/tool prompt; Gemini grades language, usefulness, tone, groundedness, and safety;
5. the model selects and the runtime executes both `search_properties` and `book_appointment` against an in-memory copy of the production tool contract;
6. OmniVoice produces non-silent, plausible-duration speech in all three languages and stays faster than real time;
7. a representative full-model workload continuously samples GPU memory/utilization and fails on CUDA OOM or peak VRAM above 16,384 MiB.

Only a clean archive of committed files and the read-only Hugging Face token are sent to the host; the local `.env` and its other credentials are never copied. The Gemini judge key stays on the GitHub runner. The workflow uploads its combined JSON report and WAV samples for 14 days. Its cleanup step destroys the Vast instance and removes the temporary SSH key even after a failure. Configure repository secrets `VAST_AI_API_KEY` (prefer a scoped CI key with user/instance/SSH-key permissions), `HF_TOKEN` (read-only), and `GEMINI_API_KEY`, then require the `GPU integration tests` status check on `main`. Run one stage on a CUDA host with:

```bash
uv sync --extra server --frozen
PYTHONPATH=. uv run --extra server python tests/gpu_integration.py asr
```

## 1. Vast.ai WhatsApp webserver

The server needs one verified RTX 4070/30-series GPU with at least 16 GB VRAM and a direct SSH port. From your Mac:

```bash
uv sync
chmod +x scripts/vast.sh
./scripts/vast.sh rent
```

The command selects the cheapest matching offer, rents it, syncs this repository plus `.env`, installs the `server` dependency profile, starts supervised FastAPI and ngrok services, waits for model prewarming, and prints the `/webhook` URL. Preview the offer without renting:

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

For the separate Sinhala conversation model, run the single reproducible Gemma 4 E4B LoRA job on a CUDA host with at least 24 GB VRAM:

```bash
uv run scripts/finetune_llm.py \
  --scripts-csv /workspace/voice_scripts.csv
```

The script turns the human-written call-center lines into grounded caller/agent pairs, validates a grouped train/validation split, fine-tunes E4B with response-only LoRA, and pushes the private dataset and adapter to Hugging Face. Use `--prepare-only` or `--train-only` to resume one stage without repeating the other.

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

## Major changes

Add a dated entry here whenever a change affects the production model, training data or workflow, deployment architecture, database/API contract, or caller-visible behavior. Routine refactors and dependency bumps do not need entries.

### 2026-08-12

- Tightened the model-level multilingual/tool contract after the production quality gate exposed Sinhala hesitation and Tamil-to-Sinhala drift. Added protocol support for Gemma's native tool-call envelope without adding intent or property keyword routing.
- Replaced the broad unit-test collection with staged GPU integration tests for webhook, real ASR, multilingual system-prompted Gemma plus both property tools, audible/latency-checked OmniVoice, and a representative 16 GiB GPU load budget. GitHub Actions rents one temporary Vast GPU, runs each stage over a separate SSH call with status `200` on success, AI-grades the three LLM languages with Gemini, preserves the report/audio samples, and always destroys the instance.
- Reviewed the latest production call and corrected caller-visible behavior without adding intent keyword routes: relaxed the multilingual system style, added interruptible language-aware progress speech before slow/tool work and every ten seconds thereafter, and prevented invented personal identities.
- Calmed OmniVoice delivery by selecting a declarative real-estate reference from the same pinned V5 dataset, using a neutral terminal cadence, and selecting Sinhala, Tamil, or English per turn. Added TTS-boundary Sinhala number verbalization and Sri Lankan place-name pronunciation while leaving stored/model text unchanged.
- Expanded the seeded Neon inventory from 4 to 16 properties across 15 locations, made seed changes synchronize existing rows, and added one safe retry for transient Neon writes and WhatsApp call-control requests.
- Replaced the stock conversation model with the private, revision-pinned Gemma 4 E4B Sinhala call-center Q4_K_M model. Added the reproducible dataset/LoRA training script and kept datasets, adapters, converted weights, and checkpoints in cloud storage.
- Moved the Vast.ai webhook and ngrok tunnel to supervised, automatically restarting services.
- Fixed live audio turn handling after production testing: the system prompt is pre-cached, the greeting is shorter, end-of-turn silence is reduced, and outbound speech is queued before echo suppression so the bot no longer transcribes itself as the caller.
- Removed transcript keyword routing and canned intent responses. The fine-tuned model now owns language choice, acknowledgements, property intent/filter extraction, tool selection, tool-result interpretation, and final wording; runtime code only validates and executes tool calls.
- Expanded the reproducible LLM training data with model-level acknowledgement, language-choice, and named-property tool examples after production evaluation exposed overly proactive replies.

### 2026-08-11

- Reduced the repository to the lean voice runtime and one primary README. Archived legacy comparisons and large local artifacts to S3.
- Promoted revision-pinned OmniVoice V5 for speech generation and retained only its cloud dataset/model references locally.

### 2026-08-02

- Added durable WhatsApp call status and transcript storage in Neon.
- Added Neon-backed property search and viewing-booking tools plus the one-command Vast.ai deployment workflow.

Archived legacy comparisons, bundled voices, and the old local MLflow database are recoverable from `s3://serendibai-models/sl-chatbot-legacy/2026-08-11/`.
