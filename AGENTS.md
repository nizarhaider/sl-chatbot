# Gemini Live runtime notes

The production voice path is Gemini Live only:

```text
Inbound WhatsApp audio -> Gemini Live -> outbound WhatsApp audio
```

`app/voice/gemini_live.py` owns the Live session, manual audio activity
signalling, native audio playback, transcription events, and Gemini function
responses. `app/voice/tools.py` owns property search, booking, and WhatsApp
message tools. `app/voice/audio_archive.py` stores the mixed call recording.

Keep the runtime small. Do not add local ASR, local LLM, local TTS, CUDA model
downloads, model servers, or hard-coded language-selection behavior. Gemini must
receive the caller's audio and use the system instruction to introduce itself,
select English/Sinhala/Tamil, and speak as a warm Sri Lankan woman.

`GEMINI_API_KEY` is required. Tool calls additionally need the existing Neon,
Pinecone, and WhatsApp environment variables. Never log or commit credentials.

Run `uv run pytest` after runtime changes. `scripts/setup_vastai.sh` deploys the
current branch to Vast and configures only the webhook and Cloudflare services.
The hosted transcript dashboard is https://dashboard.serendibai.lk/dashboard/calls.
