# Gemini Live runtime notes

The production voice path is Gemini Live only:

```text
Inbound WhatsApp audio -> Gemini Live -> outbound WhatsApp audio
```

`app/voice/gemini_live.py` owns the Live session, audio signalling, playback,
transcription events, and Gemini function responses. `app/voice/portal.py`
handles the portal tools and call reporting. `app/voice/audio_archive.py`
stores the mixed call recording.

Keep the runtime small. Do not add local ASR, local LLM, local TTS, CUDA model
downloads, model servers, or hard-coded language-selection behavior. Gemini must
receive the caller's audio and use the system instruction to introduce itself,
select English/Sinhala/Tamil, and speak as a warm Sri Lankan woman.

`GEMINI_API_KEY` is required. Portal tools use the scoped runtime token;
WhatsApp calls need the existing Meta credentials. Never log or commit credentials.

Run `uv run pytest` after runtime changes. This single test sends a private S3
speech recording to Gemini Live and requires the configured AWS and Gemini keys.
`./deploy.sh --env local` starts the
voice runtime on this Mac and connects it to the portal. Gemini inference runs
remotely and the runtime has no local model workload. `--env remote` is reserved.
The customer portal and transcripts are at https://portal.serendibai.lk/dashboard.
The deployment script fetches scoped configuration and reports heartbeats through `app/portal_runtime.py`.
