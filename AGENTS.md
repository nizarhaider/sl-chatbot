# Pipecat voice runtime

The production path is WhatsApp -> Pipecat SmallWebRTC -> Gemini Live -> WhatsApp.
Keep runtime logic in four flat modules: server, agent, store and tools.
Keep shared prompts/greetings in prompts.py and editable settings in config.py.
Follow Pipecat's WhatsApp example and use its existing pipeline, audio buffer,
context aggregators and transport instead of rebuilding them.

Use Gemini 3.8 Live with server-side VAD. Do not add local ASR, LLM, TTS,
VAD models, CUDA downloads or a second voice pipeline. Keep language selection
in the Gemini greeting and instructions. Showcase agents use Leda with a
Sri Lankan accent prompt and no explicit locale. Preserve tenant agent settings.

Keep the outbound signaling adapter small. Preserve signed callbacks, Meta
permission checks, tenant boundaries, tool permissions and private recordings.
Use background database writes and uploads; drain them on shutdown.
The narrow Gemini usage-metadata override is tied to pinned Pipecat 1.14.0.

Run `uv run pytest` after runtime changes. The live test sends a private S3
speech fixture through the actual Pipecat/Gemini pipeline and requires audio,
transcripts, usage and a stereo recording. It does not place a WhatsApp call.
`./deploy.sh --env local` starts the local runtime and production tunnel, so do
not run it beside the hosted service. Never print credentials.
