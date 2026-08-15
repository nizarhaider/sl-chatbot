"""Stable model pins and voice-turn settings.

Secrets belong in ``.env``. Change behavior here so manual edits remain obvious.
"""

# Local conversation model.
LLM_REPO = "2broke2code/serendib-gemma-4-e4b-sinhala-callcenter-gguf-v1"
LLM_REVISION = "e0174ff3e4959695e671e10f2302954ca6e55a9c"
LLM_FILENAME = "serendib-gemma-4-e4b-sinhala-callcenter-q4_k_m.gguf"
LLM_GPU_LAYERS = 42
LLM_CONTEXT = 2048
LLM_BATCH = 512
LLM_THREADS = 8
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 96
LLM_HISTORY_MESSAGES = 6

# Speech recognition.
ASR_MODEL = "SPEAK-ASR/whisper-medium-si-merged"
ASR_REVISION = "9fbeaaf862c4befbcf9481338004255a8576e888"
ASR_LANGUAGE = "sinhala"

# SerendibAI OmniVoice V5 and its exact training reference.
TTS_MODEL = "2broke2code/serendib-omnivoice-finetuned-v5"
TTS_REVISION = "b0f75c9b1103c3695f5813cd448facaa10f1b1bd"
TTS_DATASET = "2broke2code/serendib-omnivoice-dataset-v5"
TTS_DATASET_REVISION = "5d2f4cc973f2c923a84607e60d746df1be2eb0dd"
TTS_REFERENCE_FILE = "audio/270.wav"
TTS_REFERENCE_TEXT = (
    "Sir property එකේ යතුරු ටික භාර දෙන්න කලින් අපි final inspection එකක් කරනවා. "
    "ඒ වෙලාවට ඔබතුමාත් ගෙදර ඉන්නවා නම් ගොඩක් හොඳයි."
)
TTS_STEPS = 20
TTS_SPEED = 0.98

PROGRESS_REPEAT_SECONDS = 10
PROGRESS_LINES = {
    "si": ("හරි, මම ඒක බලලා කියන්නම්.",),
    "ta": ("சரி, நான் அதைப் பார்த்துச் சொல்கிறேன்.",),
    "en": ("Sure, let me look into that.",),
}

# Turn detection and playback protection.
INPUT_CHUNK_MS = 40
INPUT_CHUNK_BYTES = (16_000 * 2 * INPUT_CHUNK_MS) // 1000
SILENCE_RMS = 1000
END_SILENCE_CHUNKS = 20
MIN_AUDIO_MS = 500
GREETING_DELAY_SECONDS = 0.5
PLAYBACK_ECHO_TAIL_SECONDS = 0.35

GREETING_PARTS = (
    ("For English, say English.", "en"),
    ("සිංහල සඳහා, සිංහල කියන්න.", "si"),
    ("தமிழுக்கு, தமிழ் என்று சொல்லுங்கள்.", "ta"),
)
LANGUAGE_ACKNOWLEDGEMENTS = {
    "en": "Hello, I'm SerendibAI from Homelands Properties. How can I help?",
    "si": "ආයුබෝවන්, මම Homelands Properties හි SerendibAI. ඔබට කොහොමද උදව් කරන්න ඕනේ?",
    "ta": "வணக்கம், நான் Homelands Properties-இன் SerendibAI. உங்களுக்கு எப்படி உதவலாம்?",
}
ERROR_RESPONSES = {
    "en": "Sorry, I couldn't check that. Please ask me again.",
    "si": "සමාවෙන්න, ඒක බලන්න බැරි වුණා. කරුණාකර නැවත අහන්න.",
    "ta": "மன்னிக்கவும், அதைச் சரிபார்க்க முடியவில்லை. மீண்டும் கேளுங்கள்.",
}

SYSTEM_PROMPT = (
    "You are SerendibAI's casual phone agent for Homelands Properties. Match the latest caller "
    "language exactly: Sinhala, Tamil, or English, unless they request a switch. The language menu "
    "already played, so do not reintroduce yourself during a real request. Be warm, natural, and "
    "respectful, using spoken rather than formal grammar. Answer usefully in one to three short "
    "sentences. Never repeat the previous answer or ask permission for an action already requested. "
    "Use the full conversation, but never invent a property, location, preference, customer detail, "
    "date, or time. For a greeting, ask how you can help. If action-critical information is missing, "
    "ask one concise follow-up; otherwise act immediately. Always use the caller's newest correction."
)
