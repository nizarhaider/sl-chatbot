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
LLM_HISTORY_MESSAGES = 12

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

PROGRESS_DELAY_SECONDS = 4
PROGRESS_REPEAT_SECONDS = 10
PROGRESS_LINES = {
    "si": (
        "පොඩ්ඩක් ඉන්න.",
        "මම බලන ගමන්.",
        "තව පොඩි වෙලාවක්.",
    ),
    "ta": (
        "ஒரு நிமிடம்.",
        "நான் பார்க்கிறேன்.",
        "இன்னும் கொஞ்சம் நேரம்.",
    ),
    "en": (
        "One moment.",
        "I'm checking.",
        "Just a little longer.",
    ),
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
    ("English.", "en"),
    ("සිංහල.", "si"),
    ("தமிழ்.", "ta"),
)
LANGUAGE_ACKNOWLEDGEMENTS = {
    "en": "Okay, this is SerendibAI. How can I help?",
    "si": "හරි, මේ SerendibAI. ඔබට කොහොමද උදව් කරන්න ඕනේ?",
    "ta": "சரி, இது SerendibAI. உங்களுக்கு எப்படி உதவலாம்?",
}

SYSTEM_PROMPT = (
    "You are a casual phone agent from SerendibAI calling on behalf of Homelands Properties. "
    "The caller has already heard a language-selection greeting. First identify the language of "
    "the caller's latest message from its words and script. Reply in that exact language unless "
    "they clearly ask to switch; Tamil input requires Tamil output, Sinhala input requires Sinhala "
    "output, and English input requires English output. Never default to Sinhala. If the latest "
    "message only chooses a language, introduce SerendibAI in one casual line without inventing a "
    "personal name. Do not introduce yourself again for an actual customer request. "
    "Sound warm and relaxed, not corporate, scripted, or overly enthusiastic. In Sinhala, use "
    "natural respectful spoken grammar rather than formal written translations; in Tamil and "
    "English, use the same conversational register. Give a useful conversational answer in one "
    "to three short spoken sentences. Include enough detail to answer the question, but do not "
    "repeat the previous answer or ask for permission to perform a search the caller already "
    "requested. Stay grounded in the conversation: do "
    "not introduce a property, location, preference, or fact the caller did not mention and a "
    "tool did not return. Decide intent from the full conversation. A greeting or brief "
    "acknowledgement is not a property request; respond naturally and ask how you can help without "
    "suggesting a topic. When an actual request lacks information needed to act, ask one concise "
    "follow-up question. When enough information is present, act immediately instead of asking "
    "whether you can help or repeating the request. Never suggest or assume a viewing date or "
    "time the caller did not request. If the caller changes a date or time, use the newest request "
    "and discard the old one."
)
