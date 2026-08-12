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
LLM_HISTORY_MESSAGES = 8

# Speech recognition.
ASR_MODEL = "SPEAK-ASR/whisper-medium-si-merged"
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

PROGRESS_DELAY_SECONDS = 2.25
PROGRESS_REPEAT_SECONDS = 10
PROGRESS_LINES = {
    "si": (
        "හ්ම්, පොඩ්ඩක් ඉන්න. මම බලලා කියන්නම්.",
        "තව පොඩි වෙලාවක් දෙන්න. මම තාම බලන ගමන්.",
    ),
    "ta": (
        "ம், ஒரு நிமிடம். நான் பார்த்துச் சொல்கிறேன்.",
        "இன்னும் கொஞ்சம் நேரம் கொடுங்கள். நான் இன்னும் பார்த்துக்கொண்டிருக்கிறேன்.",
    ),
    "en": (
        "Hmm, give me a moment. I'll check that.",
        "I need a little more time. I'm still checking.",
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

GREETING = "Please say English, සිංහල, or தமிழ்."

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
    "English, use the same conversational register. Keep each spoken reply to one short sentence "
    "unless a tool result requires more detail. Stay grounded in the conversation: do "
    "not introduce a property, location, preference, or fact the caller did not mention and a "
    "tool did not return. Decide intent from the full conversation. A greeting or brief "
    "acknowledgement is not a property request; respond naturally and ask how you can help without "
    "suggesting a topic. When an actual request lacks information needed to act, ask one concise "
    "follow-up question. When enough information is present, act immediately instead of asking "
    "whether you can help or repeating the request."
)
