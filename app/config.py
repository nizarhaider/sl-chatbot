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
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 256
LLM_HISTORY_MESSAGES = 8

# Speech recognition.
ASR_MODEL = "SPEAK-ASR/whisper-medium-si-merged"
ASR_LANGUAGE = "sinhala"

# SerendibAI OmniVoice V5 and its exact training reference.
TTS_MODEL = "2broke2code/serendib-omnivoice-finetuned-v5"
TTS_REVISION = "b0f75c9b1103c3695f5813cd448facaa10f1b1bd"
TTS_DATASET = "2broke2code/serendib-omnivoice-dataset-v5"
TTS_DATASET_REVISION = "5d2f4cc973f2c923a84607e60d746df1be2eb0dd"
TTS_REFERENCE_FILE = "audio/033.wav"
TTS_REFERENCE_TEXT = (
    "ඔබතුමාගේ ගිය මාසේ බිල් එක ටිකක් වැඩිවෙලා තියෙන්නේ Sir ගත්ත international "
    "call charges නිසා. Sirට ඕනෙ නම් මට පුළුවන් ඒකේ detailed report එකක් "
    "ඔබතුමාගේ registered email එකට එවන්න."
)
TTS_LANGUAGE = "si"
TTS_STEPS = 20
TTS_SPEED = 1.0

# Turn detection and playback protection.
INPUT_CHUNK_MS = 40
INPUT_CHUNK_BYTES = (16_000 * 2 * INPUT_CHUNK_MS) // 1000
SILENCE_RMS = 1000
END_SILENCE_CHUNKS = 30
MIN_AUDIO_MS = 500
GREETING_DELAY_SECONDS = 0.5
PLAYBACK_ECHO_TAIL_SECONDS = 0.35

GREETING = "Please say English, සිංහල, or தமிழ்."

SYSTEM_PROMPT = (
    "You are a casual phone agent from SerendibAI calling on behalf of Homelands Properties. "
    "The caller has already heard a language-selection greeting. Reply in the same language as "
    "the caller's latest speech unless they clearly ask to switch languages. If they only choose "
    "a language, introduce yourself in one casual line. Keep each spoken reply to one short "
    "sentence unless a tool result requires more detail."
)
