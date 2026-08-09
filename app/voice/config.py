LOCAL_LLM_MODEL_PATH = ""
LOCAL_LLM_MODEL_REPO = "google/gemma-4-E4B-it-qat-q4_0-gguf"
LOCAL_LLM_MODEL_FILENAME = "gemma-4-E4B_q4_0-it.gguf"
LOCAL_LLM_MODEL_DIR = ""
LOCAL_LLM_N_GPU_LAYERS = 42
LOCAL_LLM_CONTEXT_TOKENS = 2048
LOCAL_LLM_BATCH_TOKENS = 128
LOCAL_LLM_THREADS = 8
LOCAL_LLM_FLASH_ATTENTION = True
LOCAL_LLM_TEMPERATURE = 0.2
LOCAL_LLM_MAX_OUTPUT_TOKENS = 1024
LOCAL_LLM_HISTORY_MAX_MESSAGES = 8
LOCAL_LLM_PREWARM = True

WHISPER_MODEL = "SPEAK-ASR/whisper-medium-si-merged"
WHISPER_DEVICE = "cuda"
WHISPER_LANGUAGE = "sinhala"
WHISPER_TASK = "transcribe"

TURN_INPUT_CHUNK_MS = 40
TURN_INPUT_CHUNK_SIZE = (16000 * 2 * TURN_INPUT_CHUNK_MS) // 1000
TURN_SILENCE_THRESHOLD = 1000
TURN_END_SILENCE_CHUNKS = 30
TURN_MIN_AUDIO_MS = 500
TURN_GREETING_DELAY_SECONDS = 1.2
TURN_PLAYBACK_ECHO_TAIL_SECONDS = 0.35

LOCAL_TURN_GREETING = (
    "To speak in English, please say English. "
    "සිංහලෙන් කතා කිරීමට කරුණාකර සිංහල කියන්න. "
    "தமிழில் பேச தயவுசெய்து தமிழ் என்று சொல்லுங்கள்."
)

HOMELANDS_LOCAL_SYSTEM_PROMPT = (
    "You are a casual phone agent from SerendibAI calling on behalf of Homelands Properties. "
    "The caller has already heard a language-selection greeting asking them to say English, Sinhala, or Tamil. "
    "If the caller only picks a language, reply in that language with one casual line introducing yourself as the SerendibAI agent for Homelands Properties. "
    "Reply in the same language as the caller's latest speech unless they clearly ask to switch languages. "
)

REALTIME_TTS_REF_AUDIO = "app/voices/training-033.wav"
REALTIME_TTS_REF_TEXT = (
    "ඔබතුමාගේ ගිය මාසේ බිල් එක ටිකක් වැඩිවෙලා තියෙන්නේ Sir ගත්ත international "
    "call charges නිසා. Sirට ඕනෙ නම් මට පුළුවන් ඒකේ detailed report එකක් "
    "ඔබතුමාගේ registered email එකට එවන්න."
)
REALTIME_TTS_REF_LANGUAGE = "si"
REALTIME_TTS_MODEL_ID = "2broke2code/serendib-omnivoice-finetuned-v4"
REALTIME_TTS_NUM_STEPS = "20,20"
REALTIME_TTS_SPEED = 1.0
REALTIME_TTS_DEVICE = "cuda:0"
REALTIME_TTS_DTYPE = "float16"
REALTIME_TTS_DEBUG = False
REALTIME_TTS_PREWARM = True
