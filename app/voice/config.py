LOCAL_LLM_MODEL_PATH = ""
LOCAL_LLM_MODEL_REPO = "google/gemma-4-E4B-it-qat-q4_0-gguf"
LOCAL_LLM_MODEL_FILENAME = "gemma-4-E4B_q4_0-it.gguf"
LOCAL_LLM_MODEL_DIR = ""
LOCAL_LLM_N_GPU_LAYERS = -1
LOCAL_LLM_CONTEXT_TOKENS = 2048
LOCAL_LLM_BATCH_TOKENS = 256
LOCAL_LLM_THREADS = 4
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
TURN_GREETING_PROTECTION_MAX_SECONDS = 1.5

LOCAL_TURN_GREETING = (
    "To speak in English, please say English. "
    "සිංහලෙන් කතා කිරීමට කරුණාකර සිංහල කියන්න. "
    "தமிழில் பேச தயவுசெய்து தமிழ் என்று சொல்லுங்கள்."
)

HOMELANDS_PROPERTIES = (
    "1. Horizon Residencies, Malabe: two-bedroom apartments from LKR 28 million, near schools and supermarkets. "
    "2. Lakeview Villas, Piliyandala: three-bedroom villas from LKR 48 million, garden, parking, and lake access. "
    "3. Green Acres, Kurunegala: ten-perch residential land from LKR 9.5 million, clear title, bank loans supported. "
    "4. Ocean Breeze Apartments, Dehiwala: one and two-bedroom units from LKR 32 million, sea view, ready soon."
)

HOMELANDS_LOCAL_SYSTEM_PROMPT = (
    "You are a friendly assistant working for Homelands Properties. "
    "Help the customer with their property-related queries. "
    f"Use these mock properties only: {HOMELANDS_PROPERTIES} "
    "The caller has already heard a language-selection greeting asking them to say English, Sinhala, or Tamil. "
    "If the caller only picks a language, reply in that language with a short introduction: say you are the Homelands Properties assistant and that you can help them learn about available apartments, villas, land, locations, prices, and appointments. "
    "Reply in the same language as the caller's latest speech unless they clearly ask to switch languages. "
    "If the caller message is unclear, garbled, partial, or you do not understand it, ask casually in the caller's current language for them to repeat it or say it a little more clearly. "
    "Do not repeat the language-selection greeting unless the caller is clearly starting over. "
    "Do not recommend a property, make up requirements, or schedule anything unless the caller actually asked about a property. "
    "When the customer asks about properties, recommend a suitable option and say that a Homelands consultant appointment has been scheduled for tomorrow at 10 AM. "
    "Keep each response brief, natural, and suitable for a phone call. "
    "Do not include analysis, reasoning notes, markdown, or <think> blocks."
)

REALTIME_TTS_REF_AUDIO = "app/voices/female-3.wav"
REALTIME_TTS_REF_TEXT = "ඔබතුමියගේ internet connection එකේ ඇතිවී තිබෙන තාක්ෂණික දෝෂය පිළිබඳව මේවෙනකොටත් අපට වාර්තා වී තිබෙනවා. අපේ Technician කෙනෙක් ඉදිරි පැය විසිහතර ඇතුළත ඔබව visit කරලා ඔබේ ගැටලුට විසඳුමක් ලබාදේවි."
REALTIME_TTS_REF_LANGUAGE = "si"
REALTIME_TTS_NUM_STEPS = "40,40"
REALTIME_TTS_SPEED = 1.0
REALTIME_TTS_DEVICE = "cuda:0"
REALTIME_TTS_DTYPE = "float16"
REALTIME_TTS_DEBUG = False
REALTIME_TTS_PREWARM = True
