from pathlib import Path


VOICE_AGENT_PROMPT_PATH = Path(__file__).with_name("prompts") / "homelands_agent.md"
VOICE_AGENT_PROMPT = VOICE_AGENT_PROMPT_PATH.read_text(encoding="utf-8").strip()

LLM_BASE_URL = "http://127.0.0.1:8000/v1"
LLM_MODEL = "unsloth/gemma-4-E4B-it-GGUF"
# Gemma 4's recommended sampling defaults; reasoning is disabled server-side.
LLM_TEMPERATURE = 1.0
LLM_PREWARM = True

WHISPER_MODEL = "openai/whisper-large-v3-turbo"
WHISPER_DEVICE = "cuda:0"
WHISPER_LANGUAGE = "sinhala"
WHISPER_TASK = "transcribe"
WHISPER_MAX_NEW_TOKENS = 64

TURN_INPUT_CHUNK_MS = 40
TURN_INPUT_CHUNK_SIZE = (16000 * 2 * TURN_INPUT_CHUNK_MS) // 1000
TURN_SILENCE_THRESHOLD = 1000
TURN_SPEECH_START_THRESHOLD = 1400
TURN_SPEECH_START_CHUNKS = 5
TURN_END_SILENCE_CHUNKS = 10
TURN_MIN_AUDIO_MS = 1000
TURN_BARGE_IN_RMS_THRESHOLD = 2200
TURN_BARGE_IN_MIN_SPEECH_CHUNKS = 12
TURN_GREETING_DELAY_SECONDS = 0.5
TURN_PLAYBACK_ECHO_TAIL_SECONDS = 1.5

LOCAL_TURN_GREETING = (
    "To speak in English, please say English. "
    "සිංහලෙන් කතා කිරීමට, කරුණාකර “සිංහල” යැයි කියන්න. "
    "தமிழில் பேச தயவுசெய்து தமிழ் என்று சொல்லுங்கள்."
)

HOMELANDS_LOCAL_SYSTEM_PROMPT = VOICE_AGENT_PROMPT

REALTIME_TTS_REF_AUDIO = "app/voices/female-004.wav"
REALTIME_TTS_REF_TEXT = (
    "Good morning sir, සර්ගේ vehicle insurance policy එක ලබන සතියෙන් expire වෙනවා. "
    "සර් කැමති නම් අපි දැන්ම ඒක renew කරන්න process එක පටන් ගන්න පුළුවන්"
)
REALTIME_TTS_REF_LANGUAGE = "si"
REALTIME_TTS_MODEL_ID = "2broke2code/serendib-omnivoice-finetuned-v2"
REALTIME_TTS_NUM_STEPS = "12,12"
REALTIME_TTS_SPEED = 1.0
REALTIME_TTS_DEVICE = "cuda:0"
REALTIME_TTS_DTYPE = "float16"
REALTIME_TTS_DEBUG = False
REALTIME_TTS_PREWARM = True
