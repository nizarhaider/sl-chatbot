from pathlib import Path


VOICE_AGENT_PROMPT_PATH = Path(__file__).with_name("prompts") / "homelands_agent.md"
VOICE_AGENT_PROMPT = VOICE_AGENT_PROMPT_PATH.read_text(encoding="utf-8").strip()

LLM_BASE_URL = "http://127.0.0.1:8000/v1"
LLM_MODEL = "unsloth/gemma-4-E4B-it-GGUF"
# Gemma 4's recommended sampling defaults; reasoning is disabled server-side.
LLM_TEMPERATURE = 1.0
LLM_PREWARM = True

WHISPER_MODEL = "SPEAK-ASR/whisper-medium-si-merged"
WHISPER_DEVICE = "cuda:0"
WHISPER_LANGUAGE = "sinhala"
WHISPER_TASK = "transcribe"

TURN_INPUT_CHUNK_MS = 40
TURN_INPUT_CHUNK_SIZE = (16000 * 2 * TURN_INPUT_CHUNK_MS) // 1000
TURN_SILENCE_THRESHOLD = 1000
# A 1.6-second boundary lets callers pause briefly to add location, bedroom,
# or budget details without receiving a reply after each fragment.
TURN_END_SILENCE_CHUNKS = 40
TURN_MIN_AUDIO_MS = 500
# Interrupting spoken output is deliberately stricter than normal turn
# detection so speaker noise and incidental sounds do not cut the bot off.
TURN_BARGE_IN_RMS_THRESHOLD = 1600
TURN_BARGE_IN_MIN_SPEECH_CHUNKS = 12
TURN_GREETING_DELAY_SECONDS = 1.2
TURN_PLAYBACK_ECHO_TAIL_SECONDS = 0.35

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
REALTIME_TTS_NUM_STEPS = "20,20"
REALTIME_TTS_SPEED = 1.0
REALTIME_TTS_DEVICE = "cuda:0"
REALTIME_TTS_DTYPE = "float16"
REALTIME_TTS_DEBUG = False
REALTIME_TTS_PREWARM = True
