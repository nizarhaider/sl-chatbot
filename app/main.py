import logging
import os
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from fastapi import FastAPI
from app.services.tts import get_tts_service
from app.webhooks.whatsapp import router as whatsapp_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
for noisy_logger in ("aioice", "google_genai", "httpx"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def prewarm_tts() -> None:
    if os.environ.get("VOICE_PIPELINE_MODE", "").strip().lower() == "realtime_turn":
        if os.environ.get("REALTIME_TTS_PREWARM", "true").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return
        from app.voice_agent.agent import voice_agent

        logger.info("Prewarming RealtimeTTS engine")
        await asyncio.to_thread(voice_agent.realtime_turn_pipeline._get_tts_stream)
        logger.info("RealtimeTTS prewarm complete")
        return

    if os.environ.get("VOICE_TTS_PREWARM", "false").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    tts_service = get_tts_service()
    if tts_service.uses_gemini_audio():
        return

    text = os.environ.get(
        "VOICE_TTS_PREWARM_TEXT",
        "ආයුබෝවන්, ඔබට කෙසේ උදව් කළ හැකිද?",
    )
    try:
        logger.info("Prewarming TTS provider with %d characters", len(text))
        await tts_service.synthesize(text)
        logger.info("TTS prewarm complete")
    except Exception:
        logger.exception("TTS prewarm failed; continuing startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await prewarm_tts()
    yield


app = FastAPI(title="WhatsApp Voice Bot", lifespan=lifespan)

# Include routers
app.include_router(whatsapp_router)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "WhatsApp Webhook Server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
