import logging
from logging.handlers import RotatingFileHandler
import os
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from fastapi import FastAPI
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


class ImportantEventFilter(logging.Filter):
    IMPORTANT_PATTERNS = (
        "WEBHOOK_VERIFIED",
        "Received call event",
        "Processing SDP Offer",
        "Received audio track",
        "Connection state",
        "terminated by peer",
        "Turn transcript",
        "Turn response",
        "Gemini response",
        "RealtimeTTS complete",
        "Turn timings",
        "Discarded",
        "input ended",
        "Stopping interrupted",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        message = record.getMessage()
        return any(pattern in message for pattern in self.IMPORTANT_PATTERNS)


def configure_important_log() -> None:
    path = os.environ.get("IMPORTANT_LOG_PATH", "run_logs/important.log")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    handler = RotatingFileHandler(
        path,
        maxBytes=int(os.environ.get("IMPORTANT_LOG_MAX_BYTES", "1048576")),
        backupCount=int(os.environ.get("IMPORTANT_LOG_BACKUPS", "3")),
    )
    handler.setLevel(logging.INFO)
    handler.addFilter(ImportantEventFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)


configure_important_log()


async def prewarm_tts() -> None:
    if os.environ.get("REALTIME_TTS_PREWARM", "true").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    from app.voice_agent.agent import voice_agent

    try:
        logger.info("Prewarming RealtimeTTS OmniVoice engine")
        await voice_agent.prewarm_tts()
        logger.info("RealtimeTTS OmniVoice prewarm complete")
    except Exception:
        logger.exception("RealtimeTTS OmniVoice prewarm failed; continuing startup")


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
