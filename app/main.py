import logging
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
import os

from dotenv import load_dotenv
from fastapi import FastAPI

from app.webhooks.whatsapp import router as whatsapp_router

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
for noisy_logger in ("aioice", "httpx", "llama_cpp"):
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
        "Turn dropped",
        "Turn response",
        "Gemma response",
        "Local Gemma model",
        "RealtimeTTS complete",
        "Greeting timings",
        "Turn timings",
        "Turn stages",
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


async def prewarm_voice_models() -> None:
    from app.voice_agent.agent import voice_agent

    try:
        logger.info("Prewarming local voice models")
        await voice_agent.prewarm_models()
        logger.info("Local voice model prewarm complete")
    except Exception:
        logger.exception("Local voice model prewarm failed")
        raise


@asynccontextmanager
async def lifespan(_: FastAPI):
    await prewarm_voice_models()
    yield


app = FastAPI(title="WhatsApp Voice Bot", lifespan=lifespan)
app.include_router(whatsapp_router)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "WhatsApp Webhook Server is running"}
