"""FastAPI entrypoint for the voice-only WhatsApp bot."""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    force=True,
)
for noisy_logger in ("aioice", "httpx", "llama_cpp"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

from app.database import call_log
from app.pipeline import voice_agent
from app.whatsapp import router, webrtc_service

logger = logging.getLogger(__name__)
REQUIRED_ENV = (
    "VERIFY_TOKEN",
    "PHONE_NUMBER_ID",
    "DATABASE_URL",
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if not (os.getenv("WHATSAPP_ACCESS_TOKEN") or os.getenv("WHATSAPP_TOKEN")):
        missing.append("WHATSAPP_ACCESS_TOKEN")
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
    logger.info("Prewarming local ASR, LLM, TTS, and database")
    await voice_agent.prewarm()
    logger.info("Voice server ready")
    try:
        yield
    finally:
        await webrtc_service.close_all()
        await voice_agent.close()
        await call_log.close()


def create_app() -> FastAPI:
    application = FastAPI(title="SerendibAI WhatsApp Voice Bot", lifespan=lifespan)
    application.include_router(router)

    @application.get("/")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
