import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.logging import configure_logging

load_dotenv()
configure_logging()
logger = logging.getLogger(__name__)


async def prewarm_voice_models() -> None:
    from app.voice.agent import voice_agent

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
    try:
        yield
    finally:
        from app.dashboard.state import dashboard_state

        dashboard_state.close()


def create_app() -> FastAPI:
    from app.dashboard.router import router as dashboard_router
    from app.integrations.whatsapp.webhook import router as whatsapp_router

    app = FastAPI(title="WhatsApp Voice Bot", lifespan=lifespan)
    app.include_router(dashboard_router)
    app.include_router(whatsapp_router)

    @app.get("/")
    def read_root():
        return {"status": "ok", "message": "WhatsApp Webhook Server is running"}

    return app
