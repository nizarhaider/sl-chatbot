import asyncio
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request

from app.api.logging import configure_logging

load_dotenv()
configure_logging()
logger = logging.getLogger(__name__)


async def prewarm_voice_models(app: FastAPI) -> None:
    from app.voice.agent import voice_agent

    try:
        logger.info("Prewarming local voice models")
        await voice_agent.prewarm_models()
        app.state.voice_ready = True
        logger.info("Local voice model prewarm complete")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        app.state.voice_startup_error = str(exc)
        logger.exception("Local voice model prewarm failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.voice_ready = False
    app.state.voice_startup_error = ""
    prewarm_task = asyncio.create_task(
        prewarm_voice_models(app),
        name="voice-model-prewarm",
    )
    try:
        yield
    finally:
        if not prewarm_task.done():
            prewarm_task.cancel()
            await asyncio.gather(prewarm_task, return_exceptions=True)
        from app.dashboard.state import dashboard_state

        dashboard_state.close()


def create_app() -> FastAPI:
    from app.dashboard.router import router as dashboard_router
    from app.integrations.whatsapp.webhook import router as whatsapp_router

    app = FastAPI(title="WhatsApp Voice Bot", lifespan=lifespan)
    app.include_router(dashboard_router)
    app.include_router(whatsapp_router)

    @app.get("/")
    def read_root(request: Request):
        if request.app.state.voice_startup_error:
            status = "error"
        elif request.app.state.voice_ready:
            status = "ready"
        else:
            status = "warming_up"
        return {"status": status, "message": "WhatsApp Webhook Server is running"}

    return app
