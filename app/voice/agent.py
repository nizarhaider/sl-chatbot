import asyncio
import logging
import re

from aiortc import MediaStreamTrack

from app.dashboard.state import dashboard_state
from app.voice.audio_track import RealtimeAudioTrack
from app.voice.turn_pipeline import LocalGemmaTurnPipeline

logger = logging.getLogger(__name__)


class VoiceAgent:
    def __init__(self):
        self.active_calls: dict[str, asyncio.Task] = {}
        self.playback_generation: dict[str, int] = {}
        self.turn_pipeline = LocalGemmaTurnPipeline(
            prepare_tts_text=self._prepare_tts_text,
            interrupt_playback=self._interrupt_playback,
        )

    async def process_audio(
        self,
        call_id: str,
        caller_phone: str,
        input_track: MediaStreamTrack,
        output_track: RealtimeAudioTrack,
    ):
        await self.cancel_call(call_id)
        self.playback_generation[call_id] = 0
        task = asyncio.create_task(
            self._run_turn_pipeline(call_id, caller_phone, input_track, output_track),
            name=f"call-{call_id}",
        )
        self.active_calls[call_id] = task
        await task

    async def cancel_call(self, call_id: str) -> None:
        task = self.active_calls.pop(call_id, None)
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logger.info("Call task %s cancelled", call_id)
        except Exception as exc:
            logger.error("Error cancelling task %s: %s", call_id, exc)

    async def prewarm_tts(self) -> None:
        await self.turn_pipeline.prewarm_tts()

    async def prewarm_models(self) -> None:
        await self.turn_pipeline.prewarm_models()

    async def close(self) -> None:
        for call_id in list(self.active_calls):
            await self.cancel_call(call_id)
        await self.turn_pipeline.close()

    def _interrupt_playback(
        self,
        call_id: str | None,
        output_track: RealtimeAudioTrack | None,
    ) -> None:
        if call_id is not None:
            self.playback_generation[call_id] = self.playback_generation.get(call_id, 0) + 1
        if output_track is not None:
            output_track.clear_buffer()

    def _prepare_tts_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        return cleaned.rstrip(",;:").strip()

    async def _run_turn_pipeline(self, call_id, caller_phone, input_track, output_track):
        try:
            await self.turn_pipeline.run(
                call_id=call_id,
                caller_phone=caller_phone,
                input_track=input_track,
                output_track=output_track,
                playback_generation=self.playback_generation,
            )
        except asyncio.CancelledError:
            logger.info("Local Gemma turn pipeline cancelled for %s", call_id)
        except Exception as exc:
            logger.error("Local Gemma turn pipeline failed for %s: %s", call_id, exc, exc_info=True)
        finally:
            self.active_calls.pop(call_id, None)
            self.playback_generation.pop(call_id, None)
            dashboard_state.end_call(call_id)
            logger.info("Cleaned up session for %s", call_id)


voice_agent = VoiceAgent()
