import asyncio
import logging

from aiortc import MediaStreamTrack

from app.dashboard.state import dashboard_state
from app.voice.audio_archive import CallAudioRecorder
from app.voice.audio_track import RealtimeAudioTrack
from app.voice.gemini_live import GeminiLivePipeline

logger = logging.getLogger(__name__)


class VoiceAgent:
    def __init__(self):
        self.active_calls: dict[str, asyncio.Task] = {}
        self.turn_pipeline = GeminiLivePipeline(interrupt_playback=self._interrupt_playback)

    async def process_audio(
        self,
        call_id: str,
        caller_phone: str,
        input_track: MediaStreamTrack,
        output_track: RealtimeAudioTrack,
    ):
        await self.cancel_call(call_id)
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

    async def prewarm_models(self) -> None:
        await self.turn_pipeline.prewarm_models()

    def _interrupt_playback(
        self,
        call_id: str | None,
        output_track: RealtimeAudioTrack | None,
    ) -> None:
        if call_id is not None:
            dashboard_state.emit(call_id, "pipeline.playback_interrupted", {})
        if output_track is not None:
            output_track.clear_buffer()

    async def _run_turn_pipeline(self, call_id, caller_phone, input_track, output_track):
        recorder = CallAudioRecorder()
        output_track.set_recording_callback(recorder.add_agent_pcm)
        try:
            await self.turn_pipeline.run(
                call_id=call_id,
                caller_phone=caller_phone,
                input_track=input_track,
                output_track=output_track,
                recorder=recorder,
            )
        except asyncio.CancelledError:
            logger.info("Gemini Live pipeline cancelled for %s", call_id)
        except Exception as exc:
            logger.error("Gemini Live pipeline failed for %s: %s", call_id, exc, exc_info=True)
        finally:
            output_track.set_recording_callback(None)
            self.active_calls.pop(call_id, None)
            dashboard_state.end_call(call_id)
            logger.info("Cleaned up session for %s", call_id)


voice_agent = VoiceAgent()
