import asyncio
import logging
import re

from aiortc import MediaStreamTrack
from num2words import num2words

from app.dashboard.state import dashboard_state
from app.voice.audio_archive import CallAudioRecorder
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

    def _interrupt_playback(
        self,
        call_id: str | None,
        output_track: RealtimeAudioTrack | None,
    ) -> None:
        if call_id is not None:
            self.playback_generation[call_id] = self.playback_generation.get(call_id, 0) + 1
            dashboard_state.emit(
                call_id,
                "pipeline.playback_interrupted",
                {"generation": self.playback_generation[call_id]},
            )
        if output_track is not None:
            output_track.clear_buffer()

    def _prepare_tts_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        spoken = re.sub(
            r"\b(?P<hour>\d{1,2})(?:[:.](?P<minute>[0-5]\d))?\s*(?P<period>a\.?m\.?|p\.?m\.?)\b",
            _time_to_words,
            cleaned,
            flags=re.IGNORECASE,
        )
        spoken = re.sub(
            r"\b\d[\d,]*(?:\.\d+)?\b",
            _number_to_words,
            spoken,
        )
        return spoken.rstrip(",;:").strip()

    async def _run_turn_pipeline(self, call_id, caller_phone, input_track, output_track):
        recorder = CallAudioRecorder()
        output_track.set_recording_callback(recorder.add_agent_pcm)
        try:
            await self.turn_pipeline.run(
                call_id=call_id,
                caller_phone=caller_phone,
                input_track=input_track,
                output_track=output_track,
                playback_generation=self.playback_generation,
                recorder=recorder,
            )
        except asyncio.CancelledError:
            logger.info("vLLM turn pipeline cancelled for %s", call_id)
        except Exception as exc:
            logger.error("vLLM turn pipeline failed for %s: %s", call_id, exc, exc_info=True)
        finally:
            output_track.set_recording_callback(None)
            self.active_calls.pop(call_id, None)
            self.playback_generation.pop(call_id, None)
            dashboard_state.end_call(call_id)
            logger.info("Cleaned up session for %s", call_id)


voice_agent = VoiceAgent()


def _number_to_words(match: re.Match[str]) -> str:
    value = match.group().replace(",", "")
    return num2words(float(value) if "." in value else int(value), lang="en")


def _time_to_words(match: re.Match[str]) -> str:
    hour = num2words(int(match.group("hour")), lang="en")
    minute = match.group("minute")
    period = match.group("period").replace(".", "").lower()
    if not minute or minute == "00":
        return f"{hour} {period}"
    if minute.startswith("0"):
        minute_words = f"oh {num2words(int(minute), lang='en')}"
    else:
        minute_words = num2words(int(minute), lang="en")
    return f"{hour} {minute_words} {period}"
