import asyncio
import logging
import re
from fractions import Fraction

import numpy as np
from aiortc import MediaStreamTrack
from av import AudioFrame
from av.audio.resampler import AudioResampler

from app.voice_agent.gemini_turn_pipeline import GeminiTurnPipeline

logger = logging.getLogger(__name__)


class RealtimeAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.queue = asyncio.Queue()
        self._pts = 0
        self._sample_rate = sample_rate
        self._channels = 2
        self._layout = "stereo"
        self._time_base = Fraction(1, self._sample_rate)
        self._samples_per_frame = self._sample_rate // 50
        self._buffer = b""
        self._start_time = None
        self._logged_non_silent_frames = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_size_bytes(self) -> int:
        return self._samples_per_frame * self._channels * 2

    def clear_buffer(self) -> None:
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._buffer = b""

    def add_audio(self, data: bytes) -> None:
        self.queue.put_nowait(data)

    def add_pcm_audio(self, pcm: bytes, sample_rate: int) -> None:
        if not pcm:
            return

        output_pcm = pcm
        input_audio = np.frombuffer(pcm, dtype=np.int16)
        input_rms = (
            float(np.sqrt(np.mean(input_audio.astype(np.float64) ** 2)))
            if input_audio.size
            else 0.0
        )

        if sample_rate != self._sample_rate:
            input_array = input_audio.reshape(1, -1)
            frame = AudioFrame.from_ndarray(input_array, format="s16", layout="mono")
            frame.sample_rate = sample_rate
            frame.time_base = Fraction(1, sample_rate)

            resampler = AudioResampler(format="s16", layout="mono", rate=self._sample_rate)
            chunks = []
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().tobytes())
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray().tobytes())
            output_pcm = b"".join(chunks)

        mono = np.frombuffer(output_pcm, dtype=np.int16)
        stereo = np.repeat(mono[:, None], self._channels, axis=1)
        output_bytes = stereo.astype(np.int16).tobytes()
        output_rms = (
            float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
            if mono.size
            else 0.0
        )
        if self._logged_non_silent_frames < 5:
            logger.info(
                "Queued outbound PCM: input_rate=%s input_bytes=%s input_rms=%.1f output_rate=%s layout=%s output_bytes=%s output_rms=%.1f",
                sample_rate,
                len(pcm),
                input_rms,
                self._sample_rate,
                self._layout,
                len(output_bytes),
                output_rms,
            )
        self.add_audio(output_bytes)

    async def recv(self):
        if self._start_time is None:
            self._start_time = asyncio.get_event_loop().time()
        next_frame_time = self._start_time + (self._pts / self._sample_rate)
        now = asyncio.get_event_loop().time()
        if next_frame_time > now:
            await asyncio.sleep(next_frame_time - now)

        target_size = self.frame_size_bytes

        while not self.queue.empty():
            try:
                self._buffer += self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        if len(self._buffer) >= target_size:
            data_to_send = self._buffer[:target_size]
            self._buffer = self._buffer[target_size:]
        else:
            data_to_send = b"\x00" * target_size

        if self._logged_non_silent_frames < 5 and data_to_send.strip(b"\x00"):
            audio = np.frombuffer(data_to_send, dtype=np.int16)
            rms = (
                float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
                if audio.size
                else 0.0
            )
            logger.info(
                "Emitting outbound audio frame: rate=%s layout=%s bytes=%s rms=%.1f buffered=%s queued_chunks=%s",
                self._sample_rate,
                self._layout,
                len(data_to_send),
                rms,
                len(self._buffer),
                self.queue.qsize(),
            )
            self._logged_non_silent_frames += 1

        audio = np.frombuffer(data_to_send, dtype=np.int16).reshape(1, -1)
        frame = AudioFrame.from_ndarray(audio, format="s16", layout=self._layout)
        frame.pts = self._pts
        frame.sample_rate = self._sample_rate
        frame.time_base = self._time_base
        self._pts += self._samples_per_frame
        return frame


class VoiceAgent:
    def __init__(self):
        self.active_calls: dict[str, asyncio.Task] = {}
        self.playback_generation: dict[str, int] = {}
        self.turn_pipeline = GeminiTurnPipeline(
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
        if call_id in self.active_calls:
            task = self.active_calls.pop(call_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info("Previous call task %s cancelled", call_id)
            except Exception as exc:
                logger.error("Error cancelling previous task %s: %s", call_id, exc)

        self.playback_generation[call_id] = 0
        task = asyncio.create_task(
            self._run_turn_pipeline(call_id, input_track, output_track),
            name=f"call-{call_id}",
        )
        self.active_calls[call_id] = task
        await task

    async def prewarm_tts(self) -> None:
        await self.turn_pipeline.prewarm_tts()

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

    async def _run_turn_pipeline(self, call_id, input_track, output_track):
        try:
            await self.turn_pipeline.run(
                call_id=call_id,
                input_track=input_track,
                output_track=output_track,
                playback_generation=self.playback_generation,
            )
        except asyncio.CancelledError:
            logger.info("Gemini STT turn pipeline cancelled for %s", call_id)
        except Exception as exc:
            logger.error("Gemini STT turn pipeline failed for %s: %s", call_id, exc, exc_info=True)
        finally:
            self.active_calls.pop(call_id, None)
            self.playback_generation.pop(call_id, None)
            logger.info("Cleaned up session for %s", call_id)


voice_agent = VoiceAgent()
