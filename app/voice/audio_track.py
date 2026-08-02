import asyncio
import logging
from fractions import Fraction

import numpy as np
from aiortc import MediaStreamTrack
from av import AudioFrame
from av.audio.resampler import AudioResampler

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
        self._pending_audio_bytes = 0
        self._start_time = None
        self._logged_non_silent_frames = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_size_bytes(self) -> int:
        return self._samples_per_frame * self._channels * 2

    @property
    def pending_audio_seconds(self) -> float:
        bytes_per_second = self._sample_rate * self._channels * 2
        return self._pending_audio_bytes / bytes_per_second

    def clear_buffer(self) -> None:
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._buffer = b""
        self._pending_audio_bytes = 0

    def add_pcm_audio(self, pcm: bytes, sample_rate: int) -> None:
        if not pcm:
            return

        mono = self._resample_if_needed(pcm, sample_rate)
        stereo = np.repeat(mono[:, None], self._channels, axis=1)
        output_bytes = stereo.astype(np.int16).tobytes()
        self._pending_audio_bytes += len(output_bytes)
        self._log_queued_audio(pcm, sample_rate, mono, output_bytes)
        self.queue.put_nowait(output_bytes)

    async def recv(self):
        await self._pace_next_frame()
        data_to_send = self._next_frame_bytes()
        self._log_emitted_audio(data_to_send)
        return self._make_audio_frame(data_to_send)

    def _resample_if_needed(self, pcm: bytes, sample_rate: int) -> np.ndarray:
        input_audio = np.frombuffer(pcm, dtype=np.int16)
        if sample_rate == self._sample_rate:
            return input_audio

        frame = AudioFrame.from_ndarray(input_audio.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = sample_rate
        frame.time_base = Fraction(1, sample_rate)
        resampler = AudioResampler(format="s16", layout="mono", rate=self._sample_rate)
        chunks = [resampled.to_ndarray().tobytes() for resampled in resampler.resample(frame)]
        chunks.extend(resampled.to_ndarray().tobytes() for resampled in resampler.resample(None))
        return np.frombuffer(b"".join(chunks), dtype=np.int16)

    async def _pace_next_frame(self) -> None:
        if self._start_time is None:
            self._start_time = asyncio.get_event_loop().time()
        next_frame_time = self._start_time + (self._pts / self._sample_rate)
        now = asyncio.get_event_loop().time()
        if next_frame_time > now:
            await asyncio.sleep(next_frame_time - now)

    def _next_frame_bytes(self) -> bytes:
        while not self.queue.empty():
            try:
                self._buffer += self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        target_size = self.frame_size_bytes
        if len(self._buffer) < target_size:
            return b"\x00" * target_size

        data_to_send = self._buffer[:target_size]
        self._buffer = self._buffer[target_size:]
        self._pending_audio_bytes = max(0, self._pending_audio_bytes - len(data_to_send))
        return data_to_send

    def _make_audio_frame(self, data: bytes) -> AudioFrame:
        audio = np.frombuffer(data, dtype=np.int16).reshape(1, -1)
        frame = AudioFrame.from_ndarray(audio, format="s16", layout=self._layout)
        frame.pts = self._pts
        frame.sample_rate = self._sample_rate
        frame.time_base = self._time_base
        self._pts += self._samples_per_frame
        return frame

    def _log_queued_audio(
        self,
        input_pcm: bytes,
        input_rate: int,
        output_mono: np.ndarray,
        output_bytes: bytes,
    ) -> None:
        if self._logged_non_silent_frames >= 5:
            return
        input_audio = np.frombuffer(input_pcm, dtype=np.int16)
        logger.info(
            "Queued outbound PCM: input_rate=%s input_bytes=%s input_rms=%.1f output_rate=%s layout=%s output_bytes=%s output_rms=%.1f",
            input_rate,
            len(input_pcm),
            _rms(input_audio),
            self._sample_rate,
            self._layout,
            len(output_bytes),
            _rms(output_mono),
        )

    def _log_emitted_audio(self, data: bytes) -> None:
        if self._logged_non_silent_frames >= 5 or not data.strip(b"\x00"):
            return
        logger.info(
            "Emitting outbound audio frame: rate=%s layout=%s bytes=%s rms=%.1f buffered=%s queued_chunks=%s",
            self._sample_rate,
            self._layout,
            len(data),
            _rms(np.frombuffer(data, dtype=np.int16)),
            len(self._buffer),
            self.queue.qsize(),
        )
        self._logged_non_silent_frames += 1


def _rms(audio: np.ndarray) -> float:
    if not audio.size:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
