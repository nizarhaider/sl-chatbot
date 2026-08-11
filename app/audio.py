"""Audio buffering, resampling, and simple turn VAD."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from aiortc import MediaStreamTrack
from av import AudioFrame
from av.audio.resampler import AudioResampler


class OutboundAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int = 48_000) -> None:
        super().__init__()
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._sample_rate = sample_rate
        self._samples_per_frame = sample_rate // 50
        self._buffer = b""
        self._pending_bytes = 0
        self._pts = 0
        self._started_at: float | None = None

    @property
    def pending_seconds(self) -> float:
        return self._pending_bytes / (self._sample_rate * 2 * 2)

    def clear(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._buffer = b""
        self._pending_bytes = 0

    def add_pcm(self, pcm: bytes, sample_rate: int) -> None:
        if not pcm:
            return
        mono = self._resample(pcm, sample_rate)
        stereo = np.repeat(mono[:, None], 2, axis=1).astype(np.int16).tobytes()
        self._pending_bytes += len(stereo)
        self._queue.put_nowait(stereo)

    async def recv(self) -> AudioFrame:
        await self._pace()
        while not self._queue.empty():
            self._buffer += self._queue.get_nowait()

        size = self._samples_per_frame * 2 * 2
        if len(self._buffer) < size:
            data = b"\x00" * size
        else:
            data, self._buffer = self._buffer[:size], self._buffer[size:]
            self._pending_bytes = max(0, self._pending_bytes - size)

        frame = AudioFrame.from_ndarray(
            np.frombuffer(data, dtype=np.int16).reshape(1, -1),
            format="s16",
            layout="stereo",
        )
        frame.pts = self._pts
        frame.sample_rate = self._sample_rate
        frame.time_base = Fraction(1, self._sample_rate)
        self._pts += self._samples_per_frame
        return frame

    def _resample(self, pcm: bytes, sample_rate: int) -> np.ndarray:
        audio = np.frombuffer(pcm, dtype=np.int16)
        if sample_rate == self._sample_rate:
            return audio
        frame = AudioFrame.from_ndarray(
            audio.reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = sample_rate
        frame.time_base = Fraction(1, sample_rate)
        resampler = AudioResampler(format="s16", layout="mono", rate=self._sample_rate)
        frames = [*resampler.resample(frame), *resampler.resample(None)]
        return np.frombuffer(
            b"".join(item.to_ndarray().tobytes() for item in frames), dtype=np.int16
        )

    async def _pace(self) -> None:
        loop = asyncio.get_running_loop()
        if self._started_at is None:
            self._started_at = loop.time()
        deadline = self._started_at + self._pts / self._sample_rate
        if deadline > loop.time():
            await asyncio.sleep(deadline - loop.time())


@dataclass(frozen=True)
class TurnAudio:
    started_at: float | None
    pcm: bytes


class VoiceActivity:
    def __init__(self) -> None:
        self.speaking = False
        self.started_at: float | None = None
        self.silence_chunks = 0
        self._buffer = bytearray()

    def start(self) -> None:
        self.speaking = True
        self.started_at = time.perf_counter()
        self.silence_chunks = 0
        self._buffer.clear()

    def add(self, chunk: bytes, speech: bool) -> None:
        self._buffer.extend(chunk)
        self.silence_chunks = 0 if speech else self.silence_chunks + 1

    def finish(self) -> TurnAudio:
        turn = TurnAudio(self.started_at, bytes(self._buffer))
        self.speaking = False
        self.started_at = None
        self.silence_chunks = 0
        self._buffer.clear()
        return turn


def pcm_rms(chunk: bytes) -> float:
    audio = np.frombuffer(chunk, dtype=np.int16)
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if audio.size else 0.0
