import time

import numpy as np


class VadState:
    def __init__(self) -> None:
        self.is_speaking = False
        self.started_at: float | None = None
        self.silence_chunks = 0
        self.speech_chunks = 0
        self._utterance_buffer = bytearray()

    def start(self) -> None:
        self.is_speaking = True
        self.started_at = time.perf_counter()
        self.silence_chunks = 0
        self.speech_chunks = 0
        self._utterance_buffer.clear()

    def add_speech(self, chunk: bytes) -> None:
        self._utterance_buffer.extend(chunk)
        self.silence_chunks = 0
        self.speech_chunks += 1

    def add_silence(self, chunk: bytes) -> None:
        self._utterance_buffer.extend(chunk)
        self.silence_chunks += 1

    def finish(self) -> "TurnAudio":
        turn = TurnAudio(started_at=self.started_at, pcm=bytes(self._utterance_buffer))
        self.is_speaking = False
        self.started_at = None
        self.silence_chunks = 0
        self.speech_chunks = 0
        self._utterance_buffer.clear()
        return turn


class TurnAudio:
    def __init__(self, started_at: float | None, pcm: bytes) -> None:
        self.started_at = started_at
        self.pcm = pcm


def pcm_rms(chunk: bytes) -> float:
    audio_np = np.frombuffer(chunk, dtype=np.int16)
    if not audio_np.size:
        return 0.0
    return float(np.sqrt(np.mean(audio_np.astype(np.float64) ** 2)))
