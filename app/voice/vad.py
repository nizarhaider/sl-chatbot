import time

import numpy as np


class VadState:
    def __init__(self) -> None:
        self.is_speaking = False
        self.started_at: float | None = None
        self.silence_chunks = 0
        self.speech_chunks = 0
        self._utterance_buffer = bytearray()
        self._candidate_chunks: list[bytes] = []

    def start(self) -> None:
        self.is_speaking = True
        self.started_at = time.perf_counter()
        self.silence_chunks = 0
        self.speech_chunks = 0
        self._utterance_buffer.clear()
        self._candidate_chunks.clear()

    def add_candidate(self, chunk: bytes) -> None:
        self._candidate_chunks.append(chunk)

    @property
    def candidate_count(self) -> int:
        return len(self._candidate_chunks)

    def promote_candidate(self) -> None:
        candidates = self._candidate_chunks
        self._candidate_chunks = []
        self.start()
        for chunk in candidates:
            self.add_speech(chunk)

    def clear_candidate(self) -> None:
        self._candidate_chunks.clear()

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
        self._candidate_chunks.clear()
        return turn

    def discard(self) -> None:
        """Drop audio captured while the agent's own playback is active."""
        self.is_speaking = False
        self.started_at = None
        self.silence_chunks = 0
        self.speech_chunks = 0
        self._utterance_buffer.clear()
        self._candidate_chunks.clear()


class TurnAudio:
    def __init__(self, started_at: float | None, pcm: bytes) -> None:
        self.started_at = started_at
        self.pcm = pcm


def pcm_rms(chunk: bytes) -> float:
    audio_np = np.frombuffer(chunk, dtype=np.int16)
    if not audio_np.size:
        return 0.0
    return float(np.sqrt(np.mean(audio_np.astype(np.float64) ** 2)))
