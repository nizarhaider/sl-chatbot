import asyncio
import logging
import time

import numpy as np

from app.voice.config import (
    REALTIME_TTS_DEBUG,
    REALTIME_TTS_DEVICE,
    REALTIME_TTS_DTYPE,
    REALTIME_TTS_MODEL_ID,
    REALTIME_TTS_NUM_STEPS,
    REALTIME_TTS_REF_AUDIO,
    REALTIME_TTS_REF_LANGUAGE,
    REALTIME_TTS_REF_TEXT,
    REALTIME_TTS_SPEED,
)

logger = logging.getLogger(__name__)


class RealtimeOmniVoiceTTS:
    def __init__(self) -> None:
        self._stream = None
        self._sample_rate = 24000
        self._lock = asyncio.Lock()

    async def prewarm(self) -> None:
        await asyncio.to_thread(self._get_stream)

    async def speak(self, text: str, on_audio_chunk) -> float:
        async with self._lock:
            chunks: list[bytes] = []

            def collect_and_forward(chunk: bytes) -> None:
                chunks.append(chunk)
                on_audio_chunk(chunk, self._sample_rate)

            started = time.perf_counter()
            await asyncio.to_thread(self._play, text, collect_and_forward)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            audio_seconds = self._audio_duration_seconds(sum(len(chunk) for chunk in chunks))
            logger.info(
                "RealtimeTTS complete: elapsed_ms=%.0f chars=%s audio_ms=%.0f chunks=%s",
                elapsed_ms,
                len(text),
                audio_seconds * 1000.0,
                len(chunks),
            )
            return audio_seconds

    def _play(self, text: str, on_audio_chunk) -> None:
        stream = self._get_stream()
        stream.feed(text)
        stream.play(muted=True, on_audio_chunk=on_audio_chunk)

    def _get_stream(self):
        if self._stream is not None:
            return self._stream

        from RealtimeTTS import OmniVoiceEngine, OmniVoiceVoice, TextToAudioStream
        import torch

        class PatchedOmniVoiceEngine(OmniVoiceEngine):
            def synthesize(self, text: str, sentence_count: int = 0) -> bool:
                super(OmniVoiceEngine, self).synthesize(text, sentence_count)
                if not self.current_voice:
                    return False

                try:
                    with torch.inference_mode():
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                        audio = self._model.generate(
                            language=_language_for_text(text, self.current_voice.language),
                            text=text,
                            ref_audio=self.current_voice.ref_audio,
                            ref_text=self.current_voice.ref_text,
                            num_step=self._get_num_steps_for_sentence(sentence_count),
                            speed=REALTIME_TTS_SPEED,
                            preprocess_prompt=self.preprocess_prompt,
                            postprocess_output=self.postprocess_output,
                        )

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                    audio_int16 = (
                        np.clip(_normalize_waveform(audio), -1.0, 1.0) * 32767
                    ).astype(np.int16)
                    self.queue.put(audio_int16.tobytes())
                    return True
                except Exception:
                    logger.exception("Patched OmniVoice synthesis failed for text: %s", text)
                    return False

        steps = [int(part.strip()) for part in REALTIME_TTS_NUM_STEPS.split(",") if part.strip()]
        voice = OmniVoiceVoice(
            name="homelands",
            ref_audio=REALTIME_TTS_REF_AUDIO,
            ref_text=REALTIME_TTS_REF_TEXT,
            language=REALTIME_TTS_REF_LANGUAGE,
        )
        engine = PatchedOmniVoiceEngine(
            model_id=REALTIME_TTS_MODEL_ID,
            voice=voice,
            device_map=REALTIME_TTS_DEVICE,
            dtype=getattr(torch, REALTIME_TTS_DTYPE),
            num_steps_schedule=steps or [12, 12],
            debug=REALTIME_TTS_DEBUG,
        )

        _, _, self._sample_rate = engine.get_stream_info()
        self._stream = TextToAudioStream(engine, muted=True)
        return self._stream

    def _audio_duration_seconds(self, audio_bytes: int) -> float:
        if audio_bytes <= 0 or self._sample_rate <= 0:
            return 0.0
        return (audio_bytes / 2) / self._sample_rate


def _normalize_waveform(audio) -> np.ndarray:
    waveform = audio[0] if isinstance(audio, (list, tuple)) else audio
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu().numpy()
    else:
        waveform = np.asarray(waveform)

    waveform = np.squeeze(waveform)
    if waveform.ndim > 1:
        waveform = waveform[0]
    return waveform.astype(np.float32, copy=False)


def _language_for_text(text: str, default: str = "si") -> str | None:
    has_sinhala = any("\u0d80" <= char <= "\u0dff" for char in text)
    has_tamil = any("\u0b80" <= char <= "\u0bff" for char in text)
    has_latin = any(char.isascii() and char.isalpha() for char in text)
    detected = [has_sinhala, has_tamil, has_latin]
    if sum(detected) > 1:
        return None
    if has_sinhala:
        return "si"
    if has_tamil:
        return "ta"
    if has_latin:
        return "en"
    return default
