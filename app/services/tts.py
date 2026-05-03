import asyncio
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SynthesizedAudio:
    pcm: bytes
    sample_rate: int
    text: str


def _detect_language_variant(text: str) -> str:
    for char in text:
        codepoint = ord(char)
        if 0x0D80 <= codepoint <= 0x0DFF:
            return "si"
        if 0x0B80 <= codepoint <= 0x0BFF:
            return "ta"
    return "en"


def _normalize_provider(value: str | None) -> str:
    provider = (value or "gemini_live").strip().lower()
    if provider in {"gemini", "gemini_live"}:
        return "gemini_live"
    if provider == "omnivoice":
        return "omnivoice"
    logger.warning("Unknown VOICE_OUTPUT_PROVIDER=%s, falling back to gemini_live", value)
    return "gemini_live"


class OmniVoiceSynthesizer:
    def __init__(self) -> None:
        self.model_id = os.environ.get("OMNIVOICE_MODEL_ID", "k2-fsa/OmniVoice")
        self.device = os.environ.get("OMNIVOICE_DEVICE", "cpu")
        self.dtype_name = os.environ.get("OMNIVOICE_DTYPE", "float32")
        self.num_step = int(os.environ.get("OMNIVOICE_NUM_STEP", "16"))
        self.speed = float(os.environ.get("OMNIVOICE_SPEED", "1.12"))
        self.sample_rate = 24000
        self._model = None
        self._load_lock = asyncio.Lock()

    async def synthesize(self, text: str) -> SynthesizedAudio:
        model = await self._get_model()
        language = _detect_language_variant(text)
        generation_kwargs = self._build_generation_kwargs(language)
        pcm = await asyncio.to_thread(
            self._generate_pcm_sync,
            model,
            text,
            generation_kwargs,
        )
        return SynthesizedAudio(pcm=pcm, sample_rate=self.sample_rate, text=text)

    async def _get_model(self):
        if self._model is not None:
            return self._model

        async with self._load_lock:
            if self._model is not None:
                return self._model

            logger.info(
                "Loading OmniVoice model %s on %s with dtype=%s",
                self.model_id,
                self.device,
                self.dtype_name,
            )
            self._model = await asyncio.to_thread(self._load_model_sync)
            return self._model

    def _load_model_sync(self):
        import torch
        from omnivoice import OmniVoice

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.dtype_name, torch.float32)
        if self.device == "cpu" and dtype is torch.float16:
            logger.warning("OMNIVOICE_DTYPE=float16 is not suitable on CPU, using float32")
            dtype = torch.float32

        return OmniVoice.from_pretrained(
            self.model_id,
            device_map=self.device,
            dtype=dtype,
        )

    def _build_generation_kwargs(self, language: str) -> dict:
        kwargs: dict[str, object] = {
            "num_step": self.num_step,
            "speed": self.speed,
        }

        ref_audio = self._env_by_language("OMNIVOICE_REF_AUDIO", language)
        ref_text = self._env_by_language("OMNIVOICE_REF_TEXT", language)
        instruct = self._env_by_language("OMNIVOICE_INSTRUCT", language)

        if ref_audio:
            kwargs["ref_audio"] = ref_audio
            if ref_text:
                kwargs["ref_text"] = ref_text
        elif instruct:
            kwargs["instruct"] = instruct

        return kwargs

    def _env_by_language(self, prefix: str, language: str) -> str | None:
        candidates = [
            f"{prefix}_{language.upper()}",
            f"{prefix}_DEFAULT",
        ]
        for key in candidates:
            value = os.environ.get(key)
            if value:
                return value
        return None

    def _generate_pcm_sync(self, model, text: str, generation_kwargs: dict) -> bytes:
        audio = model.generate(text=text, **generation_kwargs)
        if not audio:
            raise RuntimeError("OmniVoice returned no audio")

        samples = np.asarray(audio[0], dtype=np.float32)
        if samples.ndim != 1:
            samples = samples.reshape(-1)

        samples = np.clip(samples, -1.0, 1.0)
        pcm16 = (samples * 32767.0).astype(np.int16)
        return pcm16.tobytes()


class TTSService:
    def __init__(self) -> None:
        self.provider = _normalize_provider(os.environ.get("VOICE_OUTPUT_PROVIDER"))
        self._omnivoice = OmniVoiceSynthesizer() if self.provider == "omnivoice" else None

    def uses_gemini_audio(self) -> bool:
        return self.provider == "gemini_live"

    async def synthesize(self, text: str) -> SynthesizedAudio:
        if self.provider != "omnivoice" or self._omnivoice is None:
            raise RuntimeError(f"TTS provider '{self.provider}' does not support local synthesis")
        return await self._omnivoice.synthesize(text)


@lru_cache(maxsize=1)
def get_tts_service() -> TTSService:
    return TTSService()

