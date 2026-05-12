import asyncio
import json
import logging
import os
import wave
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from typing import Any

import httpx
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
    if provider in {"omnivoice", "omnivoice_remote"}:
        return "omnivoice_remote"
    if provider in {"omnivoice_local", "omnivoice-hosted-local"}:
        return "omnivoice_local"
    logger.warning("Unknown VOICE_OUTPUT_PROVIDER=%s, falling back to gemini_live", value)
    return "gemini_live"


class OmniVoiceRemoteSynthesizer:
    def __init__(self) -> None:
        self.base_url = os.environ.get(
            "OMNIVOICE_REMOTE_BASE_URL",
            "https://k2-fsa-omnivoice.hf.space",
        ).rstrip("/")
        self.api_name = os.environ.get("OMNIVOICE_REMOTE_API_NAME", "_design_fn")
        self.timeout_seconds = float(os.environ.get("OMNIVOICE_REMOTE_TIMEOUT_SECONDS", "30"))
        self.inference_steps = int(os.environ.get("OMNIVOICE_NUM_STEP", "8"))
        self.guidance_scale = float(os.environ.get("OMNIVOICE_GUIDANCE_SCALE", "2.0"))
        self.speed = float(os.environ.get("OMNIVOICE_SPEED", "1.18"))
        self.denoise = _env_bool("OMNIVOICE_DENOISE", True)
        self.preprocess_prompt = _env_bool("OMNIVOICE_PREPROCESS_PROMPT", True)
        self.postprocess_output = _env_bool("OMNIVOICE_POSTPROCESS_OUTPUT", True)
        self.english_accent = os.environ.get(
            "OMNIVOICE_ENGLISH_ACCENT",
            "Indian Accent / 印度口音",
        )
        self.default_gender = os.environ.get("OMNIVOICE_GENDER", "Auto")
        self.default_age = os.environ.get("OMNIVOICE_AGE", "Young Adult / 青年")
        self.default_pitch = os.environ.get("OMNIVOICE_PITCH", "Moderate Pitch / 中音调")
        self.default_style = os.environ.get("OMNIVOICE_STYLE", "Auto")
        self.default_chinese_dialect = os.environ.get("OMNIVOICE_CHINESE_DIALECT", "Auto")

    async def synthesize(self, text: str) -> SynthesizedAudio:
        data = await self._call_space(text)
        wav_url = data[0]["url"]
        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
            response = await client.get(wav_url)
            response.raise_for_status()
        pcm, sample_rate = await asyncio.to_thread(self._wav_bytes_to_pcm, response.content)
        return SynthesizedAudio(pcm=pcm, sample_rate=sample_rate, text=text)

    async def _call_space(self, text: str):
        endpoint = f"{self.base_url}/gradio_api/call/{self.api_name}"
        payload = {"data": self._build_inputs(text)}
        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            event_id = response.json()["event_id"]

            stream_url = f"{endpoint}/{event_id}"
            async with client.stream("GET", stream_url) as stream:
                async for line in stream.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw_data = line[6:].strip()
                    if not raw_data:
                        continue
                    decoded = json.loads(raw_data)
                    if isinstance(decoded, list) and decoded and isinstance(decoded[0], dict):
                        return decoded

        raise RuntimeError("OmniVoice remote API returned no audio result")

    def _build_inputs(self, text: str) -> list:
        language_variant = _detect_language_variant(text)
        language_name = {
            "en": "English",
            "si": "Sinhala",
            "ta": "Tamil",
        }.get(language_variant, "Auto")

        english_accent = self.english_accent if language_variant == "en" else "Auto"

        return [
            text,
            language_name,
            self.inference_steps,
            self.guidance_scale,
            self.denoise,
            self.speed,
            None,
            self.preprocess_prompt,
            self.postprocess_output,
            self.default_gender,
            self.default_age,
            self.default_pitch,
            self.default_style,
            english_accent,
            self.default_chinese_dialect,
        ]

    def _wav_bytes_to_pcm(self, wav_bytes: bytes) -> tuple[bytes, int]:
        with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())

        if sample_width != 2:
            raise RuntimeError(f"Unsupported OmniVoice sample width: {sample_width}")

        audio = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)

        return audio.tobytes(), sample_rate


class OmniVoiceLocalSynthesizer:
    def __init__(self) -> None:
        self.model_name = os.environ.get("OMNIVOICE_LOCAL_MODEL", "k2-fsa/OmniVoice")
        self.device = os.environ.get("OMNIVOICE_LOCAL_DEVICE") or self._detect_device()
        self.dtype_name = os.environ.get("OMNIVOICE_LOCAL_DTYPE", "float16")
        self.inference_steps = int(os.environ.get("OMNIVOICE_NUM_STEP", "8"))
        self.speed = float(os.environ.get("OMNIVOICE_SPEED", "1.18"))
        self.default_gender = os.environ.get("OMNIVOICE_GENDER", "Male / 男")
        self.default_age = os.environ.get("OMNIVOICE_AGE", "Young Adult / 青年")
        self.default_pitch = os.environ.get("OMNIVOICE_PITCH", "Moderate Pitch / 中音调")
        self.default_style = os.environ.get("OMNIVOICE_STYLE", "Auto")
        self.english_accent = os.environ.get(
            "OMNIVOICE_ENGLISH_ACCENT",
            "Indian Accent / 印度口音",
        )
        self.default_chinese_dialect = os.environ.get("OMNIVOICE_CHINESE_DIALECT", "Auto")
        self._model: Any | None = None
        self._lock = asyncio.Lock()

    async def synthesize(self, text: str) -> SynthesizedAudio:
        model = await self._get_model()
        pcm, sample_rate = await asyncio.to_thread(self._generate_pcm, model, text)
        return SynthesizedAudio(pcm=pcm, sample_rate=sample_rate, text=text)

    async def _get_model(self):
        if self._model is not None:
            return self._model

        async with self._lock:
            if self._model is not None:
                return self._model

            from omnivoice import OmniVoice
            import torch

            dtype = getattr(torch, self.dtype_name)
            self._model = await asyncio.to_thread(
                OmniVoice.from_pretrained,
                self.model_name,
                device_map=self.device,
                dtype=dtype,
            )
            return self._model

    def _generate_pcm(self, model, text: str) -> tuple[bytes, int]:
        instruct = self._build_instruction(text)
        audio = model.generate(
            text=text,
            instruct=instruct,
            num_step=self.inference_steps,
            speed=self.speed,
        )
        pcm = np.asarray(audio[0], dtype=np.float32)
        pcm = np.clip(pcm, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        return pcm.tobytes(), 24000

    def _build_instruction(self, text: str) -> str:
        language_variant = _detect_language_variant(text)
        english_accent = self.english_accent if language_variant == "en" else None
        parts = [
            self._extract_english_label(self.default_gender),
            self._extract_english_label(self.default_age),
            self._extract_english_label(self.default_pitch),
            self._extract_english_label(self.default_style),
            self._extract_english_label(english_accent) if english_accent else None,
            self._extract_english_label(self.default_chinese_dialect),
        ]
        return ", ".join(part for part in parts if part and part.lower() != "auto")

    def _detect_device(self) -> str:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def _extract_english_label(self, value: str | None) -> str | None:
        if not value:
            return None
        return value.split(" / ", 1)[0].strip()


def _env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class TTSService:
    def __init__(self) -> None:
        self.provider = _normalize_provider(os.environ.get("VOICE_OUTPUT_PROVIDER"))
        self._omnivoice = None
        if self.provider == "omnivoice_remote":
            self._omnivoice = OmniVoiceRemoteSynthesizer()
        elif self.provider == "omnivoice_local":
            self._omnivoice = OmniVoiceLocalSynthesizer()
        self._cache: dict[str, SynthesizedAudio] = {}
        self._cache_order: list[str] = []
        self._cache_limit = int(os.environ.get("TTS_CACHE_SIZE", "64"))

    def uses_gemini_audio(self) -> bool:
        return self.provider == "gemini_live"

    async def synthesize(self, text: str) -> SynthesizedAudio:
        if self.provider not in {"omnivoice_remote", "omnivoice_local"} or self._omnivoice is None:
            raise RuntimeError(f"TTS provider '{self.provider}' does not support synthesis")
        cache_key = self._cache_key(text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        synthesized = await self._omnivoice.synthesize(text)
        self._remember(cache_key, synthesized)
        return synthesized

    def _cache_key(self, text: str) -> str:
        return " ".join(text.split())

    def _remember(self, cache_key: str, audio: SynthesizedAudio) -> None:
        self._cache[cache_key] = audio
        self._cache_order = [key for key in self._cache_order if key != cache_key]
        self._cache_order.append(cache_key)
        while len(self._cache_order) > self._cache_limit:
            evicted = self._cache_order.pop(0)
            self._cache.pop(evicted, None)


@lru_cache(maxsize=1)
def get_tts_service() -> TTSService:
    return TTSService()
