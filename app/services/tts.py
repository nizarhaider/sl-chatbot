import asyncio
import json
import logging
import os
import wave
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO

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
        return "omnivoice"
    logger.warning("Unknown VOICE_OUTPUT_PROVIDER=%s, falling back to gemini_live", value)
    return "gemini_live"


class OmniVoiceRemoteSynthesizer:
    def __init__(self) -> None:
        self.base_url = os.environ.get(
            "OMNIVOICE_REMOTE_BASE_URL",
            "https://k2-fsa-omnivoice.hf.space",
        ).rstrip("/")
        self.api_name = os.environ.get("OMNIVOICE_REMOTE_API_NAME", "_design_fn")
        self.timeout_seconds = float(os.environ.get("OMNIVOICE_REMOTE_TIMEOUT_SECONDS", "120"))
        self.inference_steps = int(os.environ.get("OMNIVOICE_NUM_STEP", "16"))
        self.guidance_scale = float(os.environ.get("OMNIVOICE_GUIDANCE_SCALE", "2.0"))
        self.speed = float(os.environ.get("OMNIVOICE_SPEED", "1.12"))
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


def _env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class TTSService:
    def __init__(self) -> None:
        self.provider = _normalize_provider(os.environ.get("VOICE_OUTPUT_PROVIDER"))
        self._omnivoice = OmniVoiceRemoteSynthesizer() if self.provider == "omnivoice" else None

    def uses_gemini_audio(self) -> bool:
        return self.provider == "gemini_live"

    async def synthesize(self, text: str) -> SynthesizedAudio:
        if self.provider != "omnivoice" or self._omnivoice is None:
            raise RuntimeError(f"TTS provider '{self.provider}' does not support synthesis")
        return await self._omnivoice.synthesize(text)


@lru_cache(maxsize=1)
def get_tts_service() -> TTSService:
    return TTSService()
