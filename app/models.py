"""Lazy local ASR, LLM, and TTS model wrappers."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from app.config import (
    ASR_LANGUAGE,
    ASR_MODEL,
    LLM_BATCH,
    LLM_CONTEXT,
    LLM_FILENAME,
    LLM_GPU_LAYERS,
    LLM_MAX_TOKENS,
    LLM_REPO,
    LLM_REVISION,
    LLM_TEMPERATURE,
    LLM_THREADS,
    SYSTEM_PROMPT,
    TTS_DATASET,
    TTS_DATASET_REVISION,
    TTS_LANGUAGE,
    TTS_MODEL,
    TTS_REFERENCE_FILE,
    TTS_REFERENCE_TEXT,
    TTS_REVISION,
    TTS_SPEED,
    TTS_STEPS,
)
from app.database import TOOL_INSTRUCTIONS

logger = logging.getLogger(__name__)


class LocalWhisperASR:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._processor = None
        self._model = None

    def prewarm(self) -> None:
        self._get_model()

    def transcribe(self, waveform: np.ndarray) -> str:
        processor, model = self._get_model()
        inputs = processor(
            waveform,
            sampling_rate=16_000,
            return_attention_mask=True,
            return_tensors="pt",
        )
        features = inputs.input_features.to(
            self.device, dtype=next(model.parameters()).dtype
        )
        attention = getattr(inputs, "attention_mask", None)
        if attention is not None:
            attention = attention.to(self.device)

        import torch

        with torch.inference_mode():
            tokens = model.generate(
                features,
                attention_mask=attention,
                task="transcribe",
                language=ASR_LANGUAGE,
            )
        text = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        return "" if is_noise_text(text) else text

    def _get_model(self):
        if self._model is not None:
            return self._processor, self._model

        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        logger.info("Loading Whisper %s on %s", ASR_MODEL, self.device)
        self._processor = AutoProcessor.from_pretrained(ASR_MODEL)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            ASR_MODEL,
            dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa",
        ).to(self.device)
        self._model.eval()
        return self._processor, self._model


class LocalGemmaLLM:
    def __init__(self) -> None:
        self._llm = None
        self._lock = asyncio.Lock()

    async def prewarm(self) -> None:
        await asyncio.to_thread(self._prime)

    async def generate(
        self,
        transcript: str,
        history: list[dict[str, str]],
        continuation: list[dict[str, str]] | None = None,
    ) -> str:
        async with self._lock:
            return await asyncio.to_thread(
                self._generate,
                transcript,
                history,
                continuation or [],
            )

    def _generate(self, transcript, history, continuation) -> str:
        response = self._get_model().create_chat_completion(
            messages=[
                self._system_message(),
                *history,
                {"role": "user", "content": transcript},
                *continuation,
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return strip_thinking(text)

    def _prime(self) -> None:
        """Evaluate the stable system prefix once before the first live turn."""
        self._get_model().create_chat_completion(
            messages=[self._system_message(), {"role": "user", "content": " "}],
            temperature=0,
            max_tokens=1,
        )

    @staticmethod
    def _system_message() -> dict[str, str]:
        local_date = datetime.now(ZoneInfo("Asia/Colombo")).date().isoformat()
        return {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\n\n{TOOL_INSTRUCTIONS}\n\nSri Lanka date: {local_date}.",
        }

    def _get_model(self):
        if self._llm is not None:
            return self._llm

        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        model_path = hf_hub_download(
            repo_id=LLM_REPO,
            filename=LLM_FILENAME,
            revision=LLM_REVISION,
        )
        started = time.perf_counter()
        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=LLM_GPU_LAYERS,
            n_ctx=LLM_CONTEXT,
            n_batch=LLM_BATCH,
            n_threads=LLM_THREADS,
            flash_attn=True,
            verbose=False,
        )
        logger.info("Gemma loaded in %.1f seconds", time.perf_counter() - started)
        return self._llm


class OmniVoiceTTS:
    """Direct OmniVoice inference without the RealtimeTTS compatibility layer."""

    def __init__(self, device: str = "cuda:0", dtype: str = "float16") -> None:
        self.device = device
        self.dtype = dtype
        self.sample_rate = 24_000
        self._model = None
        self._reference_audio: str | None = None
        self._lock = asyncio.Lock()

    async def prewarm(self) -> None:
        await asyncio.to_thread(self._get_model)

    async def speak(self, text: str, on_audio_chunk) -> float:
        async with self._lock:
            started = time.perf_counter()
            waveform = await asyncio.to_thread(self.synthesize, text)
            pcm = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            chunk_bytes = self.sample_rate * 2 // 5  # 200 ms
            for offset in range(0, len(pcm), chunk_bytes):
                on_audio_chunk(pcm[offset : offset + chunk_bytes], self.sample_rate)
            seconds = len(waveform) / self.sample_rate
            logger.info(
                "OmniVoice complete: elapsed_ms=%.0f chars=%s audio_ms=%.0f",
                (time.perf_counter() - started) * 1000,
                len(text),
                seconds * 1000,
            )
            return seconds

    def synthesize(self, text: str, seed: int | None = None) -> np.ndarray:
        if not text.strip():
            raise ValueError("Text is required")
        model = self._get_model()

        import torch

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            audio = model.generate(
                text=text.strip(),
                language=TTS_LANGUAGE,
                ref_audio=self._reference_audio,
                ref_text=TTS_REFERENCE_TEXT,
                num_step=TTS_STEPS,
                speed=TTS_SPEED,
            )
        return normalize_waveform(audio)

    def _get_model(self):
        if self._model is not None:
            return self._model

        import torch
        from huggingface_hub import hf_hub_download, snapshot_download
        from omnivoice import OmniVoice

        self._reference_audio = hf_hub_download(
            repo_id=TTS_DATASET,
            repo_type="dataset",
            revision=TTS_DATASET_REVISION,
            filename=TTS_REFERENCE_FILE,
        )
        logger.info(
            "Loading OmniVoice %s@%s on %s", TTS_MODEL, TTS_REVISION, self.device
        )
        model_path = snapshot_download(repo_id=TTS_MODEL, revision=TTS_REVISION)
        self._model = OmniVoice.from_pretrained(
            model_path,
            device_map=self.device,
            dtype=getattr(torch, self.dtype),
            load_asr=False,
        )
        self.sample_rate = self._model.sampling_rate
        return self._model


def is_noise_text(text: str) -> bool:
    cleaned = text.strip()
    return bool(cleaned) and all(char in "_- .,\n\t" for char in cleaned)


def strip_thinking(text: str) -> str:
    text = text.strip()
    if re.search(r"<think\b", text, re.IGNORECASE) and not re.search(
        r"</think>", text, re.IGNORECASE
    ):
        return ""
    text = re.sub(
        r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(r"^.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def normalize_waveform(audio) -> np.ndarray:
    waveform = audio[0] if isinstance(audio, (list, tuple)) else audio
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    return np.squeeze(np.asarray(waveform)).astype(np.float32, copy=False)
