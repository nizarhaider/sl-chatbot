import logging
import re

import numpy as np

from app.voice.config import (
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MAX_NEW_TOKENS,
    WHISPER_MODEL,
    WHISPER_TASK,
)

logger = logging.getLogger(__name__)

class LocalWhisperASR:
    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._device = None

    def prewarm(self) -> None:
        self._get_model()

    def transcribe(self, pcm_array: np.ndarray, language: str | None = None) -> str:
        return self._transcribe(pcm_array, language=language)

    def _transcribe(self, pcm_array: np.ndarray, language: str | None = None) -> str:
        processor, model, device = self._get_model()
        inputs = processor(
            pcm_array,
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
        )
        model_dtype = next(model.parameters()).dtype
        input_features = inputs.input_features.to(device, dtype=model_dtype)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        import torch

        with torch.inference_mode():
            generated_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                language=language or WHISPER_LANGUAGE,
                max_new_tokens=WHISPER_MAX_NEW_TOKENS,
                task=WHISPER_TASK,
            )
        return processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def _get_model(self):
        if self._model is not None:
            return self._processor, self._model, self._device

        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        device = torch.device(WHISPER_DEVICE)
        dtype = torch.float16 if WHISPER_DEVICE.startswith("cuda") else torch.float32
        logger.info(
            "Loading Hugging Face Whisper ASR model: %s (device: %s dtype: %s)",
            WHISPER_MODEL,
            device,
            dtype,
        )
        self._processor = AutoProcessor.from_pretrained(WHISPER_MODEL)
        model_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
            "attn_implementation": "sdpa",
        }
        if device.type == "cuda":
            # Load directly onto the GPU. A separate CPU -> GPU `.to()` can
            # spend tens of minutes copying the sharded Whisper checkpoint.
            model_kwargs["device_map"] = {"": device.index or 0}
        logger.info("Loading Whisper weights into %s", device)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(WHISPER_MODEL, **model_kwargs)
        self._model.eval()
        self._device = device
        logger.info("Whisper model loaded into %s and set to eval", device)
        return self._processor, self._model, self._device


def is_noise_text(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return False
    if all(char in "_- .,\n\t" for char in cleaned):
        return True
    tokens = re.findall(r"\w+", cleaned.casefold())
    return len(tokens) >= 6 and len(set(tokens)) <= 2
