import logging

import numpy as np

from app.voice.config import WHISPER_DEVICE, WHISPER_LANGUAGE, WHISPER_MODEL, WHISPER_TASK

logger = logging.getLogger(__name__)


class LocalWhisperASR:
    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._device = None

    def prewarm(self) -> None:
        self._get_model()

    def transcribe(self, pcm_array: np.ndarray) -> str:
        text = self._transcribe(pcm_array)
        if is_noise_text(text):
            logger.info("Dropping noise-only transcript: %r", text)
            return ""
        return text

    def _transcribe(self, pcm_array: np.ndarray) -> str:
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

        generate_kwargs = {"task": WHISPER_TASK}
        if WHISPER_LANGUAGE:
            generate_kwargs["language"] = WHISPER_LANGUAGE

        import torch

        with torch.inference_mode():
            generated_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

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
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa",
        ).to(device)
        self._model.eval()
        self._device = device
        return self._processor, self._model, self._device


def is_noise_text(text: str) -> bool:
    cleaned = text.strip()
    return bool(cleaned) and all(char in "_- .,\n\t" for char in cleaned)
