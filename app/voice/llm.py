import asyncio
import logging
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from app.voice.config import (
    HOMELANDS_LOCAL_SYSTEM_PROMPT,
    LOCAL_LLM_BATCH_TOKENS,
    LOCAL_LLM_CONTEXT_TOKENS,
    LOCAL_LLM_FLASH_ATTENTION,
    LOCAL_LLM_MAX_OUTPUT_TOKENS,
    LOCAL_LLM_MODEL_DIR,
    LOCAL_LLM_MODEL_FILENAME,
    LOCAL_LLM_MODEL_PATH,
    LOCAL_LLM_MODEL_REPO,
    LOCAL_LLM_N_GPU_LAYERS,
    LOCAL_LLM_TEMPERATURE,
    LOCAL_LLM_THREADS,
)
from app.voice.tools import TOOL_INSTRUCTIONS

logger = logging.getLogger(__name__)


class LocalGemmaLLM:
    def __init__(self) -> None:
        self._llm = None
        self._lock = asyncio.Lock()

    async def prewarm(self) -> None:
        await asyncio.to_thread(self._get_llm)

    async def generate(
        self,
        transcript_text: str,
        history: list[dict[str, str]],
        continuation: list[dict[str, str]] | None = None,
    ) -> str:
        async with self._lock:
            return await asyncio.to_thread(self._generate_sync, transcript_text, history, continuation or [])

    def _generate_sync(
        self,
        transcript_text: str,
        history: list[dict[str, str]],
        continuation: list[dict[str, str]],
    ) -> str:
        local_time = datetime.now(ZoneInfo("Asia/Colombo")).isoformat(timespec="minutes")
        response = self._get_llm().create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{HOMELANDS_LOCAL_SYSTEM_PROMPT}\n\n{TOOL_INSTRUCTIONS}\n\n"
                        f"Current Sri Lanka date and time: {local_time}."
                    ),
                },
                *history,
                {"role": "user", "content": transcript_text},
                *continuation,
            ],
            temperature=LOCAL_LLM_TEMPERATURE,
            max_tokens=LOCAL_LLM_MAX_OUTPUT_TOKENS,
        )
        text = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        cleaned = _strip_thinking_blocks(text)
        if text and not cleaned:
            logger.info("Gemma response contained thinking without a final answer; raw_chars=%s", len(text))
        return cleaned

    def _get_llm(self):
        if self._llm is not None:
            return self._llm

        from llama_cpp import Llama

        model_path = self._resolve_model_path()
        logger.info(
            "Loading local Gemma model: path=%s n_gpu_layers=%s n_ctx=%s n_batch=%s n_threads=%s flash_attn=%s",
            model_path,
            LOCAL_LLM_N_GPU_LAYERS,
            LOCAL_LLM_CONTEXT_TOKENS,
            LOCAL_LLM_BATCH_TOKENS,
            LOCAL_LLM_THREADS,
            LOCAL_LLM_FLASH_ATTENTION,
        )
        started = time.perf_counter()
        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=LOCAL_LLM_N_GPU_LAYERS,
            n_ctx=LOCAL_LLM_CONTEXT_TOKENS,
            n_batch=LOCAL_LLM_BATCH_TOKENS,
            n_threads=LOCAL_LLM_THREADS,
            flash_attn=LOCAL_LLM_FLASH_ATTENTION,
            verbose=False,
        )
        logger.info("Local Gemma model loaded in %.0f ms", (time.perf_counter() - started) * 1000.0)
        return self._llm

    def _resolve_model_path(self) -> str:
        if LOCAL_LLM_MODEL_PATH:
            return LOCAL_LLM_MODEL_PATH

        from huggingface_hub import HfApi, hf_hub_download

        filename = LOCAL_LLM_MODEL_FILENAME
        if not filename:
            gguf_files = [
                path
                for path in HfApi().list_repo_files(LOCAL_LLM_MODEL_REPO)
                if path.lower().endswith(".gguf") and "q4" in path.lower()
            ]
            if not gguf_files:
                raise RuntimeError(
                    f"No Q4 GGUF file found in Hugging Face repo {LOCAL_LLM_MODEL_REPO!r}; "
                    "update LOCAL_LLM_MODEL_FILENAME or LOCAL_LLM_MODEL_PATH in app/voice/config.py."
                )
            filename = sorted(gguf_files)[0]

        kwargs = {"repo_id": LOCAL_LLM_MODEL_REPO, "filename": filename}
        if LOCAL_LLM_MODEL_DIR:
            kwargs["local_dir"] = LOCAL_LLM_MODEL_DIR
        return hf_hub_download(**kwargs)


def _strip_thinking_blocks(text: str) -> str:
    text = text.strip()
    if re.search(r"<think\b", text, flags=re.IGNORECASE) and not re.search(
        r"</think>",
        text,
        flags=re.IGNORECASE,
    ):
        return ""
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(
        r"(?is)^\s*(thinking|reasoning|analysis)\s*:\s*.*?(final|answer)\s*:\s*",
        "",
        text,
    )
    return text.strip()
