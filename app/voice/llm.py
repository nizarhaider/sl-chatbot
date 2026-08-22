import asyncio
import logging
import time
from datetime import datetime
from typing import Any
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

logger = logging.getLogger(__name__)


class LocalGemmaAgent:
    def __init__(self) -> None:
        self._model = None
        self._lock = asyncio.Lock()

    async def prewarm(self) -> None:
        await asyncio.to_thread(self._get_model)

    async def invoke(self, messages: list[Any], tools: list, system_prompt: str) -> dict:
        async with self._lock:
            return await asyncio.to_thread(self._invoke_sync, messages, tools, system_prompt)

    def _invoke_sync(self, messages: list[Any], tools: list, system_prompt: str) -> dict:
        from langchain.agents import create_agent

        agent = create_agent(
            model=self._get_model(),
            tools=tools,
            system_prompt=system_prompt,
        )
        return agent.invoke({"messages": messages})

    def _get_model(self):
        if self._model is not None:
            return self._model

        from langchain_community.chat_models import ChatLlamaCpp

        model_path = self._resolve_model_path()
        logger.info(
            "Loading local Gemma model through LangChain: path=%s n_gpu_layers=%s n_ctx=%s n_batch=%s n_threads=%s flash_attn=%s",
            model_path,
            LOCAL_LLM_N_GPU_LAYERS,
            LOCAL_LLM_CONTEXT_TOKENS,
            LOCAL_LLM_BATCH_TOKENS,
            LOCAL_LLM_THREADS,
            LOCAL_LLM_FLASH_ATTENTION,
        )
        started = time.perf_counter()
        self._model = ChatLlamaCpp(
            model_path=model_path,
            n_gpu_layers=LOCAL_LLM_N_GPU_LAYERS,
            n_ctx=LOCAL_LLM_CONTEXT_TOKENS,
            n_batch=LOCAL_LLM_BATCH_TOKENS,
            n_threads=LOCAL_LLM_THREADS,
            flash_attn=LOCAL_LLM_FLASH_ATTENTION,
            temperature=LOCAL_LLM_TEMPERATURE,
            max_tokens=LOCAL_LLM_MAX_OUTPUT_TOKENS,
            verbose=False,
        )
        logger.info("LangChain local Gemma model loaded in %.0f ms", (time.perf_counter() - started) * 1000.0)
        return self._model

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


def agent_system_prompt() -> str:
    local_time = datetime.now(ZoneInfo("Asia/Colombo")).isoformat(timespec="minutes")
    return f"{HOMELANDS_LOCAL_SYSTEM_PROMPT}\n\nCurrent Sri Lanka date and time: {local_time}."


def message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        ).strip()
    return str(content).strip()
