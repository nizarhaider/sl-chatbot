import asyncio
import logging
import time

from app.voice.config import (
    GEMMA_BATCH_TOKENS,
    GEMMA_CONTEXT_TOKENS,
    GEMMA_MAX_OUTPUT_TOKENS,
    GEMMA_MODEL_DIR,
    GEMMA_MODEL_FILENAME,
    GEMMA_MODEL_PATH,
    GEMMA_MODEL_REPO,
    GEMMA_N_GPU_LAYERS,
    GEMMA_TEMPERATURE,
    GEMMA_THREADS,
    HOMELANDS_LOCAL_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class LocalGemmaLLM:
    def __init__(self) -> None:
        self._llm = None
        self._lock = asyncio.Lock()

    async def prewarm(self) -> None:
        await asyncio.to_thread(self._get_llm)

    async def generate(self, transcript_text: str, history: list[dict[str, str]]) -> str:
        async with self._lock:
            return await asyncio.to_thread(self._generate_sync, transcript_text, history)

    def _generate_sync(self, transcript_text: str, history: list[dict[str, str]]) -> str:
        response = self._get_llm().create_chat_completion(
            messages=[
                {"role": "system", "content": HOMELANDS_LOCAL_SYSTEM_PROMPT},
                *history,
                {"role": "user", "content": transcript_text},
            ],
            temperature=GEMMA_TEMPERATURE,
            max_tokens=GEMMA_MAX_OUTPUT_TOKENS,
        )
        return (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

    def _get_llm(self):
        if self._llm is not None:
            return self._llm

        from llama_cpp import Llama

        model_path = self._resolve_model_path()
        logger.info(
            "Loading local Gemma model: path=%s n_gpu_layers=%s n_ctx=%s n_batch=%s",
            model_path,
            GEMMA_N_GPU_LAYERS,
            GEMMA_CONTEXT_TOKENS,
            GEMMA_BATCH_TOKENS,
        )
        started = time.perf_counter()
        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=GEMMA_N_GPU_LAYERS,
            n_ctx=GEMMA_CONTEXT_TOKENS,
            n_batch=GEMMA_BATCH_TOKENS,
            n_threads=GEMMA_THREADS,
            verbose=False,
        )
        logger.info("Local Gemma model loaded in %.0f ms", (time.perf_counter() - started) * 1000.0)
        return self._llm

    def _resolve_model_path(self) -> str:
        if GEMMA_MODEL_PATH:
            return GEMMA_MODEL_PATH

        from huggingface_hub import HfApi, hf_hub_download

        filename = GEMMA_MODEL_FILENAME
        if not filename:
            q4_files = [
                path
                for path in HfApi().list_repo_files(GEMMA_MODEL_REPO)
                if path.lower().endswith(".gguf") and "q4_0" in path.lower()
            ]
            if not q4_files:
                raise RuntimeError(
                    f"No Q4_0 GGUF file found in Hugging Face repo {GEMMA_MODEL_REPO!r}; "
                    "set GEMMA_MODEL_FILENAME or GEMMA_MODEL_PATH."
                )
            filename = sorted(q4_files)[0]

        kwargs = {"repo_id": GEMMA_MODEL_REPO, "filename": filename}
        if GEMMA_MODEL_DIR:
            kwargs["local_dir"] = GEMMA_MODEL_DIR
        return hf_hub_download(**kwargs)
