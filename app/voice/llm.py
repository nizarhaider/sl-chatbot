import asyncio
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from app.voice.config import HOMELANDS_LOCAL_SYSTEM_PROMPT, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE


class LocalLlmClient:
    """Small OpenAI-compatible client for the local llama.cpp server."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(base_url=LLM_BASE_URL, timeout=60.0)
        self._lock = asyncio.Lock()

    async def prewarm(self) -> None:
        await self.chat([{"role": "user", "content": "Reply with OK."}])

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        language: str | None = None,
    ) -> dict:
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "system", "content": agent_system_prompt(language)}, *messages],
            "temperature": LLM_TEMPERATURE,
            "max_tokens": 128,
        }
        if tools:
            payload.update({"tools": tools, "tool_choice": "auto"})

        async with self._lock:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        return response.json()["choices"][0]["message"]

    async def classify_language(self, transcript_text: str) -> str | None:
        """Classify a caller's language-selection utterance without keyword matching."""
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a language picker for a phone call. Decide whether the caller's "
                        "utterance is confidently Sinhala, Tamil, or English. Return exactly one "
                        "token: si, ta, en, or unclear. Do not explain."
                    ),
                },
                {"role": "user", "content": transcript_text},
            ],
            "temperature": 0.0,
            "max_tokens": 4,
        }
        async with self._lock:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        result = response.json()["choices"][0]["message"].get("content") or ""
        normalized = result.strip().casefold()
        for language in ("si", "ta", "en"):
            if normalized == language or normalized.startswith(language + " "):
                return language
        return None

    async def summarize_search(
        self,
        transcript_text: str,
        search_result: dict,
        language: str | None = None,
    ) -> str:
        message = await self.chat([{
            "role": "user",
            "content": (
                "A property search has already completed. Answer the caller using only its "
                "result. Do not call tools or ask again for details already present.\n"
                f"Caller request: {transcript_text}\n"
                f"Search result: {json.dumps(search_result, ensure_ascii=False)}"
            ),
        }], language=language)
        return message_text(message)


def agent_system_prompt(language: str | None = None) -> str:
    local_time = datetime.now(ZoneInfo("Asia/Colombo")).isoformat(timespec="minutes")
    language_lock = {
        "si": "The application has locked this call to Sinhala. Use Sinhala only; never switch languages.",
        "ta": "The application has locked this call to Tamil. Use Tamil only; never switch languages.",
        "en": "The application has locked this call to English. Use English only; never switch languages.",
    }.get(language, "")
    return f"{HOMELANDS_LOCAL_SYSTEM_PROMPT}\n\n{language_lock}\nCurrent Sri Lanka date and time: {local_time}."


def message_text(message: dict) -> str:
    content = message.get("content") or ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict)).strip()
    return str(content).strip()
