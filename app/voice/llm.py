import asyncio
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.voice.config import HOMELANDS_LOCAL_SYSTEM_PROMPT, VLLM_BASE_URL, VLLM_MODEL, VLLM_TEMPERATURE


class VllmAgent:
    """LangChain agent backed by the local OpenAI-compatible vLLM server."""

    def __init__(self) -> None:
        self._model = ChatOpenAI(
            base_url=VLLM_BASE_URL,
            api_key="local-vllm",
            model=VLLM_MODEL,
            temperature=VLLM_TEMPERATURE,
            max_tokens=128,
        )
        self._lock = asyncio.Lock()

    async def prewarm(self) -> None:
        await self._model.ainvoke("Reply with OK.")

    async def invoke(self, messages: list[Any], tools: list, system_prompt: str) -> dict:
        async with self._lock:
            agent = create_agent(model=self._model, tools=tools, system_prompt=system_prompt)
            return await agent.ainvoke({"messages": messages})


def agent_system_prompt() -> str:
    local_time = datetime.now(ZoneInfo("Asia/Colombo")).isoformat(timespec="minutes")
    return f"{HOMELANDS_LOCAL_SYSTEM_PROMPT}\n\nCurrent Sri Lanka date and time: {local_time}."


def message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict)).strip()
    return str(content).strip()
