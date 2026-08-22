import asyncio
import ast
import json
import re
import logging
import time
import uuid
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents import create_agent
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

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
        agent = create_agent(
            model=self._get_model(),
            tools=tools,
            system_prompt=system_prompt,
        )
        result = agent.invoke({"messages": messages})
        if _needs_search_recovery(result):
            logger.warning("Recovering a search intent with a forced LangChain tool call")
            agent = create_agent(
                model=self._get_model(),
                tools=tools,
                system_prompt=system_prompt,
                middleware=[_ForceToolCall("search_properties")],
            )
            return agent.invoke({"messages": messages})
        return result

    def _get_model(self):
        if self._model is not None:
            return self._model

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
        self._model = GemmaChatLlamaCpp(
            model_path=model_path,
            n_gpu_layers=LOCAL_LLM_N_GPU_LAYERS,
            n_ctx=LOCAL_LLM_CONTEXT_TOKENS,
            n_batch=LOCAL_LLM_BATCH_TOKENS,
            n_threads=LOCAL_LLM_THREADS,
            flash_attn=LOCAL_LLM_FLASH_ATTENTION,
            temperature=LOCAL_LLM_TEMPERATURE,
            max_tokens=LOCAL_LLM_MAX_OUTPUT_TOKENS,
            streaming=False,
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


class GemmaChatLlamaCpp(ChatLlamaCpp):
    """Adapt Gemma 4's native tool syntax to LangChain tool calls."""

    def _create_message_dicts(self, messages):
        message_dicts = super()._create_message_dicts(messages)
        rendered = []
        for message, message_dict in zip(messages, message_dicts, strict=True):
            if isinstance(message, AIMessage) and message.tool_calls:
                calls = "".join(
                    f"<|tool_call>call:{call['name']}{{{_gemma_arguments(call.get('args', {}))}}}<tool_call|>"
                    for call in message.tool_calls
                )
                message_dict["content"] = f"{message_dict.get('content') or ''}{calls}"
                message_dict.pop("tool_calls", None)
            elif isinstance(message, ToolMessage):
                result = _gemma_value(message.content)
                message_dict["role"] = "user"
                message_dict["content"] = (
                    f"<|tool_response>response:{message.name or 'tool'}{{result:{result}}}<tool_response|>"
                )
            rendered.append(message_dict)
        return rendered

    def _create_chat_result(self, response: dict) -> ChatResult:
        result = super()._create_chat_result(response)
        generations = []
        for generation in result.generations:
            message = generation.message
            if not isinstance(message, AIMessage) or not isinstance(message.content, str):
                generations.append(generation)
                continue
            parsed = _parse_gemma_tool_call(message.content)
            if parsed is None:
                generations.append(generation)
                continue
            text, name, arguments = parsed
            generations.append(ChatGeneration(
                message=AIMessage(
                    content=text,
                    tool_calls=[{
                        "name": name,
                        "args": arguments,
                        "id": f"call_{uuid.uuid4().hex}",
                        "type": "tool_call",
                    }],
                ),
                generation_info=generation.generation_info,
            ))
        return ChatResult(generations=generations, llm_output=result.llm_output)


class _ForceToolCall(AgentMiddleware):
    def __init__(self, tool_name: str) -> None:
        self._tool_name = tool_name

    def wrap_model_call(self, request: ModelRequest, handler):
        return handler(request.override(
            tool_choice={"type": "function", "function": {"name": self._tool_name}},
        ))


def _needs_search_recovery(result: dict) -> bool:
    messages = result.get("messages", [])
    if any(message.__class__.__name__ == "ToolMessage" for message in messages):
        return False
    if not messages:
        return False
    response = message_text(messages[-1]).casefold()
    all_text = " ".join(message_text(message) for message in messages).casefold()
    search_intent = (
        any(word in response for word in ("search", "check", "find", "listing", "details", "හොය", "බලන්න"))
        or ("property" in response and "තියෙන" in response)
    )
    criteria = sum(bool(marker in all_text) for marker in (
        "colombo", "කොළඹ", "කළම්බෝ", "කොලම්බු",
        "million", "මිලිය",
        "bedroom", "bedrooms", "බෙඩ්",
    ))
    return search_intent and criteria >= 3


def _parse_gemma_tool_call(text: str) -> tuple[str, str, dict] | None:
    match = re.search(r"<tool_call\s*:\s*([A-Za-z_]\w*)\s*\((.*?)\)\s*</tool_call>", text, re.IGNORECASE | re.DOTALL)
    if match is None:
        match = re.search(r"<\|tool_call>\s*call:([A-Za-z_]\w*)\s*\{(.*?)\}\s*<tool_call\|>", text, re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    arguments: dict[str, Any] = {}
    raw_arguments = match.group(2).replace('<|"|>', '"')
    for item in _split_tool_arguments(raw_arguments):
        argument = re.match(r"\s*([A-Za-z_]\w*)\s*(?:=|:)\s*(.*?)\s*$", item, re.DOTALL)
        if argument is None:
            return None
        raw_value = argument.group(2).strip()
        try:
            value = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            value = raw_value.strip('"\'')
        arguments[argument.group(1)] = value
    return text[:match.start()].strip(), match.group(1), arguments


def _gemma_arguments(arguments: dict) -> str:
    return ",".join(f"{key}:{_gemma_value(value)}" for key, value in arguments.items())


def _gemma_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return f'<|"|>{value}<|"|>'


def _split_tool_arguments(arguments: str) -> list[str]:
    items: list[str] = []
    start = 0
    quote: str | None = None
    depth = 0
    for index, char in enumerate(arguments):
        if quote:
            if char == quote and (index == 0 or arguments[index - 1] != "\\"):
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "[{(":
            depth += 1
        elif char in "]})":
            depth -= 1
        elif char == "," and depth == 0:
            items.append(arguments[start:index].strip())
            start = index + 1
    final = arguments[start:].strip()
    if final:
        items.append(final)
    return items


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
