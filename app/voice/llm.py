import asyncio
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from app.voice.config import HOMELANDS_LOCAL_SYSTEM_PROMPT, LLM_MODEL, LLM_TEMPERATURE


class LocalLlmClient:
    """In-process 4-bit Transformers client for the local Gemma checkpoint."""

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._lock = asyncio.Lock()

    async def prewarm(self) -> None:
        await self.chat([{"role": "user", "content": "Reply with OK."}])

    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        async with self._lock:
            return await asyncio.to_thread(self._generate, messages, tools)

    async def summarize_search(self, transcript_text: str, search_result: dict) -> str:
        message = await self.chat([{
            "role": "user",
            "content": (
                "A property search has already completed. Answer the caller using only its "
                "result. Do not call tools or ask again for details already present.\n"
                f"Caller request: {transcript_text}\n"
                f"Search result: {json.dumps(search_result, ensure_ascii=False)}"
            ),
        }])
        return message_text(message)

    def _generate(self, messages: list[dict], tools: list[dict] | None) -> dict:
        import torch

        processor, model = self._get_model()
        prompt = processor.apply_chat_template(
            [{"role": "system", "content": agent_system_prompt()}, *messages],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = processor(text=prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        generation = {"max_new_tokens": 128, "do_sample": LLM_TEMPERATURE > 0}
        if LLM_TEMPERATURE > 0:
            generation["temperature"] = LLM_TEMPERATURE
        with torch.inference_mode():
            outputs = model.generate(**inputs, **generation)
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        parsed = processor.parse_response(response)
        return _as_openai_message(parsed, response)

    def _get_model(self):
        if self._model is not None:
            return self._processor, self._model

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        token = os.getenv("HF_TOKEN") or None
        self._processor = AutoProcessor.from_pretrained(LLM_MODEL, token=token)
        self._model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            token=token,
            device_map="auto",
            quantization_config=quantization,
            attn_implementation="sdpa",
        )
        self._model.eval()
        return self._processor, self._model


def agent_system_prompt() -> str:
    local_time = datetime.now(ZoneInfo("Asia/Colombo")).isoformat(timespec="minutes")
    return f"{HOMELANDS_LOCAL_SYSTEM_PROMPT}\n\nCurrent Sri Lanka date and time: {local_time}."


def message_text(message: dict) -> str:
    content = message.get("content") or ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict)).strip()
    return str(content).strip()


def _as_openai_message(parsed, raw_response: str) -> dict:
    if not isinstance(parsed, dict):
        return {"role": "assistant", "content": str(parsed or raw_response).strip()}

    message = {"role": "assistant", "content": parsed.get("content") or ""}
    tool_calls = parsed.get("tool_calls") or []
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message
