"""Google ADK runtime for the local Gemma real-estate agent."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.run_config import RunConfig
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.tools import ToolContext
from google.genai import types
from pydantic import PrivateAttr

from app.config import LLM_HISTORY_MESSAGES, LLM_MAX_TOKENS, LLM_TEMPERATURE
from app.database import (
    CallContext,
    RealEstateToolService,
    ToolCall,
    parse_tool_call,
    tool_call_message,
)
from app.models import LocalGemmaLLM, strip_thinking
from app.speech import detect_language

logger = logging.getLogger(__name__)
APP_NAME = "serendibai_whatsapp"
MODEL_NAME = "local/gemma-4-26b-a4b"
LANGUAGE_NAMES = {"en": "English", "si": "Sinhala", "ta": "Tamil"}


class LocalGemmaAdkModel(BaseLlm):
    """Expose the in-process llama.cpp Gemma model through ADK's model API."""

    model: str = MODEL_NAME
    _backend: LocalGemmaLLM = PrivateAttr()

    def __init__(self, backend: LocalGemmaLLM | None = None) -> None:
        super().__init__(model=MODEL_NAME)
        self._backend = backend or LocalGemmaLLM()

    async def prewarm(self) -> None:
        await self._backend.prewarm()

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream  # The phone runtime consumes complete, non-streaming turns.
        messages = _chat_messages(llm_request)
        tools = _openai_tools(llm_request)
        message = await self._backend.chat(messages, tools)
        parts = _response_parts(message)
        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            partial=False,
        )


class PropertyAgentTools:
    """ADK function tools backed by the existing validated Neon service."""

    def __init__(self, service: RealEstateToolService | None) -> None:
        self.service = service
        self.traces: dict[str, list[dict[str, Any]]] = {}

    async def search_properties(
        self,
        tool_context: ToolContext,
        query: str = "",
        location: str = "",
        property_type: str = "",
        bedrooms: int | None = None,
        max_price_lkr: int | None = None,
    ) -> dict:
        """Find up to three relevant properties using semantic search and stated filters.

        Args:
            query: Caller's complete natural-language property request.
            location: Requested location in English.
            property_type: Requested property type in English.
            bedrooms: Minimum number of bedrooms.
            max_price_lkr: Maximum price in Sri Lankan rupees.
        """
        arguments = {
            key: value
            for key, value in {
                "query": query,
                "location": location,
                "property_type": property_type,
                "bedrooms": bedrooms,
                "max_price_lkr": max_price_lkr,
            }.items()
            if value not in (None, "")
        }
        return await self._execute("search_properties", arguments, tool_context)

    async def list_property_locations(
        self, tool_context: ToolContext, query: str = ""
    ) -> dict:
        """List every location that currently has active property inventory."""
        arguments = {"query": query} if query else {}
        result = await self._execute("list_property_locations", arguments, tool_context)
        if query:
            result["location_query"] = query
        return result

    async def book_appointment(
        self,
        tool_context: ToolContext,
        property_id: str = "",
        customer_name: str = "",
        appointment_at: str = "",
    ) -> dict:
        """Book a property viewing after all required caller details are known.

        Args:
            property_id: Exact property ID returned by a property search.
            customer_name: Caller's stated name.
            appointment_at: Caller-requested date and time in ISO 8601 format.
        """
        property_id = property_id or str(
            tool_context.state.get("last_property_id", "")
        )
        if customer_name:
            tool_context.state["booking_customer_name"] = customer_name
        if appointment_at:
            tool_context.state["booking_appointment_at"] = appointment_at
        customer_name = customer_name or str(
            tool_context.state.get("booking_customer_name", "")
        )
        appointment_at = appointment_at or str(
            tool_context.state.get("booking_appointment_at", "")
        )
        if not property_id:
            return {"ok": True, "needs_clarification": "property"}
        if not customer_name:
            return {"ok": True, "needs_clarification": "customer_name"}
        if not appointment_at:
            return {"ok": True, "needs_clarification": "appointment_at"}
        result = await self._execute(
            "book_appointment",
            {
                "property_id": property_id,
                "customer_name": customer_name,
                "appointment_at": appointment_at,
            },
            tool_context,
        )
        if result.get("ok"):
            tool_context.state["booking_customer_name"] = ""
            tool_context.state["booking_appointment_at"] = ""
        return result

    async def _execute(
        self, name: str, arguments: dict[str, Any], tool_context: ToolContext
    ) -> dict:
        call_id = str(tool_context.state.get("call_id", ""))
        safe_arguments = {
            key: "[provided]" if key == "customer_name" else value
            for key, value in arguments.items()
        }
        logger.info(
            "ADK tool call for %s: name=%s arguments=%s",
            call_id,
            name,
            safe_arguments,
        )
        if self.service is None:
            result = {"ok": False, "error": "The booking database is not configured."}
        else:
            context = CallContext(
                call_id=call_id,
                caller_phone=str(tool_context.state.get("caller_phone", "")),
            )
            result = await self.service.execute(ToolCall(name, arguments), context)
        result = dict(result)
        if name == "search_properties":
            result["search_arguments"] = arguments
        if name == "search_properties" and result.get("ok"):
            properties = result.get("properties") or []
            if len(properties) == 1 and properties[0].get("property_id"):
                tool_context.state["last_property_id"] = str(
                    properties[0]["property_id"]
                )
                tool_context.state["last_property_name"] = str(
                    properties[0].get("name") or ""
                )
            else:
                tool_context.state["last_property_id"] = ""
                tool_context.state["last_property_name"] = ""
        logger.info(
            "ADK tool result for %s: name=%s ok=%s count=%s error=%s",
            call_id,
            name,
            result.get("ok"),
            result.get("count"),
            result.get("error"),
        )
        self.traces.setdefault(call_id, []).append(
            {"name": name, "arguments": arguments, "result": result}
        )
        return result


class GemmaAgentRuntime:
    """Own ADK sessions, conversational state, tool dispatch, and model turns."""

    def __init__(
        self,
        model: LocalGemmaAdkModel | None = None,
        tool_service: RealEstateToolService | None = None,
    ) -> None:
        self.model = model or LocalGemmaAdkModel()
        self.tool_service = (
            tool_service
            if tool_service is not None
            else RealEstateToolService.from_env()
        )
        self.tools = PropertyAgentTools(self.tool_service)
        self.sessions = InMemorySessionService()
        self.agent = LlmAgent(
            name="serendibai_property_agent",
            model=self.model,
            instruction=_agent_instruction,
            tools=[
                self.tools.search_properties,
                self.tools.list_property_locations,
                self.tools.book_appointment,
            ],
            generate_content_config=types.GenerateContentConfig(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_TOKENS,
            ),
        )
        self.runner = Runner(
            app_name=APP_NAME,
            agent=self.agent,
            session_service=self.sessions,
        )
        self._users: dict[str, str] = {}

    async def prewarm(self) -> None:
        if self.tool_service:
            await self.tool_service.ensure_ready()
        await self.model.prewarm()

    async def start_session(self, call_id: str, phone: str) -> None:
        user_id = phone or call_id
        existing = await self.sessions.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=call_id
        )
        if existing is None:
            await self.sessions.create_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=call_id,
                state={"call_id": call_id, "caller_phone": phone},
            )
        self._users[call_id] = user_id

    async def respond(
        self,
        call_id: str,
        phone: str,
        transcript: str,
        language: str | None = None,
    ) -> str:
        if call_id not in self._users:
            await self.start_session(call_id, phone)
        self.tools.traces[call_id] = []
        final_text = ""
        async for event in self.runner.run_async(
            user_id=self._users[call_id],
            session_id=call_id,
            new_message=types.Content(
                role="user", parts=[types.Part(text=transcript)]
            ),
            state_delta={
                "caller_language": language or detect_language(transcript)
            },
            run_config=RunConfig(
                max_llm_calls=3,
                get_session_config=GetSessionConfig(
                    num_recent_events=LLM_HISTORY_MESSAGES
                ),
            ),
        ):
            if event.is_final_response() and event.content:
                final_text = " ".join(
                    part.text.strip()
                    for part in event.content.parts or []
                    if part.text and part.text.strip()
                )
        return strip_thinking(final_text)

    def tool_trace(self, call_id: str) -> list[dict[str, Any]]:
        return list(self.tools.traces.get(call_id, []))

    async def end_session(self, call_id: str) -> None:
        user_id = self._users.pop(call_id, None)
        self.tools.traces.pop(call_id, None)
        if user_id:
            await self.sessions.delete_session(
                app_name=APP_NAME, user_id=user_id, session_id=call_id
            )

    async def close(self) -> None:
        for call_id in list(self._users):
            await self.end_session(call_id)
        if self.tool_service:
            await self.tool_service.close()


def _chat_messages(llm_request: LlmRequest) -> list[dict]:
    messages = [{"role": "system", "content": _instruction_text(llm_request)}]
    contents = list(llm_request.contents)
    while contents and any(
        part.function_response for part in (contents[0].parts or [])
    ):
        contents.pop(0)
    for content in contents:
        text_parts: list[str] = []
        for part in content.parts or []:
            if part.text:
                text_parts.append(part.text)
            elif part.function_call:
                text_parts.append(
                    tool_call_message(
                        ToolCall(
                            part.function_call.name or "",
                            dict(part.function_call.args or {}),
                        )
                    )
                )
            elif part.function_response:
                text_parts.append(
                    "This is tool data, not caller speech. Keep the caller's language from the "
                    "system instruction.\n<tool_result>"
                    + json.dumps(
                        part.function_response.response or {}, ensure_ascii=False
                    )
                    + "</tool_result>"
                )
        if text_parts:
            messages.append(
                {
                    "role": "assistant" if content.role == "model" else "user",
                    "content": "\n".join(text_parts),
                }
            )
    return _alternating_chat_messages(messages)


def _alternating_chat_messages(messages: list[dict]) -> list[dict]:
    """Keep the newest complete turns accepted by Gemma's strict chat template."""
    system = messages[:1]
    turns: list[dict] = []
    for message in messages[1:]:
        role = message.get("role")
        if not turns and role != "user":
            continue
        if turns and turns[-1].get("role") == role:
            turns[-1] = message
        else:
            turns.append(message)
    return [*system, *turns]


def _agent_instruction(context: ReadonlyContext) -> str:
    language = str(context.state.get("caller_language", "en"))
    name = LANGUAGE_NAMES.get(language, "English")
    instruction = (
        LocalGemmaLLM.system_instruction()
        + f"\n\nContinue this call in {name} ({language}). After every tool result, "
        f"the spoken answer must remain entirely in {name}. Tool data is never caller speech. "
        "Only send a location filter when the caller clearly named a location. Noisy words "
        "meaning 'any' or 'some' are not locations; ask for a location before a broad search."
    )
    now = datetime.now(ZoneInfo("Asia/Colombo")).replace(microsecond=0).isoformat()
    instruction += (
        f"\nCurrent Sri Lanka date and time: {now}. Convert relative viewing times to ISO 8601. "
        "For a viewing, collect the caller's name and date/time one missing detail at a time. "
        "Call book_appointment only after both are known; its state preserves details across "
        "turns. If the caller sounds frustrated, acknowledge briefly and continue from the next "
        "missing detail."
    )
    if property_id := str(context.state.get("last_property_id", "")):
        property_name = str(context.state.get("last_property_name", "the property"))
        instruction += (
            f"\nThe latest searched property is {property_name}, with exact property_id "
            f"{property_id}. Use that ID when the caller refers to it in a booking follow-up."
        )
    return instruction


def _instruction_text(llm_request: LlmRequest) -> str:
    instruction = llm_request.config.system_instruction
    if isinstance(instruction, str):
        return instruction
    if instruction and instruction.parts:
        return "\n".join(part.text for part in instruction.parts if part.text)
    return ""


def _openai_tools(llm_request: LlmRequest) -> list[dict]:
    result = []
    for tool in llm_request.config.tools or []:
        for declaration in tool.function_declarations or []:
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": declaration.name,
                        "description": declaration.description or "",
                        "parameters": declaration.parameters_json_schema
                        or {"type": "object", "properties": {}},
                    },
                }
            )
    return result


def _response_parts(message: dict) -> list[types.Part]:
    parts: list[types.Part] = []
    text = strip_thinking(message.get("content") or "")
    for raw_call in message.get("tool_calls") or []:
        function = raw_call.get("function", {})
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if function.get("name") and isinstance(arguments, dict):
            parts.append(
                types.Part.from_function_call(name=function["name"], args=arguments)
            )
    if not parts and (call := parse_tool_call(text)):
        parts.append(types.Part.from_function_call(name=call.name, args=call.arguments))
    if not parts:
        parts.append(types.Part(text=text))
    return parts
