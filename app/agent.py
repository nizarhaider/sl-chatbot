"""Google ADK runtime for the local Gemma real-estate agent."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator
from difflib import SequenceMatcher
from typing import Any

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
MODEL_NAME = "local/gemma-4-e4b"
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
        if _is_post_tool_turn(messages):
            message = {"content": _grounded_fallback(messages), "tool_calls": []}
        else:
            message = await self._backend.chat(messages, tools)
            for _ in range(1):
                violations = _response_contract_violations(message, messages)
                if not violations:
                    break
                logger.info("Correcting Gemma response: %s", "; ".join(violations))
                message = await self._backend.chat(
                    [
                        *messages,
                        {
                            "role": "user",
                            "content": _correction_prompt(messages, violations),
                        },
                    ],
                    tools,
                )
            violations = _response_contract_violations(message, messages)
            if violations:
                logger.info("Using safe response fallback: %s", "; ".join(violations))
                message = {
                    "content": _safe_fallback(messages, violations),
                    "tool_calls": [],
                }
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
        """Search active properties using the caller's stated filters.

        Args:
            query: Property name or free-text property query.
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

    async def list_property_locations(self, tool_context: ToolContext) -> dict:
        """List every location that currently has active property inventory."""
        return await self._execute("list_property_locations", {}, tool_context)

    async def book_appointment(
        self,
        tool_context: ToolContext,
        property_id: str,
        customer_name: str,
        appointment_at: str,
    ) -> dict:
        """Book a property viewing after all required caller details are known.

        Args:
            property_id: Exact property ID returned by a property search.
            customer_name: Caller's stated name.
            appointment_at: Caller-requested date and time in ISO 8601 format.
        """
        return await self._execute(
            "book_appointment",
            {
                "property_id": property_id,
                "customer_name": customer_name,
                "appointment_at": appointment_at,
            },
            tool_context,
        )

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
                tool_context.state.pop("last_property_id", None)
                tool_context.state.pop("last_property_name", None)
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
    ) -> str:
        if call_id not in self._users:
            await self.start_session(call_id, phone)
        self.tools.traces[call_id] = []
        final_text = ""
        async for event in self.runner.run_async(
            user_id=self._users[call_id],
            session_id=call_id,
            new_message=types.Content(role="user", parts=[types.Part(text=transcript)]),
            state_delta={
                "caller_language": detect_language(transcript),
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
    return messages


def _agent_instruction(context: ReadonlyContext) -> str:
    language = str(context.state.get("caller_language", "en"))
    name = LANGUAGE_NAMES.get(language, "English")
    instruction = (
        LocalGemmaLLM.system_instruction()
        + f"\n\nThe latest caller message is in {name} ({language}). After every tool result, "
        f"the spoken answer must remain entirely in {name}. Tool data is never caller speech. "
        "Only send a location filter when the caller clearly named a location. Noisy words "
        "meaning 'any' or 'some' are not locations; omit location for a broad request."
    )
    if property_id := str(context.state.get("last_property_id", "")):
        property_name = str(context.state.get("last_property_name", "the property"))
        instruction += (
            f"\nThe latest searched property is {property_name}, with exact property_id "
            f"{property_id}. Use that ID when the caller refers to it in a booking follow-up."
        )
    else:
        instruction += (
            "\nNo property is currently selected from a prior search. Never say 'this property' "
            "or offer, arrange, or book a viewing unless the caller identifies an exact property "
            "by name or ID in the latest message. Never invent a date or time."
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


def _response_contract_violations(message: dict, messages: list[dict]) -> list[str]:
    if message.get("tool_calls"):
        return []
    response = strip_thinking(message.get("content") or "")
    violations = []
    if _repeats_previous_answer(response, messages):
        violations.append(
            "do not repeat the previous answer; address the latest request directly"
        )
    if re.search(r"ඔබතුමි(?:ය|යා)|ඔබතුමා", response):
        violations.append("use the gender-neutral Sinhala address ඔබ")
    if _no_property_selected(messages) and re.search(
        r"\b(?:book|booking|appointment|arrange|viewing|this property|that property)\b|"
        r"මේ property|මෙම property|booking|appointment|arrange|වෙන් කළ|හවස|උදේ|"
        r"இந்த property|appointment|பதிவு",
        response,
        re.IGNORECASE,
    ):
        violations.append(
            "no property is selected; ask the caller to choose one before a viewing"
        )
    if len(response) > 320:
        violations.append("keep the spoken reply under 320 characters")
    if len(response) >= 180 and not re.search(r"[.!?।]\s*$", response):
        violations.append("finish the reply at a complete sentence")
    return violations


def _no_property_selected(messages: list[dict]) -> bool:
    return any(
        message.get("role") == "system"
        and "No property is currently selected" in str(message.get("content", ""))
        for message in messages
    )


def _is_post_tool_turn(messages: list[dict]) -> bool:
    return bool(messages) and "<tool_result>" in str(messages[-1].get("content", ""))


def _caller_language(messages: list[dict]) -> str:
    caller_text = next(
        (
            str(item.get("content", ""))
            for item in reversed(messages)
            if item.get("role") == "user"
            and "<tool_result>" not in str(item.get("content", ""))
        ),
        "",
    )
    return detect_language(caller_text)


def _correction_prompt(messages: list[dict], violations: list[str]) -> str:
    language = _caller_language(messages)
    if language == "si":
        return (
            "අලුත්ම caller ඉල්ලීමට සෘජුව පිළිතුරු දෙන්න. කලින් පිළිතුර නැවත කියන්න එපා. "
            "ප්‍රයෝජනවත් කෙටි වාක්‍ය එකක් සිට තුනක් දක්වා සිංහලෙන් කියන්න."
        )
    if language == "ta":
        return (
            "அழைப்பாளரின் சமீபத்திய கோரிக்கைக்கு நேரடியாகப் பதிலளிக்கவும். முந்தைய பதிலை "
            "மீண்டும் சொல்ல வேண்டாம். தமிழ் எழுத்துகளை மட்டும் பயன்படுத்தி ஒன்று முதல் மூன்று "
            "பயனுள்ள குறுகிய வாக்கியங்கள் பேசவும். சிங்கள எழுத்துகளை ஒருபோதும் பயன்படுத்த வேண்டாம்; "
            "ஆங்கிலப் பெயர்களும் எண்களும் மட்டும் விதிவிலக்கு."
        )
    return (
        "Rewrite only the final spoken answer entirely in English. Fix these violations: "
        + "; ".join(violations)
        + ". Give one to three useful short sentences and answer the latest request directly."
    )


def _repeats_previous_answer(response: str, messages: list[dict]) -> bool:
    current = " ".join(re.findall(r"\w+", response.casefold()))
    if not current:
        return False
    previous = next(
        (
            str(item.get("content", ""))
            for item in reversed(messages)
            if item.get("role") == "assistant"
            and "<tool_call>" not in str(item.get("content", ""))
        ),
        "",
    )
    prior = " ".join(re.findall(r"\w+", previous.casefold()))
    if not prior:
        return False
    return current == prior or SequenceMatcher(None, current, prior).ratio() >= 0.84


def _grounded_fallback(messages: list[dict]) -> str:
    """Render a concise answer when Gemma cannot satisfy the post-tool contract."""
    language = _caller_language(messages)
    result = _latest_tool_result(messages)
    properties = result.get("properties") or []
    if properties:
        if len(properties) == 1:
            row = properties[0]
            name = str(row.get("name") or "Property")
            location = str(row.get("location") or "Sri Lanka")
            price = row.get("price_lkr")
            amount = f"{price:,}" if isinstance(price, int) else "—"
            if language == "si":
                return f"{name}, {location} — LKR {amount}. මේ listing එක ගැන වැඩි විස්තර ඕනෙද?"
            if language == "ta":
                return f"{name}, {location} — LKR {amount}. இந்த listing பற்றி மேலும் விவரம் வேண்டுமா?"
            return f"{name}, {location} — LKR {amount}. Would you like more details?"
        listings = []
        for row in properties[:3]:
            name = str(row.get("name") or "Property")
            location = str(row.get("location") or "Sri Lanka")
            price = row.get("price_lkr")
            amount = f"{price:,}" if isinstance(price, int) else "—"
            listings.append(f"{name}, {location} — LKR {amount}")
        joined = "; ".join(listings)
        if language == "si":
            return f"දැනට listings: {joined}. ඔබ කැමති property එකේ නම කියන්න."
        if language == "ta":
            return (
                f"தற்போதைய listings: {joined}. உங்களுக்கு பிடித்த property பெயரை சொல்லுங்கள்."
            )
        return f"Current listings: {joined}. Tell me which property you prefer."

    filters = result.get("search_arguments") or {}
    if filters.get("query") and len(filters) == 1:
        if language == "si":
            return "Property නම පැහැදිලිව ඇහුණේ නැහැ. කරුණාකර නම නැවත කියන්න."
        if language == "ta":
            return "Property பெயர் தெளிவாக கேட்கவில்லை. தயவுசெய்து பெயரை மீண்டும் சொல்லுங்கள்."
        return "I didn't catch the property name clearly. Please repeat it."

    locations = result.get("locations") or result.get("available_locations") or []
    if locations:
        joined = ", ".join(map(str, locations))
        if language == "si":
            return f"දැනට properties තියෙන්නේ මේ ප්‍රදේශවලයි: {joined}."
        if language == "ta":
            return f"தற்போது இந்த இடங்களில் properties உள்ளன: {joined}."
        return f"Properties are currently available in: {joined}."

    if appointment := result.get("appointment"):
        name = str(appointment.get("property_name") or "the property")
        when = str(appointment.get("appointment_at") or "the requested time")
        if language == "si":
            return f"{name} බලන්න {when} වෙලාවට appointment එක වෙන් කළා."
        if language == "ta":
            return f"{name} பார்வைக்கு {when} நேரத்தில் appointment பதிவு செய்யப்பட்டது."
        return f"Your viewing for {name} is booked for {when}."

    if language == "si":
        return "ඒ වැඩේ සම්පූර්ණ කරන්න බැරි වුණා. අවශ්‍ය විස්තර නැවත කියන්න."
    if language == "ta":
        return "அதை முடிக்க முடியவில்லை. தேவையான விவரங்களை மீண்டும் சொல்லுங்கள்."
    return "I couldn't complete that. Please repeat the required details."


def _safe_fallback(messages: list[dict], violations: list[str]) -> str:
    language = _caller_language(messages)
    needs_property = any("no property is selected" in item for item in violations)
    if language == "si":
        return (
            "මුලින් ඔබ කැමති property එකේ නම කියන්න."
            if needs_property
            else "කරුණාකර ඒක තව වරක් පැහැදිලිව කියන්න."
        )
    if language == "ta":
        return (
            "முதலில் உங்களுக்கு பிடித்த property பெயரை சொல்லுங்கள்."
            if needs_property
            else "தயவுசெய்து அதை மீண்டும் தெளிவாக சொல்லுங்கள்."
        )
    return (
        "Please choose the property first."
        if needs_property
        else "Please say that again clearly."
    )


def _latest_tool_result(messages: list[dict]) -> dict:
    for message in reversed(messages):
        content = str(message.get("content", ""))
        matches = list(
            re.finditer(r"<tool_result>(.*?)</tool_result>", content, re.DOTALL)
        )
        if matches:
            try:
                result = json.loads(matches[-1].group(1))
            except json.JSONDecodeError:
                return {}
            return result if isinstance(result, dict) else {}
    return {}
