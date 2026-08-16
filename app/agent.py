"""Google ADK runtime for the local Gemma real-estate agent."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator, Awaitable, Callable
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
from app.speech import (
    PLACE_NAMES,
    closest_location,
    detect_language,
    is_broad_property_request,
    is_property_location_request,
    known_location,
)

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
        caller_text = _latest_caller_text(messages)
        if _is_post_tool_turn(messages):
            message = {"content": _grounded_fallback(messages), "tool_calls": []}
        elif is_property_location_request(caller_text):
            message = {
                "content": None,
                "tool_calls": [
                    {"function": {"name": "list_property_locations", "arguments": {}}}
                ],
            }
        elif location := _confirmed_location_suggestion(messages):
            message = {
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_properties",
                            "arguments": {"location": location},
                        }
                    }
                ],
            }
        elif location := _location_followup(messages):
            message = {
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_properties",
                            "arguments": {
                                "query": _latest_caller_text(messages),
                                "location": location,
                            },
                        }
                    }
                ],
            }
        elif _awaiting_location(messages):
            message = {
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "name": "list_property_locations",
                            "arguments": {"query": caller_text},
                        }
                    }
                ],
            }
        elif is_broad_property_request(caller_text):
            message = {
                "content": _location_question(_caller_language(messages)),
                "tool_calls": [],
            }
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
                            "role": "assistant",
                            "content": strip_thinking(message.get("content") or ""),
                        },
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
        self.callbacks: dict[str, Callable[[str, dict[str, Any]], Awaitable[None]]] = {}

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
        if callback := self.callbacks.pop(call_id, None):
            await callback(name, arguments)
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
        on_tool_call: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> str:
        if call_id not in self._users:
            await self.start_session(call_id, phone)
        self.tools.traces[call_id] = []
        if on_tool_call:
            self.tools.callbacks[call_id] = on_tool_call
        final_text = ""
        try:
            async for event in self.runner.run_async(
                user_id=self._users[call_id],
                session_id=call_id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text=transcript)]
                ),
                state_delta={"caller_language": detect_language(transcript)},
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
        finally:
            self.tools.callbacks.pop(call_id, None)
        return strip_thinking(final_text)

    def tool_trace(self, call_id: str) -> list[dict[str, Any]]:
        return list(self.tools.traces.get(call_id, []))

    async def end_session(self, call_id: str) -> None:
        user_id = self._users.pop(call_id, None)
        self.tools.traces.pop(call_id, None)
        self.tools.callbacks.pop(call_id, None)
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
        + f"\n\nThe latest caller message is in {name} ({language}). After every tool result, "
        f"the spoken answer must remain entirely in {name}. Tool data is never caller speech. "
        "Only send a location filter when the caller clearly named a location. Noisy words "
        "meaning 'any' or 'some' are not locations; ask for a location before a broad search."
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


def _response_contract_violations(message: dict, messages: list[dict]) -> list[str]:
    response = strip_thinking(message.get("content") or "")
    if message.get("tool_calls") or parse_tool_call(response):
        return []
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
    return not any(
        message.get("role") == "system"
        and "exact property_id" in str(message.get("content", ""))
        for message in messages
    )


def _is_post_tool_turn(messages: list[dict]) -> bool:
    return bool(messages) and "<tool_result>" in str(messages[-1].get("content", ""))


def _caller_language(messages: list[dict]) -> str:
    return detect_language(_latest_caller_text(messages))


def _latest_caller_text(messages: list[dict]) -> str:
    return next(
        (
            str(item.get("content", ""))
            for item in reversed(messages)
            if item.get("role") == "user"
            and "<tool_result>" not in str(item.get("content", ""))
        ),
        "",
    )


def _location_question(language: str) -> str:
    if language == "si":
        return "හරි. ඔබ property එකක් බලන්නේ මොන ප්‍රදේශයෙන්ද?"
    if language == "ta":
        return "சரி. நீங்கள் எந்த பகுதியில் property பார்க்க விரும்புகிறீர்கள்?"
    return "Sure. Which location would you prefer for the property?"


def _location_retry(language: str) -> str:
    if language == "si":
        return "සමාවෙන්න, ප්‍රදේශයේ නම හරියට ඇහුණේ නැහැ. ආයෙත් එක පාරක් කියන්න පුළුවන්ද?"
    if language == "ta":
        return "மன்னிக்கவும், பகுதியின் பெயர் தெளிவாகக் கேட்கவில்லை. இன்னொரு முறை சொல்ல முடியுமா?"
    return "Sorry, I didn't catch the area name. Could you say it once more?"


def _location_suggestion(language: str, location: str) -> str:
    spoken = PLACE_NAMES.get(language, {}).get(location, location)
    if language == "si":
        return f"ඔබ කිව්වේ {spoken} ද?"
    if language == "ta":
        return f"நீங்கள் சொன்னது {spoken} தானா?"
    return f"Did you mean {location}?"


def _confirmed_location_suggestion(messages: list[dict]) -> str | None:
    caller = re.sub(r"[^\w\u0B80-\u0BFF\u0D80-\u0DFF]+", " ", _latest_caller_text(messages).casefold()).strip()
    affirmatives = {"yes", "yeah", "yep", "correct", "ඔව්", "හරි", "ஆம்", "ஆமாம்", "சரி"}
    if caller not in affirmatives:
        return None
    previous = next(
        (
            str(item.get("content", ""))
            for item in reversed(messages[:-1])
            if item.get("role") == "assistant"
        ),
        "",
    )
    if not any(marker in previous for marker in ("Did you mean", "ඔබ කිව්වේ", "நீங்கள் சொன்னது")):
        return None
    return known_location(previous)


def _location_followup(messages: list[dict]) -> str | None:
    location = known_location(_latest_caller_text(messages))
    return location if location and _awaiting_location(messages) else None


def _awaiting_location(messages: list[dict]) -> bool:
    previous = next(
        (
            str(item.get("content", ""))
            for item in reversed(messages[:-1])
            if item.get("role") == "assistant"
        ),
        "",
    )
    previous_caller = next(
        (
            str(item.get("content", ""))
            for item in reversed(messages[:-1])
            if item.get("role") == "user"
            and "<tool_result>" not in str(item.get("content", ""))
        ),
        "",
    )
    questions = (_location_question(language) for language in ("en", "si", "ta"))
    location_lists = (
        "Properties are currently available in:",
        "properties තියෙන්නේ මේ ප්‍රදේශවලයි:",
        "properties உள்ளன:",
    )
    folded_previous = previous.casefold()
    model_location_question = (
        any(marker in folded_previous for marker in ("location", "ප්‍රදේශ", "பகுதி"))
        and any(marker in previous for marker in ("?", "කියන්න", "சொல்ல"))
    )
    return (
        any(question in previous for question in questions)
        or is_broad_property_request(previous_caller)
        or any(marker in previous for marker in location_lists)
        or model_location_question
    )


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
    if result.get("needs_clarification") == "location":
        return _location_question(language)
    properties = result.get("properties") or []
    if properties:
        if len(properties) == 1:
            row = properties[0]
            name = str(row.get("name") or "Property")
            location = str(row.get("location") or "Sri Lanka")
            spoken_location = PLACE_NAMES.get(language, {}).get(location, location)
            price = row.get("price_lkr")
            amount = f"{price:,}" if isinstance(price, int) else "—"
            if language == "si":
                return (
                    f"{spoken_location} තියෙන {name} එක රුපියල් {amount}කට තියෙනවා. "
                    "ඒ ගැන තව විස්තර කියන්නද?"
                )
            if language == "ta":
                return (
                    f"{spoken_location} பகுதியில் உள்ள {name} விலை {amount} ரூபாய். "
                    "அதைப் பற்றி மேலும் சொல்லவா?"
                )
            return (
                f"I found {name} in {location} for LKR {amount}. "
                "Would you like to hear more about it?"
            )
        choices = []
        for row in properties[:2]:
            name = str(row.get("name") or "Property")
            location = str(row.get("location") or "Sri Lanka")
            spoken_location = PLACE_NAMES.get(language, {}).get(location, location)
            price = row.get("price_lkr")
            amount = f"{price:,}" if isinstance(price, int) else "—"
            if language == "si":
                choices.append(
                    f"{spoken_location} {name} එක රුපියල් {amount}කට"
                )
            elif language == "ta":
                choices.append(
                    f"{spoken_location} பகுதியில் {name}, {amount} ரூபாய்"
                )
            else:
                choices.append(f"{name} in {location} for LKR {amount}")
        if language == "si":
            return (
                f"ඔබට ගැළපෙන තැන් දෙකක් හම්බ වුණා. {choices[0]}, "
                f"අනිත් එක {choices[1]}. වැඩි විස්තර ඕනේ මොන එක ගැනද?"
            )
        if language == "ta":
            return (
                f"உங்களுக்கு பொருத்தமான இரண்டு வாய்ப்புகள் கிடைத்துள்ளன. {choices[0]}; "
                f"இன்னொன்று {choices[1]}. எதைப் பற்றி மேலும் கேட்க விரும்புகிறீர்கள்?"
            )
        return (
            f"I found two good options for you: {choices[0]}, and {choices[1]}. "
            "Which one would you like to hear more about?"
        )

    filters = result.get("search_arguments") or {}
    if filters.get("query") and len(filters) == 1:
        if language == "si":
            return "Property නම පැහැදිලිව ඇහුණේ නැහැ. කරුණාකර නම නැවත කියන්න."
        if language == "ta":
            return "Property பெயர் தெளிவாக கேட்கவில்லை. தயவுசெய்து பெயரை மீண்டும் சொல்லுங்கள்."
        return "I didn't catch the property name clearly. Please repeat it."

    locations = result.get("locations") or result.get("available_locations") or []
    if locations:
        if query := str(result.get("location_query") or "").strip():
            suggestion = closest_location(query, list(map(str, locations)))
            return (
                _location_suggestion(language, suggestion)
                if suggestion
                else _location_retry(language)
            )
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
