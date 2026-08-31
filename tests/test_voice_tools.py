import asyncio

import app.voice.tools as voice_tools
from app.voice.tools import CallContext, LLM_TOOLS, RealEstateToolService
from app.voice.turn_pipeline import LocalGemmaTurnPipeline


class FakeVectorStore:
    def search_properties(self, query: str) -> list[dict]:
        return [{"property_id": "property-1", "query": query}]


class FakeLlm:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = iter(responses)
        self.requests: list[list[dict]] = []

    async def chat(self, messages, tools=None, language=None) -> dict:
        self.requests.append(list(messages))
        return next(self.responses)


class FakeTools:
    async def execute(self, name: str, arguments: dict, context: CallContext) -> dict:
        assert name == "search_properties"
        assert arguments == {"query": "an apartment in Malabe"}
        assert context.caller_phone == "94770000000"
        return {"ok": True, "properties": [{"property_id": "property-1"}]}


def test_production_tool_contract_exposes_property_search() -> None:
    names = {item["function"]["name"] for item in LLM_TOOLS}

    assert {"search_properties", "book_appointment", "send_whatsapp_message"} <= names


def test_tool_service_uses_the_vector_store_for_property_search() -> None:
    service = RealEstateToolService(store=object(), vector_store=FakeVectorStore())

    result = asyncio.run(
        service.execute(
            "search_properties",
            {"query": "an apartment in Malabe"},
            CallContext(call_id="call-1", caller_phone="94770000000"),
        )
    )

    assert result == {
        "ok": True,
        "properties": [{"property_id": "property-1", "query": "an apartment in Malabe"}],
        "count": 1,
    }


def test_whatsapp_message_is_sent_to_the_callers_number(monkeypatch) -> None:
    sent_messages: list[tuple[str, str]] = []

    async def send_text_message(phone_number: str, message: str) -> bool:
        sent_messages.append((phone_number, message))
        return True

    monkeypatch.setattr(voice_tools.whatsapp_api, "send_text_message", send_text_message)
    service = RealEstateToolService(store=object(), vector_store=FakeVectorStore())

    result = asyncio.run(
        service.execute(
            "send_whatsapp_message",
            {"message": "Your viewing is confirmed."},
            CallContext(call_id="call-1", caller_phone="94742530708"),
        )
    )

    assert result == {"ok": True, "message_sent": True}
    assert sent_messages == [("94742530708", "Your viewing is confirmed.")]


def test_turn_pipeline_executes_openai_style_tool_call_before_answering() -> None:
    pipeline = LocalGemmaTurnPipeline.__new__(LocalGemmaTurnPipeline)
    pipeline._conversation_history = {}
    pipeline._call_languages = {}
    pipeline._trace_event = None
    pipeline._llm = FakeLlm([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "search-1",
                "function": {
                    "name": "search_properties",
                    "arguments": '{"query":"an apartment in Malabe"}',
                },
            }],
        },
        {"role": "assistant", "content": "I found an apartment in Malabe."},
    ])
    pipeline._tools = FakeTools()

    response = asyncio.run(
        pipeline._generate_response("call-1", "94770000000", "Find an apartment in Malabe")
    )

    assert response == "I found an apartment in Malabe."
    assert pipeline._llm.requests[1][-1]["role"] == "tool"
    assert "property-1" in pipeline._llm.requests[1][-1]["content"]
