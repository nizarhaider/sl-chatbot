import asyncio

from app.voice.tools import (
    CallContext,
    RealEstateToolService,
    ToolCall,
    _appointment_time,
    parse_tool_call,
    tool_call_message,
)
from app.voice.turn_pipeline import LocalGemmaTurnPipeline


class FakeStore:
    def __init__(self) -> None:
        self.booking = None

    def search_properties(self, arguments: dict) -> list[dict]:
        return [{"property_id": "property-1", "location": arguments.get("location")}]

    def book_appointment(self, arguments: dict, context: CallContext) -> dict:
        self.booking = (arguments, context)
        return {"status": "booked", "property_id": arguments["property_id"]}


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.continuations = []

    async def generate(self, transcript, history, continuation):
        self.continuations.append(list(continuation))
        return next(self.responses)


class FakeTools:
    def __init__(self) -> None:
        self.calls = []

    async def execute(self, call, context):
        self.calls.append((call, context))
        if call.name == "search_properties":
            return {"ok": True, "properties": [{"property_id": "property-1"}]}
        return {"ok": True, "appointment": {"status": "booked"}}


def test_tool_call_round_trip() -> None:
    call = parse_tool_call(
        '<tool_call>{"name":"search_properties","arguments":{"location":"Malabe"}}</tool_call>'
    )

    assert call == ToolCall(name="search_properties", arguments={"location": "Malabe"})
    assert parse_tool_call(tool_call_message(call)) == call
    assert parse_tool_call("Here are the properties.") is None
    assert parse_tool_call("<tool_call>not-json</tool_call>") is None


def test_tool_service_passes_call_context_to_booking() -> None:
    store = FakeStore()
    service = RealEstateToolService(store)
    context = CallContext(call_id="call-1", caller_phone="94770000000")
    call = ToolCall(
        name="book_appointment",
        arguments={
            "property_id": "property-1",
            "customer_name": "Nimal",
            "appointment_at": "2099-01-01T10:00:00+05:30",
        },
    )

    result = asyncio.run(service.execute(call, context))

    assert result["ok"] is True
    assert store.booking == (call.arguments, context)


def test_tool_service_rejects_unknown_tool() -> None:
    result = asyncio.run(
        RealEstateToolService(FakeStore()).execute(
            ToolCall(name="delete_everything", arguments={}),
            CallContext(call_id="call-1", caller_phone=""),
        )
    )

    assert result == {"ok": False, "error": "Unknown tool: delete_everything"}


def test_naive_appointment_time_uses_sri_lanka_timezone() -> None:
    parsed = _appointment_time("2099-01-01T10:00:00")

    assert parsed.isoformat() == "2099-01-01T10:00:00+05:30"


def test_turn_pipeline_can_search_then_book_before_answering() -> None:
    pipeline = LocalGemmaTurnPipeline.__new__(LocalGemmaTurnPipeline)
    pipeline._conversation_history = {}
    pipeline._llm = FakeLLM(
        [
            '<tool_call>{"name":"search_properties","arguments":{"location":"Malabe"}}</tool_call>',
            '<tool_call>{"name":"book_appointment","arguments":{"property_id":"property-1","customer_name":"Nimal","appointment_at":"2099-01-01T10:00:00+05:30"}}</tool_call>',
            "Your viewing is booked.",
        ]
    )
    pipeline._tools = FakeTools()

    response = asyncio.run(
        pipeline._generate_response("call-1", "94770000000", "Book a Malabe viewing for Nimal")
    )

    assert response == "Your viewing is booked."
    assert [call.name for call, _ in pipeline._tools.calls] == ["search_properties", "book_appointment"]
    assert pipeline._tools.calls[-1][1].caller_phone == "94770000000"
    assert "<tool_result>" in pipeline._llm.continuations[-1][-1]["content"]


def test_turn_pipeline_never_speaks_a_malformed_tool_call() -> None:
    pipeline = LocalGemmaTurnPipeline.__new__(LocalGemmaTurnPipeline)
    pipeline._conversation_history = {}
    pipeline._llm = FakeLLM(["<tool_call>not-json</tool_call>"])
    pipeline._tools = FakeTools()

    response = asyncio.run(pipeline._generate_response("call-1", "", "Find me a house"))

    assert response == "Sorry, I couldn't complete that request. Please try again."
