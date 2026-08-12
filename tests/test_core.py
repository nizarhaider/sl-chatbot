import asyncio

from app.database import (
    CallContext,
    CallLog,
    RealEstateToolService,
    ToolCall,
    appointment_time,
    format_transcript,
    parse_tool_call,
    tool_call_message,
)
from app.models import is_noise_text, strip_thinking
from app.pipeline import TurnPipeline, repetitive
from app.speech import detect_language, normalize_for_tts
from app.whatsapp import refine_sdp


class FakePropertyStore:
    def __init__(self) -> None:
        self.booking = None

    def search(self, arguments: dict) -> list[dict]:
        return [{"property_id": "property-1", "location": arguments.get("location")}]

    def book(self, arguments: dict, context: CallContext) -> dict:
        self.booking = arguments, context
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


class FakeCallStore:
    def __init__(self) -> None:
        self.calls = []

    def save(self, call: dict) -> None:
        self.calls.append(call)


class FakeTTS:
    async def speak(self, text, on_audio_chunk, language=None):
        on_audio_chunk(b"\0\0" * 240, 24_000)
        return 0.01


class FakeOutputTrack:
    def __init__(self) -> None:
        self.chunks = []

    def add_pcm(self, chunk, sample_rate):
        self.chunks.append((chunk, sample_rate))


def bare_pipeline(responses: list[str]) -> TurnPipeline:
    pipeline = TurnPipeline.__new__(TurnPipeline)
    pipeline.history = {}
    pipeline.llm = FakeLLM(responses)
    pipeline.tools = FakeTools()
    return pipeline


def test_tool_call_round_trip_and_malformed_closer() -> None:
    call = parse_tool_call(
        '<tool_call>{"name":"search_properties","arguments":{"location":"Malabe"}}</tool_call>'
    )
    malformed_close = parse_tool_call(
        '<tool_call>{"name":"search_properties","arguments":{"location":"Dehiwala"}}<tool_call|>'
    )

    assert call == ToolCall("search_properties", {"location": "Malabe"})
    assert parse_tool_call(tool_call_message(call)) == call
    assert malformed_close == ToolCall("search_properties", {"location": "Dehiwala"})
    assert parse_tool_call("<tool_call>not-json</tool_call>") is None


def test_tool_service_booking_and_unknown_tool() -> None:
    store = FakePropertyStore()
    service = RealEstateToolService(store)
    context = CallContext("call-1", "94770000000")
    booking = ToolCall(
        "book_appointment",
        {
            "property_id": "property-1",
            "customer_name": "Nimal",
            "appointment_at": "2099-01-01T10:00:00+05:30",
        },
    )

    result = asyncio.run(service.execute(booking, context))
    unknown = asyncio.run(service.execute(ToolCall("delete_everything", {}), context))

    assert result["ok"] is True
    assert store.booking == (booking.arguments, context)
    assert unknown == {"ok": False, "error": "Unknown tool: delete_everything"}


def test_pipeline_searches_then_books_without_speaking_tool_json() -> None:
    pipeline = bare_pipeline(
        [
            '<tool_call>{"name":"search_properties","arguments":{"location":"Malabe"}}</tool_call>',
            '<tool_call>{"name":"book_appointment","arguments":{"property_id":"property-1"}}</tool_call>',
            "Your viewing is booked.",
        ]
    )

    response = asyncio.run(pipeline._respond("call-1", "94770000000", "Book a viewing"))

    assert response == "Your viewing is booked."
    assert [call.name for call, _ in pipeline.tools.calls] == [
        "search_properties",
        "book_appointment",
    ]
    assert pipeline.tools.calls[-1][1].caller_phone == "94770000000"
    assert "<tool_result>" in pipeline.llm.continuations[-1][-1]["content"]


def test_pipeline_rejects_malformed_tool_json() -> None:
    pipeline = bare_pipeline(["<tool_call>not-json</tool_call>"])
    response = asyncio.run(pipeline._respond("call-1", "", "Find a house"))
    assert response == "Sorry, I couldn't complete that request. Please try again."


def test_spoken_audio_is_queued_before_echo_guard_runs() -> None:
    async def exercise():
        pipeline = TurnPipeline.__new__(TurnPipeline)
        pipeline.tts = FakeTTS()
        output = FakeOutputTrack()
        await pipeline._speak("call-1", "Hello", output, {"call-1": 0})
        return output.chunks

    chunks = asyncio.run(exercise())
    assert chunks == [(b"\0\0" * 240, 24_000)]


def test_call_log_keeps_neon_copy_readable() -> None:
    async def exercise():
        store = FakeCallStore()
        log = CallLog(store)
        log.start("call-1", "94770000000")
        log.active("call-1", "94770000000")
        log.add("call-1", "caller", "Hello")
        log.add("call-1", "assistant", "How can I help?")
        log.end("call-1")
        await log.close()
        return store.calls

    calls = asyncio.run(exercise())
    assert [call["status"] for call in calls] == [
        "connecting",
        "active",
        "active",
        "active",
        "ended",
    ]
    assert calls[-1]["status"] == "ended"
    assert (
        format_transcript(calls[-1]["transcript"])
        == "Caller: Hello · Assistant: How can I help?"
    )


def test_small_text_and_time_helpers() -> None:
    assert (
        appointment_time("2099-01-01T10:00:00").isoformat()
        == "2099-01-01T10:00:00+05:30"
    )
    assert strip_thinking("<think>private</think> Hello") == "Hello"
    assert is_noise_text("___ ---")
    assert repetitive("please wait please wait please wait while I check")
    assert not repetitive("yes yes yes")


def test_tts_uses_spoken_sinhala_numbers_and_place_names() -> None:
    spoken, language = normalize_for_tts(
        "Malabe apartment එක රුපියල් 28,000,000යි?", "si"
    )
    assert language == "si"
    assert spoken == "මාලබේ apartment එක රුපියල් මිලියන විසි අටයි."
    assert detect_language("සිංහලෙන් කතා කරන්න") == "si"
    assert detect_language("தமிழில் பேசுங்கள்") == "ta"
    assert normalize_for_tts("All set.", "en")[0] == "All set."


def test_whatsapp_sdp_is_reduced_to_expected_audio_shape() -> None:
    sdp = """v=0
o=- 1 2 IN IP4 0.0.0.0
a=group:BUNDLE 0
a=mid:0
a=setup:actpass
a=fingerprint:sha-256 AA:BB
a=extmap:1 urn:test"""
    refined = refine_sdp(sdp)
    assert "a=mid:audio" in refined
    assert "a=setup:active" in refined
    assert "a=group:BUNDLE audio" in refined
    assert "a=extmap:" not in refined
