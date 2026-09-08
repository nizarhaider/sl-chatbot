import asyncio

from app.voice.gemini_live import GeminiLivePipeline
from app.voice.tools import CallContext


class FakeTools:
    async def execute(self, name: str, arguments: dict, context: CallContext) -> dict:
        assert name == "search_properties"
        assert arguments == {"query": "Malabe"}
        assert context.caller_phone == "94770000000"
        return {"ok": True, "count": 1}


class FakeSession:
    def __init__(self) -> None:
        self.responses = []

    async def send_tool_response(self, *, function_responses) -> None:
        self.responses = function_responses


class FakeCall:
    id = "tool-1"
    name = "search_properties"
    args = {"query": "Malabe"}


def test_gemini_tool_response_uses_live_result_envelope() -> None:
    pipeline = GeminiLivePipeline.__new__(GeminiLivePipeline)
    pipeline._tools = FakeTools()
    session = FakeSession()

    asyncio.run(
        pipeline._handle_tool_calls(
            session,
            [FakeCall()],
            "call-1",
            CallContext(call_id="call-1", caller_phone="94770000000"),
        )
    )

    assert session.responses[0].response == {"result": {"ok": True, "count": 1}}


def test_gemini_instruction_leaves_language_selection_to_live_audio() -> None:
    pipeline = GeminiLivePipeline.__new__(GeminiLivePipeline)

    instruction = pipeline._session_config()["system_instruction"]

    assert "Sinhala" in instruction
    assert "Sri Lankan woman" in instruction
    assert "never wait for text input" in instruction
