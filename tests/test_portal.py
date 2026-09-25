import asyncio

from app.voice.gemini_live import GeminiLivePipeline
from app.voice.portal import PortalTools, _usage_from_events
from app.voice.tools import CallContext


def test_call_usage_preserves_each_provider_breakdown():
    usage = {"total_token_count": 42, "prompt_tokens_details": [{"modality": "AUDIO", "token_count": 25}]}
    assert _usage_from_events([
        {"kind": "gemini_live.usage", "data": {"total_tokens": 42, "usage": usage}},
        {"kind": "gemini_live.interrupted", "data": {}},
    ]) == {"provider": "google_gemini_live", "requests": [usage]}
    assert _usage_from_events([]) is None


def test_portal_instructions_replace_legacy_business_and_disable_all_tools():
    pipeline = GeminiLivePipeline.__new__(GeminiLivePipeline)
    config = pipeline._session_config({"instructions": "You help airline passengers.", "greeting": "Welcome aboard", "enabled_tools": [], "voice": "Aoede", "languages": ["English"]})
    assert "airline passengers" in config["system_instruction"]
    assert "property" not in config["system_instruction"]
    assert "tools" not in config
    assert config["speech_config"]["voice_config"]["prebuilt_voice_config"]["voice_name"] == "Aoede"


def test_portal_exposes_only_enabled_tools():
    pipeline = GeminiLivePipeline.__new__(GeminiLivePipeline)
    config = pipeline._session_config({"greeting": "Hello", "enabled_tools": ["search_products"]})
    assert [t["name"] for t in config["tools"][0]["function_declarations"]] == ["search_products"]


def test_portal_exposes_booking_tool_when_enabled():
    pipeline = GeminiLivePipeline.__new__(GeminiLivePipeline)
    config = pipeline._session_config({"greeting": "Hello", "enabled_tools": ["book_appointment"]})
    declaration = config["tools"][0]["function_declarations"][0]
    assert declaration["name"] == "book_appointment"
    assert set(declaration["parameters"]["required"]) == {"customer_name", "service", "appointment_at"}


def test_portal_exposes_order_and_ticket_tools_when_enabled():
    pipeline = GeminiLivePipeline.__new__(GeminiLivePipeline)
    config = pipeline._session_config({"greeting": "Hello", "enabled_tools": ["create_order", "create_ticket"]})
    declarations = config["tools"][0]["function_declarations"]
    assert [tool["name"] for tool in declarations] == ["create_order", "create_ticket"]


def test_disabled_tool_is_rejected_at_execution():
    tools = PortalTools.__new__(PortalTools)

    async def config():
        return {"enabled_tools": []}

    tools.agent_config = config
    result = asyncio.run(tools.execute("search_products", {"query": "flights"}, CallContext("test", "")))
    assert result == {"ok": False, "error": "This tool is disabled."}
