import asyncio

from app.voice.gemini_live import GeminiLivePipeline
from app.voice.portal import PortalTools
from app.voice.tools import CallContext


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


def test_disabled_tool_is_rejected_at_execution():
    tools = PortalTools.__new__(PortalTools)

    async def config():
        return {"enabled_tools": []}

    tools.agent_config = config
    result = asyncio.run(tools.execute("search_products", {"query": "flights"}, CallContext("test", "")))
    assert result == {"ok": False, "error": "This tool is disabled."}
