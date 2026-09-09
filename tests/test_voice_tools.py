import asyncio

import app.voice.tools as voice_tools
from app.voice.tools import CallContext, LLM_TOOLS, RealEstateToolService


class FakeVectorStore:
    def search_properties(self, query: str) -> list[dict]:
        return [{"property_id": "property-1", "query": query}]


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


def test_whatsapp_message_uses_an_alternate_recipient(monkeypatch) -> None:
    sent_messages: list[tuple[str, str]] = []

    async def send_text_message(phone_number: str, message: str) -> bool:
        sent_messages.append((phone_number, message))
        return True

    monkeypatch.setattr(voice_tools.whatsapp_api, "send_text_message", send_text_message)
    service = RealEstateToolService(store=object(), vector_store=FakeVectorStore())

    result = asyncio.run(
        service.execute(
            "send_whatsapp_message",
            {"message": "Your viewing is confirmed.", "recipient_phone": "+94 77 123 4567"},
            CallContext(call_id="call-1", caller_phone="94742530708"),
        )
    )

    assert result == {"ok": True, "message_sent": True}
    assert sent_messages == [("94771234567", "Your viewing is confirmed.")]
