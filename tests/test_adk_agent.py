"""Focused tests for the local Gemma to Google ADK boundary."""

from __future__ import annotations

import unittest

from app.agent import (
    APP_NAME,
    GemmaAgentRuntime,
    LocalGemmaAdkModel,
    _grounded_fallback,
    _is_post_tool_turn,
)
from app.database import CallContext, RealEstateToolService, ToolCall
from app.speech import selected_language


class FakeGemmaBackend:
    def __init__(self) -> None:
        self.requests: list[tuple[list[dict], list[dict]]] = []

    async def prewarm(self) -> None:
        return None

    async def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        self.requests.append((messages, tools))
        if len(self.requests) == 1:
            return {
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_properties",
                            "arguments": '{"location":"Malabe","bedrooms":2}',
                        }
                    }
                ],
            }
        if len(self.requests) == 2:
            return {"content": "මිල ලක්ෂ විසි අටයි."}
        if len(self.requests) in {3, 4}:
            return {"content": "I found one matching property for LKR 28,000,000."}
        return {
            "content": "It is in Malabe, and the listed price is LKR 28,000,000."
        }


class FakePropertyService:
    def __init__(self) -> None:
        self.calls = []

    async def ensure_ready(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def available_locations(self) -> list[str]:
        return ["Malabe", "Nugegoda"]

    async def execute(self, call, context) -> dict:
        self.calls.append((call, context))
        return {
            "ok": True,
            "count": 1,
            "properties": [
                {
                    "property_id": "property-1",
                    "location": "Malabe",
                    "bedrooms": 2,
                    "price_lkr": 28_000_000,
                }
            ],
        }


class GemmaAgentRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_adk_retains_events_and_dispatches_native_function_call(self) -> None:
        backend = FakeGemmaBackend()
        service = FakePropertyService()
        runtime = GemmaAgentRuntime(LocalGemmaAdkModel(backend), service)

        await runtime.start_session("call-1", "94770000000")
        response = await runtime.respond(
            "call-1",
            "94770000000",
            "Find a two-bedroom property in Malabe.",
        )
        first_trace = runtime.tool_trace("call-1")
        followup = await runtime.respond(
            "call-1", "94770000000", "Where is that property?"
        )
        session = await runtime.sessions.get_session(
            app_name=APP_NAME,
            user_id="94770000000",
            session_id="call-1",
        )

        self.assertEqual(response, "I found one matching property for LKR 28,000,000.")
        self.assertEqual(
            followup,
            "It is in Malabe, and the listed price is LKR 28,000,000.",
        )
        self.assertEqual(service.calls[0][0].name, "search_properties")
        self.assertEqual(service.calls[0][0].arguments["bedrooms"], 2)
        self.assertEqual(service.calls[0][1].caller_phone, "94770000000")
        self.assertEqual(len(session.events), 6)
        self.assertEqual(session.state["last_property_id"], "property-1")
        self.assertEqual(first_trace[0]["name"], "search_properties")
        self.assertEqual(first_trace[0]["result"]["count"], 1)
        self.assertEqual(runtime.tool_trace("call-1"), [])
        self.assertIn("<tool_result>", backend.requests[1][0][-1]["content"])
        self.assertIn("not caller speech", backend.requests[1][0][-1]["content"])
        self.assertIn("entirely in English", backend.requests[1][0][0]["content"])
        self.assertIn("omit location for a broad request", backend.requests[0][0][0]["content"])
        self.assertIn(
            "answer entirely in English", backend.requests[2][0][-1]["content"]
        )
        self.assertNotIn(
            "මිල ලක්ෂ විසි අටයි.",
            [message["content"] for message in backend.requests[2][0]],
        )
        self.assertIn("LKR 28,000,000", backend.requests[2][0][-1]["content"])
        self.assertTrue(
            any(
                message["content"]
                == "I found one matching property for LKR 28,000,000."
                for message in backend.requests[3][0]
            )
        )
        self.assertIn("exact property_id property-1", backend.requests[3][0][0]["content"])
        self.assertEqual(
            {tool["function"]["name"] for tool in backend.requests[0][1]},
            {"search_properties", "list_property_locations", "book_appointment"},
        )

        await runtime.end_session("call-1")
        deleted = await runtime.sessions.get_session(
            app_name=APP_NAME,
            user_id="94770000000",
            session_id="call-1",
        )
        self.assertIsNone(deleted)

    async def test_empty_search_returns_authoritative_locations(self) -> None:
        class Store:
            def search(self, arguments):
                return []

            def locations(self):
                return ["Malabe", "Nugegoda"]

        service = RealEstateToolService(Store())
        result = await service.execute(
            ToolCall("search_properties", {"location": "Manaram"}),
            CallContext("call-1", ""),
        )
        locations = await service.execute(
            ToolCall("list_property_locations", {}), CallContext("call-1", "")
        )

        self.assertEqual(result["count"], 0)
        self.assertEqual(result["available_locations"], ["Malabe", "Nugegoda"])
        self.assertEqual(locations["locations"], ["Malabe", "Nugegoda"])

    async def test_booking_rejects_model_placeholder_name(self) -> None:
        class Store:
            def book(self, arguments, context):
                raise AssertionError("invalid booking reached the store")

        service = RealEstateToolService(Store())
        result = await service.execute(
            ToolCall(
                "book_appointment",
                {
                    "property_id": "property-1",
                    "customer_name": "caller",
                    "appointment_at": "2099-01-01T10:00:00+05:30",
                },
            ),
            CallContext("call-1", "94770000000"),
        )

        self.assertFalse(result["ok"])
        self.assertIn("Ask the caller for their name", result["error"])

    def test_language_selection_accepts_repeated_choice_only(self) -> None:
        self.assertEqual(selected_language("sinhala sinhala"), "si")
        self.assertEqual(selected_language("தமிழ்"), "ta")
        self.assertIsNone(selected_language("I want an English-style apartment"))

    def test_grounded_fallback_preserves_tamil_and_exact_price(self) -> None:
        messages = [
            {"role": "user", "content": "மலபேயில் இரண்டு படுக்கையறை வீடு வேண்டும்"},
            {
                "role": "user",
                "content": '<tool_result>{"ok":true,"properties":['
                '{"name":"Horizon Residencies","location":"Malabe",'
                '"property_type":"apartment","bedrooms":2,'
                '"price_lkr":28000000}]}</tool_result>',
            },
        ]

        response = _grounded_fallback(messages)

        self.assertIn("LKR 28,000,000", response)
        self.assertIn("படுக்கையறைகள்", response)
        self.assertNotRegex(response, r"[\u0D80-\u0DFF]")

    def test_historical_tool_result_does_not_disable_followup_tools(self) -> None:
        messages = [
            {"role": "user", "content": '<tool_result>{"ok":true}</tool_result>'},
            {"role": "assistant", "content": "I found one property."},
            {"role": "user", "content": "Book it for Nimal tomorrow at 4 pm."},
        ]

        self.assertFalse(_is_post_tool_turn(messages))
        self.assertTrue(_is_post_tool_turn(messages[:1]))


if __name__ == "__main__":
    unittest.main()
