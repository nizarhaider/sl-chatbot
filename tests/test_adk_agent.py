"""Focused tests for the local Gemma to Google ADK boundary."""

from __future__ import annotations

import asyncio
import unittest

from app.agent import APP_NAME, GemmaAgentRuntime, LocalGemmaAdkModel


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
        return {"content": "I found one matching property in Malabe."}


class FakePropertyService:
    def __init__(self) -> None:
        self.calls = []

    async def ensure_ready(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def execute(self, call, context) -> dict:
        self.calls.append((call, context))
        return {
            "ok": True,
            "count": 1,
            "properties": [
                {"property_id": "property-1", "location": "Malabe", "bedrooms": 2}
            ],
        }


class GemmaAgentRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_adk_retains_events_and_dispatches_native_function_call(self) -> None:
        backend = FakeGemmaBackend()
        service = FakePropertyService()
        runtime = GemmaAgentRuntime(LocalGemmaAdkModel(backend), service)
        notices = 0

        async def notice() -> None:
            nonlocal notices
            notices += 1
            await asyncio.sleep(0)

        await runtime.start_session("call-1", "94770000000")
        response = await runtime.respond(
            "call-1",
            "94770000000",
            "Find a two-bedroom property in Malabe.",
            notice,
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

        self.assertEqual(response, "I found one matching property in Malabe.")
        self.assertEqual(followup, "I found one matching property in Malabe.")
        self.assertEqual(service.calls[0][0].name, "search_properties")
        self.assertEqual(service.calls[0][0].arguments["bedrooms"], 2)
        self.assertEqual(service.calls[0][1].caller_phone, "94770000000")
        self.assertEqual(notices, 1)
        self.assertEqual(len(session.events), 6)
        self.assertEqual(first_trace[0]["name"], "search_properties")
        self.assertEqual(first_trace[0]["result"]["count"], 1)
        self.assertEqual(runtime.tool_trace("call-1"), [])
        self.assertIn("<tool_result>", backend.requests[1][0][-1]["content"])
        self.assertIn("not caller speech", backend.requests[1][0][-1]["content"])
        self.assertIn("entirely in English", backend.requests[1][0][0]["content"])
        self.assertTrue(
            any(
                message["content"] == "I found one matching property in Malabe."
                for message in backend.requests[2][0]
            )
        )
        self.assertEqual(
            {tool["function"]["name"] for tool in backend.requests[0][1]},
            {"search_properties", "book_appointment"},
        )

        await runtime.end_session("call-1")
        deleted = await runtime.sessions.get_session(
            app_name=APP_NAME,
            user_id="94770000000",
            session_id="call-1",
        )
        self.assertIsNone(deleted)


if __name__ == "__main__":
    unittest.main()
