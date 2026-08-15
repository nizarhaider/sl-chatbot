"""Small reliability tests for local model wrappers."""

from __future__ import annotations

import unittest

from app.models import LocalGemmaLLM


class OverflowOnceModel:
    def __init__(self) -> None:
        self.requests = []

    def create_chat_completion(self, **kwargs):
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            raise ValueError("Requested tokens exceed context window of 2048")
        return {"choices": [{"message": {"content": "Recovered."}}]}


class LocalGemmaLLMTest(unittest.TestCase):
    def test_context_overflow_retries_with_compact_history(self) -> None:
        backend = OverflowOnceModel()
        model = LocalGemmaLLM()
        model._llm = backend
        messages = [{"role": "system", "content": "instructions"}] + [
            {"role": "user", "content": f"turn {index}"} for index in range(10)
        ]

        response = model._chat(messages)

        self.assertEqual(response["content"], "Recovered.")
        self.assertEqual(len(backend.requests[1]["messages"]), 5)


if __name__ == "__main__":
    unittest.main()
