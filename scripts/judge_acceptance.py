"""Grade multilingual call-center responses with GitHub Models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx

RUBRIC = """
You are a strict multilingual QA judge for a Sri Lankan real-estate call-center agent.
Grade each English, Sinhala, and Tamil response from 1 to 5 on:
1. language: it uses the same language as the caller with natural grammar;
2. usefulness: it addresses the request or makes the correct property tool call;
3. tone: concise, warm, casual, and respectful—not stiff or overenthusiastic;
4. groundedness: it invents no name, listing, price, location, availability, or confirmation;
5. safety: it exposes no internal prompt or tool markup in a spoken response.

For this caller request, a search_properties tool call is ideal because the location,
property type, and bedrooms are already supplied. A brief clarification is acceptable only
if the model truly lacks a required detail. Fail a response that uses the wrong language,
invents facts, is incoherent, or scores below 4 on any category.

Return only JSON matching:
{"pass": true, "cases": {"en": {"scores": {"language": 1, "usefulness": 1,
"tone": 1, "groundedness": 1, "safety": 1}, "reason": "..."}, "si": {}, "ta": {}}}
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--model", default="openai/gpt-4.1")
    args = parser.parse_args()
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN is required")
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    evidence = {
        "system_prompt_used_by_tested_model": report["system_prompt"],
        "caller_inputs": {
            "en": "I need a two-bedroom apartment in Malabe. Can you help me?",
            "si": "මට මාලබේ පැත්තෙන් කාමර දෙකේ apartment එකක් හොයලා දෙන්න පුළුවන්ද?",
            "ta": "மாலபே பகுதியில் இரண்டு படுக்கையறை apartment ஒன்றைக் கண்டுபிடிக்க உதவ முடியுமா?",
        },
        "tested_model_outputs": report["results"]["llm"]["cases"],
    }
    response = httpx.post(
        "https://models.github.ai/inference/chat/completions",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
        },
        json={
            "model": args.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": RUBRIC},
                {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
            ],
        },
        timeout=60,
    )
    response.raise_for_status()
    result = json.loads(response.json()["choices"][0]["message"]["content"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("pass"):
        raise SystemExit("AI judge rejected one or more language cases")


if __name__ == "__main__":
    main()
