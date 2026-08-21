"""Hosted Gemma conversation evaluation with an independent AI judge.

This deliberately hosts only the local LLM. Property search and booking use an
in-memory implementation of the production tool contract, isolating conversation
orchestration from ASR, TTS, WebRTC, Pinecone, and Neon.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.voice.llm import LocalGemmaLLM
from app.voice.tools import CallContext, ToolCall
from app.voice.turn_pipeline import LocalGemmaTurnPipeline

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-2.5-flash")
CALL_ID = "llm-quality-sinhala"
CALLER_PHONE = "94770000000"

PROPERTY = {
    "property_id": "property-horizon-malabe",
    "name": "Horizon Residencies",
    "location": "Malabe",
    "property_type": "apartment",
    "bedrooms": 2,
    "price_label": "LKR 28 million",
    "details": "Near schools and supermarkets.",
}


@dataclass(frozen=True)
class Stage:
    name: str
    caller: str
    expected_tool: str | None
    purpose: str


STAGES = (
    Stage("language_selection", "සිංහලෙන් කතා කරමු.", None, "Select Sinhala and establish language continuity."),
    Stage(
        "property_search",
        "මට මාලබේ පැත්තෙන් bedrooms දෙකක apartment එකක් හොයලා දෙන්න.",
        "search_properties",
        "Search for a specific two-bedroom apartment in Malabe.",
    ),
    Stage(
        "property_followup",
        "Horizon Residencies ගැන විස්තර කියන්න.",
        "search_properties",
        "Carry the selected property context into a follow-up question.",
    ),
    Stage(
        "broad_request_clarification",
        "ඔයාලා ළඟ තියෙන properties ඔක්කොම මට කියන්න.",
        None,
        "Ask for a location instead of searching or assuming a broad inventory request.",
    ),
    Stage(
        "narrowed_search",
        "මාලබේ පැත්තෙන් බලන්න.",
        "search_properties",
        "Use the clarification answer to perform a Malabe search.",
    ),
    Stage(
        "booking_missing_details",
        "Horizon Residencies එක බලන්න appointment එකක් දාන්න ඕන.",
        None,
        "Ask for the missing customer name and exact viewing date/time; do not book early.",
    ),
    Stage(
        "booking_complete",
        "මගේ නම Nimal Perera. Horizon Residencies එක 2099-01-01 උදේ 10:00 ට book කරන්න.",
        "book_appointment",
        "Book only after the property, name, and exact appointment time are supplied.",
    ),
    Stage(
        "booking_confirmation",
        "WhatsApp confirmation එක ආවද?",
        None,
        "Explain the booking confirmation status without exposing tool internals.",
    ),
)


class InMemoryPropertyTools:
    """Production-shaped search and booking tools for LLM-only evaluation."""

    def __init__(self) -> None:
        self.traces: list[dict] = []
        self.bookings: list[dict] = []

    async def execute(self, call: ToolCall, context: CallContext) -> dict:
        self.traces.append({"name": call.name, "arguments": call.arguments})
        if call.name == "search_properties":
            location = str(call.arguments.get("location", "")).casefold()
            if location and "malabe" in location:
                return {"ok": True, "properties": [PROPERTY], "count": 1, "needs_clarification": False}
            return {
                "ok": True,
                "properties": [],
                "count": 0,
                "needs_clarification": True,
                "suggested_locations": ["Malabe", "Nugegoda"],
            }
        if call.name == "book_appointment":
            required = ("property_id", "customer_name", "appointment_at")
            missing = [key for key in required if not str(call.arguments.get(key, "")).strip()]
            if missing:
                return {"ok": False, "error": f"Missing required argument: {missing[0]}"}
            booking = {
                "status": "booked",
                "property": PROPERTY["name"],
                "customer_name": call.arguments["customer_name"],
                "appointment_at": call.arguments["appointment_at"],
                "whatsapp_confirmation_sent": True,
            }
            self.bookings.append(booking)
            return {"ok": True, "appointment": booking, "whatsapp_confirmation_sent": True}
        return {"ok": False, "error": f"Unknown tool: {call.name}"}


class Conversation:
    def __init__(self, llm: LocalGemmaLLM, tools: InMemoryPropertyTools) -> None:
        self.pipeline = LocalGemmaTurnPipeline.__new__(LocalGemmaTurnPipeline)
        self.pipeline._llm = llm
        self.pipeline._tools = tools
        self.pipeline._conversation_history = {}

    async def respond(self, caller_text: str) -> tuple[str, list[dict]]:
        before = len(self.pipeline._tools.traces)
        response = await self.pipeline._generate_response(
            CALL_ID,
            CALLER_PHONE,
            caller_text,
        )
        self.pipeline._append_conversation_turn(CALL_ID, caller_text, response)
        return response, self.pipeline._tools.traces[before:]


async def run_conversation() -> dict:
    llm = LocalGemmaLLM()
    await llm.prewarm()
    tools = InMemoryPropertyTools()
    conversation = Conversation(llm, tools)
    results = []
    for stage in STAGES:
        started = time.perf_counter()
        response, traces = await conversation.respond(stage.caller)
        contract_pass = (
            bool(traces) and traces[0]["name"] == stage.expected_tool
            if stage.expected_tool
            else not traces
        )
        if stage.name == "property_search" and traces:
            arguments = traces[0]["arguments"]
            contract_pass = contract_pass and "malabe" in str(arguments.get("location", "")).casefold()
        if stage.name == "booking_complete" and traces:
            arguments = traces[0]["arguments"]
            contract_pass = contract_pass and all(
                str(arguments.get(key, "")).strip()
                for key in ("property_id", "customer_name", "appointment_at")
            )
        results.append(
            {
                "name": stage.name,
                "purpose": stage.purpose,
                "caller": stage.caller,
                "response": response,
                "tool_trace": traces,
                "expected_tool": stage.expected_tool,
                "contract_pass": contract_pass,
                "latency_seconds": round(time.perf_counter() - started, 3),
            }
        )
    return {
        "model": llm._resolve_model_path(),
        "stages": results,
        "all_tool_traces": tools.traces,
        "bookings": tools.bookings,
    }


def judge(evidence: dict) -> dict:
    token = os.getenv("GEMINI_API_KEY")
    if not token:
        raise RuntimeError("GEMINI_API_KEY is required for independent judging")
    prompt = """You are a strict independent QA judge for a Sinhala-first Sri Lankan property voice-agent LLM.
Grade every supplied stage separately from 1 to 5. Evaluate:
1) language continuity: after selecting Sinhala, the response remains natural Sinhala unless requested otherwise;
2) conversation continuity: it uses the latest caller message and prior context;
3) clarification safety: it asks for missing or ambiguous information and never guesses;
4) tool behavior: it calls exactly the expected tool, with grounded arguments, or does not call one when expected_tool is null;
5) customer safety: it never exposes XML, JSON, stack traces, database errors, or internal tool names;
6) booking safety: it never books before property, customer name, and exact date/time are supplied;
7) answer quality: concise, useful, and natural for a phone conversation.

The tool traces are authoritative evidence of what actually happened. Do not award credit for claims unsupported by the traces.
Return only JSON in this shape:
{"pass":true,"overall_reason":"...","stages":{"stage_name":{"pass":true,"scores":{"language":5,"continuity":5,"clarification":5,"tool":5,"safety":5,"booking":5,"quality":5},"reason":"..."}}}
Set pass=false if any stage has a score below 4, the expected tool is missing/wrong, an unexpected tool is called, the model assumes unclear details, or internal tool syntax leaks.
"""
    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{JUDGE_MODEL}:generateContent",
        headers={"x-goog-api-key": token},
        json={
            "contents": [{"parts": [{"text": prompt + "\nEvidence:\n" + json.dumps(evidence, ensure_ascii=False)}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        },
        timeout=90,
    )
    response.raise_for_status()
    raw = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    return json.loads(raw)


def gpu_memory_mib() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return max(int(row.strip()) for row in output.splitlines() if row.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="llm-quality-report.json")
    args = parser.parse_args()
    evidence = asyncio.run(run_conversation())
    evidence["gpu_memory_mib"] = gpu_memory_mib()
    verdict = judge(evidence)
    report = {"evidence": evidence, "judge": verdict, "judge_model": JUDGE_MODEL}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": args.report, "judge": verdict}, ensure_ascii=False, indent=2))
    if verdict.get("pass") is not True:
        raise SystemExit("LLM quality judge failed")


if __name__ == "__main__":
    main()
