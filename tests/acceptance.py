"""Five production acceptance checks run on an ephemeral Vast GPU."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
import numpy as np
from fastapi import FastAPI
from huggingface_hub import hf_hub_download

from app.config import SYSTEM_PROMPT, TTS_DATASET, TTS_DATASET_REVISION
from app.database import CallContext, PROPERTIES, RealEstateToolService, ToolCall
from app.models import LocalGemmaLLM, LocalWhisperASR, OmniVoiceTTS
from app.whatsapp import router

webhook_app = FastAPI()
webhook_app.include_router(router)

LANGUAGE_CASES = {
    "en": "I need a two-bedroom apartment in Malabe. Can you help me?",
    "si": "මට මාලබේ පැත්තෙන් කාමර දෙකේ apartment එකක් හොයලා දෙන්න පුළුවන්ද?",
    "ta": "மாலபே பகுதியில் இரண்டு படுக்கையறை apartment ஒன்றைக் கண்டுபிடிக்க உதவ முடியுமா?",
}
TTS_CASES = {
    "en": "Give me a moment, Sir. I will check the available properties.",
    "si": "පොඩ්ඩක් ඉන්න සර්. මම තියෙන properties බලලා කියන්නම්.",
    "ta": "ஒரு நிமிடம் சார். உள்ள properties பார்த்துச் சொல்கிறேன்.",
}


class AcceptancePropertyStore:
    """In-memory inventory with the same search and booking contract as Neon."""

    def __init__(self) -> None:
        self.rows = [
            {
                "property_id": slug,
                "name": name,
                "location": location,
                "property_type": kind,
                "bedrooms": bedrooms,
                "price_lkr": price,
                "details": details,
            }
            for slug, name, location, kind, bedrooms, price, details in PROPERTIES
        ]

    def search(self, arguments: dict) -> list[dict]:
        rows = self.rows
        for field in ("location", "property_type"):
            if value := str(arguments.get(field, "")).casefold().strip():
                rows = [row for row in rows if value in str(row[field]).casefold()]
        if bedrooms := arguments.get("bedrooms"):
            rows = [row for row in rows if (row["bedrooms"] or 0) >= int(bedrooms)]
        return rows[:5]

    def book(self, arguments: dict, context: CallContext) -> dict:
        return {
            "status": "booked",
            "property_id": arguments["property_id"],
            "customer_name": arguments["customer_name"],
            "appointment_at": arguments["appointment_at"],
            "caller_phone": context.caller_phone,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="acceptance-report.json")
    parser.add_argument("--audio-dir", default="acceptance-audio")
    args = parser.parse_args()
    started = time.perf_counter()
    results = asyncio.run(run(Path(args.audio_dir)))
    report = {
        "system_prompt": SYSTEM_PROMPT,
        "results": results,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, result in results.items():
        print(f"{name}: PASS ({result.get('latency_seconds', 0):.2f}s)")


async def run(audio_dir: Path) -> dict:
    audio_dir.mkdir(parents=True, exist_ok=True)
    results = {"webhook": check_webhook()}

    asr = LocalWhisperASR()
    llm = LocalGemmaLLM()
    tts = OmniVoiceTTS()
    started = time.perf_counter()
    await asyncio.to_thread(asr.prewarm)
    await llm.prewarm()
    await asyncio.to_thread(tts._get_model)
    model_load_seconds = time.perf_counter() - started
    results["asr"] = check_asr(asr)
    results["llm"] = await check_llm(llm)
    results["tools"] = await check_tools(llm)
    results["omnivoice"] = check_tts(tts, audio_dir)
    results["webhook"]["model_load_seconds"] = model_load_seconds
    return results


def check_webhook() -> dict:
    port, token = 8099, "acceptance-token"
    env = {
        **os.environ,
        "VERIFY_TOKEN": token,
        "PHONE_NUMBER_ID": "acceptance",
        "DATABASE_URL": "postgresql://unused",
        "WHATSAPP_ACCESS_TOKEN": "unused",
    }
    process = subprocess.Popen(
        [
            ".venv/bin/uvicorn",
            "tests.acceptance:webhook_app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    started = time.perf_counter()
    try:
        for _ in range(100):
            try:
                response = httpx.get(
                    f"http://127.0.0.1:{port}/webhook",
                    params={
                        "hub.mode": "subscribe",
                        "hub.verify_token": token,
                        "hub.challenge": "ready",
                    },
                    timeout=2,
                )
                if response.status_code == 200:
                    assert response.text == "ready"
                    return {"latency_seconds": time.perf_counter() - started}
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        raise AssertionError("webhook did not become ready")
    finally:
        process.terminate()
        process.wait(timeout=20)


def check_asr(asr: LocalWhisperASR) -> dict:
    sample = Path(
        hf_hub_download(
            repo_id=TTS_DATASET,
            repo_type="dataset",
            revision=TTS_DATASET_REVISION,
            filename="audio/202.wav",
        )
    )
    with wave.open(str(sample), "rb") as source:
        assert source.getnchannels() == 1
        sample_rate = source.getframerate()
        waveform = np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16)
    if sample_rate != 16_000:
        source_points = np.arange(len(waveform))
        target_points = np.linspace(
            0, len(waveform) - 1, round(len(waveform) * 16_000 / sample_rate)
        )
        waveform = np.interp(target_points, source_points, waveform).astype(np.int16)
    started = time.perf_counter()
    transcript = asr.transcribe(waveform.astype(np.float32) / 32768)
    latency = time.perf_counter() - started
    assert latency <= 6, f"ASR too slow: {latency:.2f}s"
    expected = {"බෙහෙත්", "නොමිලේ", "රෝහල", "prescription"}
    matches = [term for term in expected if term.casefold() in transcript.casefold()]
    assert len(matches) >= 2, f"unexpected ASR transcript: {transcript}"
    return {
        "latency_seconds": latency,
        "transcript": transcript,
        "matched_terms": matches,
    }


async def check_llm(llm: LocalGemmaLLM) -> dict:
    outputs, latencies = {}, {}
    for language, prompt in LANGUAGE_CASES.items():
        started = time.perf_counter()
        outputs[language] = await llm.generate(prompt, [], [])
        latencies[language] = time.perf_counter() - started
        assert outputs[language].strip(), f"empty {language} LLM output"
        assert latencies[language] <= 8, f"{language} LLM too slow"
    return {"latency_seconds": max(latencies.values()), "latencies": latencies, "cases": outputs}


async def check_tools(llm: LocalGemmaLLM) -> dict:
    service = RealEstateToolService(AcceptancePropertyStore())
    cases = {
        "search_properties": "Find me a two-bedroom apartment in Malabe.",
        "book_appointment": (
            "Book property horizon-residencies-malabe for Nimal Perera at "
            "2099-01-01T10:00:00+05:30."
        ),
    }
    results, latencies = {}, {}
    for expected_tool, prompt in cases.items():
        started = time.perf_counter()
        raw = await llm.generate(prompt, [], [])
        latencies[expected_tool] = time.perf_counter() - started
        from app.database import parse_tool_call

        call = parse_tool_call(raw)
        assert call and call.name == expected_tool, f"expected {expected_tool}, got {raw}"
        result = await service.execute(call, CallContext("acceptance", "94770000000"))
        assert result["ok"] is True, result
        results[expected_tool] = {"call": call.arguments, "result": result}
    unknown = await service.execute(
        ToolCall("unknown_tool", {}), CallContext("acceptance", "")
    )
    assert unknown["ok"] is False
    return {"latency_seconds": max(latencies.values()), "cases": results}


def check_tts(tts: OmniVoiceTTS, audio_dir: Path) -> dict:
    results = {}
    for language, text in TTS_CASES.items():
        started = time.perf_counter()
        waveform = tts.synthesize(text, seed=7, language=language)
        latency = time.perf_counter() - started
        duration = len(waveform) / tts.sample_rate
        rms = float(np.sqrt(np.mean(np.square(waveform.astype(np.float64)))))
        assert 0.8 <= duration <= 20, f"unexpected {language} duration: {duration}"
        assert rms >= 0.003, f"inaudible {language} output: rms={rms}"
        assert latency / duration <= 1, f"{language} TTS slower than real time"
        path = audio_dir / f"omnivoice-{language}.wav"
        write_wav(path, waveform, tts.sample_rate)
        results[language] = {
            "latency_seconds": latency,
            "duration_seconds": duration,
            "realtime_factor": latency / duration,
            "rms": rms,
            "file": str(path),
        }
    return {
        "latency_seconds": max(item["latency_seconds"] for item in results.values()),
        "cases": results,
    }


def write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(waveform, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main()
