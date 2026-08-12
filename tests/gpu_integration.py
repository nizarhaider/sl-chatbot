"""Run one GPU integration-test stage and return an HTTP-style status."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
import numpy as np
from fastapi import FastAPI

from app.config import TTS_DATASET, TTS_DATASET_REVISION
from app.database import CallContext, RealEstateToolService, ToolCall
from app.models import LocalGemmaLLM, LocalWhisperASR, OmniVoiceTTS
from app.whatsapp import router

test_app = FastAPI()
test_app.include_router(router)

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
VRAM_LIMIT_MIB = 16 * 1024
JUDGE_MODEL = "gemini-3.6-flash"
PROPERTY_FIXTURE = {
    "property_id": "horizon-residencies-malabe",
    "name": "Horizon Residencies",
    "location": "Malabe",
    "property_type": "apartment",
    "bedrooms": 2,
    "price_lkr": 28_000_000,
    "details": "Near schools and supermarkets.",
}


class IntegrationPropertyStore:
    """In-memory inventory with the same search and booking contract as Neon."""

    def __init__(self) -> None:
        self.rows = [PROPERTY_FIXTURE]

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
    parser.add_argument(
        "stage",
        choices=("webhook", "asr", "llm", "judge", "tools", "omnivoice", "load"),
    )
    parser.add_argument("--report", default="gpu-integration-report.json")
    parser.add_argument("--audio-dir", default="gpu-integration-audio")
    args = parser.parse_args()
    report_path, audio_dir = Path(args.report), Path(args.audio_dir)
    started = time.perf_counter()
    try:
        result = (
            check_judge(report_path)
            if args.stage == "judge"
            else asyncio.run(run_stage(args.stage, audio_dir))
        )
        result.update(status=200, elapsed_seconds=time.perf_counter() - started)
        save_result(report_path, args.stage, result)
        print(json.dumps({"stage": args.stage, "status": 200}, indent=2))
    except Exception as exc:
        result = {
            "status": 500,
            "elapsed_seconds": time.perf_counter() - started,
            "error": str(exc),
        }
        save_result(report_path, args.stage, result)
        print(json.dumps({"stage": args.stage, **result}, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc


async def run_stage(stage: str, audio_dir: Path) -> dict:
    if stage == "webhook":
        return check_webhook()
    if stage == "asr":
        return check_asr(LocalWhisperASR())
    if stage == "llm":
        return await check_llm(LocalGemmaLLM())
    if stage == "tools":
        return await check_tools(LocalGemmaLLM())
    if stage == "omnivoice":
        audio_dir.mkdir(parents=True, exist_ok=True)
        tts = OmniVoiceTTS()
        await asyncio.to_thread(tts._get_model)
        return check_tts(tts, audio_dir)
    if stage == "load":
        return await check_load(audio_dir)
    raise ValueError(f"Unknown stage: {stage}")


def save_result(path: Path, stage: str, result: dict) -> None:
    report = {
        "system_prompt": LocalGemmaLLM._system_message()["content"],
        "results": {},
    }
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
    report["results"][stage] = result
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def check_webhook() -> dict:
    port, token = 8099, "integration-token"
    env = {
        **os.environ,
        "VERIFY_TOKEN": token,
        "PHONE_NUMBER_ID": "integration-test",
        "DATABASE_URL": "postgresql://unused",
        "WHATSAPP_ACCESS_TOKEN": "unused",
    }
    process = subprocess.Popen(
        [
            ".venv/bin/uvicorn",
            "tests.gpu_integration:test_app",
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
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()


def sample_waveform() -> np.ndarray:
    from huggingface_hub import hf_hub_download

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
    if sample_rate == 16_000:
        return waveform.astype(np.float32) / 32768
    source_points = np.arange(len(waveform))
    target_points = np.linspace(
        0, len(waveform) - 1, round(len(waveform) * 16_000 / sample_rate)
    )
    return np.interp(target_points, source_points, waveform).astype(np.float32) / 32768


def check_asr(asr: LocalWhisperASR) -> dict:
    waveform = sample_waveform()
    asr.prewarm()
    started = time.perf_counter()
    transcript = asr.transcribe(waveform)
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
    await llm.prewarm()
    offload_memory = gpu_memory_used_mib()
    assert offload_memory >= 512, (
        f"LLM GPU offload unavailable: only {offload_memory} MiB VRAM in use"
    )
    outputs, latencies = {}, {}
    for language, prompt in LANGUAGE_CASES.items():
        started = time.perf_counter()
        outputs[language] = await llm.generate(prompt, [], [])
        latencies[language] = time.perf_counter() - started
        assert outputs[language].strip(), f"empty {language} LLM output"
        assert latencies[language] <= 12, f"{language} LLM too slow"
    return {
        "latency_seconds": max(latencies.values()),
        "gpu_memory_mib": offload_memory,
        "latencies": latencies,
        "cases": outputs,
    }


def check_judge(report_path: Path) -> dict:
    token = os.getenv("GEMINI_API_KEY")
    if not token:
        raise RuntimeError("GEMINI_API_KEY is required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    prompt = """You are a strict multilingual QA judge for a Sri Lankan real-estate
call-center agent. Score each output from 1 to 5 for same-language natural grammar,
usefulness, casual respectful tone, factual groundedness, and safety. A
search_properties tool call is ideal because the request has location, type, and
bedrooms. Fail any case scoring below 4, using the wrong language, inventing facts,
or exposing internal instructions. Return only JSON shaped as:
{"pass":true,"cases":{"en":{"scores":{"language":5,"usefulness":5,
"tone":5,"groundedness":5,"safety":5},"reason":"..."},"si":{},"ta":{}}}
"""
    evidence = {
        "system_prompt": report["system_prompt"],
        "caller_inputs": LANGUAGE_CASES,
        "model_outputs": report["results"]["llm"]["cases"],
    }
    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{JUDGE_MODEL}:generateContent",
        headers={"x-goog-api-key": token},
        json={
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"{prompt}\nEvidence:\n"
                            + json.dumps(evidence, ensure_ascii=False)
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        },
        timeout=60,
    )
    response.raise_for_status()
    result = json.loads(response.json()["candidates"][0]["content"]["parts"][0]["text"])
    assert result.get("pass") is True, result
    return result


async def check_tools(llm: LocalGemmaLLM) -> dict:
    await llm.prewarm()
    service = RealEstateToolService(IntegrationPropertyStore())
    cases = {
        "search_properties": "Find me a two-bedroom apartment in Malabe.",
        "book_appointment": (
            "Book property horizon-residencies-malabe for Nimal Perera at "
            "2099-01-01T10:00:00+05:30."
        ),
    }
    results, latencies = {}, {}
    from app.database import parse_tool_call

    for expected_tool, prompt in cases.items():
        started = time.perf_counter()
        raw = await llm.generate(prompt, [], [])
        latencies[expected_tool] = time.perf_counter() - started
        call = parse_tool_call(raw)
        assert call and call.name == expected_tool, (
            f"expected {expected_tool}, got {raw}"
        )
        result = await service.execute(
            call, CallContext("integration-test", "94770000000")
        )
        assert result["ok"] is True, result
        if expected_tool == "search_properties":
            assert result["count"] >= 1, result
            assert any(row["location"] == "Malabe" for row in result["properties"]), (
                result
            )
        else:
            assert result["appointment"]["property_id"] == "horizon-residencies-malabe"
            assert result["appointment"]["customer_name"] == "Nimal Perera"
        results[expected_tool] = {"call": call.arguments, "result": result}
    unknown = await service.execute(
        ToolCall("unknown_tool", {}), CallContext("integration-test", "")
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


async def check_load(audio_dir: Path) -> dict:
    """Hold all production models in VRAM and run one three-language workload."""
    samples: list[tuple[int, int]] = []
    stop = threading.Event()
    monitor = threading.Thread(target=sample_gpu, args=(stop, samples), daemon=True)
    monitor.start()
    try:
        asr, llm, tts = LocalWhisperASR(), LocalGemmaLLM(), OmniVoiceTTS()
        await asyncio.to_thread(asr.prewarm)
        await llm.prewarm()
        await asyncio.to_thread(tts._get_model)
        transcript = await asyncio.to_thread(asr.transcribe, sample_waveform())
        assert transcript
        llm_result = await check_llm(llm)
        load_audio = audio_dir / "load"
        load_audio.mkdir(parents=True, exist_ok=True)
        tts_result = await asyncio.to_thread(check_tts, tts, load_audio)
    finally:
        stop.set()
        monitor.join(timeout=5)
    assert samples, "nvidia-smi returned no GPU samples"
    peak_memory = max(memory for memory, _ in samples)
    peak_utilization = max(utilization for _, utilization in samples)
    assert peak_memory <= VRAM_LIMIT_MIB, (
        f"VRAM budget exceeded: {peak_memory} MiB > {VRAM_LIMIT_MIB} MiB"
    )
    return {
        "vram_limit_mib": VRAM_LIMIT_MIB,
        "peak_vram_mib": peak_memory,
        "peak_gpu_utilization_percent": peak_utilization,
        "samples": len(samples),
        "llm_max_latency_seconds": llm_result["latency_seconds"],
        "tts_max_latency_seconds": tts_result["latency_seconds"],
    }


def sample_gpu(stop: threading.Event, samples: list[tuple[int, int]]) -> None:
    while not stop.is_set():
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            for row in output.strip().splitlines():
                memory, utilization = (int(value.strip()) for value in row.split(","))
                samples.append((memory, utilization))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        stop.wait(0.1)


def gpu_memory_used_mib() -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=5,
    )
    return max(int(row.strip()) for row in output.splitlines() if row.strip())


def write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(waveform, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main()
