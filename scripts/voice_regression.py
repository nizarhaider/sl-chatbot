#!/usr/bin/env python3
"""Replay caller recordings through the local production voice pipeline.

The only doubles here are the inbound and outbound media tracks. ASR, VAD,
Gemma, OmniVoice, and the production property tools are instantiated exactly
as they are for a WhatsApp call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import av
import httpx
import numpy as np
from av import AudioFrame
from av.audio.resampler import AudioResampler
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.voice.config import (
    LLM_BASE_URL,
    TURN_END_SILENCE_CHUNKS,
    TURN_INPUT_CHUNK_MS,
    TURN_INPUT_CHUNK_SIZE,
    TURN_PLAYBACK_ECHO_TAIL_SECONDS,
    TURN_SILENCE_THRESHOLD,
)
from app.voice.turn_pipeline import LocalGemmaTurnPipeline

DEFAULT_FIXTURES = [
    "/Users/nizar/Library/Containers/com.apple.VoiceMemos/Data/tmp/.com.apple.uikit.itemprovider.temporary.93Maoz/Pragathi Mawatha.m4a",
    "/Users/nizar/Library/Containers/com.apple.VoiceMemos/Data/tmp/.com.apple.uikit.itemprovider.temporary.XNW0Im/Pragathi Mawatha 2.m4a",
    "/Users/nizar/Library/Containers/com.apple.VoiceMemos/Data/tmp/.com.apple.uikit.itemprovider.temporary.rs8kau/Pragathi Mawatha 3.m4a",
    "/Users/nizar/Library/Containers/com.apple.VoiceMemos/Data/tmp/.com.apple.uikit.itemprovider.temporary.sNg7qf/Pragathi Mawatha 4.m4a",
    "/Users/nizar/Library/Containers/com.apple.VoiceMemos/Data/tmp/.com.apple.uikit.itemprovider.temporary.BlUFCi/Pragathi Mawatha 5.m4a",
    "/Users/nizar/Library/Containers/com.apple.VoiceMemos/Data/tmp/.com.apple.uikit.itemprovider.temporary.48tReT/Pragathi Mawatha 6.m4a",
    "/Users/nizar/Library/Containers/com.apple.VoiceMemos/Data/tmp/.com.apple.uikit.itemprovider.temporary.C0kstK/Pragathi Mawatha 7.m4a",
    "/Users/nizar/Library/Containers/com.apple.VoiceMemos/Data/tmp/.com.apple.uikit.itemprovider.temporary.Spkg8M/Pragathi Mawatha 8.m4a",
]
REQUIRED_ENV = ("DATABASE_URL", "PHONE_NUMBER_ID", "PINECONE_API_KEY")


class RegressionFailure(AssertionError):
    pass


@dataclass
class Trace:
    events: list[dict[str, Any]] = field(default_factory=list)

    def add(self, kind: str, data: dict) -> None:
        self.events.append({"kind": kind, "at": time.perf_counter(), "data": data})

    def values(self, kind: str) -> list[dict]:
        return [event["data"] for event in self.events if event["kind"] == kind]


class ReplayInputTrack:
    """Inbound media transport double; it never receives outbound agent audio."""

    def __init__(self) -> None:
        self._frames: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self._pts = 0

    async def recv(self) -> AudioFrame:
        frame = await self._frames.get()
        if frame is None:
            raise RuntimeError("recording replay complete")
        return frame

    async def feed(self, pcm: bytes, pace: bool) -> float:
        last_speech_at = time.perf_counter()
        for offset in range(0, len(pcm), TURN_INPUT_CHUNK_SIZE):
            chunk = pcm[offset:offset + TURN_INPUT_CHUNK_SIZE]
            if len(chunk) < TURN_INPUT_CHUNK_SIZE:
                chunk += b"\x00" * (TURN_INPUT_CHUNK_SIZE - len(chunk))
            if _pcm_rms(chunk) > TURN_SILENCE_THRESHOLD:
                last_speech_at = time.perf_counter()
            await self._frames.put(self._frame(chunk))
            if pace:
                await asyncio.sleep(TURN_INPUT_CHUNK_MS / 1000)
        for _ in range(TURN_END_SILENCE_CHUNKS + 2):
            await self._frames.put(self._frame(b"\x00" * TURN_INPUT_CHUNK_SIZE))
            if pace:
                await asyncio.sleep(TURN_INPUT_CHUNK_MS / 1000)
        return last_speech_at

    async def close(self) -> None:
        await self._frames.put(None)

    def _frame(self, pcm: bytes) -> AudioFrame:
        audio = np.frombuffer(pcm, dtype=np.int16).reshape(1, -1)
        frame = AudioFrame.from_ndarray(audio, format="s16", layout="mono")
        frame.sample_rate = 16000
        frame.pts = self._pts
        self._pts += len(pcm) // 2
        return frame


class ReplayOutputTrack:
    """Outbound media transport double that paces playback and records audio."""

    def __init__(self) -> None:
        self._pending_seconds = 0.0
        self._changed = asyncio.Event()
        self._closed = False
        self.audio_events: list[dict[str, float | int]] = []
        self.interruptions = 0

    @property
    def pending_audio_seconds(self) -> float:
        return self._pending_seconds

    def add_pcm_audio(self, pcm: bytes, sample_rate: int) -> None:
        if not pcm or not np.any(np.frombuffer(pcm, dtype=np.int16)):
            return
        duration = len(pcm) / 2 / sample_rate
        self._pending_seconds += duration
        self.audio_events.append({"at": time.perf_counter(), "bytes": len(pcm), "sample_rate": sample_rate})
        self._changed.set()

    def clear_buffer(self) -> None:
        self._pending_seconds = 0.0
        self.interruptions += 1
        self._changed.set()

    async def drain(self) -> None:
        previous = time.perf_counter()
        while not self._closed:
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            self._pending_seconds = max(0.0, self._pending_seconds - (now - previous))
            previous = now
            self._changed.set()
            self._changed.clear()

    async def wait_for_audio_after(self, sequence: int, timeout: float = 30.0) -> dict:
        async def wait() -> dict:
            while len(self.audio_events) <= sequence:
                await self._changed.wait()
                self._changed.clear()
            return self.audio_events[sequence]
        return await asyncio.wait_for(wait(), timeout)

    async def wait_until_idle(self, timeout: float = 30.0) -> None:
        async def wait() -> None:
            while self.pending_audio_seconds > 0.01:
                await self._changed.wait()
                self._changed.clear()
        await asyncio.wait_for(wait(), timeout)

    def close(self) -> None:
        self._closed = True


def decode_recording(path: Path) -> bytes:
    resampler = AudioResampler(format="s16", layout="mono", rate=16000)
    chunks: list[bytes] = []
    with av.open(path) as container:
        stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if stream is None:
            raise RegressionFailure(f"No audio stream in fixture: {path}")
        for frame in container.decode(stream):
            chunks.extend(item.to_ndarray().tobytes() for item in resampler.resample(frame))
        chunks.extend(item.to_ndarray().tobytes() for item in resampler.resample(None))
    pcm = b"".join(chunks)
    if not pcm:
        raise RegressionFailure(f"Empty audio fixture: {path}")
    return pcm


async def wait_for_count(values: list[Any], expected: int, timeout: float = 45.0) -> None:
    started = time.perf_counter()
    while len(values) < expected:
        if time.perf_counter() - started >= timeout:
            raise RegressionFailure(f"Timed out waiting for event {expected}; received {len(values)}")
        await asyncio.sleep(0.05)


async def wait_for_interruptions(
    output_track: ReplayOutputTrack,
    expected: int,
    timeout: float = 15.0,
) -> None:
    started = time.perf_counter()
    while output_track.interruptions < expected:
        if time.perf_counter() - started >= timeout:
            raise RegressionFailure(
                f"Timed out waiting for barge-in {expected}; received {output_track.interruptions}"
            )
        await asyncio.sleep(0.05)


async def run_case(
    name: str,
    clips: list[tuple[Path, bytes]],
    expected_language: str,
    barge_in: bool,
    pace: bool,
) -> dict:
    trace = Trace()
    input_track = ReplayInputTrack()
    output_track = ReplayOutputTrack()
    generations: dict[str, int] = {}
    call_id = f"voice-regression-{name}-{uuid.uuid4().hex[:10]}"

    def interrupt(call: str, track: ReplayOutputTrack) -> None:
        generations[call] = generations.get(call, 0) + 1
        track.clear_buffer()

    pipeline = LocalGemmaTurnPipeline(
        prepare_tts_text=lambda value: value,
        interrupt_playback=interrupt,
        trace_event=trace.add,
    )
    generations[call_id] = 0
    # Match the application lifespan: load the production ASR, property tools,
    # Gemma connection, and OmniVoice before the timing-sensitive call starts.
    await pipeline.prewarm_models()
    drain_task = asyncio.create_task(output_track.drain())
    pipeline_task = asyncio.create_task(pipeline.run(call_id, "94770000000", input_track, output_track, generations))
    utterances: list[dict] = []
    try:
        await output_track.wait_for_audio_after(0, timeout=90.0)  # Greeting became audible.
        if not barge_in:
            await output_track.wait_until_idle()
            await asyncio.sleep(TURN_PLAYBACK_ECHO_TAIL_SECONDS + 0.05)
        language_audio_index = len(output_track.audio_events)
        expected_interruptions = output_track.interruptions + 1 if barge_in else None
        language_end = await input_track.feed(clips[0][1], pace)
        if expected_interruptions is not None:
            await wait_for_interruptions(output_track, expected_interruptions)
        language_first = await output_track.wait_for_audio_after(language_audio_index, timeout=45.0)
        language_utterance = _utterance(clips[0][0], language_end, language_first)
        utterances.append(language_utterance)
        if language_utterance["first_audio_latency_ms"] > 5000:
            raise RegressionFailure(
                f"First generated audio after {clips[0][0].name} was "
                f"{language_utterance['first_audio_latency_ms']:.0f} ms (> 5000 ms)"
            )
        await wait_for_count(trace.values("language.selected"), 1)
        selected = trace.values("language.selected")[-1]["language"]
        if selected != expected_language:
            raise RegressionFailure(f"Expected language {expected_language!r}, got {selected!r}")

        for clip_path, pcm in clips[1:]:
            if barge_in:
                # Start this recording while the preceding agent response is
                # still playing. The pending audio check makes the interruption
                # an actual transport-level barge-in, not merely quick turns.
                if output_track.pending_audio_seconds <= 0.01:
                    raise RegressionFailure(f"No agent audio to interrupt before {clip_path.name}")
            else:
                await output_track.wait_until_idle()
                await asyncio.sleep(TURN_PLAYBACK_ECHO_TAIL_SECONDS + 0.05)

            first_audio_index = len(output_track.audio_events)
            expected_interruptions = output_track.interruptions + 1 if barge_in else None
            speech_end = await input_track.feed(pcm, pace)
            if expected_interruptions is not None:
                await wait_for_interruptions(output_track, expected_interruptions)
            audible = await output_track.wait_for_audio_after(first_audio_index, timeout=45.0)
            item = _utterance(clip_path, speech_end, audible)
            utterances.append(item)
            if item["first_audio_latency_ms"] > 5000:
                raise RegressionFailure(
                    f"First generated audio after {clip_path.name} was {item['first_audio_latency_ms']:.0f} ms (> 5000 ms)"
                )

        await asyncio.sleep(0.1)
        transcripts = trace.values("asr.transcript")
        if len(transcripts) != len(clips):
            raise RegressionFailure(
                f"Expected exactly {len(clips)} caller transcripts; received {len(transcripts)}. "
                "This indicates agent playback or transport audio was transcribed as caller speech."
            )
        tool_calls = trace.values("tool.call")
        if not any(call["name"] == "search_properties" for call in tool_calls):
            raise RegressionFailure("No search_properties call was made for the recorded property requests")
        if barge_in and output_track.interruptions < len(clips):
            raise RegressionFailure(
                f"Expected greeting plus every property response to be interrupted ({len(clips)}); "
                f"recorded {output_track.interruptions} interruptions"
            )
        messages = trace.values("model.messages")
        if not messages or len(messages[-1]["messages"]) < len(clips):
            raise RegressionFailure("Final model request did not retain the accumulated caller history")
        final_user_messages = {
            str(message.get("content", ""))
            for message in messages[-1]["messages"]
            if message.get("role") == "user"
        }
        missing_history = [
            item["text"] for item in transcripts
            if item["text"] not in final_user_messages
        ]
        if missing_history:
            raise RegressionFailure(
                "Final model request is missing caller context: " + "; ".join(missing_history)
            )

        return {
            "case": name,
            "fixtures": [str(path) for path, _ in clips],
            "vad_segments": trace.values("vad.segment"),
            "asr_transcripts": transcripts,
            "language_selected": selected,
            "model_messages": messages,
            "tool_calls": tool_calls,
            "tool_results": trace.values("tool.result"),
            "first_audio_latency": utterances,
            "final_response": trace.values("model.output")[-1] if trace.values("model.output") else None,
            "barge_ins": trace.values("vad.barge_in"),
            "output_interruptions": output_track.interruptions,
            "status": "passed",
        }
    except Exception as exc:
        # Preserve the raw production trace even when a strict assertion
        # fails; failures are the most useful regression artifacts.
        setattr(exc, "case_artifact", {
            "case": name,
            "fixtures": [str(path) for path, _ in clips],
            "vad_segments": trace.values("vad.segment"),
            "asr_transcripts": trace.values("asr.transcript"),
            "language_selected": (trace.values("language.selected")[-1]
                                  if trace.values("language.selected") else None),
            "model_messages": trace.values("model.messages"),
            "tool_calls": trace.values("tool.call"),
            "tool_results": trace.values("tool.result"),
            "model_outputs": trace.values("model.output"),
            "first_audio_latency": utterances,
            "barge_ins": trace.values("vad.barge_in"),
            "output_interruptions": output_track.interruptions,
            "status": "failed",
            "error": str(exc),
        })
        raise
    finally:
        await input_track.close()
        await asyncio.gather(pipeline_task, return_exceptions=True)
        output_track.close()
        drain_task.cancel()
        await asyncio.gather(drain_task, return_exceptions=True)


def _utterance(path: Path, speech_end: float, audio: dict) -> dict:
    return {
        "fixture": str(path),
        "speech_end_at": speech_end,
        "first_audio_at": audio["at"],
        "first_audio_latency_ms": (float(audio["at"]) - speech_end) * 1000.0,
    }


def _pcm_rms(pcm: bytes) -> float:
    values = np.frombuffer(pcm, dtype=np.int16)
    return float(np.sqrt(np.mean(values.astype(np.float64) ** 2))) if values.size else 0.0


def preflight(fixtures: list[Path]) -> None:
    missing = [str(path) for path in fixtures if not path.is_file()]
    if missing:
        raise RegressionFailure("Missing voice fixtures:\n" + "\n".join(missing))
    missing_env = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing_env:
        raise RegressionFailure("Missing required production configuration: " + ", ".join(missing_env))
    try:
        response = httpx.get(f"{LLM_BASE_URL}/models", timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RegressionFailure(f"Production llama.cpp server is unavailable at {LLM_BASE_URL}: {exc}") from exc


def write_artifact(directory: Path, name: str, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


async def main_async(args: argparse.Namespace) -> int:
    fixtures = [Path(item).expanduser().resolve() for item in args.fixture]
    preflight(fixtures)
    clips = [(path, decode_recording(path)) for path in fixtures]
    run_dir = Path(args.artifacts_dir) / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run = {"started_at": datetime.now(UTC).isoformat(), "production_settings": True, "cases": []}
    try:
        for name, barge_in in (("normal", False), ("barge_in", True)):
            try:
                case = await run_case(name, clips, args.expected_language, barge_in, not args.no_pace)
            except Exception as exc:
                case = getattr(exc, "case_artifact", None)
                if case is not None:
                    run["cases"].append(case)
                    write_artifact(run_dir, name, case)
                raise
            run["cases"].append(case)
            write_artifact(run_dir, name, case)
    except Exception as exc:
        run["status"] = "failed"
        run["error"] = str(exc)
        write_artifact(run_dir, "failure", run)
        print(f"VOICE REGRESSION FAILED; artifact: {run_dir / 'failure.json'}", file=sys.stderr)
        raise
    run["status"] = "passed"
    write_artifact(run_dir, "summary", run)
    print(f"VOICE REGRESSION PASSED; artifacts: {run_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="append", default=[], help="M4A/WAV caller recording, in turn order")
    parser.add_argument("--expected-language", default="si")
    parser.add_argument("--artifacts-dir", default="run_logs/voice_regressions")
    parser.add_argument("--no-pace", action="store_true", help="Replay frames without 40 ms real-time pacing")
    args = parser.parse_args()
    if not args.fixture:
        args.fixture = DEFAULT_FIXTURES
    if len(args.fixture) < 2:
        parser.error("provide a language-selection recording followed by at least one property recording")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main_async(parse_args())))
    except RegressionFailure as exc:
        print(f"VOICE REGRESSION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
