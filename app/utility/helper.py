import asyncio
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime

import boto3
import numpy as np
import psutil
import uvicorn

from psycopg.types.json import Jsonb

from app.utility.tools import CallStore, connection

def pcm_rms(pcm: bytes) -> float:
    samples = np.frombuffer(pcm, dtype=np.int16)
    if not samples.size:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def join_transcript(chunks: list[str]) -> str:
    merged = ""
    for chunk in chunks:
        text = chunk.strip()
        if not text:
            continue
        if text.startswith(merged):
            merged = text
        elif not merged.endswith(text):
            merged = f"{merged} {text}".strip()
    return merged


MAX_STORED_CALLS = 100
logger = logging.getLogger(__name__)


@dataclass
class TranscriptEvent:
    speaker: str
    text: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "text": self.text,
            "timestamp": self.timestamp,
        }


@dataclass
class LiveCall:
    call_id: str
    caller_phone: str
    status: str = "connecting"
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    transcript: list[TranscriptEvent] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "caller_phone": self.caller_phone,
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "ended_at": self.ended_at,
            "transcript": [event.to_dict() for event in self.transcript],
            "events": self.events,
        }



class CallState:
    def __init__(self) -> None:
        self._call_store = CallStore() if os.environ.get("VOICE_AGENT_ID") and os.environ.get("DATABASE_URL") else None
        self._write_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="neon-call-writer") if self._call_store else None
        self._calls: dict[str, LiveCall] = {}

    def start_call(self, call_id: str, caller_phone: str = "") -> None:
        self._calls[call_id] = LiveCall(call_id=call_id, caller_phone=caller_phone)
        self.emit(call_id, "call.connected", {"caller_phone": caller_phone})
        self._persist(self._calls[call_id])

    def mark_call_active(self, call_id: str, caller_phone: str = "") -> None:
        call = self._calls.get(call_id)
        if call is None:
            call = LiveCall(call_id=call_id, caller_phone=caller_phone)
            self._calls[call_id] = call
        if caller_phone:
            call.caller_phone = caller_phone
        call.status = "active"
        call.ended_at = None
        call.updated_at = time.time()
        self.emit(call_id, "call.active", {"caller_phone": call.caller_phone})
        self._persist(call)

    def end_call(self, call_id: str) -> None:
        call = self._calls.get(call_id)
        if call is None:
            return
        if call.status == "ended":
            return
        call.status = "ended"
        call.ended_at = time.time()
        call.updated_at = call.ended_at
        self.emit(call_id, "call.ended", {})
        self._trim_old_calls()
        self._persist(call)

    def add_transcript(self, call_id: str, speaker: str, text: str) -> None:
        if not text:
            return
        call = self._calls.get(call_id)
        if call is None:
            call = LiveCall(call_id=call_id, caller_phone="", status="active")
            self._calls[call_id] = call
        call.transcript.append(TranscriptEvent(speaker=speaker, text=text))
        call.updated_at = time.time()
        self.emit(call_id, f"transcript.{speaker}", {"text": text})
        self._persist(call)

    def emit(self, call_id: str, kind: str, data: dict | None = None) -> None:
        call = self._calls.get(call_id)
        if call is None:
            call = LiveCall(call_id=call_id, caller_phone="", status="active")
            self._calls[call_id] = call
        call.events.append({
            "id": f"{call_id}:{len(call.events) + 1}",
            "kind": kind,
            "timestamp": time.time(),
            "data": data or {},
        })
        del call.events[:-500]
        call.updated_at = time.time()

    def persist(self, call_id: str) -> None:
        call = self._calls.get(call_id)
        if call is not None:
            self._persist(call)

    def _persist(self, call: LiveCall) -> None:
        self._trim_old_calls()
        if self._write_executor is not None:
            self._write_executor.submit(self._save_call, call.to_dict())

    def _save_call(self, call: dict) -> None:
        try:
            self._call_store.save_call(call)
        except Exception:
            logger.exception("Failed to persist call %s to Neon", call.get("call_id"))

    def close(self) -> None:
        if self._write_executor is not None:
            self._write_executor.shutdown(wait=True)

    def _trim_old_calls(self) -> None:
        if len(self._calls) <= MAX_STORED_CALLS:
            return
        calls = sorted(self._calls.values(), key=lambda call: call.updated_at, reverse=True)
        self._calls = {call.call_id: call for call in calls[:MAX_STORED_CALLS]}


call_state = CallState()


S3_BUCKET = os.environ.get("CALL_RECORDINGS_BUCKET", "serendibai-call-recordings-744861799976")
SAMPLE_RATE = 16_000


class CallAudioRecorder:
    """Collect a full call as stereo PCM: caller left, agent right."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started_at = clock()
        self._caller: list[tuple[int, np.ndarray]] = []
        self._agent: list[tuple[int, np.ndarray]] = []
        self._caller_cursor = 0

    def add_caller_pcm(self, pcm: bytes) -> None:
        samples = self._add(self._caller, pcm, channels=1, sample_rate=SAMPLE_RATE, offset=self._caller_cursor)
        self._caller_cursor += samples

    def add_agent_pcm(self, pcm: bytes, sample_rate: int, channels: int = 2, offset_seconds: float | None = None) -> None:
        offset = round(offset_seconds * SAMPLE_RATE) if offset_seconds is not None else None
        self._add(self._agent, pcm, channels=channels, sample_rate=sample_rate, offset=offset)

    def render_pcm16_stereo(self) -> bytes:
        length = max(
            (offset + len(samples) for channel in (self._caller, self._agent) for offset, samples in channel),
            default=0,
        )
        if not length:
            return b""
        output = np.zeros((length, 2), dtype=np.int32)
        self._mix(output, self._caller, 0)
        self._mix(output, self._agent, 1)
        return np.clip(output, -32768, 32767).astype(np.int16).tobytes()

    def _add(self, destination, pcm: bytes, channels: int, sample_rate: int, offset: int | None = None) -> int:
        if not pcm:
            return 0
        samples = np.frombuffer(pcm, dtype=np.int16)
        if channels > 1:
            samples = samples[: len(samples) // channels * channels].reshape(-1, channels)[:, 0]
        if sample_rate != SAMPLE_RATE:
            if sample_rate % SAMPLE_RATE:
                raise ValueError(f"Unsupported recording sample rate: {sample_rate}")
            samples = samples[:: sample_rate // SAMPLE_RATE]
        if samples.size:
            if offset is None:
                offset = max(0, round((self._clock() - self._started_at) * SAMPLE_RATE))
            destination.append((offset, samples.copy()))
            return len(samples)
        return 0

    @staticmethod
    def _mix(output: np.ndarray, segments, channel: int) -> None:
        for offset, samples in segments:
            output[offset : offset + len(samples), channel] += samples.astype(np.int32)


class CallAudioArchive:
    """Best-effort archival of one complete call, outside the live voice path."""

    def __init__(self, client_factory: Callable[[], object] | None = None, encode=None) -> None:
        self._client_factory = client_factory or (lambda: boto3.client("s3"))
        self._encode = encode or _pcm16_stereo_to_mp3
        self._client: object | None = None
        self._tasks: set[asyncio.Task] = set()

    def archive_call(self, call_id: str, recording: CallAudioRecorder) -> None:
        task = asyncio.create_task(self._render_encode_and_upload(call_id, recording))
        self._tasks.add(task)
        task.add_done_callback(self._log_result)

    async def _render_encode_and_upload(self, call_id: str, recording: CallAudioRecorder) -> None:
        pcm = recording.render_pcm16_stereo()
        if not pcm:
            return
        mp3 = await asyncio.to_thread(self._encode, pcm)
        key = _object_key(call_id)
        if self._client is None:
            self._client = self._client_factory()
        await asyncio.to_thread(self._client.put_object, Bucket=S3_BUCKET, Key=key, Body=mp3, ContentType="audio/mpeg", ServerSideEncryption="AES256")
        call_state.emit(call_id, "recording.archived", {"bucket": S3_BUCKET, "key": key})
        call_state.persist(call_id)
        logger.info("Archived full call MP3 for %s to s3://%s/%s", call_id, S3_BUCKET, key)

    def _log_result(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.exception("Failed to archive full call to S3")


def _pcm16_stereo_to_mp3(pcm: bytes) -> bytes:
    result = subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(SAMPLE_RATE),
        "-ac", "2", "-i", "pipe:0", "-codec:a", "libmp3lame", "-b:a", "64k", "-f", "mp3", "pipe:1",
    ], input=pcm, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode or not result.stdout:
        raise RuntimeError(f"Could not encode call recording as MP3: {result.stderr.decode('utf-8', errors='replace').strip()}")
    return result.stdout


def _object_key(call_id: str) -> str:
    safe_call_id = re.sub(r"[^A-Za-z0-9._-]+", "_", call_id).strip("_") or "unknown"
    now = datetime.now(UTC)
    return f"call-recordings/{now:%Y/%m/%d}/{safe_call_id}/{uuid.uuid4().hex}.mp3"


def run_runtime():
    from app.gemini import voice_agent
    from app.whatsapp import create_app

    app = create_app()
    agent_id = os.environ["VOICE_AGENT_ID"]
    tunnel = None
    if os.environ.get("CLOUDFLARED_TUNNEL_TOKEN"):
        tunnel = subprocess.Popen(
            ["cloudflared", "tunnel", "--no-autoupdate", "run", "--token", os.environ["CLOUDFLARED_TUNNEL_TOKEN"]],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def heartbeat():
        process = psutil.Process()
        while True:
            try:
                status = "error" if getattr(app.state, "voice_startup_error", "") else "ready" if getattr(app.state, "voice_ready", False) else "warming_up"
                telemetry = {
                    "active_calls": len(voice_agent.active_calls),
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_mb": round(process.memory_info().rss / 1048576, 1),
                    "status": status,
                    "runtime_url": "https://whatsapp.serendibai.lk" if tunnel and tunnel.poll() is None else "",
                    "error": "Gemini startup failed. Check the configured API key and model." if status == "error" else "",
                }
                with connection() as db:
                    current = db.execute(
                        "select max_calls,deployed_version from portal_agents where id=%s for update",
                        (agent_id,),
                    ).fetchone()
                    restart = current is not None and current["deployed_version"] == -1
                    row = db.execute(
                        "update portal_agents set heartbeat_at=now(),telemetry=%s,status=%s,deployed_version=version where id=%s returning max_calls",
                        (Jsonb(telemetry), "restarting" if restart else status, agent_id),
                    ).fetchone()
                if row:
                    os.environ["MAX_CALLS"] = str(row["max_calls"])
                if restart:
                    if tunnel:
                        tunnel.terminate()
                    os.execv(sys.executable, [sys.executable, "-c", "from app.utility.helper import run_runtime; run_runtime()"])
            except Exception as exc:
                logger.warning("Neon heartbeat unavailable: %s", type(exc).__name__)
            time.sleep(20)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        uvicorn.run(app, host="0.0.0.0", port=8081)
    finally:
        if tunnel:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill()
