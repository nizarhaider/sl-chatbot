import asyncio
import logging
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import boto3
import psutil
import uvicorn
from loguru import logger as pipecat_logger
from psycopg.types.json import Jsonb

from app.config import S3_BUCKET, RECORDING_SAMPLE_RATE, MP3_BITRATE, PORT, HEARTBEAT_SECONDS, SHUTDOWN_SECONDS, LOG_LEVEL
from app.tools import agent_ids, connection

logger = logging.getLogger(__name__)


class CallStore:
    def save_call(self, call):
        agent_id, customer_id = (call["agent_id"], call["customer_id"]) if call.get("agent_id") else agent_ids()
        events = call.get("events", [])
        usage_events = [event for event in events if event.get("kind") == "gemini_live.usage"]
        totals = [event["data"].get("total_tokens") for event in usage_events]
        tokens = sum(totals) if totals and all(value is not None for value in totals) else None
        requests = [event["data"]["usage"] for event in usage_events if event["data"].get("usage")]
        usage = {"provider": "google_gemini_live", "requests": requests} if requests else None
        recording = next(
            (event["data"] for event in reversed(events) if event.get("kind") == "recording.archived"), None
        )
        end = call.get("ended_at")
        with connection() as db:
            db.execute(
                """insert into portal_calls
                   (id,customer_id,agent_id,customer_phone,status,transcript,duration_seconds,
                    tokens,usage,events,recording_url,created_at)
                   values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,to_timestamp(%s))
                   on conflict(id) do update set
                   status=excluded.status,transcript=excluded.transcript,
                   duration_seconds=excluded.duration_seconds,tokens=excluded.tokens,
                   usage=excluded.usage,events=excluded.events,
                   recording_url=excluded.recording_url,updated_at=now()
                   where portal_calls.customer_id=excluded.customer_id
                     and portal_calls.agent_id=excluded.agent_id""",
                (
                    f"{agent_id}:{call['call_id']}",
                    customer_id,
                    agent_id,
                    call.get("caller_phone", ""),
                    call.get("status", "connecting"),
                    "\n\n".join(
                        f"{event['speaker'].capitalize()}: {event['text']}" for event in call.get("transcript", [])
                    ),
                    max(0, end - call["started_at"]) if end else None,
                    tokens,
                    Jsonb(usage) if usage else None,
                    Jsonb(events),
                    f"s3://{recording['bucket']}/{recording['key']}" if recording else None,
                    call["started_at"],
                ),
            )


class CallState:
    def __init__(self, persist=True):
        self.calls = {}
        self.writer = ThreadPoolExecutor(max_workers=1) if persist else None
        self.archive_tasks = set()

    def start_call(self, context):
        self.trim()
        now = time.time()
        self.calls[context.call_id] = {
            "call_id": context.call_id, "caller_phone": context.caller_phone,
            "agent_id": context.agent_id, "customer_id": context.customer_id,
            "status": "connecting", "started_at": now, "ended_at": None,
            "transcript": [], "events": [],
        }
        self.emit(context.call_id, "call.connected")
        self.persist(context.call_id)

    def emit(self, call_id, kind, data=None):
        call = self.calls.get(call_id)
        if call:
            call["events"].append({"id": uuid.uuid4().hex, "kind": kind, "timestamp": time.time(), "data": data or {}})
            call["events"] = call["events"][-500:]

    def transcript(self, call_id, speaker, text):
        call = self.calls.get(call_id)
        if call and text.strip():
            call["transcript"].append({"speaker": speaker, "text": text.strip(), "timestamp": time.time()})
            self.emit(call_id, f"transcript.{speaker}", {"text": text.strip()})
            self.persist(call_id)

    def active(self, call_id):
        if call_id in self.calls:
            self.calls[call_id]["status"] = "active"
            self.emit(call_id, "call.active")
            self.persist(call_id)

    def end_call(self, call_id):
        call = self.calls.get(call_id)
        if call and call["ended_at"] is None:
            call.update(status="ended", ended_at=time.time())
            self.emit(call_id, "call.ended")
            self.persist(call_id)

    def persist(self, call_id):
        if self.writer and call_id in self.calls:
            call = self.calls[call_id]
            self.writer.submit(
                self._save, {**call, "events": list(call["events"]), "transcript": list(call["transcript"])}
            )

    @staticmethod
    def _save(call):
        try:
            CallStore().save_call(call)
        except Exception:
            logger.exception("Call persistence failed")

    def archive(self, call_id, pcm):
        if pcm:
            task = asyncio.create_task(self._archive(call_id, pcm))
            self.archive_tasks.add(task)
            task.add_done_callback(self.archive_tasks.discard)

    async def _archive(self, call_id, pcm):
        try:
            key = _object_key(call_id)

            def upload():
                boto3.client("s3").put_object(
                    Bucket=S3_BUCKET,
                    Key=key,
                    Body=_pcm16_stereo_to_mp3(pcm),
                    ContentType="audio/mpeg",
                    ServerSideEncryption="AES256",
                )

            await asyncio.to_thread(upload)
            self.emit(call_id, "recording.archived", {"bucket": S3_BUCKET, "key": key})
            self.persist(call_id)
        except Exception:
            logger.exception("Call recording archival failed")

    def trim(self):
        removable = [key for key, call in self.calls.items() if call["ended_at"] is not None]
        if not self.archive_tasks:
            for key in removable[:max(0, len(self.calls) - 99)]:
                del self.calls[key]

    async def close(self):
        if self.archive_tasks:
            await asyncio.gather(*list(self.archive_tasks))
        if self.writer:
            await asyncio.to_thread(self.writer.shutdown, wait=True)


call_state = CallState()


def _pcm16_stereo_to_mp3(pcm: bytes) -> bytes:
    result = subprocess.run(
        shlex.split(f"ffmpeg -nostdin -hide_banner -loglevel error -f s16le -ar {RECORDING_SAMPLE_RATE} "
                    f"-ac 2 -i pipe:0 -codec:a libmp3lame -b:a {MP3_BITRATE} -f mp3 pipe:1"),
        input=pcm,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode or not result.stdout:
        raise RuntimeError(
            f"Could not encode call recording as MP3: {result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


def _object_key(call_id: str) -> str:
    safe_call_id = re.sub(r"[^A-Za-z0-9._-]+", "_", call_id).strip("_") or "unknown"
    now = datetime.now(UTC)
    return f"call-recordings/{now:%Y/%m/%d}/{safe_call_id}/{uuid.uuid4().hex}.mp3"


def run_runtime():
    from app.server import create_app

    logging.basicConfig(level=logging.INFO)
    pipecat_logger.remove()
    pipecat_logger.add(sys.stderr, level=LOG_LEVEL)

    app = create_app()
    agent_id = os.environ["VOICE_AGENT_ID"]
    tunnel = None
    if os.environ.get("CLOUDFLARED_TUNNEL_TOKEN"):
        tunnel = subprocess.Popen(
            ["cloudflared", "tunnel", "--no-autoupdate", "run", "--token", os.environ["CLOUDFLARED_TUNNEL_TOKEN"]],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    stopped = threading.Event()
    restart_requested = threading.Event()

    def heartbeat():
        process = psutil.Process()
        while not stopped.is_set():
            try:
                status = (
                    "error"
                    if getattr(app.state, "voice_startup_error", "")
                    else "ready"
                    if getattr(app.state, "voice_ready", False)
                    else "warming_up"
                )
                telemetry = {
                    "case_routing": False,
                    "case_callbacks": True,
                    "active_calls": len(app.state.calls.connections),
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_mb": round(process.memory_info().rss / 1048576, 1),
                    "status": status,
                    "runtime_url": "https://whatsapp.serendibai.lk" if tunnel and tunnel.poll() is None else "",
                    "error": "Gemini startup failed. Check the configured API key and model."
                    if status == "error"
                    else "",
                }
                with connection() as db:
                    current = db.execute(
                        "select max_calls,deployed_version from portal_agents where id=%s for update", (agent_id,)
                    ).fetchone()
                    restart = current is not None and current["deployed_version"] == -1
                    row = db.execute(
                        "update portal_agents set heartbeat_at=now(),telemetry=%s,status=%s,deployed_version=version where id=%s returning max_calls",
                        (Jsonb(telemetry), "restarting" if restart else status, agent_id),
                    ).fetchone()
                if row:
                    os.environ["MAX_CALLS"] = str(row["max_calls"])
                if restart and not app.state.calls.connections:
                    restart_requested.set()
                    server.should_exit = True
            except Exception as exc:
                logger.warning("Neon heartbeat unavailable: %s", type(exc).__name__)
            stopped.wait(HEARTBEAT_SECONDS)

    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=PORT, access_log=False, timeout_graceful_shutdown=SHUTDOWN_SECONDS))
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        server.run()
    finally:
        stopped.set()
        thread.join(timeout=15)
        if tunnel:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill()

    if restart_requested.is_set():
        os.execv(sys.executable, [sys.executable, "-m", "app.server"])
