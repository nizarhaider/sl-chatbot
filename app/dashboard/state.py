import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from app.dashboard.neon_store import NeonCallStore

SESSION_STORE_PATH = "run_logs/call_sessions.json"
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

    @classmethod
    def from_dict(cls, data: dict) -> "LiveCall":
        call = cls(
            call_id=data.get("call_id", ""),
            caller_phone=data.get("caller_phone", ""),
            status=data.get("status", "ended"),
            started_at=float(data.get("started_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
            ended_at=data.get("ended_at"),
        )
        call.transcript = [
            TranscriptEvent(
                speaker=event.get("speaker", ""),
                text=event.get("text", ""),
                timestamp=float(event.get("timestamp") or call.started_at),
            )
            for event in data.get("transcript", [])
            if event.get("text")
        ]
        call.events = list(data.get("events", []))[-500:]
        return call


class DashboardState:
    def __init__(
        self,
        call_store: NeonCallStore | None = None,
        session_store_path: str = SESSION_STORE_PATH,
    ) -> None:
        self._session_store_path = session_store_path
        self._call_store = call_store if call_store is not None else NeonCallStore.from_env()
        self._write_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="neon-call-writer")
            if self._call_store is not None
            else None
        )
        self._calls: dict[str, LiveCall] = self._load_calls()

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

    def snapshot(self) -> dict:
        calls = sorted(
            self._calls.values(),
            key=lambda call: (call.status != "active", -call.updated_at),
        )
        return {"generated_at": time.time(), "calls": [call.to_dict() for call in calls]}

    def _load_calls(self) -> dict[str, LiveCall]:
        if not os.path.exists(self._session_store_path):
            return {}
        try:
            with open(self._session_store_path, encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return {}
        calls = {}
        for item in data.get("calls", []):
            call = LiveCall.from_dict(item)
            if call.call_id:
                calls[call.call_id] = call
        return calls

    def _persist(self, call: LiveCall) -> None:
        Path(self._session_store_path).parent.mkdir(parents=True, exist_ok=True)
        self._trim_old_calls()
        payload = self.snapshot()
        with open(self._session_store_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        if self._write_executor is not None:
            self._write_executor.submit(self._save_call_to_neon, call.to_dict())

    def _save_call_to_neon(self, call: dict) -> None:
        try:
            assert self._call_store is not None
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


dashboard_state = DashboardState()
