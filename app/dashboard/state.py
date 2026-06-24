import json
import os
import time
from dataclasses import dataclass, field

SESSION_STORE_PATH = "run_logs/call_sessions.json"
MAX_STORED_CALLS = 100


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

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "caller_phone": self.caller_phone,
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "ended_at": self.ended_at,
            "transcript": [event.to_dict() for event in self.transcript],
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
        return call


class DashboardState:
    def __init__(self) -> None:
        self._calls: dict[str, LiveCall] = self._load_calls()

    def start_call(self, call_id: str, caller_phone: str = "") -> None:
        self._calls[call_id] = LiveCall(call_id=call_id, caller_phone=caller_phone)
        self._persist()

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
        self._persist()

    def end_call(self, call_id: str) -> None:
        call = self._calls.get(call_id)
        if call is None:
            return
        call.status = "ended"
        call.ended_at = time.time()
        call.updated_at = call.ended_at
        self._trim_old_calls()
        self._persist()

    def add_transcript(self, call_id: str, speaker: str, text: str) -> None:
        if not text:
            return
        call = self._calls.get(call_id)
        if call is None:
            call = LiveCall(call_id=call_id, caller_phone="", status="active")
            self._calls[call_id] = call
        call.transcript.append(TranscriptEvent(speaker=speaker, text=text))
        call.updated_at = time.time()
        self._persist()

    def snapshot(self) -> dict:
        calls = sorted(
            self._calls.values(),
            key=lambda call: (call.status != "active", -call.updated_at),
        )
        return {"calls": [call.to_dict() for call in calls]}

    def _load_calls(self) -> dict[str, LiveCall]:
        if not os.path.exists(SESSION_STORE_PATH):
            return {}
        try:
            with open(SESSION_STORE_PATH, encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return {}
        calls = {}
        for item in data.get("calls", []):
            call = LiveCall.from_dict(item)
            if call.call_id:
                calls[call.call_id] = call
        return calls

    def _persist(self) -> None:
        os.makedirs(os.path.dirname(SESSION_STORE_PATH), exist_ok=True)
        self._trim_old_calls()
        payload = self.snapshot()
        with open(SESSION_STORE_PATH, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _trim_old_calls(self) -> None:
        if len(self._calls) <= MAX_STORED_CALLS:
            return
        calls = sorted(self._calls.values(), key=lambda call: call.updated_at, reverse=True)
        self._calls = {call.call_id: call for call in calls[:MAX_STORED_CALLS]}


dashboard_state = DashboardState()
