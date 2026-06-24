import time
from dataclasses import dataclass, field


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
    transcript: list[TranscriptEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "caller_phone": self.caller_phone,
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "transcript": [event.to_dict() for event in self.transcript],
        }


class DashboardState:
    def __init__(self) -> None:
        self._calls: dict[str, LiveCall] = {}

    def start_call(self, call_id: str, caller_phone: str = "") -> None:
        self._calls[call_id] = LiveCall(call_id=call_id, caller_phone=caller_phone)

    def mark_call_active(self, call_id: str, caller_phone: str = "") -> None:
        call = self._calls.get(call_id)
        if call is None:
            call = LiveCall(call_id=call_id, caller_phone=caller_phone)
            self._calls[call_id] = call
        if caller_phone:
            call.caller_phone = caller_phone
        call.status = "active"
        call.updated_at = time.time()

    def end_call(self, call_id: str) -> None:
        self._calls.pop(call_id, None)

    def add_transcript(self, call_id: str, speaker: str, text: str) -> None:
        if not text:
            return
        call = self._calls.get(call_id)
        if call is None:
            call = LiveCall(call_id=call_id, caller_phone="", status="active")
            self._calls[call_id] = call
        call.transcript.append(TranscriptEvent(speaker=speaker, text=text))
        call.updated_at = time.time()

    def snapshot(self) -> dict:
        calls = sorted(self._calls.values(), key=lambda call: call.started_at)
        return {"calls": [call.to_dict() for call in calls]}


dashboard_state = DashboardState()
