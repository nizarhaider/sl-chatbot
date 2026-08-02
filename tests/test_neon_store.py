from app.dashboard.neon_store import _dashboard_status, _format_transcript
from app.dashboard.state import DashboardState


class FakeCallStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def save_call(self, call: dict) -> None:
        self.calls.append(call)


def test_dashboard_statuses_match_existing_dashboard_values() -> None:
    assert _dashboard_status("connecting") == "started"
    assert _dashboard_status("active") == "active"
    assert _dashboard_status("ended") == "completed"


def test_transcript_is_readable_on_the_dashboard() -> None:
    assert _format_transcript(
        [
            {"speaker": "caller", "text": "Hello"},
            {"speaker": "assistant", "text": "How can I help?"},
        ]
    ) == "Caller: Hello · Assistant: How can I help?"


def test_empty_transcript_is_null() -> None:
    assert _format_transcript([]) is None


def test_dashboard_state_queues_each_call_update(tmp_path) -> None:
    store = FakeCallStore()
    state = DashboardState(
        call_store=store,
        session_store_path=str(tmp_path / "call_sessions.json"),
    )

    state.start_call("test-call", "94770000000")
    state.mark_call_active("test-call")
    state.add_transcript("test-call", "caller", "Hello")
    state.add_transcript("test-call", "assistant", "Hi")
    state.end_call("test-call")
    state.close()

    assert len(store.calls) == 5
    assert store.calls[-1]["status"] == "ended"
    assert [event["text"] for event in store.calls[-1]["transcript"]] == ["Hello", "Hi"]
