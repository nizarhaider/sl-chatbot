import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import psycopg


CALL_ID_NAMESPACE = uuid.UUID("70b37a94-aefa-4c52-a5f8-916272bd5f8c")


@dataclass(frozen=True)
class WhatsAppNumberMapping:
    customer_id: str
    whatsapp_number_id: str


class NeonCallStore:
    """Persist WhatsApp call state in the Neon database used by the dashboard."""

    def __init__(self, database_url: str, phone_number_id: str) -> None:
        self._database_url = database_url
        self._phone_number_id = phone_number_id
        self._mapping: WhatsAppNumberMapping | None = None

    @classmethod
    def from_env(cls) -> "NeonCallStore | None":
        database_url = os.environ.get("DATABASE_URL")
        phone_number_id = os.environ.get("PHONE_NUMBER_ID")
        if not database_url or not phone_number_id:
            return None
        return cls(database_url=database_url, phone_number_id=phone_number_id)

    def save_call(self, call: dict) -> None:
        with psycopg.connect(self._database_url, connect_timeout=10) as connection:
            mapping = self._mapping or self._load_mapping(connection)
            self._mapping = mapping
            connection.execute(
                """
                insert into calls (
                    id,
                    customer_id,
                    whatsapp_number_id,
                    customer_phone,
                    status,
                    transcript,
                    created_at
                )
                values (%s, %s, %s, %s, %s, %s, %s)
                on conflict (id) do update set
                    customer_phone = excluded.customer_phone,
                    status = excluded.status,
                    transcript = excluded.transcript
                """,
                (
                    str(uuid.uuid5(CALL_ID_NAMESPACE, call["call_id"])),
                    mapping.customer_id,
                    mapping.whatsapp_number_id,
                    call.get("caller_phone") or None,
                    _dashboard_status(call.get("status", "connecting")),
                    _format_transcript(call.get("transcript", [])),
                    datetime.fromtimestamp(float(call["started_at"]), tz=timezone.utc),
                ),
            )

    def _load_mapping(self, connection: psycopg.Connection) -> WhatsAppNumberMapping:
        row = connection.execute(
            """
            select customer_id, id
            from whatsapp_numbers
            where phone_number_id = %s
              and status = 'active'
            limit 1
            """,
            (self._phone_number_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(
                "No active dashboard WhatsApp number matches PHONE_NUMBER_ID; "
                "configure the number in Neon before taking calls"
            )
        return WhatsAppNumberMapping(customer_id=str(row[0]), whatsapp_number_id=str(row[1]))


def _dashboard_status(status: str) -> str:
    return {
        "connecting": "started",
        "active": "active",
        "ended": "completed",
    }.get(status, status)


def _format_transcript(events: list[dict]) -> str | None:
    lines = []
    for event in events:
        text = str(event.get("text", "")).strip()
        if not text:
            continue
        speaker = str(event.get("speaker", "speaker")).strip().capitalize()
        lines.append(f"{speaker}: {text}")
    return " · ".join(lines) or None
