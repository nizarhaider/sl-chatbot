"""Neon call persistence and property-search tools."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import psycopg
from psycopg.errors import OperationalError, UniqueViolation
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)
COLOMBO = ZoneInfo("Asia/Colombo")
CALL_NAMESPACE = uuid.UUID("70b37a94-aefa-4c52-a5f8-916272bd5f8c")
PROPERTY_NAMESPACE = uuid.UUID("a1c78a4d-7d09-4d29-b24b-7c427ab7912f")

PROPERTIES = (
    (
        "horizon-residencies-malabe",
        "Horizon Residencies",
        "Malabe",
        "apartment",
        2,
        28_000_000,
        "Near schools and supermarkets.",
    ),
    (
        "lakeview-villas-piliyandala",
        "Lakeview Villas",
        "Piliyandala",
        "villa",
        3,
        48_000_000,
        "Garden, parking, and lake access.",
    ),
    (
        "green-acres-kurunegala",
        "Green Acres",
        "Kurunegala",
        "land",
        None,
        9_500_000,
        "Ten-perch residential land with clear title; bank loans supported.",
    ),
    (
        "ocean-breeze-apartments-dehiwala",
        "Ocean Breeze Apartments",
        "Dehiwala",
        "apartment",
        2,
        32_000_000,
        "One and two-bedroom units with sea views; ready soon.",
    ),
    (
        "city-gardens-rajagiriya",
        "City Gardens",
        "Rajagiriya",
        "apartment",
        3,
        41_500_000,
        "Three-bedroom apartment with two parking spaces and gym access.",
    ),
    (
        "capital-heights-battaramulla",
        "Capital Heights",
        "Battaramulla",
        "apartment",
        2,
        35_000_000,
        "Two-bedroom apartment near the administrative district; ready to occupy.",
    ),
    (
        "urban-nest-nugegoda",
        "Urban Nest",
        "Nugegoda",
        "apartment",
        2,
        29_500_000,
        "Apartment close to public transport, shopping, and schools.",
    ),
    (
        "palm-court-mount-lavinia",
        "Palm Court",
        "Mount Lavinia",
        "apartment",
        3,
        46_000_000,
        "Three-bedroom apartment within walking distance of the beach.",
    ),
    (
        "orchard-villas-kottawa",
        "Orchard Villas",
        "Kottawa",
        "villa",
        4,
        57_000_000,
        "Four-bedroom gated villa with garden, parking, and solar hot water.",
    ),
    (
        "lake-road-homes-maharagama",
        "Lake Road Homes",
        "Maharagama",
        "house",
        3,
        38_500_000,
        "Detached family home with three bedrooms and covered parking.",
    ),
    (
        "canal-view-homes-negombo",
        "Canal View Homes",
        "Negombo",
        "house",
        3,
        34_000_000,
        "Three-bedroom house with garden, ten minutes from Negombo town.",
    ),
    (
        "hill-country-residences-kandy",
        "Hill Country Residences",
        "Kandy",
        "apartment",
        2,
        31_500_000,
        "Two-bedroom apartment with hill views and backup power.",
    ),
    (
        "southern-breeze-villas-galle",
        "Southern Breeze Villas",
        "Galle",
        "villa",
        3,
        52_000_000,
        "Three-bedroom villa with pool access, near Galle town.",
    ),
    (
        "sunset-gardens-matara",
        "Sunset Gardens",
        "Matara",
        "house",
        3,
        30_000_000,
        "New three-bedroom house with parking and a small garden.",
    ),
    (
        "tech-park-lands-malabe",
        "Tech Park Lands",
        "Malabe",
        "land",
        None,
        12_500_000,
        "Eight-perch residential plots near the technology corridor; clear title.",
    ),
    (
        "expressway-lands-homagama",
        "Expressway Lands",
        "Homagama",
        "land",
        None,
        8_800_000,
        "Ten-perch residential plots with road, water, and electricity access.",
    ),
)

TOOL_INSTRUCTIONS = """
Property facts and viewing appointments are available only through tools. Never invent inventory,
prices, availability, locations, or confirmations. Infer the caller's intent and filters from the
full conversation. Ask a natural follow-up question when required information is missing. Emit
exactly one tool block containing valid JSON when a tool is needed.

Tools:
- search_properties: optional query, location, property_type, bedrooms, max_price_lkr.
- book_appointment: required property_id, customer_name, appointment_at (ISO 8601).
Call search_properties as soon as the caller asks what is available, names a property, or supplies
any search filter. For a broad inventory request, search with empty arguments instead of asking for
filters first. Put a named property in query. Include only filters supported by the conversation
and never invent one. Write location and property_type tool arguments in English even when the
caller speaks another language. Greetings and acknowledgements are not searches. Once you choose a
tool and its required information is present, emit the tool call immediately with no introduction,
permission question, acknowledgement, or other spoken text.
After a tool result, interpret it and answer naturally in the caller's language. Preserve all
numbers and property facts exactly as returned. A booking is confirmed only when
book_appointment returns ok=true.
""".strip()


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


@dataclass(frozen=True)
class CallContext:
    call_id: str
    caller_phone: str


def parse_tool_call(text: str) -> ToolCall | None:
    match = re.search(r"<tool_call>\s*", text, re.IGNORECASE)
    if match:
        try:
            payload, _ = json.JSONDecoder().raw_decode(text[match.end() :])
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            return None
        arguments = payload.get("arguments", {})
        return (
            ToolCall(payload["name"], normalize_tool_arguments(arguments))
            if isinstance(arguments, dict)
            else None
        )

    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.I)
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict):
        if isinstance(payload.get("tool_name"), str) and isinstance(
            payload.get("parameters", {}), dict
        ):
            return ToolCall(
                payload["tool_name"], normalize_tool_arguments(payload["parameters"])
            )
        calls = payload.get("tool_calls")
        if isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], dict):
            function, arguments = calls[0].get("function"), calls[0].get("args", {})
            if isinstance(function, str) and isinstance(arguments, dict):
                return ToolCall(function, normalize_tool_arguments(arguments))

    native = re.search(
        r"<\|tool_call>\s*call:([A-Za-z_][\w]*)\s*\{(.*?)\}\s*<tool_call\|>",
        text,
        re.DOTALL,
    )
    if not native:
        return None
    arguments: dict[str, object] = {}
    for item in re.split(r",\s*(?=[A-Za-z_]\w*\s*:)", native.group(2).strip()):
        if not item:
            continue
        key, separator, value = item.partition(":")
        if not separator or not re.fullmatch(r"[A-Za-z_]\w*", key.strip()):
            return None
        value = value.strip().strip('"\'')
        if re.fullmatch(r"-?\d+", value):
            arguments[key.strip()] = int(value)
        elif value.casefold() in {"true", "false"}:
            arguments[key.strip()] = value.casefold() == "true"
        else:
            arguments[key.strip()] = normalize_tool_value(value)
    return ToolCall(native.group(1), normalize_tool_arguments(arguments))


def normalize_tool_arguments(arguments: dict) -> dict:
    return {
        key: normalize_tool_value(value)
        for key, value in arguments.items()
        if value is not None
    }


def normalize_tool_value(value):
    if not isinstance(value, str):
        return value
    cleaned = value.strip()
    quote_token = '<|"|>'
    if cleaned.startswith(quote_token) and cleaned.endswith(quote_token):
        cleaned = cleaned[len(quote_token) : -len(quote_token)]
    return cleaned.strip('"\'')


def tool_call_message(call: ToolCall) -> str:
    payload = json.dumps(
        {"name": call.name, "arguments": call.arguments}, separators=(",", ":")
    )
    return f"<tool_call>{payload}</tool_call>"


@dataclass
class CallRecord:
    call_id: str
    caller_phone: str
    status: str = "connecting"
    started_at: float = field(default_factory=time.time)
    transcript: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "caller_phone": self.caller_phone,
            "status": self.status,
            "started_at": self.started_at,
            "transcript": list(self.transcript),
        }


class NeonCallStore:
    def __init__(self, database_url: str, phone_number_id: str) -> None:
        self.database_url = database_url
        self.phone_number_id = phone_number_id
        self._mapping: tuple[str, str] | None = None

    @classmethod
    def from_env(cls) -> NeonCallStore | None:
        url, number = os.getenv("DATABASE_URL"), os.getenv("PHONE_NUMBER_ID")
        return cls(url, number) if url and number else None

    def save(self, call: dict) -> None:
        for attempt in range(2):
            try:
                with psycopg.connect(
                    self.database_url, connect_timeout=10
                ) as connection:
                    customer_id, number_id = self._mapping or _load_mapping(
                        connection, self.phone_number_id
                    )
                    self._mapping = customer_id, number_id
                    connection.execute(
                        """
                        insert into calls (id, customer_id, whatsapp_number_id, customer_phone, status, transcript, created_at)
                        values (%s, %s, %s, %s, %s, %s, %s)
                        on conflict (id) do update set
                            customer_phone=excluded.customer_phone,
                            status=excluded.status,
                            transcript=excluded.transcript
                        """,
                        (
                            str(uuid.uuid5(CALL_NAMESPACE, call["call_id"])),
                            customer_id,
                            number_id,
                            call.get("caller_phone") or None,
                            {
                                "connecting": "started",
                                "active": "active",
                                "ended": "completed",
                            }.get(call["status"], call["status"]),
                            format_transcript(call.get("transcript", [])),
                            datetime.fromtimestamp(call["started_at"], tz=UTC),
                        ),
                    )
                return
            except OperationalError:
                self._mapping = None
                if attempt:
                    raise
                time.sleep(0.5)


class CallLog:
    """Keep active state in memory and send the durable copy to Neon."""

    def __init__(self, store: NeonCallStore | None = None) -> None:
        self.store = store if store is not None else NeonCallStore.from_env()
        self.calls: dict[str, CallRecord] = {}
        self._writes: set[asyncio.Task] = set()
        self._last_write: asyncio.Task | None = None

    def start(self, call_id: str, phone: str) -> None:
        self.calls[call_id] = CallRecord(call_id, phone)
        self._persist(self.calls[call_id])

    def active(self, call_id: str, phone: str) -> None:
        call = self.calls.setdefault(call_id, CallRecord(call_id, phone))
        call.caller_phone = phone or call.caller_phone
        call.status = "active"
        self._persist(call)

    def add(self, call_id: str, speaker: str, text: str) -> None:
        if not text:
            return
        call = self.calls.setdefault(call_id, CallRecord(call_id, "", "active"))
        call.transcript.append(
            {"speaker": speaker, "text": text, "timestamp": time.time()}
        )
        self._persist(call)

    def end(self, call_id: str) -> None:
        call = self.calls.pop(call_id, None)
        if call:
            call.status = "ended"
            self._persist(call)

    async def close(self) -> None:
        if self._writes:
            await asyncio.gather(*self._writes, return_exceptions=True)

    def _persist(self, call: CallRecord) -> None:
        if self.store is None:
            return
        previous = self._last_write
        snapshot = call.to_dict()

        async def write_in_order() -> None:
            if previous:
                await asyncio.gather(previous, return_exceptions=True)
            await asyncio.to_thread(self.store.save, snapshot)

        task = asyncio.create_task(write_in_order())
        self._last_write = task
        self._writes.add(task)
        task.add_done_callback(self._write_finished)

    def _write_finished(self, task: asyncio.Task) -> None:
        self._writes.discard(task)
        if task.cancelled():
            return
        if error := task.exception():
            logger.error(
                "Neon call write failed: %s",
                error,
                exc_info=(type(error), error, error.__traceback__),
            )


class PropertyStore:
    def __init__(self, database_url: str, phone_number_id: str) -> None:
        self.phone_number_id = phone_number_id
        self.pool = ConnectionPool(
            database_url,
            min_size=1,
            max_size=4,
            open=False,
            check=ConnectionPool.check_connection,
            timeout=5,
            kwargs={"connect_timeout": 10},
        )
        self.mapping: tuple[str, str] | None = None

    def ensure_ready(self) -> None:
        self.pool.open(wait=True, timeout=15)
        with self.pool.connection() as connection:
            self.mapping = _load_mapping(connection, self.phone_number_id)
            customer_id, _ = self.mapping
            connection.execute(
                """
                create table if not exists real_estate_properties (
                    id uuid primary key, customer_id uuid not null references customers(id) on delete cascade,
                    slug text not null, name text not null, location text not null, property_type text not null,
                    bedrooms integer, price_lkr bigint not null, details text not null,
                    status text not null default 'active', created_at timestamptz not null default now(),
                    unique (customer_id, slug)
                )
                """
            )
            connection.execute(
                """
                create table if not exists property_appointments (
                    id uuid primary key, customer_id uuid not null references customers(id) on delete cascade,
                    whatsapp_number_id uuid not null references whatsapp_numbers(id) on delete cascade,
                    property_id uuid not null references real_estate_properties(id), call_id text not null,
                    customer_phone text, customer_name text not null, appointment_at timestamptz not null,
                    status text not null default 'booked', created_at timestamptz not null default now()
                )
                """
            )
            connection.execute(
                """create unique index if not exists property_appointments_booked_slot_idx
                on property_appointments (property_id, appointment_at) where status='booked'"""
            )
            for row in PROPERTIES:
                slug, name, location, kind, bedrooms, price, details = row
                connection.execute(
                    """
                    insert into real_estate_properties
                        (id, customer_id, slug, name, location, property_type, bedrooms, price_lkr, details)
                    values (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    on conflict (customer_id, slug) do update set
                        name=excluded.name, location=excluded.location,
                        property_type=excluded.property_type, bedrooms=excluded.bedrooms,
                        price_lkr=excluded.price_lkr, details=excluded.details, status='active'
                    """,
                    (
                        uuid.uuid5(PROPERTY_NAMESPACE, f"{customer_id}:{slug}"),
                        customer_id,
                        slug,
                        name,
                        location,
                        kind,
                        bedrooms,
                        price,
                        details,
                    ),
                )

    def close(self) -> None:
        self.pool.close()

    def search(self, arguments: dict) -> list[dict]:
        customer_id, _ = self._get_mapping()
        clauses, values = ["customer_id=%s", "status='active'"], [customer_id]
        for column in ("location", "property_type"):
            if value := str(arguments.get(column, "")).strip():
                clauses.append(f"{column} ilike %s")
                values.append(f"%{value}%")
        if query := str(arguments.get("query", "")).strip():
            clauses.append("(name ilike %s or location ilike %s or details ilike %s)")
            values.extend([f"%{query}%"] * 3)
        if arguments.get("bedrooms") is not None:
            clauses.append("bedrooms >= %s")
            values.append(positive_int(arguments["bedrooms"], "bedrooms"))
        if arguments.get("max_price_lkr") is not None:
            clauses.append("price_lkr <= %s")
            values.append(positive_int(arguments["max_price_lkr"], "max_price_lkr"))
        sql = f"""select id,name,location,property_type,bedrooms,price_lkr,details
            from real_estate_properties where {" and ".join(clauses)} order by price_lkr,name limit 5"""
        with self.pool.connection() as connection:
            rows = connection.execute(sql, values).fetchall()
        keys = (
            "property_id",
            "name",
            "location",
            "property_type",
            "bedrooms",
            "price_lkr",
            "details",
        )
        return [dict(zip(keys, (str(row[0]), *row[1:]))) for row in rows]

    def book(self, arguments: dict, context: CallContext) -> dict:
        customer_id, number_id = self._get_mapping()
        property_id = required(arguments, "property_id")
        customer_name = required(arguments, "customer_name")
        appointment = appointment_time(required(arguments, "appointment_at"))
        appointment_id = uuid.uuid4()
        try:
            with self.pool.connection() as connection:
                property_row = connection.execute(
                    "select name,location from real_estate_properties where id=%s and customer_id=%s and status='active'",
                    (property_id, customer_id),
                ).fetchone()
                if property_row is None:
                    raise ValueError("That property is unavailable. Search again.")
                connection.execute(
                    """insert into property_appointments
                    (id,customer_id,whatsapp_number_id,property_id,call_id,customer_phone,customer_name,appointment_at)
                    values (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        appointment_id,
                        customer_id,
                        number_id,
                        property_id,
                        context.call_id,
                        context.caller_phone or None,
                        customer_name,
                        appointment,
                    ),
                )
        except UniqueViolation as exc:
            raise ValueError(
                "That viewing time is already booked. Ask for another time."
            ) from exc
        return {
            "appointment_id": str(appointment_id),
            "property_name": property_row[0],
            "location": property_row[1],
            "customer_name": customer_name,
            "appointment_at": appointment.isoformat(),
            "status": "booked",
        }

    def _get_mapping(self) -> tuple[str, str]:
        if self.mapping is None:
            self.ensure_ready()
        assert self.mapping
        return self.mapping


class RealEstateToolService:
    def __init__(self, store: PropertyStore) -> None:
        self.store = store

    @classmethod
    def from_env(cls) -> RealEstateToolService | None:
        url, number = os.getenv("DATABASE_URL"), os.getenv("PHONE_NUMBER_ID")
        return cls(PropertyStore(url, number)) if url and number else None

    async def ensure_ready(self) -> None:
        await asyncio.to_thread(self.store.ensure_ready)

    async def close(self) -> None:
        await asyncio.to_thread(self.store.close)

    async def execute(self, call: ToolCall, context: CallContext) -> dict:
        try:
            if call.name == "search_properties":
                rows = await asyncio.to_thread(self.store.search, call.arguments)
                return {"ok": True, "properties": rows, "count": len(rows)}
            if call.name == "book_appointment":
                row = await asyncio.to_thread(self.store.book, call.arguments, context)
                return {"ok": True, "appointment": row}
            return {"ok": False, "error": f"Unknown tool: {call.name}"}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception:
            logger.exception("Property tool %s failed", call.name)
            return {
                "ok": False,
                "error": "The booking database is temporarily unavailable.",
            }


def _load_mapping(
    connection: psycopg.Connection, phone_number_id: str
) -> tuple[str, str]:
    row = connection.execute(
        "select customer_id,id from whatsapp_numbers where phone_number_id=%s and status='active' limit 1",
        (phone_number_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError("No active WhatsApp number matches PHONE_NUMBER_ID")
    return str(row[0]), str(row[1])


def format_transcript(events: list[dict]) -> str | None:
    lines = [
        f"{str(event.get('speaker', 'speaker')).strip().capitalize()}: {str(event.get('text', '')).strip()}"
        for event in events
        if str(event.get("text", "")).strip()
    ]
    return " · ".join(lines) or None


def required(arguments: dict, name: str) -> str:
    value = str(arguments.get(name, "")).strip()
    if not value:
        raise ValueError(f"Missing required argument: {name}")
    return value


def positive_int(value: object, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive number")
    return result


def appointment_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("appointment_at must be an ISO 8601 date and time") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=COLOMBO)
    if parsed <= datetime.now(COLOMBO):
        raise ValueError("The appointment time must be in the future")
    return parsed


call_log = CallLog()
