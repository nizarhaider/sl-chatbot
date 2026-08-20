import asyncio
import ast
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import psycopg
from psycopg.errors import UniqueViolation

logger = logging.getLogger(__name__)
COLOMBO_TZ = ZoneInfo("Asia/Colombo")
PROPERTY_NAMESPACE = uuid.UUID("a1c78a4d-7d09-4d29-b24b-7c427ab7912f")

DEFAULT_PROPERTIES = (
    ("horizon-residencies-malabe", "Horizon Residencies", "Malabe", "apartment", 2, 28_000_000,
     "Near schools and supermarkets."),
    ("lakeview-villas-piliyandala", "Lakeview Villas", "Piliyandala", "villa", 3, 48_000_000,
     "Garden, parking, and lake access."),
    ("green-acres-kurunegala", "Green Acres", "Kurunegala", "land", None, 9_500_000,
     "Ten-perch residential land with clear title; bank loans supported."),
    ("ocean-breeze-apartments-dehiwala", "Ocean Breeze Apartments", "Dehiwala", "apartment", 2, 32_000_000,
     "One and two-bedroom units with sea views; ready soon."),
)

TOOL_INSTRUCTIONS = """
Property facts and viewing appointments are available only through these tools. Never invent inventory,
prices, availability, or booking confirmations. To use a tool, output only one block in this exact form:
<tool_call>{"name":"search_properties","arguments":{"location":"Malabe"}}</tool_call>

Tools:
- search_properties: optional arguments query, location, property_type, bedrooms, min_bedrooms, max_bedrooms,
  max_price_lkr. For a request such as two or three bedrooms, use min_bedrooms=2 and max_bedrooms=3.
- book_appointment: required arguments property_id, customer_name, appointment_at. appointment_at must be
  an ISO 8601 date and time; ask the caller for any missing detail before calling it.

After a tool result, either call another tool or answer the caller naturally in their language. A booking is
confirmed only when book_appointment returns ok=true. Search results include an exact price_label and
price_millions value: copy those values exactly and never calculate or invent a price. Mention only property
names, locations, prices, bedrooms, and details returned by the tool. If a search returns no results, say so
and ask whether the caller would like to broaden the location, bedroom range, or budget. Never claim that a
search was completed unless a search_properties tool call returned successfully.
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
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    raw_payload = match.group(1).strip()
    payload = None
    candidates = (
        raw_payload,
        re.sub(r",\s*([}\]])", r"\1", raw_payload),
    )
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if payload is None:
        try:
            payload = ast.literal_eval(raw_payload)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        return None
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        return None
    return ToolCall(name=payload["name"], arguments=arguments)


def tool_call_message(call: ToolCall) -> str:
    return f"<tool_call>{json.dumps({'name': call.name, 'arguments': call.arguments}, separators=(',', ':'))}</tool_call>"


class NeonRealEstateStore:
    def __init__(self, database_url: str, phone_number_id: str) -> None:
        self._database_url = database_url
        self._phone_number_id = phone_number_id
        self._customer_id: str | None = None
        self._whatsapp_number_id: str | None = None

    def ensure_schema(self) -> None:
        with psycopg.connect(self._database_url, connect_timeout=10) as connection:
            customer_id, whatsapp_number_id = self._load_mapping(connection)
            connection.execute(
                """
                create table if not exists real_estate_properties (
                    id uuid primary key,
                    customer_id uuid not null references customers(id) on delete cascade,
                    slug text not null,
                    name text not null,
                    location text not null,
                    property_type text not null,
                    bedrooms integer,
                    price_lkr bigint not null,
                    details text not null,
                    status text not null default 'active',
                    created_at timestamptz not null default now(),
                    unique (customer_id, slug)
                )
                """
            )
            connection.execute(
                """
                create table if not exists property_appointments (
                    id uuid primary key,
                    customer_id uuid not null references customers(id) on delete cascade,
                    whatsapp_number_id uuid not null references whatsapp_numbers(id) on delete cascade,
                    property_id uuid not null references real_estate_properties(id),
                    call_id text not null,
                    customer_phone text,
                    customer_name text not null,
                    appointment_at timestamptz not null,
                    status text not null default 'booked',
                    created_at timestamptz not null default now()
                )
                """
            )
            connection.execute(
                """
                create unique index if not exists property_appointments_booked_slot_idx
                on property_appointments (property_id, appointment_at)
                where status = 'booked'
                """
            )
            for slug, name, location, property_type, bedrooms, price_lkr, details in DEFAULT_PROPERTIES:
                property_id = uuid.uuid5(PROPERTY_NAMESPACE, f"{customer_id}:{slug}")
                connection.execute(
                    """
                    insert into real_estate_properties
                        (id, customer_id, slug, name, location, property_type, bedrooms, price_lkr, details)
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    on conflict (customer_id, slug) do nothing
                    """,
                    (property_id, customer_id, slug, name, location, property_type, bedrooms, price_lkr, details),
                )
            self._customer_id = customer_id
            self._whatsapp_number_id = whatsapp_number_id

    def search_properties(self, arguments: dict) -> list[dict]:
        customer_id, _ = self._mapping()
        clauses = ["customer_id = %s", "status = 'active'"]
        values: list[object] = [customer_id]
        for field in ("location", "property_type"):
            value = str(arguments.get(field, "")).strip()
            if value:
                clauses.append(f"{field} ilike %s")
                values.append(f"%{value}%")
        query = str(arguments.get("query", "")).strip()
        if query:
            clauses.append("(name ilike %s or location ilike %s or details ilike %s)")
            values.extend([f"%{query}%"] * 3)
        if arguments.get("bedrooms") is not None:
            bedrooms = arguments["bedrooms"]
            if isinstance(bedrooms, list):
                bedroom_values = [_positive_int(value, "bedrooms") for value in bedrooms]
                if not bedroom_values:
                    raise ValueError("bedrooms must contain at least one positive number")
                clauses.append("bedrooms = any(%s)")
                values.append(bedroom_values)
            elif isinstance(bedrooms, str) and (match := re.fullmatch(
                r"\s*(\d+)\s*(?:-|to|or)\s*(\d+)\s*", bedrooms, flags=re.IGNORECASE
            )):
                lower = _positive_int(match.group(1), "bedrooms")
                upper = _positive_int(match.group(2), "bedrooms")
                if lower > upper:
                    lower, upper = upper, lower
                clauses.extend(["bedrooms >= %s", "bedrooms <= %s"])
                values.extend([lower, upper])
            else:
                clauses.append("bedrooms >= %s")
                values.append(_positive_int(bedrooms, "bedrooms"))
        if arguments.get("min_bedrooms") is not None:
            clauses.append("bedrooms >= %s")
            values.append(_positive_int(arguments["min_bedrooms"], "min_bedrooms"))
        if arguments.get("max_bedrooms") is not None:
            clauses.append("bedrooms <= %s")
            values.append(_positive_int(arguments["max_bedrooms"], "max_bedrooms"))
        if arguments.get("max_price_lkr") is not None:
            clauses.append("price_lkr <= %s")
            values.append(_positive_int(arguments["max_price_lkr"], "max_price_lkr"))

        sql = f"""
            select id, name, location, property_type, bedrooms, price_lkr, details
            from real_estate_properties
            where {' and '.join(clauses)}
            order by price_lkr, name
            limit 5
        """
        with psycopg.connect(self._database_url, connect_timeout=10) as connection:
            rows = connection.execute(sql, values).fetchall()
        return [
            {
                "property_id": str(row[0]),
                "name": row[1],
                "location": row[2],
                "property_type": row[3],
                "bedrooms": row[4],
                "price_lkr": row[5],
                "price_millions": round(row[5] / 1_000_000, 2),
                "price_label": f"LKR {row[5] / 1_000_000:g} million",
                "details": row[6],
            }
            for row in rows
        ]

    def book_appointment(self, arguments: dict, context: CallContext) -> dict:
        customer_id, whatsapp_number_id = self._mapping()
        property_id = _required(arguments, "property_id")
        customer_name = _required(arguments, "customer_name")
        appointment_at = _appointment_time(_required(arguments, "appointment_at"))
        appointment_id = uuid.uuid4()

        try:
            with psycopg.connect(self._database_url, connect_timeout=10) as connection:
                property_row = connection.execute(
                    """
                    select name, location
                    from real_estate_properties
                    where id = %s and customer_id = %s and status = 'active'
                    """,
                    (property_id, customer_id),
                ).fetchone()
                if property_row is None:
                    raise ValueError("That property is not available. Search for properties again.")
                connection.execute(
                    """
                    insert into property_appointments (
                        id, customer_id, whatsapp_number_id, property_id, call_id,
                        customer_phone, customer_name, appointment_at
                    ) values (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        appointment_id, customer_id, whatsapp_number_id, property_id, context.call_id,
                        context.caller_phone or None, customer_name, appointment_at,
                    ),
                )
        except UniqueViolation as exc:
            raise ValueError("That viewing time has already been booked. Ask for another time.") from exc

        return {
            "appointment_id": str(appointment_id),
            "property_name": property_row[0],
            "location": property_row[1],
            "customer_name": customer_name,
            "appointment_at": appointment_at.isoformat(),
            "status": "booked",
        }

    def _mapping(self) -> tuple[str, str]:
        if not self._customer_id or not self._whatsapp_number_id:
            self.ensure_schema()
        assert self._customer_id and self._whatsapp_number_id
        return self._customer_id, self._whatsapp_number_id

    def _load_mapping(self, connection: psycopg.Connection) -> tuple[str, str]:
        row = connection.execute(
            """
            select customer_id, id from whatsapp_numbers
            where phone_number_id = %s and status = 'active'
            limit 1
            """,
            (self._phone_number_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("No active dashboard WhatsApp number matches PHONE_NUMBER_ID")
        return str(row[0]), str(row[1])


class RealEstateToolService:
    def __init__(self, store: NeonRealEstateStore) -> None:
        self._store = store

    @classmethod
    def from_env(cls) -> "RealEstateToolService | None":
        database_url = os.environ.get("DATABASE_URL")
        phone_number_id = os.environ.get("PHONE_NUMBER_ID")
        if not database_url or not phone_number_id:
            return None
        return cls(NeonRealEstateStore(database_url, phone_number_id))

    async def ensure_ready(self) -> None:
        await asyncio.to_thread(self._store.ensure_schema)

    async def execute(self, call: ToolCall, context: CallContext) -> dict:
        try:
            if call.name == "search_properties":
                properties = await asyncio.to_thread(self._store.search_properties, call.arguments)
                return {"ok": True, "properties": properties, "count": len(properties)}
            if call.name == "book_appointment":
                appointment = await asyncio.to_thread(self._store.book_appointment, call.arguments, context)
                return {"ok": True, "appointment": appointment}
            return {"ok": False, "error": f"Unknown tool: {call.name}"}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception:
            logger.exception("Real-estate tool %s failed", call.name)
            return {"ok": False, "error": "The booking database is temporarily unavailable."}


def _required(arguments: dict, name: str) -> str:
    value = str(arguments.get(name, "")).strip()
    if not value:
        raise ValueError(f"Missing required argument: {name}")
    return value


def _positive_int(value: object, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive number")
    return result


def _appointment_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("appointment_at must be an ISO 8601 date and time") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=COLOMBO_TZ)
    if parsed <= datetime.now(COLOMBO_TZ):
        raise ValueError("The appointment time must be in the future")
    return parsed
