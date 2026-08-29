import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
import psycopg
from psycopg.errors import UniqueViolation

from app.integrations.whatsapp.client import _normalize_phone_number, whatsapp_api
from app.voice.pinecone_store import PineconePropertyStore

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

@dataclass(frozen=True)
class CallContext:
    call_id: str
    caller_phone: str


def _property_dict(row: tuple) -> dict:
    return {
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


LLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_properties",
            "description": (
                "Search available properties after the caller has supplied enough useful "
                "requirements, such as a property type plus a location or bedroom count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The caller's confirmed property requirements.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book a viewing for an exact property returned by a search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "property_id": {"type": "string"},
                    "customer_name": {"type": "string"},
                    "appointment_at": {
                        "type": "string",
                        "description": "ISO 8601 date and time in Asia/Colombo",
                    },
                },
                "required": ["property_id", "customer_name", "appointment_at"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_whatsapp_message",
            "description": "Send text to the caller only when they explicitly request it.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
    },
]


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

    def list_active_properties(self) -> list[dict]:
        customer_id, _ = self._mapping()
        sql = f"""
            select id, name, location, property_type, bedrooms, price_lkr, details
            from real_estate_properties
            where customer_id = %s and status = 'active'
            order by price_lkr, name
        """
        with psycopg.connect(self._database_url, connect_timeout=10) as connection:
            rows = connection.execute(sql, (customer_id,)).fetchall()
        return [
            _property_dict(row)
            for row in rows
        ]

    def customer_namespace(self) -> str:
        customer_id, _ = self._mapping()
        return customer_id

    def book_appointment(self, arguments: dict, context: CallContext) -> dict:
        customer_id, whatsapp_number_id = self._mapping()
        property_reference = _required(arguments, "property_id")
        customer_name = _required(arguments, "customer_name")
        appointment_at = _appointment_time(_required(arguments, "appointment_at"))
        appointment_id = uuid.uuid4()

        try:
            with psycopg.connect(self._database_url, connect_timeout=10) as connection:
                property_row = connection.execute(
                    """
                    select id, name, location
                    from real_estate_properties
                    where customer_id = %s
                      and status = 'active'
                      and (id::text = %s or slug = %s or lower(name) = lower(%s))
                    """,
                    (customer_id, property_reference, property_reference, property_reference),
                ).fetchone()
                if property_row is None:
                    raise ValueError("That property is not available. Search for properties again.")
                existing_booking = connection.execute(
                    """
                    select 1
                    from property_appointments
                    where property_id = %s and appointment_at = %s and status = 'booked'
                    limit 1
                    """,
                    (property_row[0], appointment_at),
                ).fetchone()
                if existing_booking is not None:
                    raise ValueError("That viewing time has already been booked. Ask for another time.")
                connection.execute(
                    """
                    insert into property_appointments (
                        id, customer_id, whatsapp_number_id, property_id, call_id,
                        customer_phone, customer_name, appointment_at
                    ) values (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        appointment_id, customer_id, whatsapp_number_id, property_row[0], context.call_id,
                        context.caller_phone or None, customer_name, appointment_at,
                    ),
                )
        except UniqueViolation as exc:
            raise ValueError("That viewing time has already been booked. Ask for another time.") from exc

        return {
            "appointment_id": str(appointment_id),
            "property_id": str(property_row[0]),
            "property_name": property_row[1],
            "location": property_row[2],
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
    def __init__(self, store: NeonRealEstateStore, vector_store: PineconePropertyStore | None = None) -> None:
        self._store = store
        self._vector_store = vector_store

    @classmethod
    def from_env(cls) -> "RealEstateToolService | None":
        database_url = os.environ.get("DATABASE_URL")
        phone_number_id = os.environ.get("PHONE_NUMBER_ID")
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        if not database_url or not phone_number_id or not pinecone_api_key:
            return None
        return cls(NeonRealEstateStore(database_url, phone_number_id))

    async def ensure_ready(self) -> None:
        await asyncio.to_thread(self._store.ensure_schema)
        namespace = await asyncio.to_thread(self._store.customer_namespace)
        self._vector_store = await asyncio.to_thread(
            PineconePropertyStore,
            os.environ["PINECONE_API_KEY"],
            os.environ.get("PINECONE_INDEX_NAME", "homelands-properties"),
            namespace,
        )
        await asyncio.to_thread(
            self._vector_store.upsert_properties,
            self._store.list_active_properties(),
        )

    async def execute(self, name: str, arguments: dict, context: CallContext) -> dict:
        try:
            if name == "search_properties":
                if self._vector_store is None:
                    raise RuntimeError("The property search index is not ready.")
                search_result = await asyncio.to_thread(self._vector_store.search_properties, arguments["query"])
                if isinstance(search_result, dict):
                    return {"ok": True, **search_result}
                return {"ok": True, "properties": search_result, "count": len(search_result)}
            if name == "book_appointment":
                if not _normalize_phone_number(context.caller_phone):
                    return {"ok": False, "error": "A valid WhatsApp number is required to book a viewing."}
                appointment = await asyncio.to_thread(self._store.book_appointment, arguments, context)
                confirmation_sent = await whatsapp_api.send_text_message(
                    context.caller_phone,
                    (
                        f"Your viewing is confirmed: {appointment['property_name']} in "
                        f"{appointment['location']} on {appointment['appointment_at']}."
                    ),
                )
                logger.info("Appointment persisted for %s: appointment_id=%s", context.call_id, appointment["appointment_id"])
                return {"ok": True, "appointment": appointment, "confirmation_sent": confirmation_sent}
            if name == "send_whatsapp_message":
                sent = await whatsapp_api.send_text_message(context.caller_phone, _required(arguments, "message"))
                return {"ok": sent, "message_sent": sent}
            return {"ok": False, "error": f"Unknown tool: {name}"}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception:
            logger.exception("Real-estate tool %s failed", name)
            return {"ok": False, "error": "The booking database is temporarily unavailable."}


def _required(arguments: dict, name: str) -> str:
    value = str(arguments.get(name, "")).strip()
    if not value:
        raise ValueError(f"Missing required argument: {name}")
    return value


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
