import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class PineconePropertyStore:
    def __init__(self, api_key: str, index_name: str, namespace: str) -> None:
        from pinecone import Pinecone

        self._index_name = index_name
        self._namespace = namespace
        self._client = Pinecone(api_key=api_key)
        if not self._client.has_index(index_name):
            raise RuntimeError(
                f"Pinecone index {index_name!r} does not exist. Run scripts/build_pinecone_properties.py first."
            )
        self._index = self._client.Index(index_name)

    def upsert_properties(self, properties: list[dict]) -> None:
        records = [_property_record(property_data) for property_data in properties]
        if records:
            self._index.upsert_records(namespace=self._namespace, records=records)
            logger.info(
                "Upserted %s property records into Pinecone index=%s namespace=%s",
                len(records),
                self._index_name,
                self._namespace,
            )

    def search_properties(self, arguments: dict) -> dict:
        properties = self._search(arguments)
        requested_location = str(arguments.get("location", "")).strip()
        if properties or not requested_location:
            return {
                "properties": properties,
                "count": len(properties),
                "needs_clarification": False,
            }

        # An exact location miss is not enough evidence that the caller's
        # broader area is unavailable. Search the remaining constraints and
        # return candidate locations so the agent can ask a useful follow-up.
        broader_arguments = dict(arguments)
        broader_arguments.pop("location", None)
        broader_properties = self._search(broader_arguments)
        suggested_locations = list(dict.fromkeys(
            property_data["location"]
            for property_data in broader_properties
            if property_data.get("location")
        ))[:5]
        return {
            "properties": [],
            "count": 0,
            "needs_clarification": True,
            "requested_location": requested_location,
            "suggested_locations": suggested_locations,
        }

    def _search(self, arguments: dict) -> list[dict]:
        query = _search_text(arguments)
        query_payload: dict[str, Any] = {
            "top_k": 10,
            "inputs": {"text": query},
        }
        metadata_filter = _metadata_filter(arguments)
        if metadata_filter:
            query_payload["filter"] = metadata_filter

        result = self._index.search(
            namespace=self._namespace,
            query=query_payload,
            rerank={
                "model": "bge-reranker-v2-m3",
                "top_n": 5,
                "rank_fields": ["content"],
            },
        )
        result_data = _value(result, "result", {})
        hits = _value(result_data, "hits", []) or []
        properties = []
        for hit in hits:
            fields = _value(hit, "fields", {}) or {}
            properties.append(
                {
                    "property_id": _value(fields, "property_id", _value(hit, "_id", _value(hit, "id", ""))),
                    "name": _value(fields, "name", ""),
                    "location": _value(fields, "location", ""),
                    "property_type": _value(fields, "property_type", ""),
                    "bedrooms": _value(fields, "bedrooms"),
                    "price_lkr": _value(fields, "price_lkr"),
                    "price_millions": _value(fields, "price_millions"),
                    "price_label": _value(fields, "price_label", ""),
                    "details": _value(fields, "details", ""),
                }
            )
        return properties


def _property_record(property_data: dict) -> dict:
    record = {
        "_id": str(property_data["property_id"]),
        "content": _property_content(property_data),
        "property_id": str(property_data["property_id"]),
        "name": property_data["name"],
        "location": property_data["location"],
        "location_lower": property_data["location"].casefold(),
        "property_type": property_data["property_type"],
        "property_type_lower": property_data["property_type"].casefold(),
        "bedrooms": property_data["bedrooms"] if property_data["bedrooms"] is not None else 0,
        "price_lkr": property_data["price_lkr"],
        "price_millions": property_data["price_millions"],
        "price_label": property_data["price_label"],
        "details": property_data["details"],
    }
    return record


def _property_content(property_data: dict) -> str:
    bedroom_text = (
        f"{property_data['bedrooms']} bedrooms"
        if property_data["bedrooms"] is not None
        else "land without bedrooms"
    )
    return (
        f"{property_data['name']} is a {property_data['property_type']} in {property_data['location']}. "
        f"It has {bedroom_text}. The exact price is {property_data['price_label']}. "
        f"Details: {property_data['details']}"
    )


def _search_text(arguments: dict) -> str:
    parts = [str(arguments.get("query", "")).strip()]
    for field in ("location", "property_type"):
        value = str(arguments.get(field, "")).strip()
        if value:
            parts.append(value)
    bedrooms = arguments.get("bedrooms")
    if bedrooms is not None:
        parts.append(f"{bedrooms} bedrooms")
    if arguments.get("min_bedrooms") is not None or arguments.get("max_bedrooms") is not None:
        parts.append(
            f"{arguments.get('min_bedrooms', '')} to {arguments.get('max_bedrooms', '')} bedrooms"
        )
    if arguments.get("max_price_lkr") is not None:
        parts.append(f"under {arguments['max_price_lkr']} LKR")
    return " ".join(part for part in parts if part) or "available properties"


def _metadata_filter(arguments: dict) -> dict | None:
    filters: list[dict] = []
    location = str(arguments.get("location", "")).strip()
    if location:
        filters.append({"location_lower": {"$eq": location.casefold()}})

    property_type = str(arguments.get("property_type", "")).strip()
    if property_type:
        filters.append({"property_type_lower": {"$eq": property_type.casefold()}})

    bedrooms = arguments.get("bedrooms")
    if isinstance(bedrooms, list):
        filters.append({"bedrooms": {"$in": [int(value) for value in bedrooms]}})
    elif isinstance(bedrooms, int):
        filters.append({"bedrooms": {"$gte": bedrooms}})
    elif isinstance(bedrooms, str) and (match := re.fullmatch(
        r"\s*(\d+)\s*(?:-|to|or)\s*(\d+)\s*", bedrooms, flags=re.IGNORECASE
    )):
        lower, upper = sorted((int(match.group(1)), int(match.group(2))))
        filters.extend([{"bedrooms": {"$gte": lower}}, {"bedrooms": {"$lte": upper}}])

    if arguments.get("min_bedrooms") is not None:
        filters.append({"bedrooms": {"$gte": int(arguments["min_bedrooms"])}})
    if arguments.get("max_bedrooms") is not None:
        filters.append({"bedrooms": {"$lte": int(arguments["max_bedrooms"])}})
    if arguments.get("max_price_lkr") is not None:
        filters.append({"price_lkr": {"$lte": int(arguments["max_price_lkr"])}})

    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)
