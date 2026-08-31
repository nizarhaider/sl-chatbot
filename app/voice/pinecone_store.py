import logging

from app.voice.location_lexicon import location_search_terms

logger = logging.getLogger(__name__)


class PineconePropertyStore:
    def __init__(self, api_key: str, index_name: str, namespace: str) -> None:
        from pinecone import Pinecone

        self._namespace = namespace
        client = Pinecone(api_key=api_key)
        if not client.has_index(index_name):
            raise RuntimeError(f"Pinecone index {index_name!r} does not exist.")
        self._index = client.Index(index_name)

    def upsert_properties(self, properties: list[dict]) -> None:
        records = [_property_record(property_data) for property_data in properties]
        if records:
            self._index.upsert_records(namespace=self._namespace, records=records)
            logger.info("Upserted %s property records into Pinecone", len(records))

    def search_properties(self, query: str) -> dict:
        result = self._index.search(
            namespace=self._namespace,
            query={"top_k": 5, "inputs": {"text": query}},
        )
        hits = _value(_value(result, "result", {}), "hits", []) or []
        properties = []
        for hit in hits:
            fields = _value(hit, "fields", {}) or {}
            properties.append({
                "property_id": _value(fields, "property_id", _value(hit, "_id", _value(hit, "id", ""))),
                "name": _value(fields, "name", ""),
                "location": _value(fields, "location", ""),
                "property_type": _value(fields, "property_type", ""),
                "bedrooms": _value(fields, "bedrooms"),
                "price_lkr": _value(fields, "price_lkr"),
                "price_millions": _value(fields, "price_millions"),
                "price_label": _value(fields, "price_label", ""),
                "details": _value(fields, "details", ""),
            })
        return {"properties": properties, "count": len(properties)}


def _property_record(property_data: dict) -> dict:
    bedrooms = property_data["bedrooms"]
    bedroom_text = f"{bedrooms} bedrooms" if bedrooms is not None else "land without bedrooms"
    location_terms = location_search_terms(property_data["location"])
    return {
        "_id": str(property_data["property_id"]),
        "content": (
            f"{property_data['name']} is a {property_data['property_type']} in {location_terms}. "
            f"It has {bedroom_text}. The exact price is {property_data['price_label']}. "
            f"Details: {property_data['details']}"
        ),
        "property_id": str(property_data["property_id"]),
        "name": property_data["name"],
        "location": property_data["location"],
        "property_type": property_data["property_type"],
        "bedrooms": bedrooms if bedrooms is not None else 0,
        "price_lkr": property_data["price_lkr"],
        "price_millions": property_data["price_millions"],
        "price_label": property_data["price_label"],
        "details": property_data["details"],
    }


def _value(value, name: str, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)
