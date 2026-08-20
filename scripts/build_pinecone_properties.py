import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from pinecone import Pinecone

from app.voice.tools import NeonRealEstateStore
from app.voice.pinecone_store import PineconePropertyStore


INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "homelands-properties")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("PINECONE_API_KEY")
    database_url = os.environ.get("DATABASE_URL")
    phone_number_id = os.environ.get("PHONE_NUMBER_ID")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")
    if not database_url or not phone_number_id:
        raise RuntimeError("DATABASE_URL and PHONE_NUMBER_ID are required to migrate properties")

    pinecone = Pinecone(api_key=api_key)
    if not pinecone.has_index(INDEX_NAME):
        print(f"Creating Pinecone index {INDEX_NAME!r} with multilingual integrated embeddings")
        pinecone.create_index_for_model(
            name=INDEX_NAME,
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
            embed={
                "model": "multilingual-e5-large",
                "field_map": {"text": "content"},
            },
        )

    for _ in range(60):
        description = pinecone.describe_index(INDEX_NAME)
        status = getattr(description, "status", None) or description.get("status", {})
        ready = getattr(status, "ready", None)
        if ready is None and isinstance(status, dict):
            ready = status.get("ready")
        if ready:
            break
        time.sleep(2)
    else:
        raise RuntimeError(f"Pinecone index {INDEX_NAME!r} did not become ready")

    source = NeonRealEstateStore(database_url, phone_number_id)
    source.ensure_schema()
    properties = source.list_active_properties()
    vector_store = PineconePropertyStore(api_key, INDEX_NAME, source.customer_namespace())
    vector_store.upsert_properties(properties)
    print(f"Synced {len(properties)} active properties to Pinecone index {INDEX_NAME!r}")


if __name__ == "__main__":
    main()
