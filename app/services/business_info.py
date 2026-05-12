from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


@dataclass
class CachedBusinessInfo:
    text: str
    expires_at: datetime


class BusinessInfoProvider:
    def __init__(self):
        self.doc_id = os.environ.get("GOOGLE_BUSINESS_INFO_DOC_ID")
        self.local_path = os.environ.get("BUSINESS_INFO_PATH", "data/business_info.md")
        self.cache_ttl_seconds = int(os.environ.get("GOOGLE_BUSINESS_INFO_CACHE_SECONDS", "300"))
        self._cache: CachedBusinessInfo | None = None

    def get_text(self, fallback: str) -> str:
        local_text = self._read_local_text()
        if local_text:
            return local_text

        if not self.doc_id:
            return fallback

        now = datetime.now(timezone.utc)
        if self._cache and self._cache.expires_at > now:
            return self._cache.text

        try:
            text = self._fetch_google_doc_text()
        except Exception as exc:
            logger.error("Failed to fetch Google business info doc: %s", exc)
            return fallback

        if not text.strip():
            return fallback

        self._cache = CachedBusinessInfo(
            text=text.strip(),
            expires_at=now + timedelta(seconds=self.cache_ttl_seconds),
        )
        return self._cache.text

    def _read_local_text(self) -> str:
        if not self.local_path:
            return ""

        try:
            with open(self.local_path, encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            return ""
        except OSError as exc:
            logger.error("Failed to read local business info file %s: %s", self.local_path, exc)
            return ""

    def _fetch_google_doc_text(self) -> str:
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise RuntimeError("google-api-python-client and google-auth are required") from exc

        credentials = _service_account_credentials(
            scopes=["https://www.googleapis.com/auth/documents.readonly"]
        )
        service = build("docs", "v1", credentials=credentials, cache_discovery=False)
        document = service.documents().get(documentId=self.doc_id).execute()
        return _extract_document_text(document)


def _service_account_credentials(scopes: list[str]):
    from google.oauth2 import service_account

    json_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
    json_blob = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")

    if json_path:
        return service_account.Credentials.from_service_account_file(json_path, scopes=scopes)
    if json_blob:
        return service_account.Credentials.from_service_account_info(json.loads(json_blob), scopes=scopes)
    raise RuntimeError("Set GOOGLE_SERVICE_ACCOUNT_FILE or GOOGLE_SERVICE_ACCOUNT_JSON")


def _extract_document_text(document: dict) -> str:
    chunks: list[str] = []
    for element in document.get("body", {}).get("content", []):
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        for paragraph_element in paragraph.get("elements", []):
            text_run = paragraph_element.get("textRun")
            if text_run:
                chunks.append(text_run.get("content", ""))
    return "".join(chunks)
