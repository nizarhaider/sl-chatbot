#!/usr/bin/env python3
"""Discover, but never download, reusable Sinhala YouTube ASR candidates.

Each result must be Creative Commons Attribution, contain human-provided Sinhala
subtitles, and postdate the deployed ASR checkpoint.  Downloading happens only
after a candidate passes the separate provenance/corpus gate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.curate_asr_manifest import MODEL_CREATION_DATE


def published_date(entry: dict) -> str | None:
    value = str(entry.get("upload_date", ""))
    if len(value) != 8 or not value.isdigit():
        return None
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def qualifying_entry(entry: dict, discovered_at: str) -> dict | None:
    """Return a stable candidate record only when the license evidence is strong."""
    license_text = str(entry.get("license", "")).casefold()
    subtitles = entry.get("subtitles") or {}
    source_published_at = published_date(entry)
    if "creative commons attribution" not in license_text:
        return None
    if not any(language.casefold().replace("_", "-").startswith("si") for language in subtitles):
        return None
    if source_published_at is None or source_published_at <= MODEL_CREATION_DATE.isoformat():
        return None
    video_id = str(entry.get("id", ""))
    url = str(entry.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}")
    if not video_id or not url.startswith("https://www.youtube.com/watch"):
        return None
    return {
        "id": f"youtube-{video_id}",
        "source": "youtube-cc-by",
        "source_url": url,
        "rights_basis": "cc-by-4.0",
        "rights_evidence_url": url,
        "source_published_at": source_published_at,
        "source_revision": discovered_at,
        "video_id": video_id,
        "uploader_id": entry.get("uploader_id"),
        "duration_seconds": entry.get("duration"),
        "subtitle_languages": sorted(subtitles),
    }


def search(query: str, limit: int) -> list[dict]:
    command = [
        "yt-dlp", f"ytsearch{limit}:{query}", "--dump-single-json", "--skip-download",
        "--no-warnings", "--socket-timeout", "30", "--retries", "1",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout).get("entries", [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True, help="UTF-8 file with one search query per line")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    discovered_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    candidates: dict[str, dict] = {}
    searched = 0
    for query in args.queries.read_text(encoding="utf-8").splitlines():
        if not query.strip():
            continue
        searched += 1
        for entry in search(query, args.limit):
            candidate = qualifying_entry(entry, discovered_at)
            if candidate:
                candidates[candidate["id"]] = candidate
    args.output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in candidates.values()), encoding="utf-8"
    )
    print(f"queries={searched} candidates={len(candidates)}")


if __name__ == "__main__":
    main()
