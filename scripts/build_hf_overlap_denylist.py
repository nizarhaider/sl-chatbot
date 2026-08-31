#!/usr/bin/env python3
"""Build a hash-only overlap denylist from a public Hugging Face dataset.

Only source identifiers and hashes of normalized text are retained; audio and
transcripts are never copied.  The result can be passed to later corpus-admission
jobs to exclude a model author's known training material.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.curate_asr_manifest import normalized_text


BASE_URL = "https://datasets-server.huggingface.co"
DEFAULT_DATASET = "SPEAK-ASR/youtube-sinhala-asr"


def get_json(path: str, params: dict) -> dict:
    url = f"{BASE_URL}/{path}?{urlencode(params)}"
    for attempt in range(3):
        try:
            with urlopen(url, timeout=30) as response:  # noqa: S310 - fixed HTTPS host
                return json.load(response)
        except (HTTPError, URLError) as error:
            if attempt == 2 or isinstance(error, HTTPError) and error.code < 500:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def text_hash(value: str) -> str:
    return hashlib.sha256(normalized_text(value).encode("utf-8")).hexdigest()


def records(dataset: str, fetch: Callable[[str, dict], dict] = get_json) -> Iterable[dict]:
    splits = fetch("splits", {"dataset": dataset}).get("splits", [])
    for split_info in splits:
        config = split_info["config"]
        split = split_info["split"]
        offset = 0
        total = None
        while total is None or offset < total:
            page = fetch("rows", {
                "dataset": dataset, "config": config, "split": split, "offset": offset, "length": 100,
            })
            rows = page.get("rows", [])
            total = page.get("num_rows_total", 0)
            if not rows:
                break
            for item in rows:
                row = item.get("row", item)
                video_id = str(row.get("video_id", "")).strip()
                text = str(row.get("text", ""))
                if video_id and normalized_text(text):
                    yield {
                        "source_dataset": dataset,
                        "config": config,
                        "split": split,
                        "video_id": video_id,
                        "text_sha256": text_hash(text),
                    }
            offset += len(rows)


def parquet_records(dataset: str, fetch: Callable[[str, dict], dict] = get_json) -> Iterable[dict]:
    """Read only identifiers and text columns, avoiding Dataset Viewer rate limits."""
    try:
        import duckdb
    except ImportError as error:
        raise SystemExit("Parquet mode needs DuckDB: run with `uv run --with duckdb`") from error
    files = fetch("parquet", {"dataset": dataset}).get("parquet_files", [])
    urls = [str(item["url"]) for item in files]
    if not urls:
        return
    relation = duckdb.connect().execute(
        "SELECT video_id, text, filename FROM read_parquet(?, filename = true)", [urls]
    )
    for video_id, text, filename in relation.fetchall():
        if str(video_id).strip() and normalized_text(str(text)):
            split = next((part for part in str(filename).split("/") if part in {"train", "validation", "test"}), "unknown")
            yield {
                "source_dataset": dataset,
                "config": "default",
                "split": split,
                "video_id": str(video_id),
                "text_sha256": text_hash(str(text)),
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=("rows", "parquet"), default="rows")
    args = parser.parse_args()
    unique: dict[tuple[str, str], dict] = {}
    source_records = parquet_records(args.dataset) if args.backend == "parquet" else records(args.dataset)
    for item in source_records:
        unique[(item["video_id"], item["text_sha256"])] = item
    args.output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in unique.values()), encoding="utf-8"
    )
    print(f"dataset={args.dataset} denylist_records={len(unique)}")


if __name__ == "__main__":
    main()
