#!/usr/bin/env python3
"""Gate external ASR clips before they enter the Sinhala fine-tuning corpus.

The script keeps raw media out of Git.  It accepts JSONL records referencing local
clips, validates provenance and reuse rights, and produces an accepted manifest
plus a rejection report.  Exact audio/text hashes prevent cross-source leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path


ALLOWED_RIGHTS = {"cc0-1.0", "cc-by-4.0", "public-domain", "owner-permission"}
DENIED_SOURCES = {"openslr-52", "slr52", "openslr/slr52"}
SINHALA = re.compile(r"[\u0D80-\u0DFF]")


def normalized_text(value: str) -> str:
    """Canonicalize text without changing its words or numeric notation."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def has_sinhala(text: str) -> bool:
    letters = [character for character in text if character.isalpha()]
    return bool(letters) and sum(bool(SINHALA.fullmatch(character)) for character in letters) / len(letters) >= 0.5


def validate(record: dict, manifest_dir: Path, audio_hashes: set[str], text_hashes: set[str]) -> tuple[dict | None, str | None]:
    required = ("id", "audio", "text", "source", "source_url", "rights_basis", "rights_evidence_url", "source_revision")
    missing = [field for field in required if not str(record.get(field, "")).strip()]
    if missing:
        return None, f"missing required metadata: {', '.join(missing)}"
    source = str(record["source"]).strip().lower()
    if source in DENIED_SOURCES:
        return None, "known common Sinhala ASR training source is excluded"
    rights = str(record["rights_basis"]).strip().lower()
    if rights not in ALLOWED_RIGHTS:
        return None, "rights basis is not approved for training"
    audio = (manifest_dir / str(record["audio"])).resolve()
    if not audio.is_file():
        return None, "referenced audio is missing"
    text = normalized_text(str(record["text"]))
    if not has_sinhala(text):
        return None, "transcript is not predominantly Sinhala"
    audio_sha256 = digest_file(audio)
    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if audio_sha256 in audio_hashes:
        return None, "duplicate audio content"
    if text_sha256 in text_hashes:
        return None, "duplicate normalized transcript"
    accepted = dict(record)
    accepted.update({
        "audio": str(audio),
        "text": text,
        "audio_sha256": audio_sha256,
        "text_sha256": text_sha256,
    })
    return accepted, None


def records(path: Path) -> Iterable[tuple[int, dict]]:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            yield number, json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--accepted", required=True, type=Path)
    parser.add_argument("--rejected", required=True, type=Path)
    args = parser.parse_args()
    audio_hashes: set[str] = set()
    text_hashes: set[str] = set()
    accepted: list[dict] = []
    rejected: list[dict] = []
    for line, record in records(args.input):
        item, reason = validate(record, args.input.parent, audio_hashes, text_hashes)
        if item is None:
            rejected.append({"line": line, "id": record.get("id"), "reason": reason})
        else:
            accepted.append(item)
            audio_hashes.add(item["audio_sha256"])
            text_hashes.add(item["text_sha256"])
    args.accepted.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in accepted), encoding="utf-8")
    args.rejected.write_text(json.dumps(rejected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"accepted={len(accepted)} rejected={len(rejected)}")


if __name__ == "__main__":
    main()
