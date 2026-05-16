#!/usr/bin/env python3
"""Prepare OpenSLR SLR30 Sinhala TTS data for the Orpheus notebook.

The output CSV has the columns expected by the training notebook:

    source,audio,text

It accepts either the original OpenSLR transcript file or the Hugging Face
mirror's file_index.tsv. Audio paths are resolved against the extracted WAV
directory and written as paths relative to the CSV file.
"""

from __future__ import annotations

import argparse
import csv
import re
import tarfile
import urllib.request
from pathlib import Path


OPENSLR_AUDIO_URL = "https://www.openslr.org/resources/30/si_lk.tar.gz"
OPENSLR_TRANSCRIPT_URL = "https://www.openslr.org/resources/30/si_lk.lines.txt"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"exists: {destination}")
        return

    print(f"downloading: {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def extract_tarball(tarball: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / ".extracted"
    if marker.exists():
        print(f"already extracted: {output_dir}")
        return

    print(f"extracting: {tarball}")
    with tarfile.open(tarball) as archive:
        archive.extractall(output_dir)
    marker.write_text("ok\n", encoding="utf-8")


def looks_like_audio_id(value: str) -> bool:
    value = Path(value.strip()).stem
    return bool(re.search(r"\d", value)) and not bool(re.search(r"[\u0D80-\u0DFF]", value))


def infer_source(file_id: str) -> str:
    stem = Path(file_id).stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0] == "sin":
        return "_".join(parts[:2])
    for separator in ("_", "-", "/"):
        if separator in stem:
            return stem.split(separator, 1)[0]
    return "sinhala"


def parse_transcript_line(line: str) -> tuple[str, str, str] | None:
    line = line.strip()
    if not line:
        return None

    lower = line.lower()
    if lower.startswith(("fileid", "file_id", "id\t", "sentence ")):
        return None

    parenthesized = re.match(r'^\(\s*(?P<file_id>\S+)\s+"(?P<text>.*)"\s*\)$', line)
    if parenthesized:
        file_id = parenthesized.group("file_id")
        text = parenthesized.group("text").strip()
        return file_id, infer_source(file_id), text

    if "\t" in line:
        parts = [part.strip() for part in line.split("\t") if part.strip()]
        if len(parts) >= 3:
            file_id = parts[0]
            source = parts[1] or infer_source(file_id)
            text = parts[-1]
            return file_id, source, text
        if len(parts) == 2:
            first, second = parts
            if looks_like_audio_id(first):
                return first, infer_source(first), second
            return second, infer_source(second), first

    if " " in line:
        first, rest = line.split(maxsplit=1)
        if looks_like_audio_id(first):
            return first, infer_source(first), rest.strip()

        text, file_id = line.rsplit(maxsplit=1)
        if looks_like_audio_id(file_id):
            return file_id, infer_source(file_id), text.strip()

    return None


def build_audio_index(audio_root: Path) -> dict[str, Path]:
    audio_files = {}
    for path in audio_root.rglob("*.wav"):
        audio_files[path.stem] = path
        audio_files[path.name] = path
    return audio_files


def write_manifest(transcripts: Path, audio_root: Path, output_csv: Path) -> None:
    audio_index = build_audio_index(audio_root)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = 0
    skipped = 0

    for line in transcripts.read_text(encoding="utf-8").splitlines():
        parsed = parse_transcript_line(line)
        if parsed is None:
            skipped += 1
            continue

        file_id, source, text = parsed
        audio = audio_index.get(Path(file_id).stem) or audio_index.get(file_id)
        if audio is None:
            missing += 1
            continue

        rows.append(
            {
                "source": source,
                "audio": audio.relative_to(output_csv.parent).as_posix(),
                "text": text,
            }
        )

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "audio", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote: {output_csv}")
    print(f"rows: {len(rows)}")
    print(f"missing audio: {missing}")
    print(f"skipped lines: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("training_custom_tts/datasets/openslr_sinhala"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--transcripts", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir
    raw_dir = data_dir / "raw"
    extracted_dir = data_dir / "extracted"

    audio_tarball = raw_dir / "si_lk.tar.gz"
    transcript_file = args.transcripts or raw_dir / "si_lk.lines.txt"

    if args.download:
        download(OPENSLR_AUDIO_URL, audio_tarball)
        download(OPENSLR_TRANSCRIPT_URL, transcript_file)

    if args.extract:
        extract_tarball(audio_tarball, extracted_dir)

    audio_root = args.audio_root or extracted_dir
    output_csv = args.output_csv or data_dir / "metadata.csv"
    write_manifest(transcript_file, audio_root, output_csv)


if __name__ == "__main__":
    main()
