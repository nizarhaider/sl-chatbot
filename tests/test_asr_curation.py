import json

from scripts.curate_asr_manifest import validate


def record(audio: str) -> dict:
    return {
        "id": "clip-1",
        "audio": audio,
        "text": "මම කොළඹ සිට ගාල්ලට යනවා",
        "source": "creator-channel",
        "source_url": "https://example.test/video",
        "rights_basis": "cc-by-4.0",
        "rights_evidence_url": "https://example.test/license",
        "source_revision": "2026-09-01",
        "source_published_at": "2026-09-01",
    }


def test_accepts_traceable_sinhala_clip(tmp_path):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"audio")
    accepted, reason = validate(record(audio.name), tmp_path, set(), set())
    assert reason is None
    assert accepted["text"] == "මම කොළඹ සිට ගාල්ලට යනවා"
    assert len(accepted["audio_sha256"]) == 64


def test_rejects_common_existing_sinhala_asr_source(tmp_path):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"audio")
    item = record(audio.name)
    item["source"] = "openslr-52"
    accepted, reason = validate(item, tmp_path, set(), set())
    assert accepted is None
    assert reason == "known common Sinhala ASR training source is excluded"


def test_rejects_duplicate_normalized_transcript(tmp_path):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"audio")
    first, _ = validate(record(audio.name), tmp_path, set(), set())
    other = tmp_path / "other.mp3"
    other.write_bytes(b"other audio")
    duplicate = record(other.name)
    accepted, reason = validate(duplicate, tmp_path, set(), {first["text_sha256"]})
    assert accepted is None
    assert reason == "duplicate normalized transcript"


def test_rejects_public_clip_that_predates_the_deployed_model(tmp_path):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"audio")
    item = record(audio.name)
    item["source_published_at"] = "2024-01-01"
    accepted, reason = validate(item, tmp_path, set(), set())
    assert accepted is None
    assert reason == "no evidence that this clip postdates the deployed model"


def test_accepts_previously_unpublished_owner_permitted_clip(tmp_path):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"audio")
    item = record(audio.name)
    item["rights_basis"] = "owner-permission"
    item["source_published_at"] = "2024-01-01"
    item["previously_unpublished"] = True
    accepted, reason = validate(item, tmp_path, set(), set())
    assert reason is None
    assert accepted["rights_basis"] == "owner-permission"
