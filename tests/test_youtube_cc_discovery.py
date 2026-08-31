from scripts.discover_youtube_cc import qualifying_entry


def entry(**updates):
    result = {
        "id": "video-id",
        "license": "Creative Commons Attribution license (reuse allowed)",
        "subtitles": {"si": [{"url": "https://example.test/subtitle"}]},
        "upload_date": "20260801",
        "webpage_url": "https://www.youtube.com/watch?v=video-id",
    }
    result.update(updates)
    return result


def test_accepts_recent_cc_video_with_human_sinhala_subtitles():
    candidate = qualifying_entry(entry(), "2026-09-01T00:00:00+00:00")
    assert candidate["id"] == "youtube-video-id"
    assert candidate["rights_basis"] == "cc-by-4.0"


def test_rejects_automatic_caption_only_video():
    candidate = qualifying_entry(entry(subtitles={}), "2026-09-01T00:00:00+00:00")
    assert candidate is None


def test_rejects_video_old_enough_to_overlap_the_model():
    candidate = qualifying_entry(entry(upload_date="20260505"), "2026-09-01T00:00:00+00:00")
    assert candidate is None


def test_rejects_standard_youtube_license():
    candidate = qualifying_entry(entry(license="Standard YouTube License"), "2026-09-01T00:00:00+00:00")
    assert candidate is None
