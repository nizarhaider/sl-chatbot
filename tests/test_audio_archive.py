import asyncio
from app.voice.audio_archive import AudioClipArchive, S3_BUCKET, _object_key, _pcm16_to_mp3


class FakeS3Client:
    def __init__(self) -> None:
        self.requests = []

    def put_object(self, **kwargs) -> None:
        self.requests.append(kwargs)


def test_archives_pcm_as_private_mp3_without_waiting_for_upload() -> None:
    async def run() -> FakeS3Client:
        client = FakeS3Client()
        archive = AudioClipArchive(client_factory=lambda: client, encode=lambda pcm: b"mp3-data")
        archive.archive_turn("call/123", b"\x01\x00" * 160)
        await asyncio.gather(*archive._tasks)
        return client

    client = asyncio.run(run())
    request = client.requests[0]
    assert request["Bucket"] == S3_BUCKET
    assert request["Key"].startswith("voice-clips/")
    assert "/call_123/" in request["Key"]
    assert request["Key"].endswith(".mp3")
    assert request["ContentType"] == "audio/mpeg"
    assert request["ServerSideEncryption"] == "AES256"
    assert request["Body"] == b"mp3-data"


def test_object_key_excludes_unsafe_call_id_characters() -> None:
    key = _object_key("../../call id")
    assert key.startswith("voice-clips/")
    assert "../" not in key
    assert "call_id" in key


def test_encodes_pcm_as_playable_mp3() -> None:
    mp3 = _pcm16_to_mp3(b"\x00\x00" * 16_000)
    assert mp3.startswith(b"ID3") or mp3.startswith(b"\xff\xfb")
