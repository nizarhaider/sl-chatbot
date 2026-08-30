import asyncio
import io
import wave

from app.voice.audio_archive import AudioClipArchive, SAMPLE_RATE, S3_BUCKET, _object_key


class FakeS3Client:
    def __init__(self) -> None:
        self.requests = []

    def put_object(self, **kwargs) -> None:
        self.requests.append(kwargs)


def test_archives_pcm_as_private_wav_without_waiting_for_upload() -> None:
    async def run() -> FakeS3Client:
        client = FakeS3Client()
        archive = AudioClipArchive(client_factory=lambda: client)
        archive.archive_turn("call/123", b"\x01\x00" * 160)
        await asyncio.gather(*archive._tasks)
        return client

    client = asyncio.run(run())
    request = client.requests[0]
    assert request["Bucket"] == S3_BUCKET
    assert request["Key"].startswith("voice-clips/")
    assert "/call_123/" in request["Key"]
    assert request["ContentType"] == "audio/wav"
    assert request["ServerSideEncryption"] == "AES256"
    with wave.open(io.BytesIO(request["Body"]), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 160


def test_object_key_excludes_unsafe_call_id_characters() -> None:
    key = _object_key("../../call id")
    assert key.startswith("voice-clips/")
    assert "../" not in key
    assert "call_id" in key
