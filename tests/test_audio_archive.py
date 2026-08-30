import asyncio

import numpy as np

from app.voice.audio_archive import CallAudioArchive, CallAudioRecorder, S3_BUCKET, _object_key, _pcm16_stereo_to_mp3


class FakeS3Client:
    def __init__(self) -> None:
        self.requests = []

    def put_object(self, **kwargs) -> None:
        self.requests.append(kwargs)


def test_archives_one_call_as_private_mp3_without_waiting_for_upload() -> None:
    async def run() -> FakeS3Client:
        client = FakeS3Client()
        recording = CallAudioRecorder()
        recording.add_caller_pcm(b"\x01\x00" * 160)
        archive = CallAudioArchive(client_factory=lambda: client, encode=lambda pcm: b"mp3-data")
        archive.archive_call("call/123", recording)
        await asyncio.gather(*archive._tasks)
        return client

    request = asyncio.run(run()).requests[0]
    assert request["Bucket"] == S3_BUCKET
    assert request["Key"].startswith("call-recordings/")
    assert "/call_123/" in request["Key"]
    assert request["ContentType"] == "audio/mpeg"
    assert request["ServerSideEncryption"] == "AES256"


def test_records_caller_and_agent_in_separate_stereo_channels() -> None:
    now = [0.0]
    recorder = CallAudioRecorder(clock=lambda: now[0])
    recorder.add_caller_pcm(np.array([100, 200], dtype=np.int16).tobytes())
    agent = np.repeat(np.array([300, 400, 500, 600, 700, 800], dtype=np.int16)[:, None], 2, axis=1)
    recorder.add_agent_pcm(agent.tobytes(), sample_rate=48_000, offset_seconds=2 / 16_000)
    recorded = np.frombuffer(recorder.render_pcm16_stereo(), dtype=np.int16).reshape(-1, 2)
    assert recorded.tolist() == [[100, 0], [200, 0], [0, 300], [0, 600]]


def test_caller_recording_uses_contiguous_media_samples_not_arrival_time() -> None:
    now = [0.0]
    recorder = CallAudioRecorder(clock=lambda: now[0])
    recorder.add_caller_pcm(np.array([100, 200], dtype=np.int16).tobytes())
    now[0] = 10.0
    recorder.add_caller_pcm(np.array([300, 400], dtype=np.int16).tobytes())
    recorded = np.frombuffer(recorder.render_pcm16_stereo(), dtype=np.int16).reshape(-1, 2)
    assert recorded[:, 0].tolist() == [100, 200, 300, 400]


def test_object_key_and_mp3_encoder() -> None:
    key = _object_key("../../call id")
    assert key.startswith("call-recordings/") and "../" not in key and "call_id" in key
    mp3 = _pcm16_stereo_to_mp3(b"\x00\x00" * 16_000 * 2)
    assert mp3.startswith(b"ID3") or mp3.startswith(b"\xff\xfb")
