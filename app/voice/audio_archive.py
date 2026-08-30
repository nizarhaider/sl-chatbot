import asyncio
import logging
import re
import subprocess
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

import boto3
import numpy as np

logger = logging.getLogger(__name__)
S3_BUCKET = "serendibai-lk"
SAMPLE_RATE = 16_000


class CallAudioRecorder:
    """Collect a full call as stereo PCM: caller left, agent right."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started_at = clock()
        self._caller: list[tuple[int, np.ndarray]] = []
        self._agent: list[tuple[int, np.ndarray]] = []

    def add_caller_pcm(self, pcm: bytes) -> None:
        self._add(self._caller, pcm, channels=1, sample_rate=SAMPLE_RATE)

    def add_agent_pcm(self, pcm: bytes, sample_rate: int, channels: int = 2) -> None:
        self._add(self._agent, pcm, channels=channels, sample_rate=sample_rate)

    def render_pcm16_stereo(self) -> bytes:
        length = max(
            (offset + len(samples) for channel in (self._caller, self._agent) for offset, samples in channel),
            default=0,
        )
        if not length:
            return b""
        output = np.zeros((length, 2), dtype=np.int32)
        self._mix(output, self._caller, 0)
        self._mix(output, self._agent, 1)
        return np.clip(output, -32768, 32767).astype(np.int16).tobytes()

    def _add(self, destination, pcm: bytes, channels: int, sample_rate: int) -> None:
        if not pcm:
            return
        samples = np.frombuffer(pcm, dtype=np.int16)
        if channels > 1:
            samples = samples[: len(samples) // channels * channels].reshape(-1, channels)[:, 0]
        if sample_rate != SAMPLE_RATE:
            if sample_rate % SAMPLE_RATE:
                raise ValueError(f"Unsupported recording sample rate: {sample_rate}")
            samples = samples[:: sample_rate // SAMPLE_RATE]
        if samples.size:
            offset = max(0, round((self._clock() - self._started_at) * SAMPLE_RATE))
            destination.append((offset, samples.copy()))

    @staticmethod
    def _mix(output: np.ndarray, segments, channel: int) -> None:
        for offset, samples in segments:
            output[offset : offset + len(samples), channel] += samples.astype(np.int32)


class CallAudioArchive:
    """Best-effort archival of one complete call, outside the live voice path."""

    def __init__(self, client_factory: Callable[[], object] | None = None, encode=None) -> None:
        self._client_factory = client_factory or (lambda: boto3.client("s3"))
        self._encode = encode or _pcm16_stereo_to_mp3
        self._client: object | None = None
        self._tasks: set[asyncio.Task] = set()

    def archive_call(self, call_id: str, recording: CallAudioRecorder) -> None:
        task = asyncio.create_task(self._render_encode_and_upload(call_id, recording))
        self._tasks.add(task)
        task.add_done_callback(self._log_result)

    async def _render_encode_and_upload(self, call_id: str, recording: CallAudioRecorder) -> None:
        pcm = recording.render_pcm16_stereo()
        if not pcm:
            return
        mp3 = await asyncio.to_thread(self._encode, pcm)
        key = _object_key(call_id)
        if self._client is None:
            self._client = self._client_factory()
        await asyncio.to_thread(self._client.put_object, Bucket=S3_BUCKET, Key=key, Body=mp3, ContentType="audio/mpeg", ServerSideEncryption="AES256")
        logger.info("Archived full call MP3 for %s to s3://%s/%s", call_id, S3_BUCKET, key)

    def _log_result(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.exception("Failed to archive full call to S3")


def _pcm16_stereo_to_mp3(pcm: bytes) -> bytes:
    result = subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(SAMPLE_RATE),
        "-ac", "2", "-i", "pipe:0", "-codec:a", "libmp3lame", "-b:a", "64k", "-f", "mp3", "pipe:1",
    ], input=pcm, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode or not result.stdout:
        raise RuntimeError(f"Could not encode call recording as MP3: {result.stderr.decode('utf-8', errors='replace').strip()}")
    return result.stdout


def _object_key(call_id: str) -> str:
    safe_call_id = re.sub(r"[^A-Za-z0-9._-]+", "_", call_id).strip("_") or "unknown"
    now = datetime.now(UTC)
    return f"call-recordings/{now:%Y/%m/%d}/{safe_call_id}/{uuid.uuid4().hex}.mp3"
