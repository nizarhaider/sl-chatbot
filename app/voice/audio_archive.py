import asyncio
import io
import logging
import re
import uuid
import wave
from collections.abc import Callable
from datetime import UTC, datetime

import boto3

logger = logging.getLogger(__name__)

S3_BUCKET = "serendibai-lk"
SAMPLE_RATE = 16_000


class AudioClipArchive:
    """Best-effort archival of completed caller turns, outside the voice path."""

    def __init__(self, client_factory: Callable[[], object] | None = None) -> None:
        self._client_factory = client_factory or (lambda: boto3.client("s3"))
        self._client: object | None = None
        self._tasks: set[asyncio.Task] = set()

    def archive_turn(self, call_id: str, pcm: bytes) -> None:
        if not pcm:
            return
        task = asyncio.create_task(self._upload(call_id, _pcm16_to_wav(pcm)))
        self._tasks.add(task)
        task.add_done_callback(self._log_result)

    async def _upload(self, call_id: str, wav: bytes) -> None:
        key = _object_key(call_id)
        if self._client is None:
            self._client = self._client_factory()
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=S3_BUCKET,
            Key=key,
            Body=wav,
            ContentType="audio/wav",
            ServerSideEncryption="AES256",
        )
        logger.info("Archived caller clip for %s to s3://%s/%s", call_id, S3_BUCKET, key)

    def _log_result(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.exception("Failed to archive caller clip to S3")


def _pcm16_to_wav(pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return output.getvalue()


def _object_key(call_id: str) -> str:
    safe_call_id = re.sub(r"[^A-Za-z0-9._-]+", "_", call_id).strip("_") or "unknown"
    now = datetime.now(UTC)
    return f"voice-clips/{now:%Y/%m/%d}/{safe_call_id}/{uuid.uuid4().hex}.wav"
