import asyncio
import logging
import re
import subprocess
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

import boto3

logger = logging.getLogger(__name__)

S3_BUCKET = "serendibai-lk"
SAMPLE_RATE = 16_000


class AudioClipArchive:
    """Best-effort archival of completed caller turns, outside the voice path."""

    def __init__(
        self,
        client_factory: Callable[[], object] | None = None,
        encode: Callable[[bytes], bytes] | None = None,
    ) -> None:
        self._client_factory = client_factory or (lambda: boto3.client("s3"))
        self._encode = encode or _pcm16_to_mp3
        self._client: object | None = None
        self._tasks: set[asyncio.Task] = set()

    def archive_turn(self, call_id: str, pcm: bytes) -> None:
        if not pcm:
            return
        task = asyncio.create_task(self._encode_and_upload(call_id, pcm))
        self._tasks.add(task)
        task.add_done_callback(self._log_result)

    async def _encode_and_upload(self, call_id: str, pcm: bytes) -> None:
        mp3 = await asyncio.to_thread(self._encode, pcm)
        key = _object_key(call_id)
        if self._client is None:
            self._client = self._client_factory()
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=S3_BUCKET,
            Key=key,
            Body=mp3,
            ContentType="audio/mpeg",
            ServerSideEncryption="AES256",
        )
        logger.info("Archived caller MP3 for %s to s3://%s/%s", call_id, S3_BUCKET, key)

    def _log_result(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.exception("Failed to archive caller clip to S3")


def _pcm16_to_mp3(pcm: bytes) -> bytes:
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            "-f",
            "mp3",
            "pipe:1",
        ],
        input=pcm,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode or not result.stdout:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Could not encode caller clip as MP3: {message}")
    return result.stdout


def _object_key(call_id: str) -> str:
    safe_call_id = re.sub(r"[^A-Za-z0-9._-]+", "_", call_id).strip("_") or "unknown"
    now = datetime.now(UTC)
    return f"voice-clips/{now:%Y/%m/%d}/{safe_call_id}/{uuid.uuid4().hex}.mp3"
