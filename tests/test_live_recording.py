import asyncio
import subprocess

import boto3
from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.voice.audio_archive import S3_BUCKET
from app.voice.gemini_live import GEMINI_LIVE_MODEL


async def _check_recording():
    recording = boto3.client("s3").get_object(
        Bucket=S3_BUCKET, Key="test-fixtures/gemini-live-input.wav"
    )["Body"].read()
    audio = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1"],
        input=recording, capture_output=True, check=True,
    ).stdout
    assert audio
    client = genai.Client()
    async with client.aio.live.connect(
        model=GEMINI_LIVE_MODEL,
        config={"response_modalities": ["AUDIO"], "system_instruction": "Listen to this recording and reply with one short spoken sentence."},
    ) as session:
        async def exchange():
            for offset in range(0, len(audio), 3200):
                await session.send_realtime_input(audio=types.Blob(data=audio[offset:offset + 3200], mime_type="audio/pcm;rate=16000"))
                await asyncio.sleep(0.1)
            await session.send_realtime_input(audio_stream_end=True)
            async for response in session.receive():
                content = response.server_content
                if content and content.model_turn:
                    for part in content.model_turn.parts or []:
                        if part.inline_data and part.inline_data.data:
                            return
            raise AssertionError("Gemini Live did not return audio")

        await asyncio.wait_for(exchange(), timeout=90)


def test_gemini_live_returns_audio_from_s3_recording():
    load_dotenv(".env")
    asyncio.run(_check_recording())
