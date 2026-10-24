import asyncio
import subprocess

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from pipecat.frames.frames import (
    EndFrame,
    InputAudioRawFrame,
    LLMRunFrame,
    OutputAudioRawFrame,
)
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.workers.runner import WorkerRunner

from app.agent import build_pipeline
from app.store import S3_BUCKET, CallState
from app.tools import CallContext


class TestTransport:
    __test__ = False

    def __init__(self):
        self.source = IdentityFilter()
        self.sink = IdentityFilter()

    def input(self):
        return self.source

    def output(self):
        return self.sink


async def check_recording():
    def download():
        client = boto3.client("s3", config=Config(connect_timeout=10, read_timeout=20))
        with client.get_object(
            Bucket=S3_BUCKET, Key="test-fixtures/gemini-live-input.wav"
        )["Body"] as body:
            return body.read()

    recording = await asyncio.to_thread(download)
    audio = await asyncio.to_thread(
        subprocess.run,
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "pipe:1",
        ],
        input=recording,
        capture_output=True,
        check=True,
    )
    audio = audio.stdout
    state, context = CallState(persist=False), CallContext("fixture", "")
    state.start_call(context)
    archived = []
    state.archive = lambda call_id, pcm: archived.append(pcm)
    worker, recorder = build_pipeline(
        TestTransport(),
        {
            "name": "Samantha",
            "company_name": "SerendibAI",
            "voice": "Leda",
            "instructions": "Listen to the caller and reply with one short spoken sentence.",
            "languages": ["English"],
        },
        context,
        state,
        greet=False,
    )
    heard, started = asyncio.Event(), asyncio.Event()
    returned = bytearray()
    worker.set_reached_downstream_filter((OutputAudioRawFrame,))

    @worker.event_handler("on_frame_reached_downstream")
    async def output(worker, frame):
        returned.extend(frame.audio)
        if len(returned) >= 4800:
            heard.set()

    @worker.event_handler("on_pipeline_started")
    async def ready(worker, frame):
        await recorder.start_recording()
        await worker.queue_frames([LLMRunFrame()])
        started.set()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    task = asyncio.create_task(runner.run())
    try:
        await asyncio.wait_for(started.wait(), 30)
        await asyncio.sleep(3)
        for offset in range(0, len(audio), 3200):
            await worker.queue_frames(
                [InputAudioRawFrame(audio[offset : offset + 3200], 16000, 1)]
            )
            await asyncio.sleep(0.1)
        await worker.queue_frames([InputAudioRawFrame(bytes(32000), 16000, 1)])
        await asyncio.wait_for(heard.wait(), 60)
        await worker.queue_frames([EndFrame()])
        await asyncio.wait_for(task, 20)
    finally:
        await runner.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await state.close()
    assert returned
    assert archived and len(archived[0]) % 4 == 0
    assert any(
        e["kind"] == "gemini_live.usage" for e in state.calls[context.call_id]["events"]
    )
    assert state.calls[context.call_id]["transcript"]


def test_pipecat_returns_audio_from_s3_recording():
    load_dotenv(".env")
    asyncio.run(check_recording())
