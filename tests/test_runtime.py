import asyncio
import hashlib
import hmac
import json
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import httpx
from aiortc import RTCConfiguration, RTCPeerConnection
from fastapi import HTTPException
from google.genai import types
from pipecat.frames.frames import EndFrame, InputAudioRawFrame, InterruptionFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.whatsapp.api import WhatsAppApi
from pipecat.transports.whatsapp.client import WhatsAppClient
from pipecat.workers.runner import WorkerRunner

from app import server, store
from app.agent import CallGeminiService
from app.tools import CallContext


def test_webhook_verification_signature_and_readiness(monkeypatch):
    monkeypatch.setenv("VERIFY_TOKEN", "test-verify")
    monkeypatch.setenv("WHATSAPP_APP_SECRET", "test-secret")
    app = server.create_app()

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/")).json()["status"] == "warming_up"
            assert (await client.get("/webhook?hub.mode=subscribe&hub.verify_token=wrong")).status_code == 403
            response = await client.get("/webhook?hub.mode=subscribe&hub.verify_token=test-verify&hub.challenge=123")
            assert response.text == "123"
            body = json.dumps({"object": "whatsapp_business_account", "entry": []}).encode()
            assert (await client.post("/webhook", content=body)).status_code == 403
            signature = "sha256=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
            assert (await client.post("/webhook", content=body, headers={"x-hub-signature-256": signature})).status_code == 200
            assert (await client.post("/demo/call")).status_code == 404
    asyncio.run(check())


def test_capacity_and_duplicate_reservations(monkeypatch):
    monkeypatch.setenv("MAX_CALLS", "2")
    async def check():
        calls = server.Calls()
        assert await calls.reserve("one")
        assert not await calls.reserve("one")
        assert await calls.reserve("two")
        try:
            await calls.reserve("three")
        except HTTPException as exc:
            assert exc.status_code == 503
        else:
            raise AssertionError("Capacity limit was bypassed")
    asyncio.run(check())


def test_outbound_sdp_answer_gates_agent_and_cleanup(monkeypatch):
    state = store.CallState(persist=False)
    monkeypatch.setattr(server, "call_state", state)
    monkeypatch.setattr(server, "SmallWebRTCConnection", lambda **kwargs: SmallWebRTCConnection(ice_servers=[]))
    monkeypatch.setattr(server, "query", AsyncMock())
    monkeypatch.setattr(server.whatsapp_api, "send_call_action", AsyncMock(return_value=True))
    monkeypatch.setenv("VOICE_AGENT_ID", "host")
    started = asyncio.Event()

    async def agent(peer, context, ready):
        await ready.wait()
        started.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(server, "run_agent", agent)

    async def check():
        remote = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        calls = server.Calls()
        async def dial(phone, sdp, request_id):
            await remote.setRemoteDescription(server.RTCSessionDescription(sdp=sdp, type="offer"))
            await remote.setLocalDescription(await remote.createAnswer())
            return "outbound-test"
        monkeypatch.setattr(server.whatsapp_api, "create_call", dial)
        try:
            call_id = await calls.dial("request", CallContext("", "94770000000", "agent", "tenant"))
            await asyncio.sleep(0)
            assert not started.is_set()
            event = {"id": call_id, "event": "connect", "session": {"sdp": remote.localDescription.sdp, "sdp_type": "answer"}}
            await calls.call_event(event, {})
            await asyncio.wait_for(started.wait(), 2)
            assert calls.connections[call_id].answered
            assert state.calls[call_id]["agent_id"] == "agent"
            await calls.close_call(call_id)
            await calls.close_call(call_id)
            assert not calls.connections
            assert state.calls[call_id]["status"] == "ended"
        finally:
            await calls.close()
            await remote.close()
    asyncio.run(check())


def test_inbound_uses_pipecat_whatsapp_negotiation(monkeypatch):
    state = store.CallState(persist=False)
    monkeypatch.setattr(server, "call_state", state)
    monkeypatch.setattr(WhatsAppApi, "answer_call_to_whatsapp", AsyncMock(return_value={"success": True}))
    monkeypatch.setattr(server.whatsapp_api, "send_call_action", AsyncMock(return_value=True))
    async def agent(peer, context, ready):
        assert ready.is_set()
        await asyncio.Event().wait()
    monkeypatch.setattr(server, "run_agent", agent)

    async def check():
        remote = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        remote.addTransceiver("audio", direction="sendrecv")
        calls = server.Calls()
        async with aiohttp.ClientSession() as session:
            calls.client = WhatsAppClient("test", "test", session, ice_servers=[])
            try:
                await remote.setLocalDescription(await remote.createOffer())
                event = {"id": "inbound-test", "from": "94770000000", "to": "94774482914",
                         "event": "connect", "timestamp": "123", "direction": "inbound",
                         "session": {"sdp": remote.localDescription.sdp, "sdp_type": "offer"}}
                value = {"messaging_product": "whatsapp", "metadata": {"display_phone_number": "94774482914", "phone_number_id": "test"}, "contacts": []}
                await calls.call_event(event, value)
                assert calls.connections["inbound-test"].ready.is_set()
                assert state.calls["inbound-test"]["agent_id"] == ""
                assert WhatsAppApi.answer_call_to_whatsapp.await_count == 2
                await calls.close_call("inbound-test")
                assert not calls.connections
            finally:
                await calls.close()
                await remote.close()
    asyncio.run(check())


def test_raw_usage_fields_are_preserved(monkeypatch):
    monkeypatch.setattr(GeminiLiveLLMService, "_handle_msg_usage_metadata", AsyncMock())
    state, context = store.CallState(persist=False), CallContext("usage-test", "")
    state.start_call(context)
    usage = types.UsageMetadata(prompt_token_count=10, response_token_count=20, total_token_count=30,
                                prompt_tokens_details=[types.ModalityTokenCount(modality="AUDIO", token_count=10)])
    service = CallGeminiService(context, state, api_key="test")
    asyncio.run(service._handle_msg_usage_metadata(SimpleNamespace(usage_metadata=usage)))
    saved = state.calls[context.call_id]["events"][-1]
    assert saved["data"]["usage"] == usage.model_dump(mode="json", exclude_none=True)
    assert saved["data"]["total_tokens"] == 30


class SlowOutput(BaseOutputTransport):
    def __init__(self):
        super().__init__(TransportParams(audio_out_enabled=True, audio_out_10ms_chunks=2))
        self.sent = bytearray()

    async def start(self, frame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame):
        self.sent.extend(frame.audio)
        await asyncio.sleep(0.02)
        return True


def test_recording_tracks_and_interrupted_playback():
    async def check():
        output = SlowOutput()
        recorder = AudioBufferProcessor(sample_rate=16000, num_channels=2, auto_start_recording=True)
        worker = PipelineWorker(Pipeline([IdentityFilter(), output, recorder]), enable_rtvi=False,
                                idle_timeout_secs=None, params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=16000))
        started, recordings = asyncio.Event(), []
        @worker.event_handler("on_pipeline_started")
        async def ready(worker, frame):
            started.set()
        @recorder.event_handler("on_audio_data")
        async def recorded(processor, audio, rate, channels):
            assert (rate, channels) == (16000, 2)
            recordings.append(audio)
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        task = asyncio.create_task(runner.run())
        try:
            await asyncio.wait_for(started.wait(), 3)
            await worker.queue_frames([InputAudioRawFrame(struct.pack('<h', 1000) * 3200, 16000, 1),
                                       OutputAudioRawFrame(struct.pack('<h', 2000) * 48000, 16000, 1)])
            await asyncio.sleep(0.08)
            await worker.queue_frames([InterruptionFrame()])
            await asyncio.sleep(0.1)
            await worker.queue_frames([OutputAudioRawFrame(struct.pack('<h', 3000) * 1600, 16000, 1), EndFrame()])
            await asyncio.wait_for(task, 5)
        finally:
            await runner.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert 0 < output.sent.count(struct.pack('<h', 2000)) < 48000
        assert struct.pack('<h', 3000) in output.sent
        samples = struct.unpack('<' + 'h' * (len(recordings[0]) // 2), recordings[0])
        assert max(samples[::2]) == 1000
        assert 3000 in samples[1::2]
        assert len(recordings[0]) < 48000 * 4
    asyncio.run(check())


def test_recording_upload_flushes_and_uses_private_s3(monkeypatch):
    uploaded = []
    class S3:
        def put_object(self, **kwargs):
            uploaded.append(kwargs)
    monkeypatch.setattr(store.boto3, "client", lambda name: S3())
    monkeypatch.setattr(store, "_pcm16_stereo_to_mp3", lambda pcm: b"mp3")
    async def check():
        state, context = store.CallState(persist=False), CallContext("recording-test", "")
        state.start_call(context)
        state.archive(context.call_id, b"pcm")
        state.end_call(context.call_id)
        await state.close()
        event = state.calls[context.call_id]["events"][-1]
        assert event["kind"] == "recording.archived"
        assert uploaded[0]["ServerSideEncryption"] == "AES256"
        assert uploaded[0]["ContentType"] == "audio/mpeg"
        assert uploaded[0]["Key"].endswith('.mp3')
    asyncio.run(check())
