import asyncio
import os
import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("VOICE_OUTPUT_PROVIDER", "omnivoice")

from app.main import app
from app.services.tts import SynthesizedAudio
from app.voice_agent.agent import (
    GEMINI_LIVE_AUDIO_MODEL,
    RealtimeAudioTrack,
    TurnLatencyTracker,
    voice_agent,
)
from app.voice_agent.gemini_turn_pipeline import GeminiTurnPipeline


class WebhookPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.transport = httpx.ASGITransport(app=app)
        self.client = httpx.AsyncClient(transport=self.transport, base_url="http://testserver")

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_call_connect_webhook_schedules_offer_handling(self):
        handled = asyncio.Event()
        captured = {}

        async def fake_handle_offer(call_id: str, sdp_offer: str, caller_phone: str = ""):
            captured["call_id"] = call_id
            captured["sdp_offer"] = sdp_offer
            captured["caller_phone"] = caller_phone
            handled.set()

        payload = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "calls": [
                                    {
                                        "event": "connect",
                                        "id": "call-123",
                                        "from": "94770000000",
                                        "session": {
                                            "sdp_type": "offer",
                                            "sdp": "v=0\r\ns=test\r\n",
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            ],
        }

        with patch("app.webhooks.whatsapp.webrtc_service.handle_offer", new=fake_handle_offer):
            response = await self.client.post("/webhook", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.text, "EVENT_RECEIVED")
            await asyncio.wait_for(handled.wait(), timeout=1)

        self.assertEqual(
            captured,
            {
                "call_id": "call-123",
                "sdp_offer": "v=0\r\ns=test\r\n",
                "caller_phone": "94770000000",
            },
        )

    async def test_text_message_webhook_sends_chat_agent_reply(self):
        sent = asyncio.Event()
        captured = {}

        async def fake_get_response(text: str) -> str:
            captured["prompt"] = text
            return "Hello from the bot"

        async def fake_send_message(to: str, text: str) -> bool:
            captured["to"] = to
            captured["reply"] = text
            sent.set()
            return True

        payload = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "type": "text",
                                        "from": "94770000000",
                                        "text": {"body": "Hi"},
                                    }
                                ]
                            }
                        }
                    ]
                }
            ],
        }

        with patch("app.webhooks.whatsapp.chat_agent.get_response", new=fake_get_response):
            with patch("app.webhooks.whatsapp.whatsapp_api.send_message", new=fake_send_message):
                response = await self.client.post("/webhook", json=payload)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.text, "EVENT_RECEIVED")
                await asyncio.wait_for(sent.wait(), timeout=1)

        self.assertEqual(
            captured,
            {
                "prompt": "Hi",
                "to": "94770000000",
                "reply": "Hello from the bot",
            },
        )

    async def test_text_response_is_synthesized_and_buffered_for_whatsapp(self):
        pcm_bytes = b"\x01\x02" * 960
        output_track = RealtimeAudioTrack()
        run_config = object()
        live_request_queue = object()
        captured = {}

        async def fake_synthesize(text: str) -> SynthesizedAudio:
            captured["text"] = text
            return SynthesizedAudio(pcm=pcm_bytes, sample_rate=24000, text=text)

        class FakeRunner:
            async def run_live(self, **kwargs):
                yield SimpleNamespace(
                    input_transcription=None,
                    content=SimpleNamespace(parts=[SimpleNamespace(text="හායි, මම තරුශි.")]),
                    partial=False,
                    turn_complete=True,
                    interrupted=False,
                )

        with patch.object(voice_agent.tts_service, "synthesize", new=fake_synthesize):
            voice_agent.playback_generation["call-tts"] = 0
            await voice_agent._gemini_text_to_whatsapp(
                runner=FakeRunner(),
                call_id="call-tts",
                live_request_queue=live_request_queue,
                run_config=run_config,
                output_track=output_track,
            )

        queued_audio = []
        while not output_track.queue.empty():
            queued_audio.append(output_track.queue.get_nowait())

        self.assertEqual(captured["text"], "හායි, මම තරුශි.")
        self.assertTrue(queued_audio)
        self.assertEqual(b"".join(queued_audio), pcm_bytes)

    async def test_tts_text_is_trimmed_before_synthesis(self):
        pcm_bytes = b"\x01\x02" * 480
        output_track = RealtimeAudioTrack()
        captured = {}

        async def fake_synthesize(text: str) -> SynthesizedAudio:
            captured["text"] = text
            return SynthesizedAudio(pcm=pcm_bytes, sample_rate=24000, text=text)

        long_text = (
            "First sentence stays. "
            "Second sentence also stays. "
            "This third sentence should not be synthesized because the TTS text is trimmed."
        )

        with patch.object(voice_agent.tts_service, "synthesize", new=fake_synthesize):
            voice_agent.playback_generation["call-trim"] = 0
            await voice_agent._speak_text_response("call-trim", long_text, output_track)

        self.assertEqual(captured["text"], "First sentence stays. Second sentence also stays.")

    async def test_omnivoice_path_falls_back_to_audio_live_model_for_calls(self):
        output_track = RealtimeAudioTrack()
        runner_models = []

        async def fake_whatsapp_to_gemini(*args, **kwargs):
            return None

        async def fake_gemini_text_to_whatsapp(*args, **kwargs):
            return None

        class FakeRunner:
            def __init__(self, *args, **kwargs):
                runner_models.append(kwargs["agent"].model)

        with patch.object(voice_agent.tts_service, "uses_gemini_audio", return_value=False):
            with patch("app.voice_agent.agent.Runner", new=FakeRunner):
                with patch.object(voice_agent, "_whatsapp_to_gemini", new=fake_whatsapp_to_gemini):
                    with patch.object(voice_agent, "_gemini_text_to_whatsapp", new=fake_gemini_text_to_whatsapp):
                        await voice_agent.process_audio(
                            "call-model",
                            "94770000000",
                            SimpleNamespace(),
                            output_track,
                        )

        self.assertEqual(runner_models[-1], GEMINI_LIVE_AUDIO_MODEL)

    async def test_gemini_turn_pipeline_emits_greeting_through_tts(self):
        output_track = RealtimeAudioTrack()
        captured = {}

        async def fake_synthesize(text: str) -> SynthesizedAudio:
            captured["text"] = text
            return SynthesizedAudio(pcm=b"\x01\x02" * 480, sample_rate=24000, text=text)

        class EndTrack:
            async def recv(self):
                raise RuntimeError("done")

        pipeline = GeminiTurnPipeline(
            tts_service=SimpleNamespace(synthesize=fake_synthesize),
            prepare_tts_text=lambda text: text,
            interrupt_playback=lambda call_id, output_track: None,
        )

        await pipeline.run(
            call_id="call-turn",
            input_track=EndTrack(),
            output_track=output_track,
            playback_generation={"call-turn": 0},
        )

        self.assertIn("For English, please say English.", captured["text"])

    async def test_turn_latency_is_logged_once_on_first_response(self):
        tracker = TurnLatencyTracker()
        tracker.speech_ended_at = 100.0
        voice_agent.turn_latency["call-latency"] = tracker

        with patch("app.voice_agent.agent.time.perf_counter", side_effect=[101.25, 102.0]):
            with self.assertLogs("app.voice_agent.agent", level="INFO") as logs:
                voice_agent._note_turn_response_start("call-latency", "model audio")
                voice_agent._note_turn_response_start("call-latency", "model audio")

        self.assertIn(
            "Turn latency for call-latency: first model audio in 1250 ms after speech end",
            "\n".join(logs.output),
        )

        voice_agent.turn_latency.pop("call-latency", None)


if __name__ == "__main__":
    unittest.main()
