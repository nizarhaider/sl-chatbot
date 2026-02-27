import os
import json
import base64
import asyncio
import logging
import numpy as np
from fractions import Fraction
from aiortc import MediaStreamTrack
from av import AudioFrame

# New OpenAI Agents SDK imports
from agents.realtime import (
    RealtimeAgent,
    RealtimeRunner,
    RealtimeSession,
)

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class VoiceAgent:
    def __init__(self):
        self.agent = RealtimeAgent(
            name="SL Voice Assistant",
            instructions=(
                "You are a professional assistant on a WhatsApp voice call. "
                "Your name is SL Bot. Keep responses short and snappy. "
                "You must always respond in a conversational and helpful manner."
            )
        )
        # The runner handles the connection and session lifecycle
        self.runner = RealtimeRunner(self.agent)

    async def process_audio(self, call_id: str, input_track: MediaStreamTrack, output_track: MediaStreamTrack):
        """
        Main loop for audio processing using OpenAI Agents SDK.
        """
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not set")
            return

        model_config = {
            "model": "gpt-4o-realtime-preview",
            "initial_model_settings": {
                "modalities": ["text", "audio"],
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                }
            }
        }

        try:
            logger.info(f"Starting RealtimeSession for call {call_id} using Agents SDK")
            
            # runner.run() returns an async context manager
            async with await self.runner.run(model_config=model_config) as session:
                logger.info(f"Connected to OpenAI via Agents SDK for {call_id}")
                
                # Small wait for session stabilization
                await asyncio.sleep(0.5)
                
                # Trigger initial greeting
                logger.info("Triggering initial greeting via SDK...")
                await session.send_message("The call is connected. Please greet the user warmly and introduce yourself as the SL Voice Assistant.")

                # Start concurrent tasks
                await asyncio.gather(
                    self._whatsapp_to_openai(input_track, session),
                    self._openai_to_whatsapp(session, output_track)
                )

        except Exception as e:
            logger.error(f"Error in VoiceAgent SDK bridge for {call_id}: {e}", exc_info=True)

    async def _whatsapp_to_openai(self, track: MediaStreamTrack, session: RealtimeSession):
        """
        Reads audio from WhatsApp (aiortc) and sends it to OpenAI Session.
        """
        try:
            logger.info("Starting WhatsApp -> OpenAI (SDK) audio stream")
            while True:
                frame = await track.recv()
                audio_data = frame.to_ndarray()
                
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=0)
                else:
                    audio_data = audio_data.flatten()

                # Downsample 48k to 24k
                if frame.sample_rate == 48000:
                    audio_data = audio_data[::2]
                
                audio_bytes = audio_data.astype('int16').tobytes()
                
                # Send raw bytes to the session
                await session.send_audio(audio_bytes)
                
        except Exception as e:
            logger.info(f"WhatsApp to OpenAI stream ended: {e}")

    async def _openai_to_whatsapp(self, session: RealtimeSession, output_track):
        """
        Reads events from OpenAI Session and pushes audio to WhatsApp.
        """
        try:
            logger.info("Starting OpenAI (SDK) -> WhatsApp audio stream")
            async for event in session:
                # The SDK wraps raw events into structured dataclasses
                if event.type == "audio":
                    # event.audio is a RealtimeModelAudioEvent (or similar)
                    # We need the raw bytes. Looking at session.py:
                    # Raw model audio is in event.audio.data
                    audio_bytes = event.audio.data
                    output_track.add_audio(audio_bytes)
                
                elif event.type == "audio_interrupted":
                    logger.info("AI audio interrupted by user")
                    # In a advanced version, we would clear the output track buffer here
                
                elif event.type == "history_updated":
                    # Logs transcripts and tool calls if needed
                    # logger.debug(f"History updated: {len(event.history)} items")
                    pass
                
                elif event.type == "error":
                    logger.error(f"SDK Session Error: {event.error}")

        except Exception as e:
            logger.info(f"OpenAI to WhatsApp stream ended: {e}")

class RealtimeAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()
        self._pts = 0
        self._sample_rate = 24000
        self._time_base = Fraction(1, self._sample_rate)
        self._samples_per_frame = 480 # 20ms at 24kHz

    def add_audio(self, data: bytes):
        self.queue.put_nowait(data)

    async def recv(self):
        try:
            # wait_for avoids blocking forever
            data = await asyncio.wait_for(self.queue.get(), timeout=0.1)
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            # Return silence if no data
            data = b'\x00' * (self._samples_per_frame * 2)
        
        num_samples = len(data) // 2
        frame = AudioFrame(format='s16', layout='mono', samples=num_samples)
        frame.planes[0].update(data)
        frame.pts = self._pts
        frame.sample_rate = self._sample_rate
        frame.time_base = self._time_base
        self._pts += num_samples
        return frame

voice_agent = VoiceAgent()
