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
                "modalities": ["audio"],  # Changed from ["text", "audio"] to fix SDK error
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.6,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 800,
                    "interrupt_response": True
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
        self._sample_rate = 24000  # OpenAI sends 24kHz
        self._time_base = Fraction(1, self._sample_rate)
        self._samples_per_frame = 480  # 20ms at 24kHz
        self._buffer = b""
        self._start_time = None

    def add_audio(self, data: bytes):
        self.queue.put_nowait(data)

    async def recv(self):
        # 1. Pacing: ensure we don't return frames faster than real-time
        if self._start_time is None:
            self._start_time = asyncio.get_event_loop().time()
        
        # Calculate when the NEXT frame should be sent
        next_frame_time = self._start_time + (self._pts / self._sample_rate)
        now = asyncio.get_event_loop().time()
        
        # Sleep until it's time for the next frame
        if next_frame_time > now:
            await asyncio.sleep(next_frame_time - now)

        # 2. Frame Construction
        target_size = self._samples_per_frame * 2  # 16-bit mono = 2 bytes per sample
        
        # Pull everything available in the queue into our buffer
        while not self.queue.empty():
            self._buffer += self.queue.get_nowait()
        
        if len(self._buffer) >= target_size:
            # We have enough data for a real frame
            data_to_send = self._buffer[:target_size]
            self._buffer = self._buffer[target_size:]
        else:
            # Not enough data yet (jitter/lag), send 20ms of silence
            data_to_send = b'\x00' * target_size

        # 3. Create and return the frame
        frame = AudioFrame(format='s16', layout='mono', samples=self._samples_per_frame)
        frame.planes[0].update(data_to_send)
        frame.pts = self._pts
        frame.sample_rate = self._sample_rate
        frame.time_base = self._time_base
        self._pts += self._samples_per_frame
        
        return frame

voice_agent = VoiceAgent()
