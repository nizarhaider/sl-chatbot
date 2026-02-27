import os
import json
import base64
import asyncio
import logging
import websockets
import numpy as np
from fractions import Fraction
from aiortc import MediaStreamTrack
from av import AudioFrame

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class VoiceAgent:
    def __init__(self):
        self.output_track = None

    async def process_audio(self, call_id: str, input_track: MediaStreamTrack, output_track: MediaStreamTrack):
        """
        Main loop for audio processing using OpenAI Realtime.
        """
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not set")
            return

        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }

        try:
            # We'll try extra_headers first, which is standard for 'websockets'
            # If it continues to fail, we'll catch it.
            async with websockets.connect(
                url, 
                extra_headers=headers
            ) as openai_ws:
                logger.info(f"Connected to OpenAI Realtime for {call_id}")
                
                # 1. Initialize session
                await self._initialize_session(openai_ws)

                # 2. Start concurrent tasks
                await asyncio.gather(
                    self._whatsapp_to_openai(input_track, openai_ws),
                    self._openai_to_whatsapp(openai_ws, output_track)
                )

        except Exception as e:
            logger.error(f"Error in VoiceAgent for {call_id}: {e}")
            # If it's the extra_headers error, we might be using a different library
            if "extra_headers" in str(e):
                logger.warning("Retrying with 'headers' instead of 'extra_headers'...")
                try:
                    async with websockets.connect(url, headers=headers) as openai_ws:
                        await self._initialize_session(openai_ws)
                        await asyncio.gather(
                            self._whatsapp_to_openai(input_track, openai_ws),
                            self._openai_to_whatsapp(openai_ws, output_track)
                        )
                except Exception as e2:
                    logger.error(f"Retry failed: {e2}")

    async def _initialize_session(self, ws):
        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": (
                    "You are a helpful assistant. You are talking to a user over a WhatsApp voice call. "
                    "Keep your responses concise and conversational."
                ),
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {"type": "server_vad"}
            }
        }
        await ws.send(json.dumps(session_update))

    async def _whatsapp_to_openai(self, track, ws):
        try:
            while True:
                frame = await track.recv()
                audio_data = frame.to_ndarray()
                
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=0)
                else:
                    audio_data = audio_data.flatten()

                if frame.sample_rate == 48000:
                    audio_data = audio_data[::2]
                
                audio_bytes = audio_data.astype('int16').tobytes()
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }))
        except Exception as e:
            logger.info(f"WhatsApp to OpenAI stream ended: {e}")

    async def _openai_to_whatsapp(self, ws, output_track):
        try:
            async for message in ws:
                event = json.loads(message)
                
                if event["type"] == "response.audio.delta":
                    audio_b64 = event["delta"]
                    audio_bytes = base64.b64decode(audio_b64)
                    output_track.add_audio(audio_bytes)
                
                elif event["type"] == "response.audio_transcript.delta":
                    # Optional: Log what the AI is saying
                    pass
                    
                elif event["type"] == "error":
                    logger.error(f"OpenAI error: {event['error']}")
                    
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
        # If we have no data, return a silent frame to keep the stream alive
        if self.queue.empty():
            await asyncio.sleep(0.02)
            data = b'\x00' * (self._samples_per_frame * 2)
        else:
            data = await self.queue.get()
            # If data is not exactly 20ms, it might cause jitter
            # But aiortc is usually forgiving with pts
        
        num_samples = len(data) // 2
        frame = AudioFrame(format='s16', layout='mono', samples=num_samples)
        frame.planes[0].update(data)
        frame.pts = self._pts
        frame.sample_rate = self._sample_rate
        frame.time_base = self._time_base
        self._pts += num_samples
        return frame

voice_agent = VoiceAgent()
