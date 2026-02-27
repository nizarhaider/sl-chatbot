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
            logger.info(f"Attempting to connect to OpenAI Realtime for {call_id} at {url}")
            # In websockets 14.0+, the argument is 'additional_headers'
            async with websockets.connect(
                url, 
                additional_headers=headers
            ) as openai_ws:
                logger.info(f"Connected to OpenAI Realtime for {call_id}")
                
                # 1. Initialize session
                await self._initialize_session(openai_ws)
                await asyncio.sleep(0.5) # Wait for session to stabilize
                
                # 2. Trigger an initial greeting from the AI
                await self._send_greeting(openai_ws)

                # 3. Start concurrent tasks
                await asyncio.gather(
                    self._whatsapp_to_openai(input_track, openai_ws),
                    self._openai_to_whatsapp(openai_ws, output_track)
                )

        except Exception as e:
            logger.error(f"Error in VoiceAgent for {call_id}: {e}")
            # Fallback for older or legacy versions of websockets
            if "unexpected keyword argument" in str(e):
                logger.warning("Retrying with legacy 'extra_headers'...")
                try:
                    async with websockets.connect(url, extra_headers=headers) as openai_ws:
                        await self._initialize_session(openai_ws)
                        await asyncio.sleep(0.5)
                        await self._send_greeting(openai_ws)
                        await asyncio.gather(
                            self._whatsapp_to_openai(input_track, openai_ws),
                            self._openai_to_whatsapp(openai_ws, output_track)
                        )
                except Exception as e2:
                    logger.error(f"Legacy retry failed: {e2}")

    async def _send_greeting(self, ws):
        """
        Force the AI to speak first so the user knows it's connected.
        """
        logger.info("Triggering AI greeting...")
        # Create a conversation item
        item_create = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "The call is connected. Please greet the user warmly and introduce yourself as the SL Voice Assistant."
                    }
                ]
            }
        }
        await ws.send(json.dumps(item_create))
        
        # Trigger the response
        await ws.send(json.dumps({"type": "response.create"}))

    async def _initialize_session(self, ws):
        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": (
                    "You are a professional assistant on a WhatsApp voice call. "
                    "Your name is SL Bot. Keep responses short and snappy. "
                    "You must always respond with audio."
                ),
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5, # Slightly less sensitive to avoid background noise interruption
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                }
            }
        }
        await ws.send(json.dumps(session_update))

    async def _whatsapp_to_openai(self, track, ws):
        try:
            logger.info("Starting WhatsApp to OpenAI audio stream")
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
            logger.info("Starting OpenAI to WhatsApp audio stream")
            async for message in ws:
                event = json.loads(message)
                
                # Log non-audio events with more detail for responses
                if event["type"] == "response.done":
                    logger.info(f"OpenAI Response Done: {json.dumps(event, indent=2)}")
                elif event["type"] == "error":
                    logger.error(f"OpenAI Error: {json.dumps(event, indent=2)}")
                elif event["type"] != "audio":
                    logger.info(f"OpenAI Event: {event['type']}")
                
                if event["type"] == "response.audio.delta":
                    audio_b64 = event["delta"]
                    audio_bytes = base64.b64decode(audio_b64)
                    output_track.add_audio(audio_bytes)
                
                elif event["type"] == "response.audio_transcript.delta":
                    logger.info(f"AI Transcript: {event.get('delta')}")
                    
                elif event["type"] == "session.created":
                    logger.info("OpenAI session created successfully")
                    
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
        # logger.debug(f"Pushed {len(data)} bytes to audio queue")
        self.queue.put_nowait(data)

    async def recv(self):
        # We need to return exactly 20ms frames to match WebRTC expectations
        # OpenAI sends chunks of varying sizes. 
        # We'll just return whatever we have for now, but WebRTC prefers 20ms.
        
        try:
            # wait_for avoids blocking forever if the stream ends
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
