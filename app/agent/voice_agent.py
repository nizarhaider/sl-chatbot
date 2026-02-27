import os
import json
import base64
import asyncio
import logging
import websockets
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
            async with websockets.connect(url, extra_headers=headers) as openai_ws:
                logger.info(f"Connected to OpenAI Realtime for {call_id}")
                
                # 1. Initialize session
                await self._initialize_session(openai_ws)

                # 2. Start concurrent tasks: 
                #    - Read from WhatsApp -> Send to OpenAI
                #    - Read from OpenAI -> Send to WhatsApp
                await asyncio.gather(
                    self._whatsapp_to_openai(input_track, openai_ws),
                    self._openai_to_whatsapp(openai_ws, output_track)
                )

        except Exception as e:
            logger.error(f"Error in VoiceAgent for {call_id}: {e}")

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
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {"type": "server_vad"}
            }
        }
        await ws.send(json.dumps(session_update))

    async def _whatsapp_to_openai(self, track, ws):
        """
        Reads audio from WhatsApp (aiortc) and sends it to OpenAI.
        """
        try:
            while True:
                frame = await track.recv()
                # 1. Get audio as numpy array
                # frames from aiortc can be mono or stereo
                audio_data = frame.to_ndarray()
                
                # 2. Convert to mono if stereo
                if audio_data.ndim > 1 and audio_data.shape[0] > 1:
                    # Average the channels
                    audio_data = np.mean(audio_data, axis=0)
                else:
                    audio_data = audio_data.flatten()

                # 3. Simple downsampling from 48k to 24k if needed
                if frame.sample_rate == 48000:
                    audio_data = audio_data[::2]
                elif frame.sample_rate != 24000:
                    # In a production app, use a proper resampler like soxr or audioop
                    # For now, we assume 48k or 24k
                    pass
                
                # 4. Convert to base64
                audio_bytes = audio_data.astype('int16').tobytes()
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }))
        except Exception as e:
            logger.info(f"WhatsApp to OpenAI stream ended: {e}")

    async def _openai_to_whatsapp(self, ws, output_track):
        """
        Reads audio from OpenAI and pushes it to our output track for WhatsApp.
        """
        try:
            async for message in ws:
                event = json.loads(message)
                
                if event["type"] == "response.audio.delta":
                    audio_b64 = event["delta"]
                    audio_bytes = base64.b64decode(audio_b64)
                    
                    # Push bytes into our output track's queue
                    output_track.add_audio(audio_bytes)
                    
                elif event["type"] == "error":
                    logger.error(f"OpenAI error: {event['error']}")
                    
        except Exception as e:
            logger.info(f"OpenAI to WhatsApp stream ended: {e}")

class RealtimeAudioTrack(MediaStreamTrack):
    """
    An audio track that we can push audio data into.
    """
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()
        self._pts = 0
        self._sample_rate = 24000 # OpenAI's rate
        self._time_base = 1 / self._sample_rate

    def add_audio(self, data: bytes):
        # Data is PCM16 24kHz Mono
        self.queue.put_nowait(data)

    async def recv(self):
        # We need to return an AudioFrame
        # WhatsApp usually expects 20ms frames.
        # At 24kHz, 20ms is 480 samples. 480 * 2 bytes = 960 bytes.
        
        # This is a bit tricky because we might get fragments.
        # For simplicity, we'll wait for enough data.
        data = await self.queue.get()
        
        # Convert bytes to AudioFrame
        num_samples = len(data) // 2
        frame = AudioFrame(format='s16', layout='mono', samples=num_samples)
        frame.planes[0].update(data)
        
        frame.pts = self._pts
        frame.sample_rate = self._sample_rate
        frame.time_base = self._time_base
        
        self._pts += num_samples
        return frame

voice_agent = VoiceAgent()
