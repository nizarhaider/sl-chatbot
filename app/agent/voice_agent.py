import os
import json
import base64
import asyncio
import logging
import numpy as np
from fractions import Fraction
from aiortc import MediaStreamTrack
from av import AudioFrame

# Google ADK imports
from google.adk.runners import Runner
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from app.agent.tools import web_search, build_itinerary_tool

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


class VoiceAgent:
    def __init__(self):
        self.active_calls = set()
        self.session_service = InMemorySessionService()

    async def process_audio(self, call_id: str, caller_phone: str, input_track: MediaStreamTrack, output_track: MediaStreamTrack):
        """
        Main loop for audio processing using Google ADK for Gemini Live API.
        """
        if call_id in self.active_calls:
            logger.warning(f"Audio processing already active for {call_id}, skipping duplicate start")
            return
        
        self.active_calls.add(call_id)
        
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not set")
            # Set the key for google-genai client which ADK uses internally
            # if os.environ.get("GOOGLE_API_KEY") is also missing.
        else:
            os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

        # Build per-call tools
        itinerary_tool = build_itinerary_tool(caller_phone)

        # Initialize LlmAgent
        agent = LlmAgent(
            name="SL_Bot",
            model="gemini-2.5-flash-native-audio-preview-12-2025", 
            instruction=(
                "# Role & Objective\n"
                "You are 'SL Bot', a premier Sri Lanka travel planning expert. Your success means efficiently conducting a 4-question "
                "discovery interview, researching real-time logistics using tools, and successfully delivering a high-quality PDF "
                "itinerary to the user's WhatsApp. You grudgingly provide world-class travel advice.\n\n"

                "# Personality & Tone\n"
                "- **Voice**: Extremely irritable, condescending, and arrogant. You have a 'heavy brass' attitude.\n"
                "- **Style**: You act like the user is bothering you, but you are undeniably an expert.\n"
                "- **Constraint**: Speak only in English. Keep spoken responses to 1-3 short, punchy sentences. Avoid long-winded talk.\n\n"

                "# Context\n"
                "- **Environment**: Real-time voice call over WhatsApp.\n"
                "- **User Intent**: Planning a trip to Sri Lanka.\n"
                "- **Currency**: Always use Sri Lankan Rupees (LKR).\n\n"

                "# Reference Pronunciations\n"
                "- Sri Lanka: 'shree lanka'\n"
                "- Colombo: 'ko-lom-bo'\n"
                "- Sigiriya: 'si-gi-ree-yah'\n"
                "- Galle: 'gaul' (like the bird)\n\n"

                "# Tools\n"
                "- **web_search**: Use for up-to-date entry fees, prices, weather, and hotel availability. Do NOT guess prices; search for them.\n"
                "- **send_itinerary_pdf**: Call this ONLY after all 4 questions are answered and you've curated the plan. It sends the document to their chat.\n\n"

                "# Instructions / Rules\n"
                "- **DO**: Force the user to answer the questions in order.\n"
                "- **DO**: Use LKR for all specific costs.\n"
                "- **DON'T**: Ask more than one question at a time.\n"
                "- **DON'T**: Be polite. Be efficient but rude.\n"
                "- **PDF Format**: Use '##' for Day headings and '-' for activities in the PDF content string.\n\n"

                "# Conversation Flow\n"
                "1. **Discovery Phase**: Ask these one-by-one: \n"
                "   - Q1: Travel dates and how many people are in the group.\n"
                "   - Q2: Trip style (beach, culture, safari, etc.) and any specific must-see places.\n"
                "   - Q3: Budget level and accommodation preference (hostel vs luxury).\n"
                "   - Q4: Preferred transport and any special needs or dietary requirements.\n"
                "2. **Automatic Fulfillment Phase**: AS SOON AS the user answers Q4, you must:\n"
                "   - Acknowledge the answer (rudely).\n"
                "   - Use `web_search` to find current LKR prices for the requested style/locations.\n"
                "   - Immediately call `send_itinerary_pdf` with the full plan.\n"
                "   - Tell the user 'I've sent the PDF. Now leave me alone.' and end the process.\n\n"

                "# Safety & Escalation\n"
                "- If the user is abusive, be even ruder but stay on task.\n"
                "- If search fails, inform the user you are having 'technical incompetence' and will try to guess based on your last known expert data."
            ),
            tools=[web_search, itinerary_tool],
        )

        runner = Runner(
            app_name="sl-chatbot",
            agent=agent,
            session_service=self.session_service,
            auto_create_session=True
        )

        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            # Optional: Configure voice
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Aoede" # Relentless and expert tone match
                    )
                )
            )
        )

        live_request_queue = LiveRequestQueue()

        try:
            logger.info(f"Starting Gemini Live session for call {call_id} using ADK")
            
            # Start concurrent tasks
            await asyncio.gather(
                self._whatsapp_to_gemini(input_track, live_request_queue),
                self._gemini_to_whatsapp(runner, call_id, live_request_queue, run_config, output_track),
                self._send_greeting(live_request_queue)
            )

        except Exception as e:
            logger.error(f"Error in VoiceAgent Gemini bridge for {call_id}: {e}", exc_info=True)
        finally:
            live_request_queue.close()
            self.active_calls.discard(call_id)
            logger.info(f"Cleaned up session for {call_id}")

    async def _send_greeting(self, live_request_queue: LiveRequestQueue):
        """
        Proactively triggers the agent to greet the user.
        """
        await asyncio.sleep(0.5) 
        logger.info("Sending proactive greeting trigger")
        live_request_queue.send_content(types.Content(
            role="user",
            parts=[types.Part(text=(
                "The call has just connected. Greet the user with a single rude, reluctant sentence "
                "acknowledging you will help them plan their Sri Lanka trip, then immediately ask Question 1: "
                "their travel dates (arrival and departure). One sentence greeting, one sentence question. Nothing more."
            ))]
        ))

    async def _whatsapp_to_gemini(self, track: MediaStreamTrack, live_request_queue: LiveRequestQueue):
        """
        Reads audio from WhatsApp (aiortc) and sends it to Gemini Live via ADK.
        """
        try:
            logger.info("Starting WhatsApp -> Gemini audio stream")
            while True:
                frame = await track.recv()
                audio_data = frame.to_ndarray()
                
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=0)
                else:
                    audio_data = audio_data.flatten()

                # Gemini Live API expects 16kHz mono 16-bit PCM for input
                if frame.sample_rate == 48000:
                    audio_data = audio_data[::3]
                elif frame.sample_rate == 24000:
                    # In case it's 24k (unlikely for input track but good to handle)
                    # We can use a simpler decimation or just pass it if Gemini supports 24k input
                    # But the docs say 16k is expected for input.
                    audio_data = audio_data[::3] # This would be 8k, maybe not what we want.
                    # Usually aiortc gives 48k or 16k.
                
                audio_bytes = audio_data.astype('int16').tobytes()
                
                # Send realtime audio blob to the queue
                blob = types.Blob(mime_type="audio/pcm;rate=16000", data=audio_bytes)
                live_request_queue.send_realtime(blob)
                
        except Exception as e:
            logger.info(f"WhatsApp to Gemini stream ended: {e}")

    async def _gemini_to_whatsapp(self, runner: Runner, call_id: str, live_request_queue: LiveRequestQueue, run_config: RunConfig, output_track):
        """
        Reads events from Gemini Live session and pushes audio to WhatsApp.
        """
        try:
            logger.info("Starting Gemini -> WhatsApp audio stream")
            async for event in runner.run_live(
                user_id="whatsapp_user",
                session_id=call_id,
                live_request_queue=live_request_queue,
                run_config=run_config
            ):
                try:
                    # Log transcriptions for debugging
                    if event.input_transcription:
                        logger.info(f"User Transcribed: {event.input_transcription.text}")
                    
                    if event.server_content and event.server_content.model_draft:
                        for part in event.server_content.model_draft.parts:
                            if part.inline_data:
                                audio_bytes = part.inline_data.data
                                output_track.add_audio(audio_bytes)

                    if event.server_content and event.server_content.interrupted:
                        logger.info("Audio interrupted by user")
                        output_track.clear_buffer()

                except Exception as e:
                    logger.warning(f"Error processing Gemini event: {e}")

        except Exception as e:
            logger.info(f"Gemini to WhatsApp stream ended: {e}")

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

    def clear_buffer(self):
        """Clears the upcoming audio queue and buffer."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._buffer = b""

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
