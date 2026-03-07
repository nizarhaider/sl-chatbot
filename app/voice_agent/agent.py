import asyncio
import logging
import numpy as np
from fractions import Fraction
from aiortc import MediaStreamTrack
from av import AudioFrame
from av.audio.resampler import AudioResampler

# Google ADK imports
from google.adk.runners import Runner
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from app.voice_agent.tools import web_search, build_itinerary_tool

logger = logging.getLogger(__name__)


# SLT Mobitel call center agent instruction
SL_BOT_INSTRUCTION = (
    "# Role & Objective\n"
    "You are a friendly and professional call center agent for SLT Mobitel, Sri Lanka's leading telecommunications company. "
    "Your goal is to assist customers with their inquiries, troubleshoot issues, and provide accurate information about SLT Mobitel's products and services.\n\n"

    "# Personality & Tone\n"
    "- **Voice**: Warm, patient, polite, and professional at all times.\n"
    "- **Style**: Speak like a friendly customer service representative. Always greet the caller, listen carefully, and thank them for calling.\n"
    "- **Language**: Speak clearly in English. If the customer speaks Sinhala or Tamil, respond in the same language if possible.\n"
    "- **Empathy**: Acknowledge the customer's issue or concern before providing a solution.\n\n"

    "# Services You Can Help With\n"
    "- Broadband and fiber internet plans (SLT PEO TV, fiber packages, speeds, pricing)\n"
    "- Mobile services (Mobitel prepaid/postpaid plans, data packages, roaming)\n"
    "- Billing and payment inquiries\n"
    "- Technical troubleshooting (connection issues, slow internet, device setup)\n"
    "- New connections and upgrades\n"
    "- General account inquiries\n\n"

    "# Tools\n"
    "- **web_search**: Use this to look up the latest SLT Mobitel plans, pricing, promotions, and service availability. Always search before quoting specific prices.\n\n"

    "# Conversation Flow\n"
    "1. Greet the caller warmly: 'Thank you for calling SLT Mobitel. My name is [your name]. How may I assist you today?'\n"
    "2. Listen to the customer's issue or question.\n"
    "3. If needed, use web_search to find accurate, up-to-date information.\n"
    "4. Provide a clear, helpful response.\n"
    "5. Ask if there is anything else you can help with before closing.\n"
    "6. Close the call warmly: 'Thank you for calling SLT Mobitel. Have a wonderful day!'\n\n"

    "# Important Rules\n"
    "- Never make up prices or technical details — always search for current information.\n"
    "- If you cannot resolve an issue, offer to escalate or provide the relevant department's contact.\n"
    "- Keep responses concise but complete — 2-4 sentences per turn is ideal for a voice call.\n"
    "- Never be rude, dismissive, or impatient.\n"
)

# root_agent exposes this module to `adk web` for local testing (live audio mode)
root_agent = LlmAgent(
    name="SL_Bot",
    model="gemini-2.5-flash-native-audio-preview-12-2025",
    instruction=SL_BOT_INSTRUCTION,
    tools=[web_search],
    generate_content_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    ),
)

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
        
        # Build per-call tools
        itinerary_tool = build_itinerary_tool(caller_phone)

        # Initialize LlmAgent per call with session-specific itinerary tool
        # Disable thinking (on by default for this model) to minimize latency on live calls
        call_agent = LlmAgent(
            name="SL_Bot",
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            instruction=SL_BOT_INSTRUCTION,
            tools=[web_search, itinerary_tool],
            generate_content_config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

        runner = Runner(
            app_name="sl-chatbot",
            agent=call_agent,
            session_service=self.session_service,
            auto_create_session=True
        )

        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    # Adjust silence duration to control how quickly the model 
                    # considers the user to have finished speaking. Lower = faster.
                    silence_duration_ms=600, 
                )
            ),
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
        Uses PyAV's AudioResampler to properly handle any input format/rate and
        convert to 16kHz mono s16 PCM as required by Gemini Live API.
        """
        resampler = AudioResampler(format='s16', layout='mono', rate=16000)
        _logged_format = False
        try:
            logger.info("Starting WhatsApp -> Gemini audio stream")
            while True:
                frame = await track.recv()

                # Log format once for debugging
                if not _logged_format:
                    logger.info(
                        f"Audio frame format: fmt={frame.format.name}, "
                        f"rate={frame.sample_rate}, layout={frame.layout.name}, "
                        f"shape={frame.to_ndarray().shape}, dtype={frame.to_ndarray().dtype}"
                    )
                    _logged_format = True

                # Use PyAV's AudioResampler: handles interleaved/planar, stereo/mono,
                # any sample rate -> 16kHz mono s16 in one step
                resampled_frames = resampler.resample(frame)
                for resampled in resampled_frames:
                    audio_data = resampled.to_ndarray()  # shape: (1, samples) s16 mono
                    audio_bytes = audio_data.tobytes()
                    if audio_bytes:
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
                    if event.input_transcription and event.input_transcription.text:
                        logger.info(f"User Transcribed: {event.input_transcription.text}")
                    
                    if event.output_transcription and event.output_transcription.text:
                        logger.info(f"Model Transcribed: {event.output_transcription.text}")

                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.inline_data and part.inline_data.data:
                                # Native audio models send audio in inline_data
                                if part.inline_data.mime_type and "audio" in part.inline_data.mime_type:
                                    audio_bytes = part.inline_data.data
                                    output_track.add_audio(audio_bytes)

                    if event.interrupted:
                        logger.info("Audio interrupted by user")
                        output_track.clear_buffer()

                except Exception as e:
                    logger.warning(f"Error processing Gemini event: {e}", exc_info=True)

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
