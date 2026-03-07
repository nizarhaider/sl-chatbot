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
from google.adk.agents.run_config import RunConfig, StreamingMode, ContextWindowCompressionConfig
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from app.voice_agent.tools import web_search, send_whatsapp_status

logger = logging.getLogger(__name__)


# SLT Mobitel Sinhala call center agent instruction
SL_BOT_INSTRUCTION = (
    "**Persona:**\n"
    "You are Sam, a friendly and professional senior call center agent at SLT Mobitel, Sri Lanka. "
    "You are helpful, patient, and expert in troubleshooting. You always aim to solve customer inquiries "
    "promptly and ensure they feel valued. You only speak in formal Sinhala.\n\n"

    "**Conversational Rules:**\n"
    "RESPOND UNMISTAKABLY IN FORMAL SINHALA. YOU MUST RESPOND UNMISTAKABLY IN FORMAL SINHALA.\n\n"
    "1. **Greet**: Start with a warm greeting in Sinhala (e.g., ආයුබෝවන්! SLT Mobitel Customer Care වෙත ඔබව සාදරෙයන පිළිගන්නවා. මගේ නම Sam. අද ඔබට සහය වන්නෙ කෙසේද ?') and ask how you can help.\n"
    "2. **Issue Discussion**: Listen to the customer's issue. If it's a technical issue about SLT services, discuss it briefly to understand the core problem. DO NOT repeat what the client is saying back to them.\n"
    "3. **Mock Verification**: Ask for the customer's full name and their WhatsApp number for verification purposes.\n"
    "4. **Action**: Once the details are gathered and the issue is understood, invoke `send_whatsapp_status` with the customer's name, phone number, issue, and status set to 'ක්‍රියාත්මක වෙමින් පවතී' (Processing).\n"
    "5. **Confirmation & Next Steps**: Inform the customer to check their WhatsApp for the status message. Ask if there is anything else they need help with. Let this conversational loop continue as long as they have questions.\n"
    "6. **Close**: If they say 'thank you' or indicate they are finished, wish them a great day ('සුභ දවසක්!') and end the call.\n\n"

    "**General Guidelines:**\n"
    "Keep your responses short (2-3 sentences max) to maintain a natural voice flow. Provide net new information in every turn. "
    "If the customer speaks English, politely insist on continuing in Sinhala.\n\n"

    "**Guardrails:**\n"
    "Never provide personal contact details or internal SLT Mobitel employee information. "
    "If a client gets frustrated, remain calm and professional. Avoid using industry jargon; explain things in simple formal Sinhala."
)

# root_agent exposes this module to `adk web` for local testing (live audio mode)
root_agent = LlmAgent(
    name="SL_Bot",
    model="gemini-2.5-flash-native-audio-preview-12-2025",
    instruction=SL_BOT_INSTRUCTION,
    tools=[send_whatsapp_status, web_search],
    generate_content_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    ),
)

class VoiceAgent:
    def __init__(self):
        self.active_calls: dict[str, asyncio.Task] = {}

    async def process_audio(self, call_id: str, caller_phone: str, input_track: MediaStreamTrack, output_track: MediaStreamTrack):
        """
        Main loop for audio processing using Google ADK for Gemini Live API.
        Supports multiple concurrent calls, each with isolated session state.
        """
        if call_id in self.active_calls:
            logger.warning(f"Audio processing already active for {call_id}, cancelling old session first")
            self.active_calls[call_id].cancel()
            try:
                await self.active_calls[call_id]
            except (asyncio.CancelledError, Exception):
                pass

        # Each call gets its own fresh session service to avoid state bleed
        session_service = InMemorySessionService()

        # Initialize LlmAgent per call
        call_agent = LlmAgent(
            name="SL_Bot",
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            instruction=SL_BOT_INSTRUCTION,
            tools=[send_whatsapp_status, web_search],
            generate_content_config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

        runner = Runner(
            app_name="sl-chatbot",
            agent=call_agent,
            session_service=session_service,
            auto_create_session=True
        )

        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            # Optimize for long sessions as per best practices
            context_window_compression=ContextWindowCompressionConfig(),
            # Configure voice
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Achird"
                    )
                )
            )
        )

        live_request_queue = LiveRequestQueue()

        async def _run_call():
            try:
                logger.info(f"Starting Gemini Live session for call {call_id}")
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._whatsapp_to_gemini(input_track, live_request_queue), name=f"mic-{call_id}")
                    tg.create_task(self._gemini_to_whatsapp(runner, call_id, live_request_queue, run_config, output_track), name=f"speaker-{call_id}")
                    tg.create_task(self._send_greeting(live_request_queue), name=f"greeting-{call_id}")
            except* asyncio.CancelledError:
                pass
            except* Exception as eg:
                for exc in eg.exceptions:
                    logger.error(f"Error in call task for {call_id}: {exc}", exc_info=exc)
            finally:
                live_request_queue.close()
                self.active_calls.pop(call_id, None)
                logger.info(f"Cleaned up session for {call_id}")

        task = asyncio.create_task(_run_call(), name=f"call-{call_id}")
        self.active_calls[call_id] = task
        await task

    async def _send_greeting(self, live_request_queue: LiveRequestQueue):
        """
        Proactively triggers the agent to greet the user in Sinhala.
        """
        await asyncio.sleep(0.5) 
        logger.info("Sending proactive greeting trigger in Sinhala")
        live_request_queue.send_content(types.Content(
            role="user",
            parts=[types.Part(text=(
                "The call has just connected. Start the conversation in Sinhala. "
                "This is Sam from SLT Mobitel! How can I help you today?"
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
