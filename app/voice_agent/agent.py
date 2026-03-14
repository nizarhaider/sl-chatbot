import asyncio
import logging
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

from app.voice_agent.tools import web_search, send_whatsapp_status

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SL_BOT_INSTRUCTION = (
    "**Persona:** You are Sam, a friendly senior SLT Mobitel agent in formal Sinhala.\n"
    "Keep responses short, helpful, and polite.\n"
    "Always respond in formal Sinhala.\n"
)

root_agent = LlmAgent(
    name="SL_Bot",
    model="gemini-2.5-flash-native-audio-preview-12-2025",
    instruction=SL_BOT_INSTRUCTION,
    tools=[web_search],
)


class RealtimeAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate=24000):
        super().__init__()
        self.queue = asyncio.Queue()
        self._pts = 0
        self._sample_rate = sample_rate
        self._time_base = Fraction(1, self._sample_rate)
        self._samples_per_frame = 480  # 20ms frames
        self._buffer = b""
        self._start_time = None

    def clear_buffer(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._buffer = b""

    def add_audio(self, data: bytes):
        self.queue.put_nowait(data)

    async def recv(self):
        if self._start_time is None:
            self._start_time = asyncio.get_event_loop().time()
        next_frame_time = self._start_time + (self._pts / self._sample_rate)
        now = asyncio.get_event_loop().time()
        if next_frame_time > now:
            await asyncio.sleep(next_frame_time - now)

        target_size = self._samples_per_frame * 2  # 16-bit mono

        while not self.queue.empty():
            try:
                self._buffer += self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        if len(self._buffer) >= target_size:
            data_to_send = self._buffer[:target_size]
            self._buffer = self._buffer[target_size:]
        else:
            data_to_send = b'\x00' * target_size

        frame = AudioFrame(format='s16', layout='mono', samples=self._samples_per_frame)
        frame.planes[0].update(data_to_send)
        frame.pts = self._pts
        frame.sample_rate = self._sample_rate
        frame.time_base = self._time_base
        self._pts += self._samples_per_frame
        return frame


class VoiceAgent:
    def __init__(self):
        self.active_calls: dict[str, asyncio.Task] = {}
        self.greetings_sent: dict[str, bool] = {}

    async def process_audio(self, call_id: str, caller_phone: str, input_track: MediaStreamTrack, output_track: RealtimeAudioTrack):
        # Cancel previous session if exists
        if call_id in self.active_calls:
            task = self.active_calls.pop(call_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Previous call task {call_id} cancelled")
            except Exception as e:
                logger.error(f"Error cancelling previous task {call_id}: {e}")

        self.greetings_sent[call_id] = False
        session_service = InMemorySessionService()
        call_agent = LlmAgent(
            name="SL_Bot",
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            instruction=SL_BOT_INSTRUCTION,
            tools=[web_search],
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
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Erinome")
                )
            )
        )

        live_request_queue = LiveRequestQueue()

        async def _run_call():
            try:
                # Send greeting once
                if not self.greetings_sent[call_id]:
                    live_request_queue.send_content(types.Content(
                        role="user",
                        parts=[types.Part(text="The call has just connected. Start in Sinhala. This is Sam from SLT Mobitel! How can I help you today?")]
                    ))
                    self.greetings_sent[call_id] = True

                await asyncio.gather(
                    self._whatsapp_to_gemini(input_track, live_request_queue),
                    self._gemini_to_whatsapp(runner, call_id, live_request_queue, run_config, output_track)
                )
            except asyncio.CancelledError:
                logger.info(f"Call {call_id} cancelled")
            except Exception as e:
                logger.error(f"Unexpected error in call {call_id}: {e}")
            finally:
                live_request_queue.close()
                self.active_calls.pop(call_id, None)
                self.greetings_sent.pop(call_id, None)
                logger.info(f"Cleaned up session for {call_id}")

        task = asyncio.create_task(_run_call(), name=f"call-{call_id}")
        self.active_calls[call_id] = task
        await task

    async def _whatsapp_to_gemini(self, track: MediaStreamTrack, live_request_queue: LiveRequestQueue):
        resampler = AudioResampler(format='s16', layout='mono', rate=16000)
        try:
            while True:
                frame = await track.recv()
                resampled_frames = resampler.resample(frame)
                for resampled in resampled_frames:
                    audio_bytes = resampled.to_ndarray().tobytes()
                    if audio_bytes:
                        blob = types.Blob(mime_type="audio/pcm;rate=16000", data=audio_bytes)
                        live_request_queue.send_realtime(blob)
        except asyncio.CancelledError:
            logger.info("WhatsApp -> Gemini stream cancelled")
        except Exception as e:
            logger.info(f"WhatsApp -> Gemini stream ended: {e}")

    async def _gemini_to_whatsapp(self, runner: Runner, call_id: str, live_request_queue: LiveRequestQueue, run_config: RunConfig, output_track: RealtimeAudioTrack):
        """
        Streams partial TTS immediately, avoiding repeated audio.
        """
        try:
            sent_audio = 0  # tracks already sent bytes
            async for event in runner.run_live(
                user_id="whatsapp_user",
                session_id=call_id,
                live_request_queue=live_request_queue,
                run_config=run_config
            ):
                if event.input_transcription and event.input_transcription.text:
                    logger.info(f"User Transcribed: {event.input_transcription.text}")

                if event.output_transcription and event.output_transcription.text:
                    logger.info(f"Model Transcribed: {event.output_transcription.text}")

                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.inline_data and part.inline_data.data:
                            audio_bytes = part.inline_data.data
                            # Send only new audio
                            if len(audio_bytes) > sent_audio:
                                new_bytes = audio_bytes[sent_audio:]
                                output_track.add_audio(new_bytes)
                                sent_audio = len(audio_bytes)

                if event.interrupted:
                    output_track.clear_buffer()
                    sent_audio = 0
        except asyncio.CancelledError:
            logger.info(f"Gemini -> WhatsApp stream for call {call_id} cancelled")
        except Exception as e:
            logger.info(f"Gemini -> WhatsApp stream ended for call {call_id}: {e}")


voice_agent = VoiceAgent()