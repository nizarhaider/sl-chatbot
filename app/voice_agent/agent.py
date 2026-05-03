import asyncio
import logging
from datetime import datetime
from fractions import Fraction
from aiortc import MediaStreamTrack
from av import AudioFrame
from av.audio.resampler import AudioResampler
import numpy as np

from google.adk.runners import Runner
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from app.services.tts import get_tts_service
from app.voice_agent.tools import web_search, send_whatsapp_status

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SL_BOT_INSTRUCTION = (
    "**Persona:** You are Sam, a friendly senior SLT Mobitel agent. Speak fast with a fast Sri Lankan accent \n"
    "**Task 1 (Language Selection):** At the beginning of the call, ask the user to mention their preferred language. You must say exactly:\n"
    '"සිංහලෙන් කතා කිරීමට සිංහල කියන්න. தமிழ் பேசுவதற்கு தமிழ் என்று கூறவும். For English, please say English."\n'
    "Wait for the user to mention their preferred language.\n"
    "**Task 2 (Assistance):** Once the user selects a language, smoothly transition into a helpful, polite customer service agent in that chosen language for the rest of the call.\n"
    "Keep responses short, helpful, and professional."
)

root_agent = LlmAgent(
    name="SL_Bot",
    model="gemini-2.5-flash-native-audio-preview-09-2025",
    instruction=SL_BOT_INSTRUCTION,
    tools=[web_search, send_whatsapp_status],
)


class RealtimeAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate=24000):
        super().__init__()
        self.queue = asyncio.Queue()
        self._pts = 0
        self._sample_rate = sample_rate
        self._time_base = Fraction(1, self._sample_rate)
        self._samples_per_frame = 480
        self._buffer = b""
        self._start_time = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_size_bytes(self) -> int:
        return self._samples_per_frame * 2

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

        target_size = self._samples_per_frame * 2

        while not self.queue.empty():
            try:
                self._buffer += self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        if len(self._buffer) >= target_size:
            data_to_send = self._buffer[:target_size]
            self._buffer = self._buffer[target_size:]
        else:
            data_to_send = b"\x00" * target_size

        frame = AudioFrame(format="s16", layout="mono", samples=self._samples_per_frame)
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
        self.playback_generation: dict[str, int] = {}
        self.tts_service = get_tts_service()

    async def process_audio(
        self,
        call_id: str,
        caller_phone: str,
        input_track: MediaStreamTrack,
        output_track: RealtimeAudioTrack,
    ):
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
        self.playback_generation[call_id] = 0
        session_service = InMemorySessionService()
        call_agent = LlmAgent(
            name="SL_Bot",
            model="gemini-2.5-flash-native-audio-preview-09-2025",
            instruction=SL_BOT_INSTRUCTION,
            tools=[web_search, send_whatsapp_status],
        )

        runner = Runner(
            app_name="sl-chatbot",
            agent=call_agent,
            session_service=session_service,
            auto_create_session=True,
        )

        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=(
                ["AUDIO"] if self.tts_service.uses_gemini_audio() else ["TEXT"]
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=(
                types.AudioTranscriptionConfig()
                if self.tts_service.uses_gemini_audio()
                else None
            ),
            speech_config=(
                types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="zephyr"
                        )
                    )
                )
                if self.tts_service.uses_gemini_audio()
                else None
            ),
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=4000,
                sliding_window=types.SlidingWindow(target_tokens=2000)
            ),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=True
                )
            ),
        )

        live_request_queue = LiveRequestQueue()

        async def _run_call():
            try:
                if not self.greetings_sent[call_id]:
                    live_request_queue.send_content(
                        types.Content(
                            role="user",
                            parts=[
                                types.Part(
                                    text="The call has just connected. Start by speaking the exact language selection menu provided in your instructions."
                                )
                            ],
                        )
                    )
                    self.greetings_sent[call_id] = True

                await asyncio.gather(
                    self._whatsapp_to_gemini(
                        call_id,
                        input_track,
                        output_track,
                        live_request_queue,
                    ),
                    (
                        self._gemini_audio_to_whatsapp(
                            runner, call_id, live_request_queue, run_config, output_track
                        )
                        if self.tts_service.uses_gemini_audio()
                        else self._gemini_text_to_whatsapp(
                            runner, call_id, live_request_queue, run_config, output_track
                        )
                    ),
                )
            except asyncio.CancelledError:
                logger.info(f"Call {call_id} cancelled")
            except Exception as e:
                logger.error(f"Unexpected error in call {call_id}: {e}")
            finally:
                live_request_queue.close()
                self.active_calls.pop(call_id, None)
                self.greetings_sent.pop(call_id, None)
                self.playback_generation.pop(call_id, None)
                logger.info(f"Cleaned up session for {call_id}")

        task = asyncio.create_task(_run_call(), name=f"call-{call_id}")
        self.active_calls[call_id] = task
        await task

    async def _whatsapp_to_gemini(
        self,
        call_id: str,
        track: MediaStreamTrack,
        output_track: RealtimeAudioTrack,
        live_request_queue: LiveRequestQueue,
    ):
        resampler = AudioResampler(format="s16", layout="mono", rate=16000)
        buffer = bytearray()
        # 16kHz * 2 bytes/sample (16-bit) * 0.1 seconds = 3200 bytes
        CHUNK_SIZE = 3200  
        
        is_speaking = False
        silence_threshold = 1000  # RMS threshold
        silence_duration_chunks = 0
        MAX_SILENCE_CHUNKS = 8  # ~800ms of silence to trigger end
        
        try:
            while True:
                frame = await track.recv()
                resampled_frames = resampler.resample(frame)
                for resampled in resampled_frames:
                    audio_bytes = resampled.to_ndarray().tobytes()
                    if audio_bytes:
                        buffer.extend(audio_bytes)
                        while len(buffer) >= CHUNK_SIZE:
                            chunk = bytes(buffer[:CHUNK_SIZE])
                            del buffer[:CHUNK_SIZE]
                            
                            # Simple VAD using RMS
                            audio_np = np.frombuffer(chunk, dtype=np.int16)
                            rms = np.sqrt(np.mean(audio_np.astype(np.float64)**2))
                            
                            if rms > silence_threshold:
                                if not is_speaking:
                                    logger.info("VAD: Speech started")
                                    is_speaking = True
                                    self._interrupt_playback(call_id, output_track)
                                    live_request_queue.send_activity_start()
                                silence_duration_chunks = 0
                            else:
                                if is_speaking:
                                    silence_duration_chunks += 1
                                    if silence_duration_chunks >= MAX_SILENCE_CHUNKS:
                                        logger.info("VAD: Speech ended")
                                        is_speaking = False
                                        silence_duration_chunks = 0
                                        live_request_queue.send_activity_end()
                            
                            blob = types.Blob(
                                mime_type="audio/pcm;rate=16000", data=chunk
                            )
                            live_request_queue.send_realtime(blob)
        except asyncio.CancelledError:
            logger.info("WhatsApp -> Gemini stream cancelled")
        except Exception as e:
            logger.info(f"WhatsApp -> Gemini stream ended: {e}")

    async def _gemini_audio_to_whatsapp(
        self,
        runner: Runner,
        call_id: str,
        live_request_queue: LiveRequestQueue,
        run_config: RunConfig,
        output_track: RealtimeAudioTrack,
    ):
        """
        Streams Gemini audio continuously, without skipping any word or repeating.
        """
        try:
            async for event in runner.run_live(
                user_id="whatsapp_user",
                session_id=call_id,
                live_request_queue=live_request_queue,
                run_config=run_config,
            ):
                if event.input_transcription and event.input_transcription.text:
                    time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    logger.info(
                        f"[{time_str}] User Transcribed: {event.input_transcription.text}"
                    )

                if event.output_transcription and event.output_transcription.text:
                    time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    logger.info(
                        f"[{time_str}] Model Transcribed: {event.output_transcription.text}"
                    )

                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.inline_data and part.inline_data.data:
                            # Send each audio chunk immediately
                            output_track.add_audio(part.inline_data.data)

                if event.interrupted:
                    self._interrupt_playback(call_id, output_track)
        except asyncio.CancelledError:
            logger.info(f"Gemini -> WhatsApp stream for call {call_id} cancelled")
        except Exception as e:
            logger.info(f"Gemini -> WhatsApp stream ended for call {call_id}: {e}")

    async def _gemini_text_to_whatsapp(
        self,
        runner: Runner,
        call_id: str,
        live_request_queue: LiveRequestQueue,
        run_config: RunConfig,
        output_track: RealtimeAudioTrack,
    ):
        latest_response_text = ""
        try:
            async for event in runner.run_live(
                user_id="whatsapp_user",
                session_id=call_id,
                live_request_queue=live_request_queue,
                run_config=run_config,
            ):
                if event.input_transcription and event.input_transcription.text:
                    time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    logger.info(
                        f"[{time_str}] User Transcribed: {event.input_transcription.text}"
                    )

                if event.content and event.content.parts:
                    text_parts = [
                        part.text.strip()
                        for part in event.content.parts
                        if part.text and part.text.strip()
                    ]
                    if text_parts and not event.partial:
                        latest_response_text = " ".join(text_parts)
                        time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        logger.info(
                            f"[{time_str}] Model Text: {latest_response_text}"
                        )

                if event.turn_complete and latest_response_text:
                    await self._speak_text_response(
                        call_id=call_id,
                        text=latest_response_text,
                        output_track=output_track,
                    )
                    latest_response_text = ""

                if event.interrupted:
                    self._interrupt_playback(call_id, output_track)
                    latest_response_text = ""
        except asyncio.CancelledError:
            logger.info(f"Gemini text -> WhatsApp stream for call {call_id} cancelled")
        except Exception as e:
            logger.info(f"Gemini text -> WhatsApp stream ended for call {call_id}: {e}")

    async def _speak_text_response(
        self,
        call_id: str,
        text: str,
        output_track: RealtimeAudioTrack,
    ):
        generation_id = self.playback_generation.get(call_id, 0)
        synthesized = await self.tts_service.synthesize(text)
        if self.playback_generation.get(call_id, 0) != generation_id:
            logger.info("Discarding stale TTS output for %s", call_id)
            return

        frame_size = output_track.frame_size_bytes
        for offset in range(0, len(synthesized.pcm), frame_size):
            if self.playback_generation.get(call_id, 0) != generation_id:
                logger.info("Stopping interrupted TTS playback for %s", call_id)
                return
            output_track.add_audio(synthesized.pcm[offset:offset + frame_size])

    def _interrupt_playback(
        self,
        call_id: str | None,
        output_track: RealtimeAudioTrack | None,
    ) -> None:
        if call_id is not None:
            self.playback_generation[call_id] = self.playback_generation.get(call_id, 0) + 1
        if output_track is not None:
            output_track.clear_buffer()


voice_agent = VoiceAgent()
