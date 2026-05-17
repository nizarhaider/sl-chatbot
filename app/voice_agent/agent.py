import asyncio
import logging
import os
import re
import time
from datetime import datetime
from dataclasses import dataclass
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
from app.voice_agent.gemma_audio_turn_pipeline import GemmaAudioTurnPipeline
from app.voice_agent.gemini_turn_pipeline import GeminiTurnPipeline
from app.voice_agent.realtime_turn_pipeline import RealtimeTurnPipeline
from app.voice_agent.tools import web_search, send_whatsapp_status

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
VOICE_INPUT_SAMPLE_RATE = 16000
VOICE_INPUT_BYTES_PER_SAMPLE = 2
VOICE_INPUT_CHUNK_MS = max(20, int(os.environ.get("VOICE_INPUT_CHUNK_MS", "20")))
VOICE_INPUT_CHUNK_SIZE = (
    VOICE_INPUT_SAMPLE_RATE * VOICE_INPUT_BYTES_PER_SAMPLE * VOICE_INPUT_CHUNK_MS
) // 1000
VOICE_SILENCE_THRESHOLD = int(os.environ.get("VOICE_SILENCE_THRESHOLD", "1000"))
VOICE_END_SILENCE_CHUNKS = max(2, int(os.environ.get("VOICE_END_SILENCE_CHUNKS", "5")))
GEMINI_LIVE_USE_AUTOMATIC_VAD = os.environ.get(
    "GEMINI_LIVE_USE_AUTOMATIC_VAD",
    "true",
).strip().lower() in {"1", "true", "yes", "on"}
GEMINI_LIVE_VAD_PREFIX_PADDING_MS = int(
    os.environ.get("GEMINI_LIVE_VAD_PREFIX_PADDING_MS", "20")
)
GEMINI_LIVE_VAD_SILENCE_MS = int(os.environ.get("GEMINI_LIVE_VAD_SILENCE_MS", "100"))
GEMINI_LIVE_ENABLE_INPUT_TRANSCRIPTION = os.environ.get(
    "GEMINI_LIVE_ENABLE_INPUT_TRANSCRIPTION",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}
GEMINI_LIVE_ENABLE_OUTPUT_TRANSCRIPTION = os.environ.get(
    "GEMINI_LIVE_ENABLE_OUTPUT_TRANSCRIPTION",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}
GEMINI_LIVE_ENABLE_TOOLS = os.environ.get(
    "GEMINI_LIVE_ENABLE_TOOLS",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}
GEMINI_LIVE_AUDIO_MODEL = os.environ.get(
    "GEMINI_LIVE_AUDIO_MODEL",
    "gemini-2.5-flash-native-audio-preview-12-2025",
)
GEMINI_LIVE_TEXT_MODEL = os.environ.get(
    "GEMINI_LIVE_TEXT_MODEL",
    "gemini-2.5-flash-native-audio-preview-12-2025",
)
GEMINI_LIVE_AUDIO_FALLBACK_FOR_CALLS = os.environ.get(
    "GEMINI_LIVE_AUDIO_FALLBACK_FOR_CALLS",
    "true",
).strip().lower() in {"1", "true", "yes", "on"}
VOICE_PIPELINE_MODE = os.environ.get("VOICE_PIPELINE_MODE", "live").strip().lower()

SL_BOT_INSTRUCTION = (
    "You are a call center agent for Homelands, a Sri Lankan property business. "
    "At the beginning of the call, ask the customer exactly: "
    '"සිංහලෙන් කතා කිරීමට සිංහල කියන්න. தமிழ் பேசுவதற்கு தமிழ் என்று கூறவும். For English, please say English." '
    "After the customer chooses English, Sinhala, or Tamil, continue naturally in that language. "
    "Help with property inquiries using these mock properties only: "
    "Horizon Residencies, Malabe: two-bedroom apartments from LKR 28 million. "
    "Lakeview Villas, Piliyandala: three-bedroom villas from LKR 48 million. "
    "Green Acres, Kurunegala: ten-perch residential land from LKR 9.5 million. "
    "Ocean Breeze Apartments, Dehiwala: one and two-bedroom units from LKR 32 million. "
    "When the customer asks about properties, recommend a suitable option and tell them you have scheduled "
    "an appointment with a Homelands consultant for tomorrow at 10 AM. "
    "Keep responses brief and natural for a phone call."
)

root_agent = LlmAgent(
    name="SL_Bot",
    model=GEMINI_LIVE_AUDIO_MODEL,
    instruction=SL_BOT_INSTRUCTION,
    tools=[web_search, send_whatsapp_status] if GEMINI_LIVE_ENABLE_TOOLS else [],
)


@dataclass
class TurnLatencyTracker:
    speech_started_at: float | None = None
    speech_ended_at: float | None = None
    response_started_at: float | None = None

    def mark_speech_started(self) -> None:
        self.speech_started_at = time.perf_counter()
        self.response_started_at = None

    def mark_speech_ended(self) -> None:
        self.speech_ended_at = time.perf_counter()

    def mark_response_started(self) -> float | None:
        if self.speech_ended_at is None or self.response_started_at is not None:
            return None
        self.response_started_at = time.perf_counter()
        return (self.response_started_at - self.speech_ended_at) * 1000.0

    def reset(self) -> None:
        self.speech_started_at = None
        self.speech_ended_at = None
        self.response_started_at = None


class RealtimeAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate=48000):
        super().__init__()
        self.queue = asyncio.Queue()
        self._pts = 0
        self._sample_rate = sample_rate
        self._channels = 2
        self._layout = "stereo"
        self._time_base = Fraction(1, self._sample_rate)
        self._samples_per_frame = self._sample_rate // 50
        self._buffer = b""
        self._start_time = None
        self._logged_non_silent_frames = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_size_bytes(self) -> int:
        return self._samples_per_frame * self._channels * 2

    def clear_buffer(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._buffer = b""

    def add_audio(self, data: bytes):
        self.queue.put_nowait(data)

    def add_pcm_audio(self, pcm: bytes, sample_rate: int):
        if not pcm:
            return
        output_pcm = pcm
        input_audio = np.frombuffer(pcm, dtype=np.int16)
        input_rms = float(np.sqrt(np.mean(input_audio.astype(np.float64) ** 2))) if input_audio.size else 0.0

        if sample_rate != self._sample_rate:
            input_array = input_audio.reshape(1, -1)
            frame = AudioFrame.from_ndarray(
                input_array,
                format="s16",
                layout="mono",
            )
            frame.sample_rate = sample_rate
            frame.time_base = Fraction(1, sample_rate)

            resampler = AudioResampler(
                format="s16",
                layout="mono",
                rate=self._sample_rate,
            )
            chunks = []
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().tobytes())
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray().tobytes())
            output_pcm = b"".join(chunks)

        mono = np.frombuffer(output_pcm, dtype=np.int16)
        stereo = np.repeat(mono[:, None], self._channels, axis=1)
        output_bytes = stereo.astype(np.int16).tobytes()
        output_rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) if mono.size else 0.0
        logger.info(
            "Queued outbound PCM: input_rate=%s input_bytes=%s input_rms=%.1f output_rate=%s layout=%s output_bytes=%s output_rms=%.1f",
            sample_rate,
            len(pcm),
            input_rms,
            self._sample_rate,
            self._layout,
            len(output_bytes),
            output_rms,
        )
        self.add_audio(output_bytes)

    async def recv(self):
        if self._start_time is None:
            self._start_time = asyncio.get_event_loop().time()
        next_frame_time = self._start_time + (self._pts / self._sample_rate)
        now = asyncio.get_event_loop().time()
        if next_frame_time > now:
            await asyncio.sleep(next_frame_time - now)

        target_size = self.frame_size_bytes

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

        if self._logged_non_silent_frames < 5 and data_to_send.strip(b"\x00"):
            audio = np.frombuffer(data_to_send, dtype=np.int16)
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if audio.size else 0.0
            logger.info(
                "Emitting outbound audio frame: rate=%s layout=%s bytes=%s rms=%.1f buffered=%s queued_chunks=%s",
                self._sample_rate,
                self._layout,
                len(data_to_send),
                rms,
                len(self._buffer),
                self.queue.qsize(),
            )
            self._logged_non_silent_frames += 1

        audio = np.frombuffer(data_to_send, dtype=np.int16).reshape(1, -1)
        frame = AudioFrame.from_ndarray(
            audio,
            format="s16",
            layout=self._layout,
        )
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
        self.turn_latency: dict[str, TurnLatencyTracker] = {}
        self.tts_service = get_tts_service()
        self.turn_pipeline = GeminiTurnPipeline(
            tts_service=self.tts_service,
            prepare_tts_text=self._prepare_tts_text,
            interrupt_playback=self._interrupt_playback,
        )
        self.gemma_audio_pipeline = GemmaAudioTurnPipeline(
            tts_service=self.tts_service,
            prepare_tts_text=self._prepare_tts_text,
            interrupt_playback=self._interrupt_playback,
        )
        self.realtime_turn_pipeline = RealtimeTurnPipeline(
            prepare_tts_text=self._prepare_tts_text,
            interrupt_playback=self._interrupt_playback,
        )

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
        self.turn_latency[call_id] = TurnLatencyTracker()
        if VOICE_PIPELINE_MODE == "gemini_turn":
            task = asyncio.create_task(
                self._run_gemini_turn_pipeline(call_id, input_track, output_track),
                name=f"call-{call_id}",
            )
            self.active_calls[call_id] = task
            await task
            return
        if VOICE_PIPELINE_MODE == "gemma_audio_turn":
            task = asyncio.create_task(
                self._run_gemma_audio_turn_pipeline(call_id, input_track, output_track),
                name=f"call-{call_id}",
            )
            self.active_calls[call_id] = task
            await task
            return
        if VOICE_PIPELINE_MODE == "realtime_turn":
            task = asyncio.create_task(
                self._run_realtime_turn_pipeline(call_id, input_track, output_track),
                name=f"call-{call_id}",
            )
            self.active_calls[call_id] = task
            await task
            return
        use_gemini_audio_for_call = self._uses_gemini_audio_for_call()
        live_model = (
            GEMINI_LIVE_AUDIO_MODEL if use_gemini_audio_for_call else GEMINI_LIVE_TEXT_MODEL
        )
        session_service = InMemorySessionService()
        call_agent = LlmAgent(
            name="SL_Bot",
            model=live_model,
            instruction=SL_BOT_INSTRUCTION,
            tools=[web_search, send_whatsapp_status] if GEMINI_LIVE_ENABLE_TOOLS else [],
        )

        runner = Runner(
            app_name="sl-chatbot",
            agent=call_agent,
            session_service=session_service,
            auto_create_session=True,
        )

        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=(["AUDIO"] if use_gemini_audio_for_call else ["TEXT"]),
            input_audio_transcription=(
                types.AudioTranscriptionConfig()
                if GEMINI_LIVE_ENABLE_INPUT_TRANSCRIPTION
                else None
            ),
            output_audio_transcription=(
                types.AudioTranscriptionConfig()
                if use_gemini_audio_for_call and GEMINI_LIVE_ENABLE_OUTPUT_TRANSCRIPTION
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
                if use_gemini_audio_for_call
                else None
            ),
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=4000,
                sliding_window=types.SlidingWindow(target_tokens=2000)
            ),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=not GEMINI_LIVE_USE_AUTOMATIC_VAD,
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                    prefix_padding_ms=GEMINI_LIVE_VAD_PREFIX_PADDING_MS,
                    silence_duration_ms=GEMINI_LIVE_VAD_SILENCE_MS,
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
                        if use_gemini_audio_for_call
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
                self.turn_latency.pop(call_id, None)
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
        is_speaking = False
        silence_duration_chunks = 0
        tracker = self.turn_latency.setdefault(call_id, TurnLatencyTracker())

        try:
            while True:
                frame = await track.recv()
                resampled_frames = resampler.resample(frame)
                for resampled in resampled_frames:
                    audio_bytes = resampled.to_ndarray().tobytes()
                    if audio_bytes:
                        buffer.extend(audio_bytes)
                        while len(buffer) >= VOICE_INPUT_CHUNK_SIZE:
                            chunk = bytes(buffer[:VOICE_INPUT_CHUNK_SIZE])
                            del buffer[:VOICE_INPUT_CHUNK_SIZE]

                            # Simple VAD using RMS
                            audio_np = np.frombuffer(chunk, dtype=np.int16)
                            rms = np.sqrt(np.mean(audio_np.astype(np.float64) ** 2))

                            if rms > VOICE_SILENCE_THRESHOLD:
                                if not is_speaking:
                                    logger.info("VAD: Speech started")
                                    is_speaking = True
                                    tracker.mark_speech_started()
                                    self._interrupt_playback(call_id, output_track)
                                    if not GEMINI_LIVE_USE_AUTOMATIC_VAD:
                                        live_request_queue.send_activity_start()
                                silence_duration_chunks = 0
                            else:
                                if is_speaking:
                                    silence_duration_chunks += 1
                                    if silence_duration_chunks >= VOICE_END_SILENCE_CHUNKS:
                                        logger.info("VAD: Speech ended")
                                        is_speaking = False
                                        silence_duration_chunks = 0
                                        tracker.mark_speech_ended()
                                        if not GEMINI_LIVE_USE_AUTOMATIC_VAD:
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
                    self._note_turn_response_start(
                        call_id,
                        source="model transcript",
                    )
                    time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    logger.info(
                        f"[{time_str}] Model Transcribed: {event.output_transcription.text}"
                    )

                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.inline_data and part.inline_data.data:
                            self._note_turn_response_start(
                                call_id,
                                source="model audio",
                            )
                            output_track.add_pcm_audio(
                                part.inline_data.data,
                                sample_rate=24000,
                            )

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
                        self._note_turn_response_start(
                            call_id,
                            source="model text",
                        )
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
        prepared_text = self._prepare_tts_text(text)
        if not prepared_text:
            return
        synthesized = await self.tts_service.synthesize(prepared_text)
        if self.playback_generation.get(call_id, 0) != generation_id:
            logger.info("Discarding stale TTS output for %s", call_id)
            return

        if self.playback_generation.get(call_id, 0) != generation_id:
            logger.info("Stopping interrupted TTS playback for %s", call_id)
            return
        output_track.add_pcm_audio(synthesized.pcm, synthesized.sample_rate)

    def _interrupt_playback(
        self,
        call_id: str | None,
        output_track: RealtimeAudioTrack | None,
    ) -> None:
        if call_id is not None:
            self.playback_generation[call_id] = self.playback_generation.get(call_id, 0) + 1
        if output_track is not None:
            output_track.clear_buffer()

    def _note_turn_response_start(self, call_id: str, source: str) -> None:
        tracker = self.turn_latency.get(call_id)
        if tracker is None:
            return
        latency_ms = tracker.mark_response_started()
        if latency_ms is None:
            return
        logger.info(
            "Turn latency for %s: first %s in %.0f ms after speech end",
            call_id,
            source,
            latency_ms,
        )

    def _prepare_tts_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        return cleaned.rstrip(",;:").strip()

    def _uses_gemini_audio_for_call(self) -> bool:
        return self.tts_service.uses_gemini_audio() or GEMINI_LIVE_AUDIO_FALLBACK_FOR_CALLS

    async def _run_gemini_turn_pipeline(self, call_id, input_track, output_track):
        try:
            await self.turn_pipeline.run(
                call_id=call_id,
                input_track=input_track,
                output_track=output_track,
                playback_generation=self.playback_generation,
            )
        except asyncio.CancelledError:
            logger.info("Gemini turn pipeline cancelled for %s", call_id)
        except Exception as exc:
            logger.error("Gemini turn pipeline failed for %s: %s", call_id, exc, exc_info=True)
        finally:
            self.active_calls.pop(call_id, None)
            self.greetings_sent.pop(call_id, None)
            self.playback_generation.pop(call_id, None)
            self.turn_latency.pop(call_id, None)
            logger.info(f"Cleaned up session for {call_id}")

    async def _run_gemma_audio_turn_pipeline(self, call_id, input_track, output_track):
        try:
            await self.gemma_audio_pipeline.run(
                call_id=call_id,
                input_track=input_track,
                output_track=output_track,
                playback_generation=self.playback_generation,
            )
        except asyncio.CancelledError:
            logger.info("Gemma audio turn pipeline cancelled for %s", call_id)
        except Exception as exc:
            logger.error("Gemma audio turn pipeline failed for %s: %s", call_id, exc, exc_info=True)
        finally:
            self.active_calls.pop(call_id, None)
            self.greetings_sent.pop(call_id, None)
            self.playback_generation.pop(call_id, None)
            self.turn_latency.pop(call_id, None)
            logger.info(f"Cleaned up session for {call_id}")

    async def _run_realtime_turn_pipeline(self, call_id, input_track, output_track):
        try:
            await self.realtime_turn_pipeline.run(
                call_id=call_id,
                input_track=input_track,
                output_track=output_track,
                playback_generation=self.playback_generation,
            )
        except asyncio.CancelledError:
            logger.info("Realtime turn pipeline cancelled for %s", call_id)
        except Exception as exc:
            logger.error("Realtime turn pipeline failed for %s: %s", call_id, exc, exc_info=True)
        finally:
            self.active_calls.pop(call_id, None)
            self.greetings_sent.pop(call_id, None)
            self.playback_generation.pop(call_id, None)
            self.turn_latency.pop(call_id, None)
            logger.info(f"Cleaned up session for {call_id}")


voice_agent = VoiceAgent()
