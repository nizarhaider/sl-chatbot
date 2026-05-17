import asyncio
import logging
import os
import time

from av.audio.resampler import AudioResampler
from google import genai
from google.genai import types

from app.voice_agent.gemini_turn_pipeline import (
    GEMINI_THINKING_LEVEL,
    GEMINI_TURN_GREETING,
    HOMELANDS_SYSTEM_PROMPT,
    TURN_GREETING_DELAY_SECONDS,
)

GEMINI_LLM_MODEL = os.environ.get("GEMINI_LLM_MODEL", "gemini-3.0-flash")

logger = logging.getLogger(__name__)

REALTIME_STT_MODEL = os.environ.get("REALTIME_STT_MODEL", "small")
REALTIME_STT_LANGUAGE = os.environ.get("REALTIME_STT_LANGUAGE", "").strip()
REALTIME_STT_DEVICE = os.environ.get("REALTIME_STT_DEVICE", "cuda")
REALTIME_STT_COMPUTE_TYPE = os.environ.get("REALTIME_STT_COMPUTE_TYPE", "float16")
REALTIME_STT_POST_SPEECH_SILENCE = float(os.environ.get("REALTIME_STT_POST_SPEECH_SILENCE", "0.55"))
REALTIME_STT_MIN_RECORDING_SECONDS = float(os.environ.get("REALTIME_STT_MIN_RECORDING_SECONDS", "0.45"))

REALTIME_TTS_ENGINE = os.environ.get("REALTIME_TTS_ENGINE", "system").strip().lower()
REALTIME_TTS_VOICE = os.environ.get("REALTIME_TTS_VOICE", "")
REALTIME_TTS_REF_TEXT = os.environ.get(
    "REALTIME_TTS_REF_TEXT",
    "ආයුබෝවන්, හෝම්ලෑන්ඩ්ස් වෙත ඔබව සාදරයෙන් පිළිගන්නවා.",
)
REALTIME_TTS_REF_LANGUAGE = os.environ.get("REALTIME_TTS_REF_LANGUAGE", "si")
REALTIME_TTS_NUM_STEPS = os.environ.get("REALTIME_TTS_NUM_STEPS", "12,12")


class RealtimeTurnPipeline:
    def __init__(self, prepare_tts_text, interrupt_playback):
        self._prepare_tts_text = prepare_tts_text
        self._interrupt_playback = interrupt_playback
        self._client = None
        self._tts_stream = None
        self._tts_sample_rate = 24000

    @property
    def client(self):
        if self._client is None:
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            self._client = genai.Client(api_key=api_key)
        return self._client

    async def run(self, call_id, input_track, output_track, playback_generation):
        if TURN_GREETING_DELAY_SECONDS:
            await asyncio.sleep(TURN_GREETING_DELAY_SECONDS)

        greeting_seconds = await self._speak(
            call_id,
            GEMINI_TURN_GREETING,
            output_track,
            playback_generation,
        )
        if greeting_seconds:
            logger.info("Protecting RealtimeSTT language prompt for %.2f seconds", greeting_seconds)
            await self._discard_input_audio(input_track, greeting_seconds + 0.25)

        loop = asyncio.get_running_loop()
        recorder = await asyncio.to_thread(self._create_recorder, call_id, output_track, loop)
        stop_event = asyncio.Event()
        feeder_task = asyncio.create_task(
            self._feed_recorder(call_id, input_track, recorder, stop_event),
            name=f"realtime-stt-feed-{call_id}",
        )
        transcript_history: list[tuple[str, str]] = [("assistant", GEMINI_TURN_GREETING)]

        try:
            while not stop_event.is_set():
                transcript_task = asyncio.create_task(
                    asyncio.to_thread(recorder.text),
                    name=f"realtime-stt-text-{call_id}",
                )
                stop_task = asyncio.create_task(stop_event.wait())
                done, pending = await asyncio.wait(
                    {transcript_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                if stop_task in done:
                    break

                started = time.perf_counter()
                transcript = (transcript_task.result() or "").strip()
                if not transcript:
                    continue
                logger.info("RealtimeSTT transcript for %s: %s", call_id, transcript)
                transcript_history.append(("user", transcript))

                llm_started = time.perf_counter()
                response_text = await self._respond_gemini(transcript_history)
                llm_ms = (time.perf_counter() - llm_started) * 1000.0
                if not response_text:
                    logger.info("Gemini returned empty response for %s in %.0f ms", call_id, llm_ms)
                    continue
                logger.info("Gemini response for %s in %.0f ms: %s", call_id, llm_ms, response_text)
                transcript_history.append(("assistant", response_text))

                tts_started = time.perf_counter()
                await self._speak(call_id, response_text, output_track, playback_generation)
                logger.info(
                    "Realtime turn timings for %s: response_total=%.0f ms llm=%.0f ms tts=%.0f ms",
                    call_id,
                    (time.perf_counter() - started) * 1000.0,
                    llm_ms,
                    (time.perf_counter() - tts_started) * 1000.0,
                )
        finally:
            stop_event.set()
            feeder_task.cancel()
            await asyncio.gather(feeder_task, return_exceptions=True)
            await asyncio.to_thread(recorder.shutdown)

    def _create_recorder(self, call_id, output_track, loop):
        from RealtimeSTT import AudioToTextRecorder

        def on_vad_start():
            loop.call_soon_threadsafe(self._interrupt_playback, call_id, output_track)

        return AudioToTextRecorder(
            use_microphone=False,
            model=REALTIME_STT_MODEL,
            language=REALTIME_STT_LANGUAGE,
            device=REALTIME_STT_DEVICE,
            compute_type=REALTIME_STT_COMPUTE_TYPE,
            post_speech_silence_duration=REALTIME_STT_POST_SPEECH_SILENCE,
            min_length_of_recording=REALTIME_STT_MIN_RECORDING_SECONDS,
            spinner=False,
            no_log_file=True,
            start_callback_in_new_thread=True,
            on_vad_start=on_vad_start,
        )

    async def _feed_recorder(self, call_id, input_track, recorder, stop_event: asyncio.Event):
        resampler = AudioResampler(format="s16", layout="mono", rate=16000)
        while not stop_event.is_set():
            try:
                frame = await input_track.recv()
            except Exception as exc:
                logger.info("RealtimeSTT input ended for %s: %s", call_id, exc)
                stop_event.set()
                return

            for resampled in resampler.resample(frame):
                audio_bytes = resampled.to_ndarray().tobytes()
                if audio_bytes:
                    recorder.feed_audio(audio_bytes, original_sample_rate=16000)

    async def _respond_gemini(self, transcript_history: list[tuple[str, str]]) -> str:
        conversation_lines = []
        for role, text in transcript_history[-10:]:
            prefix = "Caller" if role == "user" else "Agent"
            conversation_lines.append(f"{prefix}: {text}")
        prompt = "\n".join(conversation_lines)

        response = await self.client.aio.models.generate_content(
            model=GEMINI_LLM_MODEL,
            contents=("Reply to the latest caller message using the Homelands instructions.\n" f"{prompt}"),
            config=types.GenerateContentConfig(
                system_instruction=HOMELANDS_SYSTEM_PROMPT,
                temperature=0,
                max_output_tokens=160,
                thinking_config=(
                    types.ThinkingConfig(thinking_level=GEMINI_THINKING_LEVEL)
                    if GEMINI_THINKING_LEVEL
                    else None
                ),
            ),
        )
        return (response.text or "").strip()

    async def _speak(self, call_id, text, output_track, playback_generation):
        prepared = self._prepare_tts_text(text)
        if not prepared:
            return 0.0

        generation_id = playback_generation.get(call_id, 0)
        chunks: list[bytes] = []
        loop = asyncio.get_running_loop()

        def on_audio_chunk(chunk: bytes):
            if playback_generation.get(call_id, 0) != generation_id:
                return
            chunks.append(chunk)
            loop.call_soon_threadsafe(output_track.add_pcm_audio, chunk, self._tts_sample_rate)

        started = time.perf_counter()
        await asyncio.to_thread(self._play_tts, prepared, on_audio_chunk)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        audio_bytes = sum(len(chunk) for chunk in chunks)
        logger.info(
            "RealtimeTTS complete for %s: elapsed_ms=%.0f chars=%s audio_ms=%.0f chunks=%s",
            call_id,
            elapsed_ms,
            len(prepared),
            self._audio_duration_seconds(audio_bytes, self._tts_sample_rate) * 1000.0,
            len(chunks),
        )
        return self._audio_duration_seconds(audio_bytes, self._tts_sample_rate)

    def _play_tts(self, text: str, on_audio_chunk) -> None:
        stream = self._get_tts_stream()
        stream.feed(text)
        stream.play(muted=True, on_audio_chunk=on_audio_chunk)

    def _get_tts_stream(self):
        if self._tts_stream is not None:
            return self._tts_stream

        from RealtimeTTS import TextToAudioStream
        
        if REALTIME_TTS_ENGINE == "system":
            from RealtimeTTS import SystemEngine
            engine = SystemEngine(voice=REALTIME_TTS_VOICE if REALTIME_TTS_VOICE else None)
        elif REALTIME_TTS_ENGINE == "coqui":
            from RealtimeTTS import CoquiEngine
            engine = CoquiEngine(voice=REALTIME_TTS_VOICE if REALTIME_TTS_VOICE else "")
        elif REALTIME_TTS_ENGINE == "openai":
            from RealtimeTTS import OpenAIEngine
            engine = OpenAIEngine(voice=REALTIME_TTS_VOICE if REALTIME_TTS_VOICE else "nova")
        elif REALTIME_TTS_ENGINE == "edge":
            from RealtimeTTS import EdgeEngine
            engine = EdgeEngine(voice=REALTIME_TTS_VOICE if REALTIME_TTS_VOICE else "en-US-AriaNeural")
        else:
            raise RuntimeError(f"Unsupported REALTIME_TTS_ENGINE={REALTIME_TTS_ENGINE}")

        _, _, self._tts_sample_rate = engine.get_stream_info()
        self._tts_stream = TextToAudioStream(engine, muted=True)
        return self._tts_stream

    async def _discard_input_audio(self, input_track, duration_seconds: float) -> None:
        deadline = time.perf_counter() + max(0.0, duration_seconds)
        discarded_frames = 0
        while time.perf_counter() < deadline:
            try:
                timeout = max(0.01, deadline - time.perf_counter())
                await asyncio.wait_for(input_track.recv(), timeout=timeout)
                discarded_frames += 1
            except asyncio.TimeoutError:
                break
            except Exception as exc:
                logger.info("Realtime protected prompt input drain ended: %s", exc)
                break
        logger.info("Discarded %s inbound frames during Realtime protected prompt", discarded_frames)

    def _audio_duration_seconds(self, audio_bytes: int, sample_rate: int) -> float:
        if audio_bytes <= 0 or sample_rate <= 0:
            return 0.0
        return (audio_bytes / 2) / sample_rate
