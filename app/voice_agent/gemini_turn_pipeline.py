import asyncio
import logging
import os
import time
import wave
from io import BytesIO

import numpy as np
from av.audio.resampler import AudioResampler
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

GEMINI_STT_MODEL = os.environ.get("GEMINI_STT_MODEL", "gemini-2.5-flash-lite")
GEMINI_LLM_MODEL = os.environ.get("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite")
GEMINI_THINKING_LEVEL = os.environ.get("GEMINI_THINKING_LEVEL", "").strip().lower()

TURN_INPUT_CHUNK_MS = max(20, int(os.environ.get("TURN_INPUT_CHUNK_MS", "40")))
TURN_INPUT_CHUNK_SIZE = (16000 * 2 * TURN_INPUT_CHUNK_MS) // 1000
TURN_SILENCE_THRESHOLD = int(os.environ.get("TURN_SILENCE_THRESHOLD", "1000"))
TURN_END_SILENCE_CHUNKS = max(2, int(os.environ.get("TURN_END_SILENCE_CHUNKS", "20")))
TURN_GREETING_DELAY_SECONDS = max(
    0.0,
    float(os.environ.get("TURN_GREETING_DELAY_SECONDS", "1.2")),
)
TURN_GREETING_PROTECTION_MAX_SECONDS = max(
    0.0,
    float(os.environ.get("TURN_GREETING_PROTECTION_MAX_SECONDS", "1.5")),
)

GEMINI_TURN_GREETING = os.environ.get(
    "GEMINI_TURN_GREETING",
    "Hello, this is Homelands. Please say Sinhala, Tamil, or English.",
)
HOMELANDS_PROPERTIES = (
    "1. Horizon Residencies, Malabe: two-bedroom apartments from LKR 28 million, near schools and supermarkets. "
    "2. Lakeview Villas, Piliyandala: three-bedroom villas from LKR 48 million, garden, parking, and lake access. "
    "3. Green Acres, Kurunegala: ten-perch residential land from LKR 9.5 million, clear title, bank loans supported. "
    "4. Ocean Breeze Apartments, Dehiwala: one and two-bedroom units from LKR 32 million, sea view, ready soon."
)
HOMELANDS_SYSTEM_PROMPT = (
    "You are a call center agent for Homelands, a Sri Lankan property business. "
    "The first assistant message has already asked the caller to choose Sinhala, Tamil, or English; do not repeat it. "
    "After the customer chooses English, Sinhala, or Tamil, continue naturally in that language. "
    "If they only say a language name, briefly acknowledge and ask what property type, location, or budget they prefer. "
    f"Help with property inquiries using these mock properties only: {HOMELANDS_PROPERTIES} "
    "When the customer asks about properties, recommend a suitable option and tell them you have scheduled "
    "an appointment with a Homelands consultant for tomorrow at 10 AM. "
    "Keep each response brief and natural for a phone call, usually one short sentence."
)

REALTIME_TTS_REF_AUDIO = os.environ.get(
    "REALTIME_TTS_REF_AUDIO",
    "app/voices/sample_si_lk.mp3",
)
REALTIME_TTS_REF_TEXT = os.environ.get(
    "REALTIME_TTS_REF_TEXT",
    "ආයුබෝවන්, හෝම්ලෑන්ඩ්ස් වෙත ඔබව සාදරයෙන් පිළිගන්නවා.",
)
REALTIME_TTS_REF_LANGUAGE = os.environ.get("REALTIME_TTS_REF_LANGUAGE", "si")
REALTIME_TTS_NUM_STEPS = os.environ.get("REALTIME_TTS_NUM_STEPS", "12,12")
REALTIME_TTS_DEVICE = os.environ.get("REALTIME_TTS_DEVICE", "cuda:0")
REALTIME_TTS_DTYPE = os.environ.get("REALTIME_TTS_DTYPE", "float16")
REALTIME_TTS_DEBUG = os.environ.get("REALTIME_TTS_DEBUG", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class RealtimeOmniVoiceTTS:
    def __init__(self) -> None:
        self._stream = None
        self._sample_rate = 24000
        self._lock = asyncio.Lock()

    async def prewarm(self) -> None:
        await asyncio.to_thread(self._get_stream)

    async def speak(self, text: str, on_audio_chunk) -> float:
        async with self._lock:
            chunks: list[bytes] = []

            def collect_and_forward(chunk: bytes) -> None:
                chunks.append(chunk)
                on_audio_chunk(chunk, self._sample_rate)

            started = time.perf_counter()
            await asyncio.to_thread(self._play, text, collect_and_forward)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            audio_bytes = sum(len(chunk) for chunk in chunks)
            audio_seconds = self._audio_duration_seconds(audio_bytes)
            logger.info(
                "RealtimeTTS complete: elapsed_ms=%.0f chars=%s audio_ms=%.0f chunks=%s",
                elapsed_ms,
                len(text),
                audio_seconds * 1000.0,
                len(chunks),
            )
            return audio_seconds

    def _play(self, text: str, on_audio_chunk) -> None:
        stream = self._get_stream()
        stream.feed(text)
        stream.play(muted=True, on_audio_chunk=on_audio_chunk)

    def _get_stream(self):
        if self._stream is not None:
            return self._stream

        from RealtimeTTS import OmniVoiceEngine, OmniVoiceVoice, TextToAudioStream
        import torch

        steps = [
            int(part.strip())
            for part in REALTIME_TTS_NUM_STEPS.split(",")
            if part.strip()
        ]
        voice = OmniVoiceVoice(
            name="homelands",
            ref_audio=REALTIME_TTS_REF_AUDIO,
            ref_text=REALTIME_TTS_REF_TEXT,
            language=REALTIME_TTS_REF_LANGUAGE,
        )
        engine = OmniVoiceEngine(
            voice=voice,
            device_map=REALTIME_TTS_DEVICE,
            dtype=getattr(torch, REALTIME_TTS_DTYPE),
            num_steps_schedule=steps or [12, 12],
            debug=REALTIME_TTS_DEBUG,
        )
        _, _, self._sample_rate = engine.get_stream_info()
        self._stream = TextToAudioStream(engine, muted=True)
        return self._stream

    def _audio_duration_seconds(self, audio_bytes: int) -> float:
        if audio_bytes <= 0 or self._sample_rate <= 0:
            return 0.0
        return (audio_bytes / 2) / self._sample_rate


class GeminiTurnPipeline:
    def __init__(self, prepare_tts_text, interrupt_playback, tts: RealtimeOmniVoiceTTS | None = None):
        self._prepare_tts_text = prepare_tts_text
        self._interrupt_playback = interrupt_playback
        self._client = None
        self._tts = tts or RealtimeOmniVoiceTTS()

    @property
    def client(self):
        if self._client is None:
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            self._client = genai.Client(api_key=api_key)
        return self._client

    async def prewarm_tts(self) -> None:
        await self._tts.prewarm()

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
            protected_seconds = min(
                greeting_seconds + 0.25,
                TURN_GREETING_PROTECTION_MAX_SECONDS,
            )
            logger.info(
                "Protecting Homelands language prompt for %.2f seconds before VAD; audio was %.2f seconds",
                protected_seconds,
                greeting_seconds,
            )
            await self._discard_input_audio(input_track, protected_seconds)

        resampler = AudioResampler(format="s16", layout="mono", rate=16000)
        chunk_buffer = bytearray()
        utterance_buffer = bytearray()
        transcript_history: list[tuple[str, str]] = [("assistant", GEMINI_TURN_GREETING)]
        silence_chunks = 0
        is_speaking = False
        turn_started_at = None

        while True:
            try:
                frame = await input_track.recv()
            except Exception as exc:
                logger.info("Gemini turn input ended for %s: %s", call_id, exc)
                return

            for resampled in resampler.resample(frame):
                audio_bytes = resampled.to_ndarray().tobytes()
                if not audio_bytes:
                    continue
                chunk_buffer.extend(audio_bytes)

                while len(chunk_buffer) >= TURN_INPUT_CHUNK_SIZE:
                    chunk = bytes(chunk_buffer[:TURN_INPUT_CHUNK_SIZE])
                    del chunk_buffer[:TURN_INPUT_CHUNK_SIZE]
                    utterance_buffer.extend(chunk)

                    audio_np = np.frombuffer(chunk, dtype=np.int16)
                    rms = np.sqrt(np.mean(audio_np.astype(np.float64) ** 2))

                    if rms > TURN_SILENCE_THRESHOLD:
                        if not is_speaking:
                            logger.info("Turn VAD: Speech started")
                            is_speaking = True
                            turn_started_at = time.perf_counter()
                            silence_chunks = 0
                            self._interrupt_playback(call_id, output_track)
                    elif is_speaking:
                        silence_chunks += 1
                        if silence_chunks >= TURN_END_SILENCE_CHUNKS:
                            logger.info("Turn VAD: Speech ended")
                            is_speaking = False
                            silence_chunks = 0
                            utterance_pcm = bytes(utterance_buffer)
                            utterance_buffer.clear()
                            await self._handle_utterance(
                                call_id,
                                utterance_pcm,
                                transcript_history,
                                output_track,
                                playback_generation,
                                turn_started_at=turn_started_at,
                            )
                            turn_started_at = None

    async def _handle_utterance(
        self,
        call_id,
        utterance_pcm: bytes,
        transcript_history: list[tuple[str, str]],
        output_track,
        playback_generation,
        turn_started_at: float | None = None,
    ):
        if len(utterance_pcm) < 3200:
            return

        turn_end_at = time.perf_counter()

        stt_started_at = time.perf_counter()
        transcript = await self._transcribe(utterance_pcm)
        stt_ms = (time.perf_counter() - stt_started_at) * 1000.0
        if not transcript:
            logger.info("Turn STT for %s returned empty transcript in %.0f ms", call_id, stt_ms)
            return
        logger.info("Turn transcript for %s in %.0f ms: %s", call_id, stt_ms, transcript)

        transcript_history.append(("user", transcript))

        llm_started_at = time.perf_counter()
        response_text = await self._respond(transcript_history)
        llm_ms = (time.perf_counter() - llm_started_at) * 1000.0
        if not response_text:
            logger.info("Turn LLM returned empty response for %s in %.0f ms", call_id, llm_ms)
            return
        logger.info("Turn response for %s in %.0f ms: %s", call_id, llm_ms, response_text)

        transcript_history.append(("assistant", response_text))
        tts_started_at = time.perf_counter()
        await self._speak(call_id, response_text, output_track, playback_generation)
        tts_ms = (time.perf_counter() - tts_started_at) * 1000.0

        after_vad_ms = (time.perf_counter() - turn_end_at) * 1000.0
        total_turn_ms = (
            (time.perf_counter() - turn_started_at) * 1000.0
            if turn_started_at is not None
            else None
        )
        if total_turn_ms is None:
            logger.info(
                "Turn timings for %s: post-VAD=%.0f ms stt=%.0f ms llm=%.0f ms tts=%.0f ms",
                call_id,
                after_vad_ms,
                stt_ms,
                llm_ms,
                tts_ms,
            )
        else:
            logger.info(
                "Turn timings for %s: total=%.0f ms post-VAD=%.0f ms stt=%.0f ms llm=%.0f ms tts=%.0f ms",
                call_id,
                total_turn_ms,
                after_vad_ms,
                stt_ms,
                llm_ms,
                tts_ms,
            )

    async def _transcribe(self, utterance_pcm: bytes) -> str:
        wav_bytes = self._pcm_to_wav(utterance_pcm, sample_rate=16000)
        response = await self.client.aio.models.generate_content(
            model=GEMINI_STT_MODEL,
            contents=[
                "Transcribe the caller speech exactly. Return only the transcript text. If there is no clear speech, return an empty string.",
                types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
            ],
            config={
                "temperature": 0,
                "max_output_tokens": 200,
            },
        )
        return (response.text or "").strip()

    async def _respond(self, transcript_history: list[tuple[str, str]]) -> str:
        conversation_lines = []
        for role, text in transcript_history[-10:]:
            prefix = "Caller" if role == "user" else "Agent"
            conversation_lines.append(f"{prefix}: {text}")
        prompt = "\n".join(conversation_lines)

        response = await self.client.aio.models.generate_content(
            model=GEMINI_LLM_MODEL,
            contents=(
                "Reply to the latest caller message using the Homelands instructions.\n"
                f"{prompt}"
            ),
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
        loop = asyncio.get_running_loop()

        def on_audio_chunk(chunk: bytes, sample_rate: int) -> None:
            if playback_generation.get(call_id, 0) != generation_id:
                return
            loop.call_soon_threadsafe(output_track.add_pcm_audio, chunk, sample_rate)

        audio_seconds = await self._tts.speak(prepared, on_audio_chunk)
        await asyncio.sleep(0)
        if playback_generation.get(call_id, 0) != generation_id:
            logger.info("Stopping interrupted RealtimeTTS playback for %s", call_id)
            return 0.0
        return audio_seconds

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
                logger.info("Protected prompt input drain ended: %s", exc)
                break
        logger.info("Discarded %s inbound frames during protected prompt", discarded_frames)

    def _pcm_to_wav(self, pcm_bytes: bytes, sample_rate: int) -> bytes:
        buf = BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        return buf.getvalue()
