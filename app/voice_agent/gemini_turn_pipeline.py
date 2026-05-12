import asyncio
import logging
import os
import time
import wave
from io import BytesIO

import numpy as np
from aiortc import MediaStreamTrack
from av.audio.resampler import AudioResampler
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

GEMINI_STT_MODEL = os.environ.get("GEMINI_STT_MODEL", "gemini-2.5-flash")
GEMINI_LLM_MODEL = os.environ.get("GEMINI_LLM_MODEL", "gemini-2.5-flash")
TURN_INPUT_CHUNK_MS = max(20, int(os.environ.get("TURN_INPUT_CHUNK_MS", "40")))
TURN_INPUT_CHUNK_SIZE = (16000 * 2 * TURN_INPUT_CHUNK_MS) // 1000
TURN_SILENCE_THRESHOLD = int(os.environ.get("TURN_SILENCE_THRESHOLD", "1000"))
TURN_END_SILENCE_CHUNKS = max(2, int(os.environ.get("TURN_END_SILENCE_CHUNKS", "5")))
GEMINI_TURN_GREETING = (
    'සිංහලෙන් කතා කිරීමට සිංහල කියන්න. தமிழ் பேசுவதற்கு தமிழ் என்று கூறவும். '
    "For English, please say English."
)
GEMINI_TURN_SYSTEM_PROMPT = (
    "You are Sam, a friendly senior SLT Mobitel agent. "
    "Help the caller in the language they choose. "
    "Keep replies short, helpful, and professional. "
    "For voice calls, default to one short sentence and keep most replies under 18 words unless the caller explicitly asks for more detail."
)


class GeminiTurnPipeline:
    def __init__(self, tts_service, prepare_tts_text, interrupt_playback):
        self.tts_service = tts_service
        self._prepare_tts_text = prepare_tts_text
        self._interrupt_playback = interrupt_playback
        self._client = None

    @property
    def client(self):
        if self._client is None:
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            self._client = genai.Client(api_key=api_key)
        return self._client

    async def run(self, call_id, input_track, output_track, playback_generation):
        await self._speak(call_id, GEMINI_TURN_GREETING, output_track, playback_generation)

        resampler = AudioResampler(format="s16", layout="mono", rate=16000)
        chunk_buffer = bytearray()
        utterance_buffer = bytearray()
        transcript_history: list[tuple[str, str]] = []
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
            contents=prompt or "The call has just connected. Start with the language menu.",
            config={
                "system_instruction": GEMINI_TURN_SYSTEM_PROMPT,
                "temperature": 0.6,
                "max_output_tokens": 120,
            },
        )
        return (response.text or "").strip()

    async def _speak(self, call_id, text, output_track, playback_generation):
        prepared = self._prepare_tts_text(text)
        if not prepared:
            return

        generation_id = playback_generation.get(call_id, 0)
        synthesized = await self.tts_service.synthesize(prepared)
        if playback_generation.get(call_id, 0) != generation_id:
            logger.info("Discarding stale turn TTS output for %s", call_id)
            return

        frame_size = output_track.frame_size_bytes
        for offset in range(0, len(synthesized.pcm), frame_size):
            if playback_generation.get(call_id, 0) != generation_id:
                logger.info("Stopping interrupted turn TTS playback for %s", call_id)
                return
            output_track.add_audio(synthesized.pcm[offset:offset + frame_size])

    def _pcm_to_wav(self, pcm_bytes: bytes, sample_rate: int) -> bytes:
        buf = BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        return buf.getvalue()
