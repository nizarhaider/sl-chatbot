import asyncio
import logging
import os
import time
import wave
from io import BytesIO

import httpx
import numpy as np
from aiortc import MediaStreamTrack
from av.audio.resampler import AudioResampler
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

GEMINI_STT_MODEL = os.environ.get("GEMINI_STT_MODEL", "gemini-2.5-flash")
GEMINI_LLM_MODEL = os.environ.get("GEMINI_LLM_MODEL", "gemini-2.5-flash")
GEMINI_THINKING_LEVEL = os.environ.get("GEMINI_THINKING_LEVEL", "").strip().lower()
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
LOCAL_LLM_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8081/v1").rstrip("/")
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "local")
LOCAL_LLM_TIMEOUT_SECONDS = float(os.environ.get("LOCAL_LLM_TIMEOUT_SECONDS", "10"))
LOCAL_LLM_MAX_TOKENS = max(8, int(os.environ.get("LOCAL_LLM_MAX_TOKENS", "40")))
TURN_INPUT_CHUNK_MS = max(20, int(os.environ.get("TURN_INPUT_CHUNK_MS", "40")))
TURN_INPUT_CHUNK_SIZE = (16000 * 2 * TURN_INPUT_CHUNK_MS) // 1000
TURN_SILENCE_THRESHOLD = int(os.environ.get("TURN_SILENCE_THRESHOLD", "1000"))
TURN_END_SILENCE_CHUNKS = max(2, int(os.environ.get("TURN_END_SILENCE_CHUNKS", "20")))
TURN_MIN_TRANSCRIPT_WORDS = max(1, int(os.environ.get("TURN_MIN_TRANSCRIPT_WORDS", "3")))
TURN_MIN_TRANSCRIPT_CHARS = max(1, int(os.environ.get("TURN_MIN_TRANSCRIPT_CHARS", "12")))
TURN_GREETING_DELAY_SECONDS = max(
    0.0,
    float(os.environ.get("TURN_GREETING_DELAY_SECONDS", "1.2")),
)
GEMINI_TURN_GREETING = os.environ.get(
    "GEMINI_TURN_GREETING",
    "සිංහලෙන් කතා කිරීමට සිංහල කියන්න. தமிழ் பேசுவதற்கு தமிழ் என்று கூறவும். For English, please say English.",
)
HOMELANDS_PROPERTIES = (
    "1. Horizon Residencies, Malabe: two-bedroom apartments from LKR 28 million, near schools and supermarkets. "
    "2. Lakeview Villas, Piliyandala: three-bedroom villas from LKR 48 million, garden, parking, and lake access. "
    "3. Green Acres, Kurunegala: ten-perch residential land from LKR 9.5 million, clear title, bank loans supported. "
    "4. Ocean Breeze Apartments, Dehiwala: one and two-bedroom units from LKR 32 million, sea view, ready soon."
)
HOMELANDS_SYSTEM_PROMPT = (
    "You are a call center agent for Homelands, a Sri Lankan property business. "
    "At the beginning of the call, ask the customer exactly: "
    '"සිංහලෙන් කතා කිරීමට සිංහල කියන්න. தமிழ் பேசுவதற்கு தமிழ் என்று கூறவும். For English, please say English." '
    "This exact language menu is already spoken as the first assistant message; do not repeat it after the customer answers. "
    "After the customer chooses English, Sinhala, or Tamil, continue naturally in that language. "
    "If they only say a language name, briefly acknowledge and ask what property type, location, or budget they prefer. "
    "Help with property inquiries using these mock properties only: "
    f"{HOMELANDS_PROPERTIES} "
    "When the customer asks about properties, recommend a suitable option and tell them you have scheduled "
    "an appointment with a Homelands consultant for tomorrow at 10 AM. "
    "Keep each response brief and natural for a phone call, usually one short sentence."
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
        if TURN_GREETING_DELAY_SECONDS:
            await asyncio.sleep(TURN_GREETING_DELAY_SECONDS)
        greeting_seconds = await self._speak(
            call_id,
            GEMINI_TURN_GREETING,
            output_track,
            playback_generation,
        )
        if greeting_seconds:
            logger.info(
                "Protecting Homelands language prompt for %.2f seconds before VAD",
                greeting_seconds,
            )
            await self._discard_input_audio(input_track, greeting_seconds + 0.25)

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

    async def _respond(self, transcript_history: list[tuple[str, str]]) -> str:
        if LLM_PROVIDER in {"local", "openai", "openai_compatible"}:
            return await self._respond_local_llm(transcript_history)
        return await self._respond_gemini(transcript_history)

    async def _respond_gemini(self, transcript_history: list[tuple[str, str]]) -> str:
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

    async def _respond_local_llm(self, transcript_history: list[tuple[str, str]]) -> str:
        messages = [{"role": "system", "content": HOMELANDS_SYSTEM_PROMPT}]
        for role, text in transcript_history[-10:]:
            messages.append(
                {
                    "role": "assistant" if role == "assistant" else "user",
                    "content": text,
                }
            )
        messages.append(
            {
                "role": "user",
                "content": "Reply to the latest caller message using the Homelands instructions.",
            }
        )

        payload = {
            "model": LOCAL_LLM_MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max(LOCAL_LLM_MAX_TOKENS, 160),
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=LOCAL_LLM_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{LOCAL_LLM_BASE_URL}/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
        except Exception:
            logger.exception("Local LLM request failed")
            return ""

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            logger.warning("Local LLM returned no choices: %s", data)
            return ""
        message = choices[0].get("message") or {}
        return (message.get("content") or "").strip()

    async def _speak(self, call_id, text, output_track, playback_generation):
        prepared = self._prepare_tts_text(text)
        if not prepared:
            return 0.0

        generation_id = playback_generation.get(call_id, 0)
        synthesized = await self.tts_service.synthesize(prepared)
        if playback_generation.get(call_id, 0) != generation_id:
            logger.info("Discarding stale turn TTS output for %s", call_id)
            return 0.0

        if playback_generation.get(call_id, 0) != generation_id:
            logger.info("Stopping interrupted turn TTS playback for %s", call_id)
            return 0.0
        output_track.add_pcm_audio(synthesized.pcm, synthesized.sample_rate)
        return self._audio_duration_seconds(synthesized.pcm, synthesized.sample_rate)

    def _audio_duration_seconds(self, pcm: bytes, sample_rate: int) -> float:
        if not pcm or sample_rate <= 0:
            return 0.0
        return (len(pcm) / 2) / sample_rate

    def _pcm_to_wav(self, pcm_bytes: bytes, sample_rate: int) -> bytes:
        buf = BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        return buf.getvalue()
