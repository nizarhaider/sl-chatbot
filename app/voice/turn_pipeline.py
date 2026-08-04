import asyncio
import json
import logging
import re
import time

import numpy as np
from av.audio.resampler import AudioResampler

from app.dashboard.state import dashboard_state
from app.voice.asr import LocalWhisperASR, is_noise_text
from app.voice.config import (
    LOCAL_LLM_HISTORY_MAX_MESSAGES,
    LOCAL_LLM_PREWARM,
    LOCAL_TURN_GREETING,
    REALTIME_TTS_PREWARM,
    TURN_END_SILENCE_CHUNKS,
    TURN_GREETING_DELAY_SECONDS,
    TURN_INPUT_CHUNK_SIZE,
    TURN_MIN_AUDIO_MS,
    TURN_PLAYBACK_ECHO_TAIL_SECONDS,
    TURN_SILENCE_THRESHOLD,
)
from app.voice.llm import LocalGemmaLLM
from app.voice.tts import RealtimeOmniVoiceTTS
from app.voice.tools import CallContext, RealEstateToolService, parse_tool_call, tool_call_message
from app.voice.vad import VadState, pcm_rms

logger = logging.getLogger(__name__)


class LocalGemmaTurnPipeline:
    def __init__(
        self,
        prepare_tts_text,
        interrupt_playback,
        tts: RealtimeOmniVoiceTTS | None = None,
    ):
        self._prepare_tts_text = prepare_tts_text
        self._interrupt_playback = interrupt_playback
        self._asr = LocalWhisperASR()
        self._tts = tts or RealtimeOmniVoiceTTS()
        self._llm = LocalGemmaLLM()
        self._tools = RealEstateToolService.from_env()
        self._conversation_history: dict[str, list[dict[str, str]]] = {}

    async def prewarm_tts(self) -> None:
        await self._tts.prewarm()

    async def prewarm_models(self) -> None:
        if self._tools is not None:
            await self._tools.ensure_ready()
        await asyncio.to_thread(self._asr.prewarm)
        if LOCAL_LLM_PREWARM:
            await self._llm.prewarm()
        if REALTIME_TTS_PREWARM:
            await self._tts.prewarm()

    async def run(self, call_id, caller_phone, input_track, output_track, playback_generation):
        try:
            await self._play_greeting(call_id, input_track, output_track, playback_generation)
            await self._run_turn_loop(call_id, caller_phone, input_track, output_track, playback_generation)
        finally:
            self._conversation_history.pop(call_id, None)

    async def _play_greeting(self, call_id, input_track, output_track, playback_generation) -> None:
        if TURN_GREETING_DELAY_SECONDS:
            await asyncio.sleep(TURN_GREETING_DELAY_SECONDS)

        greeting_started_at = time.perf_counter()
        greeting_seconds = await self._speak(
            call_id,
            LOCAL_TURN_GREETING,
            output_track,
            playback_generation,
        )

        logger.info(
            "Greeting timings for %s: tts_wall=%.0f ms tts_audio=%.0f ms",
            call_id,
            (time.perf_counter() - greeting_started_at) * 1000.0,
            greeting_seconds * 1000.0,
        )
        if not greeting_seconds:
            return

        await self._discard_playback_echo(
            call_id,
            input_track,
            output_track,
            greeting_seconds,
        )

    async def _run_turn_loop(self, call_id, caller_phone, input_track, output_track, playback_generation) -> None:
        vad = VadState()
        resampler = AudioResampler(format="s16", layout="mono", rate=16000)
        chunk_buffer = bytearray()

        while True:
            try:
                frame = await input_track.recv()
            except Exception as exc:
                logger.info("Input ended for %s: %s", call_id, exc)
                return

            for resampled in resampler.resample(frame):
                chunk_buffer.extend(resampled.to_ndarray().tobytes())
                await self._consume_chunks(
                    call_id,
                    caller_phone,
                    input_track,
                    chunk_buffer,
                    vad,
                    output_track,
                    playback_generation,
                )

    async def _consume_chunks(
        self,
        call_id,
        caller_phone,
        input_track,
        chunk_buffer: bytearray,
        vad: VadState,
        output_track,
        playback_generation,
    ) -> None:
        while len(chunk_buffer) >= TURN_INPUT_CHUNK_SIZE:
            chunk = bytes(chunk_buffer[:TURN_INPUT_CHUNK_SIZE])
            del chunk_buffer[:TURN_INPUT_CHUNK_SIZE]

            if pcm_rms(chunk) > TURN_SILENCE_THRESHOLD:
                if not vad.is_speaking:
                    logger.info("Turn VAD: Speech started")
                    vad.start()
                    self._interrupt_playback(call_id, output_track)
                vad.add_speech(chunk)
                continue

            if not vad.is_speaking:
                continue

            vad.add_silence(chunk)
            if vad.silence_chunks < TURN_END_SILENCE_CHUNKS:
                continue

            logger.info("Turn VAD: Speech ended")
            turn = vad.finish()
            await self._handle_turn(
                call_id=call_id,
                caller_phone=caller_phone,
                input_track=input_track,
                output_track=output_track,
                playback_generation=playback_generation,
                turn_started_at=turn.started_at,
                turn_end_at=time.perf_counter(),
                utterance_pcm=turn.pcm,
            )

    async def _handle_turn(
        self,
        call_id,
        caller_phone,
        input_track,
        output_track,
        playback_generation,
        turn_started_at: float | None,
        turn_end_at: float,
        utterance_pcm: bytes,
    ) -> None:
        audio_ms = (len(utterance_pcm) / 2) / 16000 * 1000.0
        if audio_ms < TURN_MIN_AUDIO_MS:
            logger.info("Skipping short utterance for %s: %.0f ms", call_id, audio_ms)
            return

        transcript_text, transcript_ms = await self._timed_transcribe(call_id, utterance_pcm)
        if not transcript_text:
            return
        dashboard_state.add_transcript(call_id, "caller", transcript_text)

        response_text, llm_ms = await self._timed_response(call_id, caller_phone, transcript_text)
        if not response_text or is_noise_text(response_text):
            logger.info("Dropping empty/noise Gemma response for %s: %r", call_id, response_text)
            return

        logger.info("Turn response for %s in %.0f ms: %s", call_id, llm_ms, response_text)
        dashboard_state.add_transcript(call_id, "assistant", response_text)
        self._append_conversation_turn(call_id, transcript_text, response_text)
        tts_audio_seconds, tts_ms = await self._timed_speak(
            call_id,
            response_text,
            output_track,
            playback_generation,
        )
        await self._discard_playback_echo(
            call_id,
            input_track,
            output_track,
            tts_audio_seconds,
        )
        self._log_turn_timings(
            call_id,
            turn_started_at,
            turn_end_at,
            transcript_ms,
            llm_ms,
            tts_ms,
            tts_audio_seconds,
        )

    async def _timed_transcribe(self, call_id, utterance_pcm: bytes) -> tuple[str, float]:
        started = time.perf_counter()
        transcript_text = await self._transcribe_pcm16(utterance_pcm)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if transcript_text:
            logger.info("Turn transcript for %s in %.0f ms: %s", call_id, elapsed_ms, transcript_text)
        else:
            logger.info("Empty transcript for %s in %.0f ms", call_id, elapsed_ms)
        return transcript_text, elapsed_ms

    async def _timed_response(self, call_id, caller_phone, transcript_text: str) -> tuple[str, float]:
        started = time.perf_counter()
        response_text = await self._generate_response(call_id, caller_phone, transcript_text)
        if _is_repetitive_response(response_text):
            logger.info("Dropping repetitive Gemma response for %s: %r", call_id, response_text)
            response_text = ""
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not response_text:
            logger.info("Empty Gemma response for %s in %.0f ms", call_id, elapsed_ms)
        return response_text, elapsed_ms

    async def _timed_speak(self, call_id, response_text, output_track, playback_generation):
        started = time.perf_counter()
        audio_seconds = await self._speak(call_id, response_text, output_track, playback_generation)
        return audio_seconds, (time.perf_counter() - started) * 1000.0

    async def _transcribe_pcm16(self, pcm: bytes) -> str:
        pcm_array = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, lambda: self._asr.transcribe(pcm_array))
        if not text:
            logger.warning("Whisper returned empty transcription")
        return text

    async def _generate_response(self, call_id, caller_phone, transcript_text: str) -> str:
        history = list(self._conversation_history.get(call_id, []))
        continuation: list[dict[str, str]] = []
        context = CallContext(call_id=call_id, caller_phone=caller_phone)

        for _ in range(2):
            response = await self._llm.generate(transcript_text, history, continuation)
            tool_call = parse_tool_call(response)
            if tool_call is None:
                if "<tool_call" in response.casefold():
                    logger.warning("Gemma emitted a malformed tool call for %s", call_id)
                    return "Sorry, I couldn't complete that request. Please try again."
                return response
            if self._tools is None:
                result = {"ok": False, "error": "The booking database is not configured."}
            else:
                result = await self._tools.execute(tool_call, context)
            logger.info("Tool call for %s: name=%s ok=%s", call_id, tool_call.name, result.get("ok"))
            continuation.extend(
                [
                    {"role": "assistant", "content": tool_call_message(tool_call)},
                    {
                        "role": "user",
                        "content": f"<tool_result>{json.dumps(result, ensure_ascii=False)}</tool_result>",
                    },
                ]
            )

        response = await self._llm.generate(transcript_text, history, continuation)
        if parse_tool_call(response) is not None:
            logger.warning("Gemma exceeded the tool-call limit for %s", call_id)
            return "Sorry, I couldn't complete that request. Please try again."
        return response

    def _append_conversation_turn(self, call_id, transcript_text: str, response_text: str) -> None:
        history = self._conversation_history.setdefault(call_id, [])
        history.extend(
            [
                {"role": "user", "content": transcript_text},
                {"role": "assistant", "content": response_text},
            ]
        )
        del history[:-LOCAL_LLM_HISTORY_MAX_MESSAGES]

    async def _speak(self, call_id, text, output_track, playback_generation):
        prepared = self._prepare_tts_text(text)
        if not prepared:
            return 0.0

        generation_id = playback_generation.get(call_id, 0)
        loop = asyncio.get_running_loop()

        def on_audio_chunk(chunk: bytes, sample_rate: int) -> None:
            if playback_generation.get(call_id, 0) == generation_id:
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

        logger.info("Discarded %s inbound frames during protected playback", discarded_frames)

    async def _discard_playback_echo(
        self,
        call_id,
        input_track,
        output_track,
        generated_audio_seconds: float,
    ) -> None:
        if not generated_audio_seconds:
            return

        pending_seconds = min(
            generated_audio_seconds,
            output_track.pending_audio_seconds,
        )
        protected_seconds = pending_seconds + TURN_PLAYBACK_ECHO_TAIL_SECONDS
        logger.info(
            "Suppressing inbound audio for %s during playback: pending=%.2f seconds tail=%.2f seconds",
            call_id,
            pending_seconds,
            TURN_PLAYBACK_ECHO_TAIL_SECONDS,
        )
        await self._discard_input_audio(input_track, protected_seconds)

    def _log_turn_timings(
        self,
        call_id,
        turn_started_at,
        turn_end_at,
        transcript_ms,
        llm_ms,
        tts_ms,
        tts_audio_seconds,
    ) -> None:
        after_vad_ms = (time.perf_counter() - turn_end_at) * 1000.0
        total_turn_ms = (
            (time.perf_counter() - turn_started_at) * 1000.0
            if turn_started_at is not None
            else after_vad_ms
        )
        logger.info(
            "Turn stages for %s: stt=%.0f ms llm=%.0f ms tts_wall=%.0f ms tts_audio=%.0f ms post_vad=%.0f ms total=%.0f ms",
            call_id,
            transcript_ms,
            llm_ms,
            tts_ms,
            tts_audio_seconds * 1000.0,
            after_vad_ms,
            total_turn_ms,
        )


def _is_repetitive_response(text: str) -> bool:
    words = re.findall(r"\S+", text.casefold())
    if len(words) < 8:
        return False
    for size in range(2, 6):
        grams = [" ".join(words[index : index + size]) for index in range(len(words) - size + 1)]
        if any(grams.count(gram) >= 3 for gram in set(grams)):
            return True
    return False
