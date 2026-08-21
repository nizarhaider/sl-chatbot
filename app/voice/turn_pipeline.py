import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable

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
            logger.info("Voice tool service ready")
        await asyncio.to_thread(self._asr.prewarm)
        logger.info("Whisper prewarm complete")
        if LOCAL_LLM_PREWARM:
            logger.info("Starting Gemma prewarm")
            await self._llm.prewarm()
            logger.info("Gemma prewarm complete")
        if REALTIME_TTS_PREWARM:
            logger.info("Starting OmniVoice prewarm")
            await self._tts.prewarm()
            logger.info("OmniVoice prewarm complete")

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
        turn_task: asyncio.Task | None = None

        try:
            while True:
                try:
                    frame = await input_track.recv()
                except Exception as exc:
                    logger.info("Input ended for %s: %s", call_id, exc)
                    return

                for resampled in resampler.resample(frame):
                    chunk_buffer.extend(resampled.to_ndarray().tobytes())
                    turn_task = await self._consume_chunks(
                        call_id,
                        caller_phone,
                        input_track,
                        chunk_buffer,
                        vad,
                        output_track,
                        playback_generation,
                        turn_task,
                    )
        finally:
            if turn_task is not None and not turn_task.done():
                turn_task.cancel()
            if turn_task is not None:
                await asyncio.gather(turn_task, return_exceptions=True)

    async def _consume_chunks(
        self,
        call_id,
        caller_phone,
        input_track,
        chunk_buffer: bytearray,
        vad: VadState,
        output_track,
        playback_generation,
        turn_task: asyncio.Task | None,
    ) -> asyncio.Task | None:
        while len(chunk_buffer) >= TURN_INPUT_CHUNK_SIZE:
            chunk = bytes(chunk_buffer[:TURN_INPUT_CHUNK_SIZE])
            del chunk_buffer[:TURN_INPUT_CHUNK_SIZE]

            if pcm_rms(chunk) > TURN_SILENCE_THRESHOLD:
                if not vad.is_speaking:
                    logger.info("Turn VAD: Speech started")
                    dashboard_state.emit(call_id, "pipeline.speech_started", {})
                    vad.start()
                    self._interrupt_playback(call_id, output_track)
                    if turn_task is not None:
                        if not turn_task.done():
                            turn_task.cancel()
                        await asyncio.gather(turn_task, return_exceptions=True)
                        turn_task = None
                vad.add_speech(chunk)
                continue

            if not vad.is_speaking:
                continue

            vad.add_silence(chunk)
            if vad.silence_chunks < TURN_END_SILENCE_CHUNKS:
                continue

            logger.info("Turn VAD: Speech ended")
            dashboard_state.emit(call_id, "pipeline.speech_ended", {})
            turn = vad.finish()
            if turn_task is not None and turn_task.done():
                await asyncio.gather(turn_task, return_exceptions=True)
                turn_task = None
            turn_task = asyncio.create_task(self._handle_turn(
                call_id=call_id,
                caller_phone=caller_phone,
                input_track=input_track,
                output_track=output_track,
                playback_generation=playback_generation,
                turn_started_at=turn.started_at,
                turn_end_at=time.perf_counter(),
                utterance_pcm=turn.pcm,
            ), name=f"turn-{call_id}")

        return turn_task

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
        dashboard_state.emit(call_id, "pipeline.asr_complete", {"text": transcript_text, "duration_ms": transcript_ms})

        async def announce_tool(tool_call) -> None:
            if tool_call.name not in {"search_properties", "book_appointment"}:
                return
            hold_text = _tool_wait_message(transcript_text, tool_call.name)
            dashboard_state.emit(call_id, "tool.announced", {"name": tool_call.name, "text": hold_text})
            dashboard_state.add_transcript(call_id, "assistant", hold_text)
            hold_audio_seconds, _ = await self._timed_speak(
                call_id,
                hold_text,
                output_track,
                playback_generation,
            )

        response_text, llm_ms = await self._timed_response(
            call_id,
            caller_phone,
            transcript_text,
            announce_tool,
        )
        if not response_text or is_noise_text(response_text):
            logger.info("Dropping empty/noise Gemma response for %s: %r", call_id, response_text)
            return

        logger.info("Turn response for %s in %.0f ms: %s", call_id, llm_ms, response_text)
        dashboard_state.emit(call_id, "pipeline.response_ready", {"text": response_text, "duration_ms": llm_ms})
        dashboard_state.add_transcript(call_id, "assistant", response_text)
        self._append_conversation_turn(call_id, transcript_text, response_text)
        tts_audio_seconds, tts_ms = await self._timed_speak(
            call_id,
            response_text,
            output_track,
            playback_generation,
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

    async def _timed_response(
        self,
        call_id,
        caller_phone,
        transcript_text: str,
        announce_tool: Callable[[object], Awaitable[None]] | None = None,
    ) -> tuple[str, float]:
        started = time.perf_counter()
        response_text = await self._generate_response(
            call_id,
            caller_phone,
            transcript_text,
            announce_tool,
        )
        if _is_repetitive_response(response_text):
            logger.info("Dropping repetitive Gemma response for %s: %r", call_id, response_text)
            response_text = _repetition_fallback(response_text)
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

    async def _generate_response(
        self,
        call_id,
        caller_phone,
        transcript_text: str,
        announce_tool: Callable[[object], Awaitable[None]] | None = None,
    ) -> str:
        history = list(self._conversation_history.get(call_id, []))
        continuation: list[dict[str, str]] = []
        context = CallContext(call_id=call_id, caller_phone=caller_phone)

        for _ in range(2):
            dashboard_state.emit(call_id, "model.request", {
                "transcript": transcript_text,
                "history": history,
                "continuation": continuation,
            })
            response = await self._llm.generate(transcript_text, history, continuation)
            dashboard_state.emit(call_id, "model.output", {"text": response})
            tool_call = parse_tool_call(response)
            if tool_call is None:
                if _contains_internal_control_text(response):
                    logger.warning("Gemma emitted an unparseable tool call for %s: %r", call_id, response[:500])
                    return _tool_recovery_response(transcript_text)
                return response
            if self._tools is None:
                result = {"ok": False, "error": "The booking database is not configured."}
            else:
                if announce_tool is not None:
                    await announce_tool(tool_call)
                result = await self._tools.execute(tool_call, context)
            logger.info("Tool call for %s: name=%s ok=%s", call_id, tool_call.name, result.get("ok"))
            dashboard_state.emit(call_id, "tool.call", {"name": tool_call.name, "arguments": tool_call.arguments})
            dashboard_state.emit(call_id, "tool.result", {"name": tool_call.name, "result": result})
            if not result.get("ok"):
                logger.warning(
                    "Tool call failed for %s: name=%s reason=%s",
                    call_id,
                    tool_call.name,
                    result.get("error", "unknown error"),
                )
                return _tool_failure_response(
                    transcript_text,
                    tool_call.name,
                    str(result.get("error", "")),
                )
            continuation.extend(
                [
                    {"role": "assistant", "content": tool_call_message(tool_call)},
                    {
                        "role": "user",
                        "content": f"<tool_result>{json.dumps(result, ensure_ascii=False)}</tool_result>",
                    },
                ]
            )

        dashboard_state.emit(call_id, "model.request", {
            "transcript": transcript_text,
            "history": history,
            "continuation": continuation,
        })
        response = await self._llm.generate(transcript_text, history, continuation)
        dashboard_state.emit(call_id, "model.output", {"text": response})
        if parse_tool_call(response) is not None or _contains_internal_control_text(response):
            logger.warning("Gemma exceeded the tool-call limit for %s", call_id)
            return _tool_recovery_response(transcript_text)
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
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    words = normalized.split()
    if len(words) < 12:
        return False
    segments = re.split(r"(?:\r?\n)+|(?<=[.!?。！？])\s+", text.casefold())
    segment_counts: dict[str, int] = {}
    for segment in segments:
        segment = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", segment).strip()
        if len(segment.split()) >= 6:
            segment_counts[segment] = segment_counts.get(segment, 0) + 1
    if any(count >= 2 for count in segment_counts.values()):
        return True

    for size in (6, 8, 10):
        grams = [" ".join(words[index : index + size]) for index in range(len(words) - size + 1)]
        if any(
            grams.count(gram) >= 3 and (grams.count(gram) * size) / len(words) >= 0.35
            for gram in set(grams)
        ):
            return True
    return False


def _contains_internal_control_text(text: str) -> bool:
    lowered = text.casefold()
    return (
        "<tool_call" in lowered
        or "<tool_result" in lowered
        or ("\"name\"" in lowered and "\"arguments\"" in lowered)
    )


def _repetition_fallback(text: str) -> str:
    if re.search(r"[\u0D80-\u0DFF]", text):
        return "සමාවෙන්න, මට ඒක පැහැදිලිව කියන්න බැරි වුණා. කරුණාකර නැවත කියන්න පුළුවන්ද?"
    return "I didn't catch that clearly. Could you say it again?"


def _tool_recovery_response(transcript_text: str) -> str:
    if re.search(r"[\u0D80-\u0DFF]", transcript_text):
        return "මට ඒක පැහැදිලිව තේරුම් ගන්න බැරි වුණා. location එක, budget එක සහ bedrooms ගණන ආයෙත් කියන්න පුළුවන්ද?"
    if re.search(r"[\u0B80-\u0BFF]", transcript_text):
        return "எனக்கு அது தெளிவாகப் புரியவில்லை. இடம், budget மற்றும் bedrooms எண்ணிக்கையை மீண்டும் சொல்ல முடியுமா?"
    return "I didn't catch that clearly. Could you repeat the location, budget, and number of bedrooms?"


def _tool_failure_response(transcript_text: str, tool_name: str, error: str) -> str:
    """Turn internal tool failures into a short, caller-safe clarification."""
    lowered = error.casefold()
    if "missing required argument: property_id" in lowered:
        english = "Which property would you like to view? Please tell me its name or location."
        sinhala = "ඔබට බලන්න අවශ්‍ය property එක මොකක්ද? ඒකේ නම හෝ location එක කියන්න පුළුවන්ද?"
        tamil = "எந்த property-ஐ பார்க்க விரும்புகிறீர்கள்? அதன் பெயர் அல்லது location-ஐ சொல்ல முடியுமா?"
    elif "missing required argument: customer_name" in lowered:
        english = "What name should I use for the appointment?"
        sinhala = "Appointment එකට යොදන්න ඕන නම මොකක්ද?"
        tamil = "Appointment-க்கு எந்த பெயரை பயன்படுத்த வேண்டும்?"
    elif "missing required argument: appointment_at" in lowered:
        english = "What exact date and time would you like for the viewing?"
        sinhala = "Viewing එකට ඔබට අවශ්‍ය exact date එක සහ time එක මොකක්ද?"
        tamil = "Viewing-க்கு உங்களுக்கு வேண்டிய சரியான date மற்றும் time என்ன?"
    elif "already been booked" in lowered:
        english = "That time is already booked. What other exact date and time would work for you?"
        sinhala = "ඒ වෙලාව දැනටමත් book කරලා. වෙනත් exact date එකක් සහ time එකක් කියන්න පුළුවන්ද?"
        tamil = "அந்த நேரம் ஏற்கனவே book செய்யப்பட்டுள்ளது. வேறு சரியான date மற்றும் time சொல்ல முடியுமா?"
    else:
        english = "I couldn't complete that just now. Could you repeat the details so I can try again?"
        sinhala = "ඒක දැන් complete කරන්න බැරි වුණා. Details ටික ආයෙත් කියන්න පුළුවන්ද?"
        tamil = "அதை இப்போது முடிக்க முடியவில்லை. Details-ஐ மீண்டும் சொல்ல முடியுமா?"

    if re.search(r"[\u0D80-\u0DFF]", transcript_text):
        return sinhala
    if re.search(r"[\u0B80-\u0BFF]", transcript_text):
        return tamil
    return english


def _tool_wait_message(transcript_text: str, tool_name: str) -> str:
    if re.search(r"[\u0D80-\u0DFF]", transcript_text):
        if tool_name == "book_appointment":
            return "හරි, මේ appointment එක confirm කරලා කියන්නම්. කරුණාකර පොඩ්ඩක් රැඳී සිටින්න."
        return "හරි, ඔබතුමා කියපු විස්තර අනුව මම දැන් බලලා කියන්නම්. කරුණාකර පොඩ්ඩක් රැඳී සිටින්න."
    if re.search(r"[\u0B80-\u0BFF]", transcript_text):
        if tool_name == "book_appointment":
            return "சரி, இந்த appointment-ஐ confirm செய்து சொல்கிறேன். தயவுசெய்து சிறிது நேரம் காத்திருக்கவும்."
        return "சரி, நீங்கள் சொன்ன விவரங்களை இப்போது பார்க்கிறேன். தயவுசெய்து சிறிது நேரம் காத்திருக்கவும்."
    if tool_name == "book_appointment":
        return "Okay, I’ll confirm that appointment now. Please hold for a moment."
    return "Okay, I’ll check those details now. Please hold for a moment."
