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
    LOCAL_TURN_GREETING,
    REALTIME_TTS_PREWARM,
    TURN_BARGE_IN_MIN_SPEECH_CHUNKS,
    TURN_BARGE_IN_RMS_THRESHOLD,
    TURN_END_SILENCE_CHUNKS,
    TURN_GREETING_DELAY_SECONDS,
    TURN_INPUT_CHUNK_MS,
    TURN_INPUT_CHUNK_SIZE,
    TURN_MIN_AUDIO_MS,
    TURN_SPEECH_START_CHUNKS,
    TURN_SPEECH_START_THRESHOLD,
    TURN_PLAYBACK_ECHO_TAIL_SECONDS,
    TURN_SILENCE_THRESHOLD,
    LLM_PREWARM,
)
from app.voice.llm import LocalLlmClient, message_text
from app.voice.tts import RealtimeOmniVoiceTTS
from app.voice.tools import LLM_TOOLS, CallContext, RealEstateToolService
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
        self._llm = LocalLlmClient()
        self._tools = RealEstateToolService.from_env()
        self._conversation_history: dict[str, list] = {}
        self._language_selection_pending: set[str] = set()
        self._call_languages: dict[str, str] = {}

    async def prewarm_tts(self) -> None:
        await self._tts.prewarm()

    async def prewarm_models(self) -> None:
        if self._tools is not None:
            await self._tools.ensure_ready()
            logger.info("Voice tool service ready")
        await asyncio.to_thread(self._asr.prewarm)
        logger.info("Whisper prewarm complete")
        if LLM_PREWARM:
            logger.info("Starting Gemma prewarm")
            await self._llm.prewarm()
            logger.info("Gemma prewarm complete")
        if REALTIME_TTS_PREWARM:
            logger.info("Starting OmniVoice prewarm")
            await self._tts.prewarm()
            logger.info("OmniVoice prewarm complete")

    async def run(self, call_id, caller_phone, input_track, output_track, playback_generation):
        self._language_selection_pending.add(call_id)
        dashboard_state.emit(call_id, "language.selection_pending", {})
        greeting_task = asyncio.create_task(
            self._play_greeting(call_id, output_track, playback_generation),
            name=f"greeting-{call_id}",
        )
        try:
            await self._run_turn_loop(call_id, caller_phone, input_track, output_track, playback_generation)
        finally:
            greeting_task.cancel()
            await asyncio.gather(greeting_task, return_exceptions=True)
            self._conversation_history.pop(call_id, None)
            self._language_selection_pending.discard(call_id)
            self._call_languages.pop(call_id, None)

    async def _play_greeting(self, call_id, output_track, playback_generation) -> None:
        if TURN_GREETING_DELAY_SECONDS:
            await asyncio.sleep(TURN_GREETING_DELAY_SECONDS)

        greeting_started_at = time.perf_counter()
        greeting_seconds = 0.0
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

    async def _run_turn_loop(self, call_id, caller_phone, input_track, output_track, playback_generation) -> None:
        vad = VadState()
        playback_echo_state = {"until": 0.0, "was_playing": False}
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
                        playback_echo_state,
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
        playback_echo_state: dict[str, float | bool],
    ) -> asyncio.Task | None:
        while len(chunk_buffer) >= TURN_INPUT_CHUNK_SIZE:
            chunk = bytes(chunk_buffer[:TURN_INPUT_CHUNK_SIZE])
            del chunk_buffer[:TURN_INPUT_CHUNK_SIZE]

            now = time.monotonic()
            if output_track.pending_audio_seconds > 0:
                playback_echo_state["was_playing"] = True
            elif playback_echo_state["was_playing"]:
                playback_echo_state["was_playing"] = False
                playback_echo_state["until"] = now + TURN_PLAYBACK_ECHO_TAIL_SECONDS
            if now < float(playback_echo_state["until"]):
                vad.discard()
                continue

            chunk_rms = pcm_rms(chunk)
            if not vad.is_speaking and chunk_rms > TURN_SPEECH_START_THRESHOLD:
                if not vad.is_speaking:
                    vad.add_candidate(chunk)
                    if vad.candidate_count < TURN_SPEECH_START_CHUNKS:
                        continue
                    logger.info("Turn VAD: Speech started after sustained energy")
                    dashboard_state.emit(call_id, "pipeline.speech_started", {})
                    vad.promote_candidate()
                    continue
            if vad.is_speaking and chunk_rms > TURN_SILENCE_THRESHOLD:
                vad.add_speech(chunk)
                if (
                    vad.speech_chunks == TURN_BARGE_IN_MIN_SPEECH_CHUNKS
                    and output_track.pending_audio_seconds > 0
                    and pcm_rms(chunk) > TURN_BARGE_IN_RMS_THRESHOLD
                ):
                    logger.info(
                        "Turn VAD: confirmed barge-in after %.0f ms",
                        vad.speech_chunks * TURN_INPUT_CHUNK_MS,
                    )
                    self._interrupt_playback(call_id, output_track)
                    if turn_task is not None:
                        if not turn_task.done():
                            turn_task.cancel()
                        await asyncio.gather(turn_task, return_exceptions=True)
                        turn_task = None
                continue

            if not vad.is_speaking:
                vad.clear_candidate()
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

        if call_id in self._language_selection_pending:
            language = _detect_language_selection(transcript_text)
            if language is None:
                dashboard_state.emit(call_id, "language.selection_retry", {"transcript": transcript_text})
                logger.info("Language selection not recognized for %s; remaining silent", call_id)
                return
            self._call_languages[call_id] = language
            self._language_selection_pending.discard(call_id)
            dashboard_state.emit(call_id, "language.selected", {"language": language})
            response_text, llm_ms = await self._timed_response(
                call_id, caller_phone, transcript_text, None, language=language, selecting_language=True
            )
            if response_text:
                dashboard_state.add_transcript(call_id, "assistant", response_text)
                await self._timed_speak(call_id, response_text, output_track, playback_generation)
            return

        if _is_wait_request(transcript_text):
            response_text = _wait_response(transcript_text)
            dashboard_state.emit(call_id, "pipeline.response_ready", {"text": response_text, "duration_ms": 0})
            dashboard_state.add_transcript(call_id, "assistant", response_text)
            self._append_conversation_turn(call_id, transcript_text, response_text)
            await self._timed_speak(call_id, response_text, output_track, playback_generation)
            return

        async def announce_tool(tool_name: str) -> None:
            if tool_name not in {"search_properties", "book_appointment", "send_whatsapp_message"}:
                return
            hold_text = _tool_wait_message(transcript_text, tool_name)
            dashboard_state.emit(call_id, "tool.announced", {"name": tool_name, "text": hold_text})
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

        spoken_response_text = self._prepare_tts_text(response_text)
        logger.info("Turn response for %s in %.0f ms: %s", call_id, llm_ms, response_text)
        dashboard_state.emit(call_id, "pipeline.response_ready", {"text": spoken_response_text, "duration_ms": llm_ms})
        dashboard_state.add_transcript(call_id, "assistant", spoken_response_text)
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
        language: str | None = None,
        selecting_language: bool = False,
    ) -> tuple[str, float]:
        started = time.perf_counter()
        response_text = await self._generate_response(
            call_id,
            caller_phone,
            transcript_text,
            announce_tool,
            language=language,
            selecting_language=selecting_language,
        )
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
        language: str | None = None,
        selecting_language: bool = False,
    ) -> str:
        context = CallContext(call_id=call_id, caller_phone=caller_phone)
        history = list(self._conversation_history.get(call_id, []))
        dashboard_state.emit(call_id, "model.request", {"transcript": transcript_text})
        if selecting_language:
            messages = [{"role": "user", "content": (
                "The caller has selected the language identified by the application. Acknowledge "
                "the selection briefly in that language and say you are ready to listen. Do not "
                "ask about properties or process the caller's request during this turn."
            )}, *history, {"role": "user", "content": transcript_text}]
        else:
            messages = [*history, {"role": "user", "content": transcript_text}]
        search_query = _property_search_query(history, transcript_text)
        if not selecting_language and self._tools is not None and search_query:
            return await self._search_properties(
                call_id, caller_phone, search_query, announce_tool,
                language=language or self._call_languages.get(call_id),
            )
        for _ in range(4):
            assistant = await self._llm.chat(messages, LLM_TOOLS if self._tools else None, language=language or self._call_languages.get(call_id))
            messages.append(assistant)
            tool_calls = assistant.get("tool_calls") or []
            if not tool_calls:
                break
            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                name = function.get("name", "")
                raw_arguments = function.get("arguments") or {}
                try:
                    arguments = (
                        json.loads(raw_arguments)
                        if isinstance(raw_arguments, str)
                        else dict(raw_arguments)
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    arguments = {}
                dashboard_state.emit(call_id, "tool.call", {
                    "name": name,
                    "arguments": arguments,
                })
                if announce_tool is not None:
                    await announce_tool(name)
                result = await self._tools.execute(name, arguments, context)
                dashboard_state.emit(call_id, "tool.result", {
                    "name": name,
                    "result": result,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", name),
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                })
        else:
            assistant = {"content": "Sorry, I couldn't complete that request. Please try again."}
            messages.append({"role": "assistant", "content": assistant["content"]})

        self._conversation_history[call_id] = messages
        response = message_text(assistant)
        dashboard_state.emit(call_id, "model.output", {"text": response})
        return response

    async def _search_properties(
        self,
        call_id: str,
        caller_phone: str,
        transcript_text: str,
        announce_tool: Callable[[object], Awaitable[None]] | None,
        language: str | None = None,
    ) -> str:
        assert self._tools is not None
        arguments = {"query": transcript_text}
        dashboard_state.emit(call_id, "tool.call", {
            "name": "search_properties", "arguments": arguments,
        })
        if announce_tool is not None:
            await announce_tool("search_properties")
        result = await self._tools.execute(
            "search_properties", arguments, CallContext(call_id=call_id, caller_phone=caller_phone)
        )
        dashboard_state.emit(call_id, "tool.result", {
            "name": "search_properties", "result": result,
        })
        response = await self._llm.summarize_search(transcript_text, result, language=language)
        self._append_conversation_turn(call_id, transcript_text, response)
        dashboard_state.emit(call_id, "model.output", {"text": response})
        return response

    def _append_conversation_turn(self, call_id, transcript_text: str, response_text: str) -> None:
        history = self._conversation_history.setdefault(call_id, [])
        history.extend([
            {"role": "user", "content": transcript_text},
            {"role": "assistant", "content": response_text},
        ])

    async def _speak(self, call_id, text, output_track, playback_generation):
        prepared = self._prepare_tts_text(text)
        if not prepared:
            return 0.0

        generation_id = playback_generation.get(call_id, 0)
        loop = asyncio.get_running_loop()

        def on_audio_chunk(chunk: bytes, sample_rate: int) -> None:
            if playback_generation.get(call_id, 0) == generation_id:
                loop.call_soon_threadsafe(output_track.add_pcm_audio, chunk, sample_rate)

        audio_seconds = await self._tts.speak(
            prepared, on_audio_chunk, language=self._call_languages.get(call_id)
        )
        await asyncio.sleep(0)
        if playback_generation.get(call_id, 0) != generation_id:
            logger.info("Stopping interrupted RealtimeTTS playback for %s", call_id)
            return 0.0
        return audio_seconds


def _detect_language_selection(text: str) -> str | None:
    normalized = re.sub(r"[^a-zA-Z\u0b80-\u0dff]+", " ", text.casefold()).strip()
    if re.search(r"(?:\bsinhala(?:\s+language)?\b|සිංහල(?:ෙන්| භාෂාව)?)", normalized):
        return "si"
    if re.search(r"(?:\btamil(?:\s+language)?\b|தமிழ்(?:இல்|ல்| மொழி)?)", normalized):
        return "ta"
    if re.search(r"(?:\benglish(?:\s+language)?\b|ඉංග්‍රීසි(?:යෙන්| භාෂාව)?|ஆங்கிலம்(?:ல்| மொழி)?)", normalized):
        return "en"
    return None

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


def _property_search_query(history: list, transcript_text: str) -> str | None:
    """Return a usable property query without relying on the model to choose a tool."""
    caller_turns = [
        message_text(message)
        for message in history
        if message.get("role") == "user" and message_text(message)
    ]
    query = " ".join([*caller_turns[-3:], transcript_text]).strip()
    normalized = query.casefold()
    latest = transcript_text.casefold()
    property_type_pattern = (
        r"\b(?:apartment|apartments|house|villa|land|property)\b|"
        r"(?:අපාර්ට්මන්ට්|ගෙයක්|නිවසක්|විලා|ඉඩමක්|දේපලක්)"
    )
    location_pattern = (
        r"\b(?:colombo|malabe|battaramulla|kottawa|dehiwala|piliyandala|"
        r"kurunegala|nugegoda|rajagiriya|maharagama)\b|"
        r"(?:කොළඹ|මාලබේ|බත්තරමුල්ල|කොට්ටාව|දෙහිවල|පිළියන්දල|"
        r"කුරුණෑගල|නුගේගොඩ|රාජගිරිය|මහරගම)"
    )
    bedroom_pattern = (
        r"\b(?:one|two|three|four|five|[1-9])\s*(?:bed|bedroom|bedrooms)\b|"
        r"(?:නිදන\s*කාමර|කාමර\s*(?:එකක්|දෙකක්|තුනක්|හතරක්|පහක්))"
    )
    property_type = re.search(property_type_pattern, normalized)
    location = re.search(location_pattern, normalized)
    bedrooms = re.search(bedroom_pattern, normalized)
    latest_detail = re.search(
        f"{property_type_pattern}|{location_pattern}|{bedroom_pattern}", latest
    )
    has_usable_request = (
        property_type and (location or bedrooms)
    ) or (location and bedrooms)
    if has_usable_request and latest_detail:
        return query
    return None


def _is_wait_request(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return bool(re.search(
        r"\b(?:please\s+)?(?:wait|hold\s+on|one\s+moment)\b|"
        r"(?:පොඩ්ඩක්|ටිකක්)\s*(?:ඉන්න|රැඳී|හිටින්න)",
        normalized,
    ))


def _wait_response(text: str) -> str:
    if re.search(r"[\u0D80-\u0DFF]", text):
        return "හරි, මම ඉන්නම්."
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "சரி, நான் காத்திருக்கிறேன்."
    return "Sure, I’ll wait."
