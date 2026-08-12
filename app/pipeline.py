"""One conversational turn pipeline and active-call owner."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time

import numpy as np
from aiortc import MediaStreamTrack
from av.audio.resampler import AudioResampler

from app.audio import OutboundAudioTrack, VoiceActivity, pcm_rms
from app.config import (
    END_SILENCE_CHUNKS,
    GREETING,
    GREETING_DELAY_SECONDS,
    INPUT_CHUNK_BYTES,
    LLM_HISTORY_MESSAGES,
    MIN_AUDIO_MS,
    PLAYBACK_ECHO_TAIL_SECONDS,
    SILENCE_RMS,
)
from app.database import (
    CallContext,
    RealEstateToolService,
    ToolCall,
    call_log,
    ground_search_call,
    parse_tool_call,
    tool_call_message,
)
from app.models import LocalGemmaLLM, LocalWhisperASR, OmniVoiceTTS, is_noise_text

logger = logging.getLogger(__name__)


class TurnPipeline:
    def __init__(self, tts: OmniVoiceTTS | None = None) -> None:
        self.asr = LocalWhisperASR()
        self.llm = LocalGemmaLLM()
        self.tts = tts or OmniVoiceTTS()
        self.tools = RealEstateToolService.from_env()
        self.history: dict[str, list[dict[str, str]]] = {}

    async def prewarm(self) -> None:
        if self.tools:
            await self.tools.ensure_ready()
        await asyncio.to_thread(self.asr.prewarm)
        await self.llm.prewarm()
        await self.tts.prewarm()

    async def close(self) -> None:
        if self.tools:
            await self.tools.close()

    async def run(
        self,
        call_id: str,
        phone: str,
        input_track: MediaStreamTrack,
        output_track: OutboundAudioTrack,
        playback_generation: dict[str, int],
    ) -> None:
        try:
            await asyncio.sleep(GREETING_DELAY_SECONDS)
            greeting_seconds = await self._speak(
                call_id, GREETING, output_track, playback_generation
            )
            await self._discard_echo(input_track, output_track, greeting_seconds)
            await self._listen(
                call_id, phone, input_track, output_track, playback_generation
            )
        finally:
            self.history.pop(call_id, None)

    async def _listen(
        self, call_id, phone, input_track, output_track, playback_generation
    ) -> None:
        vad = VoiceActivity()
        resampler = AudioResampler(format="s16", layout="mono", rate=16_000)
        buffer = bytearray()
        while True:
            try:
                frame = await input_track.recv()
            except Exception as exc:  # noqa: BLE001 - aiortc tracks expose backend errors here.
                logger.info("Input ended for %s: %s", call_id, exc)
                return
            for converted in resampler.resample(frame):
                buffer.extend(converted.to_ndarray().tobytes())
                await self._consume(
                    call_id,
                    phone,
                    input_track,
                    output_track,
                    playback_generation,
                    vad,
                    buffer,
                )

    async def _consume(
        self, call_id, phone, input_track, output_track, generations, vad, buffer
    ) -> None:
        while len(buffer) >= INPUT_CHUNK_BYTES:
            chunk = bytes(buffer[:INPUT_CHUNK_BYTES])
            del buffer[:INPUT_CHUNK_BYTES]
            speech = pcm_rms(chunk) > SILENCE_RMS
            if speech and not vad.speaking:
                logger.info("Turn VAD: Speech started")
                vad.start()
                generations[call_id] = generations.get(call_id, 0) + 1
                output_track.clear()
            if vad.speaking:
                vad.add(chunk, speech)
            if vad.speaking and vad.silence_chunks >= END_SILENCE_CHUNKS:
                logger.info("Turn VAD: Speech ended")
                turn = vad.finish()
                await self._handle_turn(
                    call_id,
                    phone,
                    input_track,
                    output_track,
                    generations,
                    turn.started_at,
                    turn.pcm,
                )

    async def _handle_turn(
        self, call_id, phone, input_track, output_track, generations, started_at, pcm
    ) -> None:
        if (len(pcm) / 2) / 16_000 * 1000 < MIN_AUDIO_MS:
            return
        stage_started = time.perf_counter()
        waveform = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
        transcript = await asyncio.to_thread(self.asr.transcribe, waveform)
        stt_ms = (time.perf_counter() - stage_started) * 1000
        if not transcript:
            return
        logger.info("Turn transcript for %s: %s", call_id, transcript)
        call_log.add(call_id, "caller", transcript)

        stage_started = time.perf_counter()
        response = await self._respond(call_id, phone, transcript)
        llm_ms = (time.perf_counter() - stage_started) * 1000
        if not response or is_noise_text(response) or repetitive(response):
            return
        logger.info("Turn response for %s: %s", call_id, response)
        call_log.add(call_id, "assistant", response)
        self._remember(call_id, transcript, response)

        stage_started = time.perf_counter()
        seconds = await self._speak(call_id, response, output_track, generations)
        tts_ms = (time.perf_counter() - stage_started) * 1000
        await self._discard_echo(input_track, output_track, seconds)
        total_ms = (
            (time.perf_counter() - started_at) * 1000
            if started_at
            else stt_ms + llm_ms + tts_ms
        )
        logger.info(
            "Turn timings for %s: stt=%.0fms llm=%.0fms tts=%.0fms total=%.0fms",
            call_id,
            stt_ms,
            llm_ms,
            tts_ms,
            total_ms,
        )

    async def _respond(self, call_id: str, phone: str, transcript: str) -> str:
        if reply := language_selection_reply(transcript):
            return reply
        if reply := acknowledgement_reply(transcript):
            return reply
        history = list(self.history.get(call_id, []))
        continuation: list[dict[str, str]] = []
        context = CallContext(call_id, phone)
        direct_call = direct_search_call(transcript)
        if direct_call and not wants_booking(transcript):
            result = (
                await self.tools.execute(direct_call, context)
                if self.tools
                else {"ok": False, "error": "The booking database is not configured."}
            )
            return property_result_reply(result, transcript)
        if reply := inventory_clarification_reply(transcript):
            return reply
        for _ in range(2):
            response = await self.llm.generate(transcript, history, continuation)
            call = parse_tool_call(response)
            if call is None:
                if "<tool_call" in response.casefold():
                    return "Sorry, I couldn't complete that request. Please try again."
                return response
            call = ground_search_call(transcript, call)
            result = (
                await self.tools.execute(call, context)
                if self.tools
                else {"ok": False, "error": "The booking database is not configured."}
            )
            if call.name == "search_properties" and not wants_booking(transcript):
                return property_result_reply(result, transcript)
            if call.name == "book_appointment":
                return booking_result_reply(result, transcript)
            continuation.extend(
                [
                    {"role": "assistant", "content": tool_call_message(call)},
                    {
                        "role": "user",
                        "content": f"<tool_result>{json.dumps(result, ensure_ascii=False)}</tool_result>",
                    },
                ]
            )
        response = await self.llm.generate(transcript, history, continuation)
        return (
            "Sorry, I couldn't complete that request. Please try again."
            if parse_tool_call(response)
            else response
        )

    def _remember(self, call_id: str, transcript: str, response: str) -> None:
        history = self.history.setdefault(call_id, [])
        history.extend(
            [
                {"role": "user", "content": transcript},
                {"role": "assistant", "content": response},
            ]
        )
        del history[:-LLM_HISTORY_MESSAGES]

    async def _speak(self, call_id, text, output_track, generations) -> float:
        text = re.sub(r"\s+", " ", text).strip().rstrip(",;:")
        if not text:
            return 0
        generation = generations.get(call_id, 0)

        def forward(chunk: bytes, sample_rate: int) -> None:
            if generations.get(call_id, 0) == generation:
                output_track.add_pcm(chunk, sample_rate)

        seconds = await self.tts.speak(text, forward)
        return seconds if generations.get(call_id, 0) == generation else 0

    async def _discard_echo(
        self, input_track, output_track, audio_seconds: float
    ) -> None:
        if not audio_seconds:
            return
        deadline = (
            time.perf_counter()
            + min(audio_seconds, output_track.pending_seconds)
            + PLAYBACK_ECHO_TAIL_SECONDS
        )
        while time.perf_counter() < deadline:
            try:
                await asyncio.wait_for(
                    input_track.recv(),
                    timeout=max(0.01, deadline - time.perf_counter()),
                )
            except TimeoutError:
                break
            except Exception as exc:  # noqa: BLE001 - draining ends on any closed media backend.
                logger.info("Echo drain ended: %s", exc)
                break


class VoiceAgent:
    def __init__(self) -> None:
        self.pipeline = TurnPipeline()
        self.calls: dict[str, asyncio.Task] = {}
        self.generations: dict[str, int] = {}

    async def prewarm(self) -> None:
        await self.pipeline.prewarm()

    async def process(self, call_id, phone, input_track, output_track) -> None:
        await self.cancel(call_id)
        self.generations[call_id] = 0
        task = asyncio.create_task(
            self.pipeline.run(
                call_id, phone, input_track, output_track, self.generations
            ),
            name=f"call-{call_id}",
        )
        self.calls[call_id] = task
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Voice pipeline failed for %s", call_id)
        finally:
            self.calls.pop(call_id, None)
            self.generations.pop(call_id, None)
            call_log.end(call_id)

    async def cancel(self, call_id: str) -> None:
        task = self.calls.pop(call_id, None)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def close(self) -> None:
        await asyncio.gather(*(self.cancel(call_id) for call_id in list(self.calls)))
        await self.pipeline.close()


def repetitive(text: str) -> bool:
    words = re.findall(r"\S+", text.casefold())
    if len(words) < 8:
        return False
    for size in range(2, 6):
        grams = [
            " ".join(words[index : index + size])
            for index in range(len(words) - size + 1)
        ]
        if any(grams.count(gram) >= 3 for gram in set(grams)):
            return True
    return False


def language_selection_reply(transcript: str) -> str | None:
    """Answer a bare language choice immediately without invoking the LLM."""
    text = re.sub(r"[^\w\u0D80-\u0DFF\u0B80-\u0BFF]+", " ", transcript.casefold()).strip()
    if not text or len(text) > 40 or len(text.split()) > 4:
        return None
    if "සිංහල" in text or "sinhala" in text:
        return "හරි, අපි සිංහලෙන් කතා කරමු. ඔබට කොහොමද උදව් කරන්න ඕනේ?"
    if "english" in text:
        return "Sure, let's speak in English. How can I help you?"
    if "தமிழ்" in text or "tamil" in text:
        return "சரி, தமிழில் பேசலாம். நான் உங்களுக்கு எப்படி உதவலாம்?"
    return None


def acknowledgement_reply(transcript: str) -> str | None:
    """Keep short acknowledgements from turning into invented sales topics."""
    text = re.sub(r"[^\w\u0D80-\u0DFF]+", "", transcript.casefold())
    if text in {"ඔකේ", "ඕකේ", "හරි", "ඔව්", "okay", "ok", "yes", "yeah"}:
        if has_sinhala(transcript):
            return "හරි, ඔබට මොන වගේ දේකට උදව් ඕනේද?"
        return "Sure. What can I help you with?"
    return None


def direct_search_call(transcript: str) -> ToolCall | None:
    """Build an inventory search directly from explicit caller terms."""
    call = ground_search_call(transcript, ToolCall("search_properties", {}))
    arguments = dict(call.arguments)
    text = transcript.casefold()
    property_types = {
        "apartment": ("apartment", "apartments", "අපාර්ට්මන්ට්"),
        "villa": ("villa", "villas", "විලා", "house", "ගෙයක්"),
        "land": ("land", "ඉඩම", "ඉඩමක්"),
    }
    for property_type, aliases in property_types.items():
        if any(alias in text for alias in aliases):
            arguments["property_type"] = property_type
            break
    return ToolCall(call.name, arguments) if arguments else None


def inventory_clarification_reply(transcript: str) -> str | None:
    text = transcript.casefold()
    vague = ("මොකක් තියෙන්නේ", "මොනවද තියෙන්නේ", "what do you have", "what is available")
    if any(phrase in text for phrase in vague):
        return (
            "ඔබ බලන්නේ මොන location එකේ, මොන property type එකක්ද?"
            if has_sinhala(transcript)
            else "Which location and property type are you interested in?"
        )
    return None


def wants_booking(transcript: str) -> bool:
    text = transcript.casefold()
    return any(
        phrase in text
        for phrase in ("book", "viewing", "appointment", "බලන්න", "වෙලාවක්", "හමුවක්")
    )


def property_result_reply(result: dict, transcript: str) -> str:
    """Speak database facts exactly, without a second model generation."""
    sinhala = has_sinhala(transcript)
    if not result.get("ok"):
        error = result.get("error", "Property search failed.")
        return f"Property search එක කරන්න බැරි වුණා. {error}" if sinhala else str(error)
    rows = result.get("properties") or []
    if not rows:
        return (
            "ඒ විස්තරයට ගැළපෙන property එකක් දැන් නැහැ. වෙන location එකක් බලමුද?"
            if sinhala
            else "I couldn't find a matching property. Would you like another location?"
        )
    row = rows[0]
    bedrooms = row.get("bedrooms")
    bedroom_text = f", bedrooms {bedrooms}" if bedrooms else ""
    price = exact_price(row.get("price_lkr"))
    if sinhala:
        return (
            f"{row['location']} වල {row['name']} තියෙනවා{bedroom_text}. "
            f"මිල රුපියල් මිලියන {price}."
        )
    return (
        f"{row['name']} in {row['location']} is available{bedroom_text}, "
        f"priced at {price} million rupees."
    )


def booking_result_reply(result: dict, transcript: str) -> str:
    sinhala = has_sinhala(transcript)
    if not result.get("ok"):
        error = result.get("error", "The booking could not be completed.")
        return f"Booking එක කරන්න බැරි වුණා. {error}" if sinhala else str(error)
    appointment = result.get("appointment") or {}
    name = appointment.get("property_name", "the property")
    when = appointment.get("appointment_at", "the requested time")
    return (
        f"{name} viewing එක {when}ට confirm කළා."
        if sinhala
        else f"Your viewing for {name} is confirmed for {when}."
    )


def exact_price(value: object) -> str:
    millions = int(value or 0) / 1_000_000
    return f"{millions:g}"


def has_sinhala(text: str) -> bool:
    return bool(re.search(r"[\u0D80-\u0DFF]", text))


voice_agent = VoiceAgent()
