"""One conversational turn pipeline and active-call owner."""

from __future__ import annotations

import asyncio
import logging
import re
import time

import numpy as np
from aiortc import MediaStreamTrack
from av.audio.resampler import AudioResampler

from app.agent import GemmaAgentRuntime
from app.audio import OutboundAudioTrack, VoiceActivity, pcm_rms
from app.config import (
    END_SILENCE_CHUNKS,
    GREETING,
    GREETING_DELAY_SECONDS,
    INPUT_CHUNK_BYTES,
    MIN_AUDIO_MS,
    PLAYBACK_ECHO_TAIL_SECONDS,
    PROGRESS_DELAY_SECONDS,
    PROGRESS_LINES,
    PROGRESS_REPEAT_SECONDS,
    SILENCE_RMS,
)
from app.database import call_log
from app.models import LocalWhisperASR, OmniVoiceTTS, is_noise_text
from app.speech import detect_language

logger = logging.getLogger(__name__)


class TurnPipeline:
    def __init__(self, tts: OmniVoiceTTS | None = None) -> None:
        self.asr = LocalWhisperASR()
        self.agent = GemmaAgentRuntime()
        self.tts = tts or OmniVoiceTTS()
        self.progress_index: dict[str, int] = {}

    async def prewarm(self) -> None:
        await asyncio.to_thread(self.asr.prewarm)
        await self.agent.prewarm()
        await self.tts.prewarm()

    async def close(self) -> None:
        await self.agent.close()

    async def run(
        self,
        call_id: str,
        phone: str,
        input_track: MediaStreamTrack,
        output_track: OutboundAudioTrack,
        playback_generation: dict[str, int],
    ) -> None:
        await self.agent.start_session(call_id, phone)
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
            await self.agent.end_session(call_id)
            self.progress_index.pop(call_id, None)

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
        language = detect_language(transcript)
        last_notice = 0.0
        notice_count = 0
        notice_lock = asyncio.Lock()

        async def progress(followup: bool = False) -> None:
            nonlocal last_notice, notice_count
            async with notice_lock:
                if notice_count and not followup:
                    return
                if (
                    followup
                    and time.perf_counter() - last_notice < PROGRESS_REPEAT_SECONDS
                ):
                    return
                lines = PROGRESS_LINES[language]
                index = self.progress_index.get(call_id, 0)
                line = lines[index % len(lines)]
                self.progress_index[call_id] = index + 1
                call_log.add(call_id, "assistant", line)
                logger.info("Turn progress for %s: %s", call_id, line)
                seconds = await self._speak(
                    call_id, line, output_track, generations, language
                )
                last_notice = time.perf_counter()
                notice_count += 1
                await self._discard_echo(input_track, output_track, seconds)

        response_task = asyncio.create_task(
            self._respond(call_id, phone, transcript, progress)
        )
        try:
            try:
                response = await asyncio.wait_for(
                    asyncio.shield(response_task), PROGRESS_DELAY_SECONDS
                )
            except TimeoutError:
                await progress()
                while True:
                    response, interrupted = await self._wait_for_response_or_speech(
                        response_task,
                        call_id,
                        input_track,
                        output_track,
                        generations,
                        PROGRESS_REPEAT_SECONDS,
                    )
                    if interrupted:
                        response_task.cancel()
                        await asyncio.gather(response_task, return_exceptions=True)
                        await self._handle_turn(
                            call_id,
                            phone,
                            input_track,
                            output_track,
                            generations,
                            interrupted.started_at,
                            interrupted.pcm,
                        )
                        return
                    if response is not None:
                        break
                    await progress(followup=True)
        finally:
            if not response_task.done():
                response_task.cancel()
                await asyncio.gather(response_task, return_exceptions=True)
        llm_ms = (time.perf_counter() - stage_started) * 1000
        if not response or is_noise_text(response) or repetitive(response):
            return
        logger.info("Turn response for %s: %s", call_id, response)
        call_log.add(call_id, "assistant", response)

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

    async def _wait_for_response_or_speech(
        self,
        response_task,
        call_id,
        input_track,
        output_track,
        generations,
        timeout,
    ):
        """Keep listening during slow work so caller speech can cancel the turn."""
        vad = VoiceActivity()
        resampler = AudioResampler(format="s16", layout="mono", rate=16_000)
        buffer = bytearray()
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return None, None
            receive = asyncio.create_task(input_track.recv())
            done, _ = await asyncio.wait(
                (response_task, receive),
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if response_task in done:
                receive.cancel()
                await asyncio.gather(receive, return_exceptions=True)
                return response_task.result(), None
            if receive not in done:
                receive.cancel()
                await asyncio.gather(receive, return_exceptions=True)
                return None, None
            try:
                frame = receive.result()
            except Exception as exc:
                logger.info("Input ended during slow turn for %s: %s", call_id, exc)
                response_task.cancel()
                raise
            for converted in resampler.resample(frame):
                buffer.extend(converted.to_ndarray().tobytes())
            while len(buffer) >= INPUT_CHUNK_BYTES:
                chunk = bytes(buffer[:INPUT_CHUNK_BYTES])
                del buffer[:INPUT_CHUNK_BYTES]
                speech = pcm_rms(chunk) > SILENCE_RMS
                if speech and not vad.speaking:
                    logger.info("Turn interrupted by caller speech")
                    vad.start()
                    generations[call_id] = generations.get(call_id, 0) + 1
                    output_track.clear()
                if vad.speaking:
                    vad.add(chunk, speech)
                if vad.speaking and vad.silence_chunks >= END_SILENCE_CHUNKS:
                    return None, vad.finish()

    async def _respond(
        self, call_id: str, phone: str, transcript: str, on_tool=None
    ) -> str:
        return await self.agent.respond(call_id, phone, transcript, on_tool)

    async def _speak(
        self, call_id, text, output_track, generations, language=None
    ) -> float:
        text = re.sub(r"\s+", " ", text).strip().rstrip(",;:")
        if not text:
            return 0
        generation = generations.get(call_id, 0)

        def forward(chunk: bytes, sample_rate: int) -> None:
            if generations.get(call_id, 0) == generation:
                output_track.add_pcm(chunk, sample_rate)

        seconds = await self.tts.speak(text, forward, language)
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


voice_agent = VoiceAgent()
