import asyncio
import logging
import os
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo

from av.audio.resampler import AudioResampler
from google import genai
from google.genai import types
import numpy as np
from websockets.exceptions import ConnectionClosed

from app.dashboard.state import dashboard_state
from app.voice.audio_archive import CallAudioArchive, CallAudioRecorder
from app.voice.config import VOICE_AGENT_PROMPT
from app.voice.tools import CallContext, LLM_TOOLS, RealEstateToolService

logger = logging.getLogger(__name__)

GEMINI_LIVE_MODEL = "gemini-3.1-flash-live-preview"
INPUT_RATE = 16_000
OUTPUT_RATE = 24_000
INPUT_CHUNK_BYTES = INPUT_RATE * 2 // 10  # 100 ms of mono PCM16.
SPEECH_RMS_THRESHOLD = 650
SPEECH_START_CHUNKS = 2
TURN_END_SILENCE_CHUNKS = 7  # 700 ms at the 100 ms input chunk size.
PREFIX_CHUNKS = 3
MAX_PLAYBACK_BUFFER_SECONDS = 0.8
LIVE_SESSION_ATTEMPTS = 2


class GeminiLivePipeline:
    """One Gemini Live session per WhatsApp call.

    Gemini performs the speech recognition, turn detection, reasoning, and speech
    synthesis. The runtime only bridges WebRTC PCM and executes the existing
    property/appointment functions requested by Gemini.
    """

    def __init__(self, interrupt_playback) -> None:
        self._interrupt_playback = interrupt_playback
        self._tools = RealEstateToolService.from_env()
        self._audio_archive = CallAudioArchive()
        self._client: genai.Client | None = None

    async def prewarm_models(self) -> None:
        client = self._get_client()
        if self._tools is not None:
            await self._tools.ensure_ready()
            logger.info("Voice tool service ready")
        # Establishing the WebSocket catches invalid keys/model access before a
        # caller reaches the webhook. No media or model turn is generated.
        async with client.aio.live.connect(
            model=GEMINI_LIVE_MODEL,
            config=self._session_config(),
        ):
            logger.info("Gemini Live prewarm connection established")

    async def run(self, call_id, caller_phone, input_track, output_track, playback_generation, recorder: CallAudioRecorder):
        client = self._get_client()
        context = CallContext(call_id=call_id, caller_phone=caller_phone)
        dashboard_state.emit(call_id, "gemini_live.connecting", {"model": GEMINI_LIVE_MODEL})
        try:
            for attempt in range(1, LIVE_SESSION_ATTEMPTS + 1):
                try:
                    async with client.aio.live.connect(
                        model=GEMINI_LIVE_MODEL,
                        config=self._session_config(),
                    ) as session:
                        dashboard_state.emit(call_id, "gemini_live.connected", {"model": GEMINI_LIVE_MODEL, "attempt": attempt})
                        # Gemini Live generates the multilingual opening directly; it is
                        # deliberately a Live input rather than a local prerecorded/TTS greeting.
                        if attempt == 1:
                            await session.send_realtime_input(
                                text=(
                                    "Start the phone call now. Say only the language-selection greeting: "
                                    "ask the caller to say English, Sinhala, or Tamil."
                                )
                            )
                        receive_task = asyncio.create_task(
                            self._receive(session, call_id, context, output_track, playback_generation),
                            name=f"gemini-receive-{call_id}-{attempt}",
                        )
                        try:
                            await self._send_input(session, call_id, input_track, recorder)
                            return
                        finally:
                            receive_task.cancel()
                            await asyncio.gather(receive_task, return_exceptions=True)
                except ConnectionClosed as exc:
                    if attempt == LIVE_SESSION_ATTEMPTS:
                        raise
                    logger.warning("Gemini Live closed for %s; reconnecting once: %s", call_id, exc)
                    dashboard_state.emit(call_id, "gemini_live.reconnecting", {"attempt": attempt, "reason": "connection_closed"})
                    await asyncio.sleep(0.25)
        finally:
            self._audio_archive.archive_call(call_id, recorder)

    def _get_client(self) -> genai.Client:
        if self._client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is required for the Gemini Live voice runtime")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _session_config(self) -> dict:
        declarations = [tool["function"] for tool in LLM_TOOLS]
        today = datetime.now(ZoneInfo("Asia/Colombo")).date().isoformat()
        return {
            "response_modalities": ["AUDIO"],
            "system_instruction": (
                f"{VOICE_AGENT_PROMPT}\n\n"
                f"Today is {today} in Asia/Colombo. You are a live phone agent: "
                "respond with native audio only. Do not mention transcripts, tools, or "
                "implementation details. Speak as a warm, professional Sri Lankan woman. "
                "Use natural Sri Lankan English and Sinhala pronunciation; never imitate an "
                "American accent. Start with the language-selection greeting. Once the caller "
                "chooses English, Sinhala, or Tamil, acknowledge that choice in the selected "
                "language and immediately ask what property they are looking for. Listen to and "
                "respond to every caller turn; never wait for text input or an external language "
                "selection signal."
            ),
            "speech_config": {
                "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}
            },
            "input_audio_transcription": {},
            "output_audio_transcription": {},
            "realtime_input_config": {
                "automatic_activity_detection": {
                    "disabled": True,
                }
            },
            "tools": [{"function_declarations": declarations}],
        }

    async def _send_input(self, session, call_id, input_track, recorder: CallAudioRecorder) -> None:
        resampler = AudioResampler(format="s16", layout="mono", rate=INPUT_RATE)
        buffer = bytearray()
        prefix: deque[bytes] = deque(maxlen=PREFIX_CHUNKS)
        speaking = False
        speech_chunks = 0
        silence_chunks = 0
        while True:
            try:
                frame = await input_track.recv()
            except Exception as exc:
                logger.info("Gemini Live input ended for %s: %s", call_id, exc)
                if speaking:
                    await session.send_realtime_input(activity_end=types.ActivityEnd())
                return
            for resampled in resampler.resample(frame):
                pcm = resampled.to_ndarray().tobytes()
                recorder.add_caller_pcm(pcm)
                buffer.extend(pcm)
                while len(buffer) >= INPUT_CHUNK_BYTES:
                    chunk = bytes(buffer[:INPUT_CHUNK_BYTES])
                    del buffer[:INPUT_CHUNK_BYTES]
                    rms = _pcm_rms(chunk)
                    if not speaking:
                        prefix.append(chunk)
                        if rms < SPEECH_RMS_THRESHOLD:
                            speech_chunks = 0
                            continue
                        speech_chunks += 1
                        if speech_chunks < SPEECH_START_CHUNKS:
                            continue
                        speaking = True
                        speech_chunks = 0
                        silence_chunks = 0
                        dashboard_state.emit(call_id, "pipeline.speech_started", {"provider": "audio_activity"})
                        logger.info("Gemini Live activity start for %s (rms=%.0f)", call_id, rms)
                        await session.send_realtime_input(activity_start=types.ActivityStart())
                        for buffered_chunk in prefix:
                            await session.send_realtime_input(
                                audio=types.Blob(data=buffered_chunk, mime_type="audio/pcm;rate=16000")
                            )
                        prefix.clear()
                        continue
                    await session.send_realtime_input(
                        audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                    )
                    if rms >= SPEECH_RMS_THRESHOLD:
                        silence_chunks = 0
                        continue
                    silence_chunks += 1
                    if silence_chunks >= TURN_END_SILENCE_CHUNKS:
                        speaking = False
                        silence_chunks = 0
                        dashboard_state.emit(call_id, "pipeline.speech_ended", {"provider": "audio_activity"})
                        logger.info("Gemini Live activity end for %s", call_id)
                        await session.send_realtime_input(activity_end=types.ActivityEnd())

    async def _receive(self, session, call_id, context, output_track, playback_generation) -> None:
        caller_text: list[str] = []
        assistant_text: list[str] = []
        # The SDK exposes one receive iterator per completed Live turn.  Start
        # the next iterator after each turn so a call remains conversational
        # after Gemini has delivered its opening greeting.
        while True:
            async for response in session.receive():
                if response.tool_call:
                    await self._handle_tool_calls(session, response.tool_call.function_calls, call_id, context)
                content = response.server_content
                if content is None:
                    continue
                if content.interrupted:
                    self._interrupt_playback(call_id, output_track)
                    dashboard_state.emit(call_id, "gemini_live.interrupted", {})
                if content.input_transcription and content.input_transcription.text:
                    caller_text.append(content.input_transcription.text)
                if content.output_transcription and content.output_transcription.text:
                    assistant_text.append(content.output_transcription.text)
                if content.model_turn:
                    for part in content.model_turn.parts or []:
                        if part.inline_data and part.inline_data.data:
                            while output_track.pending_audio_seconds >= MAX_PLAYBACK_BUFFER_SECONDS:
                                await asyncio.sleep(0.02)
                            output_track.add_pcm_audio(part.inline_data.data, OUTPUT_RATE)
                if content.turn_complete:
                    logger.info("Gemini Live turn complete for %s", call_id)
                    self._publish_transcripts(call_id, caller_text, assistant_text)
                    caller_text.clear()
                    assistant_text.clear()

    async def _handle_tool_calls(self, session, calls, call_id: str, context: CallContext) -> None:
        responses = []
        for call in calls or []:
            arguments = dict(call.args or {})
            dashboard_state.emit(call_id, "tool.call", {"name": call.name, "arguments": arguments})
            if self._tools is None:
                result = {"ok": False, "error": "Property tools are not configured."}
            else:
                result = await self._tools.execute(call.name, arguments, context)
            dashboard_state.emit(call_id, "tool.result", {"name": call.name, "result": result})
            # Live API function responses use a `result` envelope. Supplying the
            # raw tool object leaves Gemini waiting for a completed tool turn.
            responses.append(types.FunctionResponse(id=call.id, name=call.name, response={"result": result}))
        if responses:
            await session.send_tool_response(function_responses=responses)

    @staticmethod
    def _publish_transcripts(call_id: str, caller_text: list[str], assistant_text: list[str]) -> None:
        caller = _join_transcript(caller_text)
        assistant = _join_transcript(assistant_text)
        if caller:
            dashboard_state.add_transcript(call_id, "caller", caller)
            dashboard_state.emit(call_id, "pipeline.asr_complete", {"text": caller, "provider": "gemini_live"})
        if assistant:
            dashboard_state.add_transcript(call_id, "assistant", assistant)
            dashboard_state.emit(call_id, "pipeline.response_ready", {"text": assistant, "provider": "gemini_live"})


def _pcm_rms(pcm: bytes) -> float:
    samples = np.frombuffer(pcm, dtype=np.int16)
    if not samples.size:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def _join_transcript(chunks: list[str]) -> str:
    """Live transcription chunks can be cumulative or incremental by model version."""
    merged = ""
    for chunk in chunks:
        text = chunk.strip()
        if not text:
            continue
        if text.startswith(merged):
            merged = text
        elif not merged.endswith(text):
            merged = f"{merged} {text}".strip()
    return merged
