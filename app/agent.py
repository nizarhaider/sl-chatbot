import asyncio
import io
import logging
import os
import re
import wave

from google import genai
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.google.gemini_live.llm import (
    GeminiLiveLLMService,
    GeminiVADParams,
)
from pipecat.services.google.tts import GeminiTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from app.config import (
    DEFAULT_VOICE,
    GEMINI_LIVE_MODEL,
    INPUT_SAMPLE_RATE,
    LANGUAGE_CODE,
    OUTPUT_SAMPLE_RATE,
    RECORDING_SAMPLE_RATE,
)
from app.prompts import START_CALL, opening_greeting, system_prompt
from app.store import call_state
from app.tools import VOICE_TOOLS, DatabaseTools

greeting_audio = {}


def greeting_key(text):
    return re.sub(r"[^\w]+", "", text.casefold())


def agent_settings(agent_config):
    replicated = bool(os.environ.get("REPLICATED_VOICE_ID"))
    return GeminiLiveLLMService.Settings(
        model=GEMINI_LIVE_MODEL,
        voice=agent_config.get("voice", DEFAULT_VOICE),
        language=LANGUAGE_CODE,
        system_instruction=system_prompt(agent_config) + (
            " If the caller is unclear, ask them to repeat rather than guessing." if replicated else ""
        ),
        vad=GeminiVADParams(disabled=False, silence_duration_ms=750 if replicated else None),
    )


class CallGeminiService(GeminiLiveLLMService):
    def __init__(self, context, state=call_state, **kwargs):
        super().__init__(**kwargs)
        self.call_context, self.call_state = context, state
        self.first_speech = asyncio.Event()
        self.replicated = bool(os.environ.get("REPLICATED_VOICE_ID"))
        self.speaking = False
        self.interrupted = False

    async def process_frame(self, frame, direction):
        if self.replicated and isinstance(frame, BotStartedSpeakingFrame):
            self.speaking, self.interrupted = True, False
        elif self.replicated and isinstance(frame, BotStoppedSpeakingFrame):
            self.speaking, self.interrupted = False, False
        await super().process_frame(frame, direction)

    async def _handle_msg_input_transcription(self, message):
        if message.server_content and message.server_content.input_transcription and message.server_content.input_transcription.text:
            self.first_speech.set()
            if self.replicated and self.speaking and not self.interrupted:
                self.interrupted = True
                await self._handle_interruption()
                await self.broadcast_interruption()
        await super()._handle_msg_input_transcription(message)

    async def _handle_msg_usage_metadata(self, message):
        if message.usage_metadata:
            usage = message.usage_metadata.model_dump(mode="json", exclude_none=True)
            self.call_state.emit(
                self.call_context.call_id,
                "gemini_live.usage",
                {"total_tokens": usage.get("total_token_count"), "usage": usage},
            )
            self.call_state.persist(self.call_context.call_id)
        await super()._handle_msg_usage_metadata(message)


class LiveSpeechFilter(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(
            frame, (TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame, TTSTextFrame)
        ):
            return
        await self.push_frame(frame, direction)


class CachedGeminiTTSService(GeminiTTSService):
    async def run_tts(self, text, context_id):
        audio = greeting_audio.get(greeting_key(text))
        if audio:
            yield TTSAudioRawFrame(audio, OUTPUT_SAMPLE_RATE, 1, context_id=context_id)
        else:
            async for frame in super().run_tts(text, context_id):
                yield frame


async def prewarm():
    config = await DatabaseTools().agent_config()
    voice = os.environ.get("REPLICATED_VOICE_ID")
    if voice and os.environ.get("SERENDIB_INBOUND") == "1":
        try:
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            sentences = re.findall(r"[^.!?]+[.!?]?", opening_greeting(config))

            async def synthesize(text):
                result = await client.aio.models.generate_content(
                    model="gemini-3.8-flash-tts", contents=text,
                    config={"response_modalities": ["AUDIO"], "speech_config": {
                        "voice_config": {"prebuilt_voice_config": {"voice_name": voice}}
                    }},
                )
                audio = result.candidates[0].content.parts[0].inline_data.data
                if audio.startswith(b"RIFF"):
                    with wave.open(io.BytesIO(audio), "rb") as wav:
                        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (OUTPUT_SAMPLE_RATE, 1, 2):
                            raise ValueError("Unexpected greeting audio format")
                        audio = wav.readframes(wav.getnframes())
                greeting_audio[greeting_key(text)] = audio

            results = await asyncio.wait_for(
                asyncio.gather(*(synthesize(t.strip()) for t in sentences), return_exceptions=True), 15
            )
            if any(isinstance(result, BaseException) for result in results):
                logging.getLogger(__name__).warning("Greeting audio prewarm failed")
        except TimeoutError:
            logging.getLogger(__name__).warning("Greeting audio prewarm timed out")
    async with genai.Client(api_key=os.environ["GEMINI_API_KEY"]).aio.live.connect(
        model=GEMINI_LIVE_MODEL,
        config={
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {"prebuilt_voice_config": {"voice_name": DEFAULT_VOICE}}
            },
        },
    ):
        pass


def build_pipeline(transport, config, context, state=call_state):
    enabled = [tool for tool in VOICE_TOOLS if tool.name in config.get("enabled_tools", [])]
    llm = CallGeminiService(
        context,
        state,
        api_key=os.environ["GEMINI_API_KEY"],
        settings=agent_settings(config),
        tools=ToolsSchema(standard_tools=enabled),
        inference_on_context_initialization=not context.outbound,
    )
    history = LLMContext(messages=[{"role": "user", "content": START_CALL.format(greeting=opening_greeting(config))}])
    user, assistant = LLMContextAggregatorPair(
        history, realtime_service_mode=True,
        user_params=LLMUserAggregatorParams(user_turn_strategies=ExternalUserTurnStrategies()),
    )
    recording = AudioBufferProcessor(sample_rate=RECORDING_SAMPLE_RATE, num_channels=2)
    processors = [transport.input(), user, llm]
    voice = os.environ.get("REPLICATED_VOICE_ID")
    if voice:
        processors.extend([
            LiveSpeechFilter(),
            CachedGeminiTTSService(
                api_key=os.environ["GEMINI_API_KEY"],
                settings=GeminiTTSService.Settings(
                    model="gemini-3.8-flash-tts", voice=voice, language=None
                ),
            ),
        ])
    processors.extend([transport.output(), recording, assistant])
    worker = PipelineWorker(
        Pipeline(processors),
        params=PipelineParams(
            audio_in_sample_rate=INPUT_SAMPLE_RATE,
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        enable_rtvi=False,
        idle_timeout_secs=None,
    )

    async def execute(params):
        state.emit(context.call_id, "tool.call", {"name": params.function_name, "arguments": params.arguments})
        result = await DatabaseTools().execute(
            params.function_name, params.arguments, context
        )
        state.emit(context.call_id, "tool.result", {"name": params.function_name, "result": result})
        state.persist(context.call_id)
        await params.result_callback(result)
        if params.function_name == "end_call":
            await worker.stop_when_done()

    for tool in enabled:
        llm.register_function(tool.name, execute, cancel_on_interruption=False)

    @user.event_handler("on_user_turn_message_added")
    async def caller_turn(aggregator, message):
        state.transcript(context.call_id, "caller", message.content or "")

    @assistant.event_handler("on_assistant_turn_stopped")
    async def agent_turn(aggregator, message):
        state.transcript(context.call_id, "assistant", message.content or "")

    @worker.event_handler("on_pipeline_error")
    async def failed(worker, frame):
        state.emit(context.call_id, "pipeline.error", {"error": "Voice processing failed", "fatal": frame.fatal})

    @recording.event_handler("on_audio_data")
    async def archive(processor, audio, sample_rate, channels):
        state.archive(context.call_id, audio)

    return worker, recording, llm


async def run_agent(connection, context, ready, state=call_state, config=None):
    config = config or await DatabaseTools().agent_config(context)
    transport = SmallWebRTCTransport(
        connection,
        TransportParams(
            audio_in_enabled=True, audio_out_enabled=True, audio_out_10ms_chunks=2
        ),
    )
    worker, recording, llm = build_pipeline(transport, config, context, state)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    greeting_task = None

    async def begin_outbound():
        try:
            await asyncio.wait_for(llm.first_speech.wait(), 5)
        except TimeoutError:
            pass
        await worker.queue_frames([LLMMessagesAppendFrame(messages=[{"role": "user", "content": "Start now with the supplied greeting."}], run_llm=True)])

    @transport.event_handler("on_client_connected")
    async def connected(transport, client):
        nonlocal greeting_task
        await ready.wait()
        state.active(context.call_id)
        await recording.start_recording()
        if context.outbound:
            greeting_task = asyncio.create_task(begin_outbound())
        else:
            await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def disconnected(transport, client):
        await worker.cancel()

    try:
        await runner.run()
    finally:
        if greeting_task:
            greeting_task.cancel()
        await recording.stop_recording()
        state.end_call(context.call_id)
