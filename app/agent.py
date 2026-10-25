import os

from google import genai
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.services.google.gemini_live.llm import (
    GeminiLiveLLMService,
    GeminiVADParams,
)
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
from app.prompts import AUDITION_PROMPT, START_CALL, opening_greeting, system_prompt
from app.store import call_state
from app.tools import VOICE_TOOLS, DatabaseTools


def agent_settings(agent_config):
    return GeminiLiveLLMService.Settings(
        model=GEMINI_LIVE_MODEL,
        voice=agent_config.get("voice", DEFAULT_VOICE),
        language=LANGUAGE_CODE,
        system_instruction=system_prompt(agent_config),
        vad=GeminiVADParams(disabled=False),
    )


class CallGeminiService(GeminiLiveLLMService):
    def __init__(self, context, state=call_state, **kwargs):
        super().__init__(**kwargs)
        self.call_context, self.call_state = context, state

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


async def prewarm():
    config = await DatabaseTools().agent_config()
    opening_greeting(config)
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


def build_pipeline(transport, config, context, state=call_state, greet=True):
    enabled = [
        tool for tool in VOICE_TOOLS if tool.name in config.get("enabled_tools", [])
    ]
    schemas = ToolsSchema(standard_tools=enabled)
    llm = CallGeminiService(
        context,
        state,
        api_key=os.environ["GEMINI_API_KEY"],
        settings=agent_settings(config),
        tools=schemas,
        inference_on_context_initialization=greet,
    )
    history = LLMContext(
        messages=[
            {
                "role": "user",
                "content": START_CALL.format(greeting=opening_greeting(config))
                if greet
                else AUDITION_PROMPT,
            }
        ],
    )
    user, assistant = LLMContextAggregatorPair(
        history,
        realtime_service_mode=True,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=ExternalUserTurnStrategies()
        ),
    )
    recording = AudioBufferProcessor(sample_rate=RECORDING_SAMPLE_RATE, num_channels=2)
    worker = PipelineWorker(
        Pipeline(
            [transport.input(), user, llm, transport.output(), recording, assistant]
        ),
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
        state.emit(
            context.call_id,
            "tool.call",
            {"name": params.function_name, "arguments": params.arguments},
        )
        result = await DatabaseTools().execute(
            params.function_name, params.arguments, context
        )
        state.emit(
            context.call_id,
            "tool.result",
            {"name": params.function_name, "result": result},
        )
        state.persist(context.call_id)
        await params.result_callback(result)

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
        state.emit(
            context.call_id,
            "pipeline.error",
            {"error": "Voice processing failed", "fatal": frame.fatal},
        )

    @recording.event_handler("on_audio_data")
    async def archive(processor, audio, sample_rate, channels):
        state.archive(context.call_id, audio)

    return worker, recording


async def run_agent(connection, context, ready, state=call_state, config=None):
    config = config or await DatabaseTools().agent_config(context)
    transport = SmallWebRTCTransport(
        connection,
        TransportParams(
            audio_in_enabled=True, audio_out_enabled=True, audio_out_10ms_chunks=2
        ),
    )
    worker, recording = build_pipeline(transport, config, context, state)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def connected(transport, client):
        await ready.wait()
        state.active(context.call_id)
        await recording.start_recording()
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def disconnected(transport, client):
        await worker.cancel()

    try:
        await runner.run()
    finally:
        await recording.stop_recording()
        state.end_call(context.call_id)
