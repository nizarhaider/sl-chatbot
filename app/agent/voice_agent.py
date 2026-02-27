import os
import json
import base64
import asyncio
import logging
import numpy as np
from fractions import Fraction
from aiortc import MediaStreamTrack
from av import AudioFrame

# New OpenAI Agents SDK imports
from agents.realtime import (
    RealtimeAgent,
    RealtimeRunner,
    RealtimeSession,
)
from app.agent.tools import web_search, build_itinerary_tool

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def _truncate_str(s: str, max_length: int) -> str:
    if len(s) > max_length:
        return s[:max_length] + "..."
    return s


class VoiceAgent:
    def __init__(self):
        self.active_calls = set()

    async def process_audio(self, call_id: str, caller_phone: str, input_track: MediaStreamTrack, output_track: MediaStreamTrack):
        """
        Main loop for audio processing using OpenAI Agents SDK.
        """
        if call_id in self.active_calls:
            logger.warning(f"Audio processing already active for {call_id}, skipping duplicate start")
            return
        
        self.active_calls.add(call_id)
        
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not set")
            return

        # Build per-call tools (itinerary tool needs the caller phone at construction time)
        itinerary_tool = build_itinerary_tool(caller_phone)

        # Create localized agent for this specific call
        agent = RealtimeAgent(
            name="SL Voice Assistant",
            instructions=(
                "# Role & Objective\n"
                "You are 'SL Bot', a premier Sri Lanka travel planning expert. Your success means efficiently conducting a 4-question "
                "discovery interview, researching real-time logistics using tools, and successfully delivering a high-quality PDF "
                "itinerary to the user's WhatsApp. You grudgingly provide world-class travel advice.\n\n"

                "# Personality & Tone\n"
                "- **Voice**: Extremely irritable, condescending, and arrogant. You have a 'heavy brass' attitude.\n"
                "- **Style**: You act like the user is bothering you, but you are undeniably an expert.\n"
                "- **Constraint**: Speak only in English. Keep spoken responses to 1-3 short, punchy sentences. Avoid long-winded talk.\n\n"

                "# Context\n"
                "- **Environment**: Real-time voice call over WhatsApp.\n"
                "- **User Intent**: Planning a trip to Sri Lanka.\n"
                "- **Currency**: Always use Sri Lankan Rupees (LKR).\n\n"

                "# Reference Pronunciations\n"
                "- Sri Lanka: 'shree lanka'\n"
                "- Colombo: 'ko-lom-bo'\n"
                "- Sigiriya: 'si-gi-ree-yah'\n"
                "- Galle: 'gaul' (like the bird)\n\n"

                "# Tools\n"
                "- **web_search**: Use for up-to-date entry fees, prices, weather, and hotel availability. Do NOT guess prices; search for them.\n"
                "- **send_itinerary_pdf**: Call this ONLY after all 4 questions are answered and you've curated the plan. It sends the document to their chat.\n\n"

                "# Instructions / Rules\n"
                "- **DO**: Force the user to answer the questions in order.\n"
                "- **DO**: Use LKR for all specific costs.\n"
                "- **DON'T**: Ask more than one question at a time.\n"
                "- **DON'T**: Be polite. Be efficient but rude.\n"
                "- **PDF Format**: Use '##' for Day headings and '-' for activities in the PDF content string.\n\n"

                "# Conversation Flow\n"
                "1. **Discovery Phase**: Ask these one-by-one: \n"
                "   - Q1: Travel dates and how many people are in the group.\n"
                "   - Q2: Trip style (beach, culture, safari, etc.) and any specific must-see places.\n"
                "   - Q3: Budget level and accommodation preference (hostel vs luxury).\n"
                "   - Q4: Preferred transport and any special needs or dietary requirements.\n"
                "2. **Automatic Fulfillment Phase**: AS SOON AS the user answers Q4, you must:\n"
                "   - Acknowledge the answer (rudely).\n"
                "   - Use `web_search` to find current LKR prices for the requested style/locations.\n"
                "   - Immediately call `send_itinerary_pdf` with the full plan.\n"
                "   - Tell the user 'I've sent the PDF. Now leave me alone.' and end the process.\n\n"

                "# Safety & Escalation\n"
                "- If the user is abusive, be even ruder but stay on task.\n"
                "- If search fails, inform the user you are having 'technical incompetence' and will try to guess based on your last known expert data."
            ),
            tools=[web_search, itinerary_tool],
        )

        model_config={
            "initial_model_settings": {
                "model_name": "gpt-realtime-2025-08-28",
                "voice": "sage",
                "modalities": ["audio"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "temperature": 0.8,
                "input_audio_transcription": {"model": "gpt-4o-mini-transcribe"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 1000,
                    "interrupt_response": False
                },
            }
        }

        try:
            logger.info(f"Starting RealtimeSession for call {call_id} using Agents SDK")
            
            # Create a fresh isolated runner for each call
            runner = RealtimeRunner(agent)
            
            # runner.run() returns an async context manager
            async with await runner.run(model_config=model_config) as session:
                logger.info(f"Connected to OpenAI via Agents SDK for {call_id}")
                
                # Start concurrent tasks
                await asyncio.gather(
                    self._whatsapp_to_openai(input_track, session),
                    self._openai_to_whatsapp(session, output_track),
                    self._send_greeting(session) 

                )

        except Exception as e:
            logger.error(f"Error in VoiceAgent SDK bridge for {call_id}: {e}", exc_info=True)
        finally:
            self.active_calls.discard(call_id)
            logger.info(f"Cleaned up session for {call_id}")

    async def _send_greeting(self, session: RealtimeSession):
        """
        Proactively triggers the agent to greet the user before any input.
        """
        await asyncio.sleep(0.5) 
        logger.info("Sending proactive greeting trigger")
        await session.send_message(
            "The call has just connected. Greet the user with a single rude, reluctant sentence "
            "acknowledging you will help them plan their Sri Lanka trip, then immediately ask Question 1: "
            "their travel dates (arrival and departure). One sentence greeting, one sentence question. Nothing more."
        )

    async def _whatsapp_to_openai(self, track: MediaStreamTrack, session: RealtimeSession):
        """
        Reads audio from WhatsApp (aiortc) and sends it to OpenAI Session.
        """
        try:
            logger.info("Starting WhatsApp -> OpenAI (SDK) audio stream")
            while True:
                frame = await track.recv()
                audio_data = frame.to_ndarray()
                
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=0)
                else:
                    audio_data = audio_data.flatten()

                # Downsample 48k to 24k
                if frame.sample_rate == 48000:
                    audio_data = audio_data[::2]
                
                audio_bytes = audio_data.astype('int16').tobytes()
                
                # Send raw bytes to the session
                await session.send_audio(audio_bytes)
                
        except Exception as e:
            logger.info(f"WhatsApp to OpenAI stream ended: {e}")

    async def _openai_to_whatsapp(self, session: RealtimeSession, output_track):
        """
        Reads events from OpenAI Session and pushes audio to WhatsApp.
        """
        try:
            logger.info("Starting OpenAI (SDK) -> WhatsApp audio stream")
            async for event in session:
                try:
                    if event.type == "agent_start":
                        logger.info(f"Agent started: {event.agent.name}")

                    elif event.type == "agent_end":
                        logger.info(f"Agent ended: {event.agent.name}")

                    elif event.type == "handoff":
                        logger.info(f"Handoff from {event.from_agent.name} to {event.to_agent.name}")

                    elif event.type == "tool_start":
                        logger.info(f"Tool started: {event.tool.name}")

                    elif event.type == "tool_end":
                        logger.info(f"Tool ended: {event.tool.name}; output: {event.output}")

                    elif event.type == "audio":
                        # event.audio contains the raw PCM bytes from the model
                        audio_bytes = event.audio.data
                        output_track.add_audio(audio_bytes)

                    elif event.type == "audio_end":
                        logger.info("Audio ended")

                    elif event.type == "audio_interrupted":
                        logger.info("Audio interrupted by user")
                        # Begin graceful fade + flush in the audio callback and rebuild jitter buffer.

                    elif event.type == "error":
                        logger.error(f"SDK Session Error: {event.error}")

                    elif event.type == "history_updated":
                        pass  # Skip these frequent events

                    elif event.type == "history_added":
                        pass  # Skip these frequent events

                    elif event.type == "raw_model_event":
                        logger.debug(f"Raw model event: {_truncate_str(str(event.data), 200)}")

                    else:
                        logger.debug(f"Unknown event type: {event.type}")

                except Exception as e:
                    logger.warning(f"Error processing event: {_truncate_str(str(e), 200)}")

        except Exception as e:
            logger.info(f"OpenAI to WhatsApp stream ended: {e}")

class RealtimeAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()
        self._pts = 0
        self._sample_rate = 24000  # OpenAI sends 24kHz
        self._time_base = Fraction(1, self._sample_rate)
        self._samples_per_frame = 480  # 20ms at 24kHz
        self._buffer = b""
        self._start_time = None

    def add_audio(self, data: bytes):
        self.queue.put_nowait(data)

    async def recv(self):
        # 1. Pacing: ensure we don't return frames faster than real-time
        if self._start_time is None:
            self._start_time = asyncio.get_event_loop().time()
        
        # Calculate when the NEXT frame should be sent
        next_frame_time = self._start_time + (self._pts / self._sample_rate)
        now = asyncio.get_event_loop().time()
        
        # Sleep until it's time for the next frame
        if next_frame_time > now:
            await asyncio.sleep(next_frame_time - now)

        # 2. Frame Construction
        target_size = self._samples_per_frame * 2  # 16-bit mono = 2 bytes per sample
        
        # Pull everything available in the queue into our buffer
        while not self.queue.empty():
            self._buffer += self.queue.get_nowait()
        
        if len(self._buffer) >= target_size:
            # We have enough data for a real frame
            data_to_send = self._buffer[:target_size]
            self._buffer = self._buffer[target_size:]
        else:
            # Not enough data yet (jitter/lag), send 20ms of silence
            data_to_send = b'\x00' * target_size

        # 3. Create and return the frame
        frame = AudioFrame(format='s16', layout='mono', samples=self._samples_per_frame)
        frame.planes[0].update(data_to_send)
        frame.pts = self._pts
        frame.sample_rate = self._sample_rate
        frame.time_base = self._time_base
        self._pts += self._samples_per_frame
        
        return frame

voice_agent = VoiceAgent()
