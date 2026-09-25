import asyncio
import hashlib
import hmac
import logging
import os
import time

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field, ValidationError

from app.gemini import RealtimeAudioTrack, voice_agent
from app.utility.helper import call_state
from app.utility.tools import whatsapp_api


class ImportantEventFilter(logging.Filter):
    IMPORTANT_PATTERNS = (
        "WEBHOOK_VERIFIED",
        "Received call event",
        "Processing SDP Offer",
        "Received audio track",
        "Connection state",
        "terminated by peer",
        "Turn transcript",
        "Turn dropped",
        "Turn response",
        "Gemini Live",
        "gemini_live",
        "Voice tool service",
        "Discarded",
        "input ended",
        "Stopping interrupted",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        message = record.getMessage()
        return any(pattern in message for pattern in self.IMPORTANT_PATTERNS)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    for noisy_logger in ("aioice", "httpx"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    configure_important_log()


def configure_important_log() -> None:
    path = "run_logs/important.log"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handler = RotatingFileHandler(
        path,
        maxBytes=1048576,
        backupCount=3,
    )
    handler.setLevel(logging.INFO)
    handler.addFilter(ImportantEventFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)


load_dotenv()
configure_logging()
logger = logging.getLogger(__name__)


async def prewarm_gemini_live(app: FastAPI) -> None:
    try:
        logger.info("Prewarming Gemini Live runtime")
        await voice_agent.prewarm_models()
        app.state.voice_ready = True
        logger.info("Gemini Live prewarm complete")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        app.state.voice_startup_error = str(exc)
        logger.exception("Gemini Live prewarm failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.voice_ready = False
    app.state.voice_startup_error = ""
    prewarm_task = asyncio.create_task(
        prewarm_gemini_live(app),
        name="gemini-live-prewarm",
    )
    try:
        yield
    finally:
        if not prewarm_task.done():
            prewarm_task.cancel()
            await asyncio.gather(prewarm_task, return_exceptions=True)
        call_state.close()


def create_app() -> FastAPI:
    app = FastAPI(title="WhatsApp Voice Bot", lifespan=lifespan)
    app.include_router(router)

    @app.get("/")
    def read_root(request: Request):
        if request.app.state.voice_startup_error:
            status = "error"
        elif request.app.state.voice_ready:
            status = "ready"
        else:
            status = "warming_up"
        return {"status": status, "message": "WhatsApp Webhook Server is running"}

    return app


router = APIRouter()

VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "my_secure_verify_token_123")


_demo_requests: dict[str, float] = {}


class DemoCall(BaseModel):
    id: str = Field(pattern=r"^[a-f0-9]{32}$")
    phone: str = Field(pattern=r"^[1-9][0-9]{7,14}$")
    timestamp: int


@router.post("/demo/call")
async def demo_call(request: Request):
    secret = os.environ.get("VERIFY_TOKEN", "")
    if os.environ.get("PORTAL_DEMO_ENABLED") != "1" or not secret:
        raise HTTPException(503, "Demo unavailable")
    raw = await request.body()
    expected = hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, request.headers.get("x-demo-signature", "")):
        raise HTTPException(403, "Invalid signature")
    try:
        payload = DemoCall.model_validate_json(raw)
    except ValidationError:
        raise HTTPException(400, "Invalid request")
    if abs(time.time() - payload.timestamp) > 60:
        raise HTTPException(403, "Expired request")
    if not request.app.state.voice_ready:
        raise HTTPException(503, "Demo unavailable")
    for key, expiry in list(_demo_requests.items()):
        if expiry < time.time():
            _demo_requests.pop(key, None)
    if payload.id in _demo_requests:
        raise HTTPException(409, "Demo already requested")
    _demo_requests[payload.id] = time.time() + 300
    try:
        call_id = await webrtc_service.dial(payload.id, payload.phone)
        return {"call_id": call_id}
    except Exception:
        logger.exception("Outbound demo connection failed")
        raise HTTPException(502, "Could not connect demo")


@router.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if not mode or not token:
        raise HTTPException(status_code=400, detail="Missing parameters")
    if mode != "subscribe" or token != VERIFY_TOKEN:
        raise HTTPException(status_code=403, detail="Verification token mismatch")

    logger.info("WEBHOOK_VERIFIED")
    return Response(content=challenge, media_type="text/plain")


@router.post("/webhook")
async def receive_webhook(request: Request):
    secret = os.environ.get("WHATSAPP_APP_SECRET")
    if secret:
        expected = "sha256=" + hmac.new(secret.encode(), await request.body(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, request.headers.get("x-hub-signature-256", "")):
            raise HTTPException(status_code=403, detail="Invalid webhook signature")
    body = await request.json()
    if body.get("object") != "whatsapp_business_account":
        raise HTTPException(status_code=404, detail="Not a WhatsApp API event")

    has_call_event = any(
        change.get("value", {}).get("calls")
        for entry in body.get("entry", [])
        for change in entry.get("changes", [])
    )
    if has_call_event and not request.app.state.voice_ready:
        logger.warning("Rejecting WhatsApp call event while voice models are not ready")
        return Response(content="VOICE_MODELS_NOT_READY", status_code=503)

    try:
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                await _handle_change(change.get("value", {}))
        return Response(content="EVENT_RECEIVED", status_code=200)
    except Exception as exc:
        logger.error("Error processing webhook: %s", exc, exc_info=True)
        return Response(content="ERROR", status_code=500)


async def _handle_change(value: dict) -> None:
    for status in value.get("statuses", []):
        logger.info("Received status update: %s", status)

    for message in value.get("messages", []):
        logger.info(
            "Ignoring unsupported WhatsApp message type: type=%s from=%s",
            message.get("type"),
            message.get("from"),
        )

    for call in value.get("calls", []):
        await _handle_call_event(call)


async def _handle_call_event(call: dict) -> None:
    event = call.get("event")
    call_id = call.get("id")
    caller_phone = call.get("from", "")
    logger.info("Received call event: %s for call %s from %s", event, call_id, caller_phone)

    if not call_id:
        logger.warning("Received call event without call id: %s", call)
        return

    if event == "connect":
        session = call.get("session", {})
        if session.get("sdp_type") == "answer":
            await webrtc_service.handle_answer(call_id, session.get("sdp", ""), call.get("biz_opaque_callback_data", ""))
        elif session.get("sdp_type") == "offer":
            logger.info("Processing SDP Offer for %s", call_id)
            asyncio.create_task(
                webrtc_service.handle_offer(call_id, session.get("sdp", ""), caller_phone)
            )
    elif event == "terminate":
        logger.info("Call %s terminated by peer, cleaning up.", call_id)
        asyncio.create_task(webrtc_service.close_call(call_id))


class WebRTCService:
    def __init__(self) -> None:
        self.pcs: dict[str, RTCPeerConnection] = {}
        self._caller_phones: dict[str, str] = {}
        self._outbound: dict[str, RTCPeerConnection] = {}
        self._dial_lock = asyncio.Lock()
        self._timers: dict[str, asyncio.Task] = {}

    async def handle_offer(self, call_id: str, sdp_offer: str, caller_phone: str = "") -> None:
        if call_id in self.pcs:
            return
        if len(self.pcs) >= int(os.environ.get("PORTAL_MAX_CALLS", "20")):
            await whatsapp_api.send_call_action(call_id, "reject")
            return
        await self.close_call(call_id)
        self._caller_phones[call_id] = caller_phone
        call_state.start_call(call_id, caller_phone)

        pc = RTCPeerConnection(configuration=_rtc_configuration())
        self.pcs[call_id] = pc
        output_track = RealtimeAudioTrack()
        pc.addTrack(output_track)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            logger.info("Connection state for %s is %s", call_id, pc.connectionState)
            call_state.emit(call_id, "webrtc.connection_state", {"state": pc.connectionState})
            if pc.connectionState in ["failed", "closed", "disconnected"]:
                await self.close_call(call_id, close_peer=pc.connectionState != "closed")

        @pc.on("track")
        def on_track(track) -> None:
            if track.kind != "audio":
                return
            phone = self._caller_phones.get(call_id, "")
            logger.info("Received audio track from WhatsApp for %s (caller: %s)", call_id, phone)
            call_state.emit(call_id, "webrtc.audio_track", {"caller_phone": phone})
            call_state.mark_call_active(call_id, phone)
            asyncio.create_task(voice_agent.process_audio(call_id, phone, track, output_track))

        offer = RTCSessionDescription(sdp=sdp_offer, type="offer")
        logger.info("Incoming audio SDP for %s: %s", call_id, _summarize_audio_sdp(sdp_offer))
        call_state.emit(call_id, "webrtc.offer_received", {"audio_sdp": _summarize_audio_sdp(sdp_offer)})
        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        refined_sdp = _refine_sdp(pc.localDescription.sdp)
        logger.info("Answer audio SDP for %s: %s", call_id, _summarize_audio_sdp(refined_sdp))
        call_state.emit(call_id, "webrtc.answer_created", {"audio_sdp": _summarize_audio_sdp(refined_sdp)})
        session = {"sdp": refined_sdp, "sdp_type": "answer"}

        if not await whatsapp_api.send_call_action(call_id, "pre_accept", session=session):
            call_state.emit(call_id, "whatsapp.pre_accept_failed", {})
            await self.close_call(call_id)
            return
        await whatsapp_api.send_call_action(call_id, "accept", session=session)
        call_state.emit(call_id, "whatsapp.accept_sent", {})

    async def dial(self, request_id: str, phone: str) -> str:
        async with self._dial_lock:
            if len(self.pcs) + len(self._outbound) >= int(os.environ.get("PORTAL_MAX_CALLS", "3")):
                raise RuntimeError("Demo capacity reached")
            pc = RTCPeerConnection(configuration=_rtc_configuration())
            self._outbound[request_id] = pc
        output_track = RealtimeAudioTrack()
        pc.addTrack(output_track)
        bound = asyncio.Event()
        call_id = ""

        @pc.on("track")
        def on_track(track):
            async def process():
                await bound.wait()
                if track.kind == "audio" and call_id in self.pcs:
                    call_state.mark_call_active(call_id, phone)
                    await voice_agent.process_audio(call_id, phone, track, output_track)
            asyncio.create_task(process())

        @pc.on("connectionstatechange")
        async def on_state():
            if call_id in self.pcs and pc.connectionState in ("failed", "closed", "disconnected"):
                await whatsapp_api.send_call_action(call_id, "terminate")
                await self.close_call(call_id, close_peer=pc.connectionState != "closed")

        try:
            await pc.setLocalDescription(await pc.createOffer())
            call_id = await whatsapp_api.initiate_call(phone, pc.localDescription.sdp, request_id)
            self.pcs[call_id] = pc
            self._caller_phones[call_id] = phone
            call_state.start_call(call_id, phone)
            call_state.emit(call_id, "whatsapp.outbound_started", {"request_id": request_id})
            bound.set()
            self._timers[call_id] = asyncio.create_task(self._end_demo(call_id))
            return call_id
        except Exception:
            bound.set()
            await pc.close()
            raise
        finally:
            self._outbound.pop(request_id, None)

    async def handle_answer(self, call_id: str, sdp: str, request_id: str = "") -> None:
        pc = self.pcs.get(call_id) or self._outbound.get(request_id)
        if pc is not None and pc.signalingState == "have-local-offer":
            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="answer"))

    async def _end_demo(self, call_id: str) -> None:
        await asyncio.sleep(180)
        await whatsapp_api.send_call_action(call_id, "terminate")
        await self.close_call(call_id)

    async def close_call(self, call_id: str, close_peer: bool = True) -> None:
        timer = self._timers.pop(call_id, None)
        if timer is not None and timer is not asyncio.current_task():
            timer.cancel()
        self._caller_phones.pop(call_id, None)
        pc = self.pcs.pop(call_id, None)
        await voice_agent.cancel_call(call_id)
        call_state.end_call(call_id)
        if close_peer and pc is not None:
            await pc.close()


def _rtc_configuration() -> RTCConfiguration:
    return RTCConfiguration(
        iceServers=[
            RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
            RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
        ]
    )


def _refine_sdp(sdp: str) -> str:
    refined_lines = []
    fingerprint_added = False

    for line in sdp.splitlines():
        if line.startswith("a=fingerprint:"):
            if not fingerprint_added and "sha-256" in line.lower():
                refined_lines.append(line.replace("sha-256", "SHA-256").replace("sha256", "SHA-256"))
                fingerprint_added = True
            continue
        if line.startswith("a=mid:"):
            refined_lines.append("a=mid:audio")
            continue
        if line.startswith("a=setup:"):
            refined_lines.append("a=setup:active")
            continue
        if line.startswith("a=group:BUNDLE"):
            refined_lines.append("a=group:BUNDLE audio")
            continue
        if line.startswith("o="):
            refined_lines.append(_fix_origin_address(line))
            continue
        if _drop_sdp_line(line):
            continue
        refined_lines.append(line)

    return "\r\n".join(refined_lines) + "\r\n"


def _fix_origin_address(line: str) -> str:
    parts = line.split()
    if len(parts) >= 6 and parts[5] == "0.0.0.0":
        parts[5] = "127.0.0.1"
        return " ".join(parts)
    return line


def _drop_sdp_line(line: str) -> bool:
    return any(
        token in line
        for token in (
            "a=extmap:",
            "a=msid-semantic:",
            "a=msid:",
            "a=ssrc:",
            "a=rtcp:",
            "c=IN IP4 0.0.0.0",
            "a=end-of-candidates",
        )
    )


def _summarize_audio_sdp(sdp: str) -> list[str]:
    audio_lines = []
    in_audio = False
    for line in sdp.splitlines():
        if line.startswith("m="):
            in_audio = line.startswith("m=audio")
            if in_audio:
                audio_lines.append(line)
            continue
        if in_audio and line.startswith(
            (
                "a=rtpmap:",
                "a=fmtp:",
                "a=ptime:",
                "a=maxptime:",
                "a=sendrecv",
                "a=sendonly",
                "a=recvonly",
                "a=inactive",
            )
        ):
            audio_lines.append(line)
    return audio_lines


webrtc_service = WebRTCService()
