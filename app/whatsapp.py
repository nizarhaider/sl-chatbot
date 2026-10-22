import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import time

import httpx

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response

from app.gemini import RealtimeAudioTrack, voice_agent
from app.utility.helper import call_state
from app.utility.tools import CallContext, claim_case, confirm_case, connection, whatsapp_api


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    for noisy_logger in ("aioice", "httpx"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


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


@router.post("/demo/call")
async def demo_call(request: Request):
    secret = os.environ.get("DEMO_SIGNING_SECRET")
    if os.environ.get("PORTAL_DEMO_ENABLED") != "1" or not secret:
        raise HTTPException(status_code=404, detail="Demo calling is disabled")
    body = await request.body()
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, request.headers.get("x-demo-signature", "")):
        raise HTTPException(status_code=403, detail="Invalid signature")
    data = json.loads(body)
    if not re.fullmatch(r"[a-f0-9]{32}", data.get("id", "")) or abs(time.time() - data.get("timestamp", 0)) > 60:
        raise HTTPException(status_code=400, detail="Invalid demo request")
    if not request.app.state.voice_ready or len(webrtc_service.pcs) >= int(os.environ.get("MAX_CALLS", "20")):
        raise HTTPException(status_code=503, detail="Voice agent unavailable")
    with connection() as db:
        claimed = db.execute(
            "update portal_demo_requests set call_id='pending' where id=%s and phone=%s and state='calling' and call_id is null and expires_at>now() returning id",
            (data["id"], data.get("phone")),
        ).fetchone()
    if not claimed:
        raise HTTPException(status_code=409, detail="Demo already handled or expired")
    try:
        call_id = await webrtc_service.start_demo(data["id"], data["phone"])
        with connection() as db:
            db.execute("update portal_demo_requests set call_id=%s where id=%s", (call_id, data["id"]))
        return {"call_id": call_id}
    except Exception:
        with connection() as db:
            db.execute("update portal_demo_requests set state='failed',call_id=null where id=%s", (data["id"],))
        logger.exception("Could not start demo call")
        raise HTTPException(status_code=502, detail="WhatsApp could not start the call")


@router.post("/cases/call")
async def case_call(request: Request):
    secret = os.environ.get("DEMO_SIGNING_SECRET")
    if not secret or os.environ.get("PORTAL_DEMO_ENABLED") != "1":
        raise HTTPException(status_code=404, detail="Demo calling is disabled")
    body = await request.body()
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, request.headers.get("x-demo-signature", "")):
        raise HTTPException(status_code=403, detail="Invalid signature")
    data = json.loads(body)
    if not re.fullmatch(r"[a-f0-9]{32}", data.get("id", "")) or abs(time.time() - data.get("timestamp", 0)) > 60:
        raise HTTPException(status_code=400, detail="Invalid demo request")
    if not request.app.state.voice_ready:
        raise HTTPException(status_code=503, detail="Voice agent unavailable")
    with connection() as db:
        row = db.execute(
            """update portal_case_requests set state='preparing' where id=%s and phone=%s
               and runtime_agent_id=%s and direction='outbound' and state='pending' and expires_at>now() returning id""",
            (data["id"], data.get("phone"), os.environ["VOICE_AGENT_ID"]),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=409, detail="Demo already handled or expired")
    await webrtc_service.request_case(data["id"])
    return {"ok": True}


async def relay_demo_events(body: dict) -> None:
    secret = os.environ.get("DEMO_SIGNING_SECRET")
    if os.environ.get("PORTAL_DEMO_ENABLED") != "1" or not secret:
        return
    if not any(
        value.get("messages") or value.get("statuses") or any(call.get("event") == "terminate" for call in value.get("calls", []))
        for entry in body.get("entry", []) for change in entry.get("changes", [])
        for value in [change.get("value", {})]
    ):
        return
    payload = json.dumps(body).encode()
    signature = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    async with httpx.AsyncClient(timeout=35) as client:
        response = await client.post(
            f"{os.environ['PORTAL_URL']}/api/demo/webhook", content=payload,
            headers={"Content-Type": "application/json", "x-demo-signature": signature},
        )
        response.raise_for_status()


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
        await relay_demo_events(body)
        return Response(content="EVENT_RECEIVED", status_code=200)
    except Exception as exc:
        logger.error("Error processing webhook: %s", exc, exc_info=True)
        return Response(content="ERROR", status_code=500)


async def _handle_change(value: dict) -> None:
    for status in value.get("statuses", []):
        logger.info("Received status update: %s", status)
        if status.get("status") == "ACCEPTED":
            webrtc_service.mark_answered(status.get("id"))
        if status.get("status") in ("failed", "REJECTED"):
            with connection() as db:
                db.execute(
                    """update portal_case_requests set state='failed',error='WhatsApp could not deliver the permission request or call. Please try again later.'
                       where runtime_agent_id=%s and direction='outbound' and state in ('permission','calling') and (message_id=%s or call_id=%s)""",
                    (os.environ["VOICE_AGENT_ID"], status.get("id"), status.get("id")),
                )

    for message in value.get("messages", []):
        reply = message.get("interactive", {}).get("call_permission_reply", {})
        if reply.get("response_source") == "user_action" and message.get("context", {}).get("id"):
            with connection() as db:
                row = db.execute(
                    """select id from portal_case_requests where phone=%s and message_id=%s
                       and runtime_agent_id=%s and direction='outbound' and state='permission' and expires_at>now()""",
                    (message.get("from"), message["context"]["id"], os.environ["VOICE_AGENT_ID"]),
                ).fetchone()
                if row and reply.get("response") == "reject":
                    db.execute("update portal_case_requests set state='declined' where id=%s", (row["id"],))
            if row and reply.get("response") == "accept":
                await webrtc_service.start_case(row["id"])
            continue
        token = re.fullmatch(r"CASE ([a-f0-9]{32})", message.get("text", {}).get("body", "").strip(), re.IGNORECASE)
        if token:
            selected = await confirm_case(message.get("from", ""), token[1].lower())
            if selected:
                await whatsapp_api.send_text_message(message["from"], f"Your {selected['title']} demo is ready. Tap the voice call button in this chat within ten minutes. This AI demo call will be recorded and transcribed. Use fictional details.")
            continue
        logger.info(
            "Ignoring unsupported WhatsApp message type: type=%s from=%s",
            message.get("type"),
            message.get("from"),
        )

    for call in value.get("calls", []):
        if call.get("event") == "terminate":
            with connection() as db:
                db.execute(
                    "update portal_case_requests set state=%s where call_id=%s and runtime_agent_id=%s and direction='outbound' and state='calling'",
                    ("failed" if call.get("status") == "FAILED" else "completed", call.get("id"), os.environ["VOICE_AGENT_ID"]),
                )
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
        if session.get("sdp_type") == "answer" and call_id in webrtc_service.pcs:
            await webrtc_service.pcs[call_id].setRemoteDescription(RTCSessionDescription(sdp=session["sdp"], type="answer"))
            webrtc_service.mark_answered(call_id)
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
        self._demo_timers: dict[str, asyncio.TimerHandle] = {}
        self._demo_answered: dict[str, asyncio.Event] = {}

    def _case_request(self, request_id: str):
        with connection() as db:
            return db.execute(
                """select q.*,a.id as agent_id,a.customer_id from portal_case_requests q
                   join portal_agent_cases c on c.slug=q.case_slug and c.runtime_agent_id=q.runtime_agent_id
                   join portal_agents a on a.id=c.agent_id where q.id=%s and q.runtime_agent_id=%s
                   and q.direction='outbound' and q.expires_at>now() and a.status<>'archived'""",
                (request_id, os.environ["VOICE_AGENT_ID"]),
            ).fetchone()

    def _fail_case(self, request_id: str, message: str):
        with connection() as db:
            db.execute("update portal_case_requests set state='failed',error=%s where id=%s and runtime_agent_id=%s", (message, request_id, os.environ["VOICE_AGENT_ID"]))

    async def request_case(self, request_id: str):
        row = self._case_request(request_id)
        if not row or row["state"] != "preparing":
            return
        try:
            permission = await whatsapp_api.call_permissions(row["phone"])
            allowed = {a["action_name"] for a in permission.get("actions", []) if a.get("can_perform_action")}
            if "start_call" in allowed:
                await self.start_case(request_id)
            elif "send_call_permission_request" in allowed:
                message_id = await whatsapp_api.request_call_permission(row["phone"])
                with connection() as db:
                    db.execute("update portal_case_requests set state='permission',message_id=%s where id=%s and state='preparing'", (message_id, request_id))
            else:
                self._fail_case(request_id, "WhatsApp cannot request call permission for this number right now. Please try again later.")
        except Exception:
            logger.exception("Could not prepare case callback")
            self._fail_case(request_id, "WhatsApp could not prepare your callback. Please check your number and try again.")

    async def start_case(self, request_id: str):
        row = self._case_request(request_id)
        if not row or row["state"] not in ("preparing", "permission"):
            return
        try:
            permission = await whatsapp_api.call_permissions(row["phone"])
            if not any(a.get("action_name") == "start_call" and a.get("can_perform_action") for a in permission.get("actions", [])):
                return
            if len(self.pcs) >= int(os.environ.get("MAX_CALLS", "20")):
                self._fail_case(request_id, "Our demo agents are busy. Please try again shortly.")
                return
            with connection() as db:
                claimed = db.execute("update portal_case_requests set state='calling' where id=%s and state in ('preparing','permission') and expires_at>now() returning id", (request_id,)).fetchone()
            if not claimed:
                return
            context = CallContext("", row["phone"], str(row["agent_id"]), str(row["customer_id"]))
            call_id = await self.start_demo(request_id, row["phone"], context)
            with connection() as db:
                db.execute("update portal_case_requests set call_id=%s where id=%s", (call_id, request_id))
        except Exception:
            logger.exception("Could not start case callback")
            self._fail_case(request_id, "WhatsApp could not place your callback. Please try again shortly.")

    async def start_demo(self, demo_id: str, phone: str, context=None) -> str:
        pc = RTCPeerConnection(configuration=_rtc_configuration())
        output_track = RealtimeAudioTrack()
        pc.addTrack(output_track)
        try:
            await pc.setLocalDescription(await pc.createOffer())
            call_id = await whatsapp_api.create_call(phone, pc.localDescription.sdp, demo_id)
            if context:
                context = CallContext(call_id, phone, context.agent_id, context.customer_id)
            self._demo_answered[call_id] = asyncio.Event()
            self._register_peer(call_id, phone, pc, output_track, context)
            self._demo_timers[call_id] = asyncio.get_running_loop().call_later(
                300 if context else 180, lambda: asyncio.create_task(self.end_demo(call_id)),
            )
            call_state.emit(call_id, "whatsapp.demo_dialing", {"demo_id": demo_id})
            return call_id
        except Exception:
            await pc.close()
            raise

    async def end_demo(self, call_id: str) -> None:
        await whatsapp_api.send_call_action(call_id, "terminate")
        await self.close_call(call_id)

    def mark_answered(self, call_id: str) -> None:
        answered = self._demo_answered.get(call_id)
        if answered:
            answered.set()

    async def handle_offer(self, call_id: str, sdp_offer: str, caller_phone: str = "") -> None:
        if call_id in self.pcs:
            return
        if len(self.pcs) >= int(os.environ.get("MAX_CALLS", "20")):
            await whatsapp_api.send_call_action(call_id, "reject")
            return
        await self.close_call(call_id)
        context = await claim_case(call_id, caller_phone)
        pc = RTCPeerConnection(configuration=_rtc_configuration())
        output_track = RealtimeAudioTrack()
        pc.addTrack(output_track)
        self._register_peer(call_id, caller_phone, pc, output_track, context)
        if context:
            self._demo_timers[call_id] = asyncio.get_running_loop().call_later(
                300, lambda: asyncio.create_task(self.end_demo(call_id)),
            )

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

    def _register_peer(self, call_id: str, caller_phone: str, pc, output_track, context=None) -> None:
        self._caller_phones[call_id] = caller_phone
        self.pcs[call_id] = pc
        call_state.start_call(call_id, caller_phone, context)

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
            answered = self._demo_answered.get(call_id)

            async def process_track():
                if answered:
                    await answered.wait()
                if call_id not in self.pcs:
                    return
                call_state.mark_call_active(call_id, phone)
                await voice_agent.process_audio(call_id, phone, track, output_track, context)

            asyncio.create_task(process_track())

    async def close_call(self, call_id: str, close_peer: bool = True) -> None:
        timer = self._demo_timers.pop(call_id, None)
        if timer:
            timer.cancel()
        self._caller_phones.pop(call_id, None)
        pc = self.pcs.pop(call_id, None)
        answered = self._demo_answered.get(call_id)
        if pc is not None and answered is not None:
            with connection() as db:
                db.execute(
                    """update portal_case_requests set state=%s,error=%s where call_id=%s
                       and runtime_agent_id=%s and direction='outbound' and state='calling'""",
                    ("completed" if answered.is_set() else "failed", "" if answered.is_set() else "The call was not answered. Please try again when you are ready.", call_id, os.environ["VOICE_AGENT_ID"]),
                )
        self.mark_answered(call_id)
        self._demo_answered.pop(call_id, None)
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
