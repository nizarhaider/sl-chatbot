import asyncio
import logging
import os
import hashlib
import hmac
import time

from pydantic import BaseModel, Field, ValidationError

from fastapi import APIRouter, HTTPException, Request, Response

from app.integrations.whatsapp.webrtc import webrtc_service

logger = logging.getLogger(__name__)
router = APIRouter()

VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "my_secure_verify_token_123")


_demo_requests: dict[str, float] = {}


class DemoCall(BaseModel):
    id: str = Field(pattern=r"^[a-f0-9]{32}$")
    phone: str = Field(pattern=r"^[1-9][0-9]{7,14}$")
    timestamp: int


@router.post("/demo/call")
async def demo_call(request: Request):
    secret = os.environ.get("WHATSAPP_APP_SECRET", "")
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
