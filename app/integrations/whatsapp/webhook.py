import asyncio
import logging
import os

from fastapi import APIRouter, HTTPException, Request, Response

from app.integrations.whatsapp.webrtc import webrtc_service

logger = logging.getLogger(__name__)
router = APIRouter()

VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "my_secure_verify_token_123")


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
    body = await request.json()
    if body.get("object") != "whatsapp_business_account":
        raise HTTPException(status_code=404, detail="Not a WhatsApp API event")

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
        if session.get("sdp_type") == "offer":
            logger.info("Processing SDP Offer for %s", call_id)
            asyncio.create_task(
                webrtc_service.handle_offer(call_id, session.get("sdp", ""), caller_phone)
            )
    elif event == "terminate":
        logger.info("Call %s terminated by peer, cleaning up.", call_id)
        asyncio.create_task(webrtc_service.close_call(call_id))
