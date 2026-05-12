import os
import asyncio
import logging
from fastapi import APIRouter, Request, HTTPException, Response, Header, status
from pydantic import BaseModel
from app.services.webrtc import webrtc_service
from app.chat_agent import chat_agent
from app.services.whatsapp_api import whatsapp_api

logger = logging.getLogger(__name__)
router = APIRouter()

VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "my_secure_verify_token_123")
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY")


class SendWhatsAppMessageRequest(BaseModel):
    phone_number: str
    message: str

@router.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == VERIFY_TOKEN:
            logger.info("WEBHOOK_VERIFIED")
            return Response(content=challenge, media_type="text/plain")
        else:
            raise HTTPException(status_code=403, detail="Verification token mismatch")
    raise HTTPException(status_code=400, detail="Missing parameters")

@router.post("/webhook")
async def receive_webhook(request: Request):
    body = await request.json()
    
    if body.get("object") == "whatsapp_business_account":
        try:
            for entry in body.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    
                    if "messages" in value:
                        for message in value["messages"]:
                            if message.get("type") == "text":
                                sender_id = message.get("from")
                                text = message.get("text", {}).get("body", "")
                                logger.info(f"Received text message from {sender_id}: {text}")
                                
                                # Process message in background to not block webhook response
                                asyncio.create_task(handle_text_message(sender_id, text))
                            
                    if "statuses" in value:
                        for status in value["statuses"]:
                            logger.info(f"Received status update: {status}")

                    if "calls" in value:
                        for call in value["calls"]:
                            event = call.get("event")
                            call_id = call.get("id")
                            caller_phone = call.get("from", "")
                            logger.info(f"Received call event: {event} for call {call_id} from {caller_phone}")

                            if event == "connect":
                                session = call.get("session", {})
                                if session.get("sdp_type") == "offer":
                                    sdp_offer = session.get("sdp")
                                    logger.info(f"Processing SDP Offer for {call_id}")
                                    asyncio.create_task(webrtc_service.handle_offer(call_id, sdp_offer, caller_phone))
                            elif event == "terminate":
                                # Immediately cleanup PCs on hangup
                                if call_id in webrtc_service.pcs:
                                    logger.info(f"Call {call_id} terminated by peer, cleaning up.")
                                    pc = webrtc_service.pcs.pop(call_id)
                                    asyncio.create_task(pc.close())

            return Response(content="EVENT_RECEIVED", status_code=200)
        except Exception as e:
            logger.error(f"Error processing webhook: {e}")
            return Response(content="ERROR", status_code=500)
    else:
        raise HTTPException(status_code=404, detail="Not a WhatsApp API event")


@router.post("/send-message")
async def send_whatsapp_message(
    payload: SendWhatsAppMessageRequest,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
):
    """
    Internal endpoint to send a WhatsApp message.

    This is intended to be called from trusted backend services (e.g. your Next.js app),
    not directly from the public internet or frontend clients.
    """
    if not INTERNAL_API_KEY:
        logger.error("INTERNAL_API_KEY is not set on the server")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server not configured for internal messaging",
        )

    if x_api_key != INTERNAL_API_KEY:
        logger.warning("Unauthorized attempt to call /send-message")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    success = await whatsapp_api.send_message(
        to=payload.phone_number,
        text=payload.message,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to send WhatsApp message",
        )

    return {"status": "sent"}


async def handle_text_message(sender_id: str, text: str):
    try:
        response_text = await chat_agent.get_response(text)
        if not response_text:
            logger.error("Chat agent returned an empty response for sender %s", sender_id)
            response_text = "Sorry, I could not process that right now. Please try again shortly."

        success = await whatsapp_api.send_message(sender_id, response_text)
        if not success:
            logger.error("Failed to send WhatsApp reply to %s", sender_id)
    except Exception as e:
        logger.error("Error handling text message from %s: %s", sender_id, e, exc_info=True)
