import os
import asyncio
import logging
from fastapi import APIRouter, Request, HTTPException, Response
from app.services.webrtc import webrtc_service
from app.voice_agent.chat_agent import chat_agent
from app.services.whatsapp_api import whatsapp_api

logger = logging.getLogger(__name__)
router = APIRouter()

VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "my_secure_verify_token_123")

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

async def handle_text_message(sender_id: str, text: str):
    response_text = await chat_agent.get_response(text)
    await whatsapp_api.send_message(sender_id, response_text)
