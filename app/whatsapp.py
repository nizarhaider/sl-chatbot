"""WhatsApp webhook, Graph API calls, and WebRTC bridge."""

from __future__ import annotations

import asyncio
import logging
import os
import random

import httpx
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from fastapi import APIRouter, HTTPException, Request, Response

from app.audio import OutboundAudioTrack
from app.database import call_log
from app.pipeline import voice_agent

logger = logging.getLogger(__name__)
router = APIRouter()


async def send_call_action(
    call_id: str, action: str, session: dict | None = None
) -> bool:
    token = os.getenv("WHATSAPP_ACCESS_TOKEN") or os.getenv("WHATSAPP_TOKEN")
    number_id = os.getenv("PHONE_NUMBER_ID")
    if not token or not number_id:
        logger.error("WHATSAPP_ACCESS_TOKEN and PHONE_NUMBER_ID are required")
        return False
    payload = {"messaging_product": "whatsapp", "call_id": call_id, "action": action}
    if session:
        payload["session"] = session
    url = f"https://graph.facebook.com/v22.0/{number_id}/calls"
    async with httpx.AsyncClient(timeout=httpx.Timeout(20, connect=10)) as client:
        for attempt in range(2):
            try:
                response = await client.post(
                    url, headers={"Authorization": f"Bearer {token}"}, json=payload
                )
                if response.status_code != 200:
                    logger.error("WhatsApp %s failed: %s", action, response.text)
                return response.status_code == 200
            except httpx.HTTPError:
                if attempt:
                    logger.exception("WhatsApp %s request failed", action)
                    return False
                await asyncio.sleep(0.4 + random.random() * 0.2)
    return False


class WebRTCService:
    def __init__(self) -> None:
        self.connections: dict[str, RTCPeerConnection] = {}
        self.phones: dict[str, str] = {}

    async def handle_offer(self, call_id: str, sdp: str, phone: str) -> None:
        await self.close(call_id)
        self.phones[call_id] = phone
        call_log.start(call_id, phone)
        pc = RTCPeerConnection(
            RTCConfiguration(
                iceServers=[
                    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                    RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
                ]
            )
        )
        self.connections[call_id] = pc
        output = OutboundAudioTrack()
        pc.addTrack(output)

        @pc.on("connectionstatechange")
        async def state_changed() -> None:
            logger.info("Connection state for %s: %s", call_id, pc.connectionState)
            if pc.connectionState in {"failed", "closed", "disconnected"}:
                await self.close(call_id, close_peer=pc.connectionState != "closed")

        @pc.on("track")
        def track_received(track) -> None:
            if track.kind == "audio":
                caller = self.phones.get(call_id, "")
                logger.info("Received audio track for %s", call_id)
                call_log.active(call_id, caller)
                asyncio.create_task(voice_agent.process(call_id, caller, track, output))

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        session = {"sdp": refine_sdp(pc.localDescription.sdp), "sdp_type": "answer"}
        if not await send_call_action(call_id, "pre_accept", session):
            await self.close(call_id)
            return
        await send_call_action(call_id, "accept", session)

    async def close(self, call_id: str, close_peer: bool = True) -> None:
        self.phones.pop(call_id, None)
        pc = self.connections.pop(call_id, None)
        await voice_agent.cancel(call_id)
        call_log.end(call_id)
        if close_peer and pc:
            await pc.close()

    async def close_all(self) -> None:
        await asyncio.gather(
            *(self.close(call_id) for call_id in list(self.connections))
        )


@router.get("/webhook")
async def verify_webhook(request: Request) -> Response:
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    if not mode or not token:
        raise HTTPException(400, "Missing parameters")
    if mode != "subscribe" or token != os.getenv("VERIFY_TOKEN"):
        raise HTTPException(403, "Verification token mismatch")
    logger.info("WEBHOOK_VERIFIED")
    return Response(challenge, media_type="text/plain")


@router.post("/webhook")
async def receive_webhook(request: Request) -> Response:
    body = await request.json()
    if body.get("object") != "whatsapp_business_account":
        raise HTTPException(404, "Not a WhatsApp API event")
    try:
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                await handle_change(change.get("value", {}))
    except Exception:
        logger.exception("Webhook processing failed")
        return Response("ERROR", status_code=500)
    return Response("EVENT_RECEIVED")


async def handle_change(value: dict) -> None:
    for message in value.get("messages", []):
        logger.info("Ignoring WhatsApp message type %s", message.get("type"))
    for call in value.get("calls", []):
        event, call_id = call.get("event"), call.get("id")
        if not call_id:
            continue
        logger.info("Received call event: %s for %s", event, call_id)
        if event == "connect" and call.get("session", {}).get("sdp_type") == "offer":
            asyncio.create_task(
                webrtc_service.handle_offer(
                    call_id,
                    call["session"].get("sdp", ""),
                    call.get("from", ""),
                )
            )
        elif event == "terminate":
            asyncio.create_task(webrtc_service.close(call_id))


def refine_sdp(sdp: str) -> str:
    lines: list[str] = []
    fingerprint = False
    dropped = (
        "a=extmap:",
        "a=msid-semantic:",
        "a=msid:",
        "a=ssrc:",
        "a=rtcp:",
        "c=IN IP4 0.0.0.0",
        "a=end-of-candidates",
    )
    for line in sdp.splitlines():
        if line.startswith("a=fingerprint:"):
            if not fingerprint and "sha-256" in line.lower():
                lines.append(
                    line.replace("sha-256", "SHA-256").replace("sha256", "SHA-256")
                )
                fingerprint = True
            continue
        if line.startswith("a=mid:"):
            line = "a=mid:audio"
        elif line.startswith("a=setup:"):
            line = "a=setup:active"
        elif line.startswith("a=group:BUNDLE"):
            line = "a=group:BUNDLE audio"
        elif line.startswith("o="):
            parts = line.split()
            if len(parts) >= 6 and parts[5] == "0.0.0.0":
                parts[5] = "127.0.0.1"
                line = " ".join(parts)
        if not any(token in line for token in dropped):
            lines.append(line)
    return "\r\n".join(lines) + "\r\n"


webrtc_service = WebRTCService()
