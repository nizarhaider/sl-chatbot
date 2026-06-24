import asyncio
import logging

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

from app.integrations.whatsapp.client import whatsapp_api
from app.voice.agent import voice_agent
from app.voice.audio_track import RealtimeAudioTrack

logger = logging.getLogger(__name__)


class WebRTCService:
    def __init__(self) -> None:
        self.pcs: dict[str, RTCPeerConnection] = {}
        self._caller_phones: dict[str, str] = {}

    async def handle_offer(self, call_id: str, sdp_offer: str, caller_phone: str = "") -> None:
        await self.close_call(call_id)
        self._caller_phones[call_id] = caller_phone

        pc = RTCPeerConnection(configuration=_rtc_configuration())
        self.pcs[call_id] = pc
        output_track = RealtimeAudioTrack()
        pc.addTrack(output_track)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            logger.info("Connection state for %s is %s", call_id, pc.connectionState)
            if pc.connectionState in ["failed", "closed", "disconnected"]:
                await self.close_call(call_id, close_peer=pc.connectionState != "closed")

        @pc.on("track")
        def on_track(track) -> None:
            if track.kind != "audio":
                return
            phone = self._caller_phones.get(call_id, "")
            logger.info("Received audio track from WhatsApp for %s (caller: %s)", call_id, phone)
            asyncio.create_task(voice_agent.process_audio(call_id, phone, track, output_track))

        offer = RTCSessionDescription(sdp=sdp_offer, type="offer")
        logger.info("Incoming audio SDP for %s: %s", call_id, _summarize_audio_sdp(sdp_offer))
        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        refined_sdp = _refine_sdp(pc.localDescription.sdp)
        logger.info("Answer audio SDP for %s: %s", call_id, _summarize_audio_sdp(refined_sdp))
        session = {"sdp": refined_sdp, "sdp_type": "answer"}

        if not await whatsapp_api.send_call_action(call_id, "pre_accept", session=session):
            await self.close_call(call_id)
            return
        await whatsapp_api.send_call_action(call_id, "accept", session=session)

    async def close_call(self, call_id: str, close_peer: bool = True) -> None:
        self._caller_phones.pop(call_id, None)
        pc = self.pcs.pop(call_id, None)
        await voice_agent.cancel_call(call_id)
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
