import asyncio
import logging

from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)

from app.services.whatsapp_api import whatsapp_api
from app.voice_agent.agent import RealtimeAudioTrack, voice_agent

logger = logging.getLogger(__name__)


class WebRTCService:
    def __init__(self) -> None:
        self.pcs: dict[str, RTCPeerConnection] = {}
        self._caller_phones: dict[str, str] = {}

    async def handle_offer(self, call_id: str, sdp_offer: str, caller_phone: str = "") -> None:
        if call_id in self.pcs:
            logger.warning("Call %s already exists, closing old connection for re-entry.", call_id)
            old_pc = self.pcs.pop(call_id)
            await old_pc.close()

        self._caller_phones[call_id] = caller_phone

        configuration = RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
            ]
        )
        pc = RTCPeerConnection(configuration=configuration)
        self.pcs[call_id] = pc

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            logger.info("Connection state for %s is %s", call_id, pc.connectionState)
            if pc.connectionState in ["failed", "closed", "disconnected"]:
                self.pcs.pop(call_id, None)
                self._caller_phones.pop(call_id, None)
                if pc.connectionState != "closed":
                    await pc.close()

        output_track = RealtimeAudioTrack()
        pc.addTrack(output_track)

        @pc.on("track")
        def on_track(track) -> None:
            if track.kind == "audio":
                phone = self._caller_phones.get(call_id, "")
                logger.info("Received audio track from WhatsApp for %s (caller: %s)", call_id, phone)
                asyncio.create_task(voice_agent.process_audio(call_id, phone, track, output_track))

        offer = RTCSessionDescription(sdp=sdp_offer, type="offer")
        logger.info("Incoming audio SDP for %s: %s", call_id, self._summarize_audio_sdp(sdp_offer))
        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        refined_sdp = self._refine_sdp(pc.localDescription.sdp)
        logger.info("Answer audio SDP for %s: %s", call_id, self._summarize_audio_sdp(refined_sdp))
        session = {"sdp": refined_sdp, "sdp_type": "answer"}

        success = await whatsapp_api.send_call_action(call_id, "pre_accept", session=session)
        if not success:
            await pc.close()
            return

        await whatsapp_api.send_call_action(call_id, "accept", session=session)

    def _refine_sdp(self, sdp: str) -> str:
        lines = sdp.splitlines()
        refined_lines = []
        fingerprint_added = False

        for line in lines:
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
                parts = line.split()
                if len(parts) >= 6 and parts[5] == "0.0.0.0":
                    parts[5] = "127.0.0.1"
                    line = " ".join(parts)
                refined_lines.append(line)
                continue
            if any(
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
            ):
                continue
            refined_lines.append(line)

        return "\r\n".join(refined_lines) + "\r\n"

    def _summarize_audio_sdp(self, sdp: str) -> list[str]:
        audio_lines = []
        in_audio = False
        for line in sdp.splitlines():
            if line.startswith("m="):
                in_audio = line.startswith("m=audio")
                if in_audio:
                    audio_lines.append(line)
                continue
            if not in_audio:
                continue
            if line.startswith(
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
