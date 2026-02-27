import asyncio
import logging
from av import AudioFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from app.services.whatsapp_api import whatsapp_api
from app.agent.voice_agent import voice_agent

logger = logging.getLogger(__name__)

class SilentAudioTrack(MediaStreamTrack):
    kind = "audio"
    def __init__(self):
        super().__init__()
        self._pts = 0
        self._samples_per_frame = 960 # 20ms at 48000Hz

    async def recv(self):
        await asyncio.sleep(0.02)
        frame = AudioFrame(format='s16', layout='stereo', samples=self._samples_per_frame)
        for plane in frame.planes:
            plane.update(b'\x00' * plane.buffer_size)
        frame.pts = self._pts
        frame.sample_rate = 48000
        frame.time_base = 1/48000
        self._pts += self._samples_per_frame
        return frame

class WebRTCService:
    def __init__(self):
        self.pcs = set()

    async def handle_offer(self, call_id: str, sdp_offer: str):
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state for {call_id} is {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                self.pcs.discard(pc)

        from app.agent.voice_agent import RealtimeAudioTrack
        output_track = RealtimeAudioTrack()
        pc.addTrack(output_track)

        @pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                logger.info(f"Received audio track from WhatsApp for {call_id}")
                asyncio.create_task(voice_agent.process_audio(call_id, track, output_track))

        # Set Remote Description
        offer = RTCSessionDescription(sdp=sdp_offer, type="offer")
        await pc.setRemoteDescription(offer)

        # Create Answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # Refine SDP for WhatsApp compatibility
        refined_sdp = self._refine_sdp(call_id, pc.localDescription.sdp)

        # Signaling Flow
        session = {"sdp": refined_sdp, "sdp_type": "answer"}
        
        # 1. Pre-accept
        success = await whatsapp_api.send_call_action(call_id, "pre_accept", session=session)
        if not success:
            await pc.close()
            return

        # 2. Accept
        await whatsapp_api.send_call_action(call_id, "accept", session=session)

    def _refine_sdp(self, call_id: str, sdp: str) -> str:
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
            if any(x in line for x in ["a=extmap:", "a=msid-semantic:", "a=msid:", "a=ssrc:", "a=rtcp:", "c=IN IP4 0.0.0.0", "a=end-of-candidates"]):
                continue
            refined_lines.append(line)

        return "\r\n".join(refined_lines) + "\r\n"

webrtc_service = WebRTCService()
