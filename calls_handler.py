import os
import httpx
import logging
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer

logger = logging.getLogger(__name__)

WHATSAPP_ACCESS_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.environ.get("PHONE_NUMBER_ID")
GRAPH_API_VERSION = "v22.0"

class CallsHandler:
    def __init__(self):
        self.pcs = set()

    async def handle_offer(self, call_id: str, sdp_offer: str):
        """
        Handles an incoming SDP offer from WhatsApp.
        """
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state for {call_id} is {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                self.pcs.discard(pc)

        # 1. Add an audio track to ensure m=audio is in the answer
        # This is often required by Meta to validate the SDP
        class DummyAudioTrack(MediaStreamTrack):
            kind = "audio"
            async def recv(self):
                import asyncio
                await asyncio.sleep(100) # Just a placeholder

        pc.addTrack(DummyAudioTrack())

        # 2. Set Remote Description
        offer = RTCSessionDescription(sdp=sdp_offer, type="offer")
        await pc.setRemoteDescription(offer)

        # 3. Create Answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # 4. Refine SDP for WhatsApp compatibility
        sdp = pc.localDescription.sdp
        
        # Meta often requires uppercase SHA-256
        sdp = sdp.replace("a=fingerprint:sha-256", "a=fingerprint:SHA-256")
        
        # WhatsApp/Meta often issues validation errors if it sees setup:active
        # Since Meta sends actpass, we should respond with passive
        sdp = sdp.replace("a=setup:active", "a=setup:passive")
        
        # Remove unsupported headers/attributes that can trigger validation errors
        filtered_lines = []
        for line in sdp.splitlines():
            # Remove extmap and msid-semantic which are often problematic
            if any(x in line for x in ["a=extmap:", "a=msid-semantic:", "a=group:BUNDLE audio video"]):
                continue
            filtered_lines.append(line)
        
        sdp = "\r\n".join(filtered_lines) + "\r\n"

        # 5. Send Answer back via WhatsApp Calls API
        success = await self.send_sdp_answer(call_id, sdp)
        
        if success:
            logger.info(f"Successfully connected call {call_id}")
        else:
            logger.error(f"Failed to connect call {call_id}")
            await pc.close()
            self.pcs.discard(pc)

    async def send_sdp_answer(self, call_id: str, sdp_answer: str):
        """
        Sends the generated SDP answer back to Meta using the WhatsApp Calls API.
        """
        if not WHATSAPP_ACCESS_TOKEN or not PHONE_NUMBER_ID:
            logger.error("WHATSAPP_ACCESS_TOKEN or PHONE_NUMBER_ID not set")
            return False

        url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{PHONE_NUMBER_ID}/calls"
        
        headers = {
            "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                # Step 1: Pre-accept
                pre_accept_payload = {
                    "messaging_product": "whatsapp",
                    "call_id": call_id,
                    "action": "pre-accept"
                }
                logger.info(f"Sending pre-accept for {call_id}")
                await client.post(url, headers=headers, json=pre_accept_payload)

                # Step 2: Send Answer
                # Note: The 'action' is usually not needed here if 'session' is present
                # but we'll include it if requested. Actually, Meta docs usually show
                # just the session for the answer step.
                answer_payload = {
                    "messaging_product": "whatsapp",
                    "call_id": call_id,
                    "session": {
                        "sdp": sdp_answer,
                        "sdp_type": "answer"
                    }
                }
                logger.info(f"Sending SDP answer for {call_id}")
                response = await client.post(url, headers=headers, json=answer_payload)
                response.raise_for_status()
                
                # Step 3: Accept
                accept_payload = {
                    "messaging_product": "whatsapp",
                    "call_id": call_id,
                    "action": "accept"
                }
                logger.info(f"Sending accept for {call_id}")
                await client.post(url, headers=headers, json=accept_payload)
                
                return True
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error sending SDP answer: {e.response.text}")
                return False
            except Exception as e:
                logger.error(f"Error sending SDP answer: {e}")
                return False

# Global handler instance
calls_handler = CallsHandler()
