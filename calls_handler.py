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
        1. Set remote description
        2. Create answer
        3. Set local description
        4. Send answer back to WhatsApp
        """
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state is {pc.connectionState}")
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        # Handle tracks (e.g., if you want to send audio)
        # For now, we'll just handle the signaling
        
        # Set Remote Description
        offer = RTCSessionDescription(sdp=sdp_offer, type="offer")
        await pc.setRemoteDescription(offer)

        # Create Answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # Send Answer back via WhatsApp Calls API
        success = await self.send_sdp_answer(call_id, pc.localDescription.sdp)
        
        if success:
            logger.info(f"Successfully sent SDP answer for call {call_id}")
        else:
            logger.error(f"Failed to send SDP answer for call {call_id}")
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
        
        payload = {
            "call_id": call_id,
            "messaging_product": "whatsapp", 
            "action": "pre_accept",
            "session": {
                "sdp": sdp_answer,
                "sdp_type": "answer"
            }
        }
        
        async with httpx.AsyncClient() as client:
            try:
                # Pre-accept the call first as recommended
                # Step 1: Pre-accept
                pre_accept_payload = {
                    "call_id": call_id,
                    "action": "pre-accept"
                }
                await client.post(url, headers=headers, json=pre_accept_payload)

                # Step 2: Send Answer
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                
                # Step 3: Accept the call
                accept_payload = {
                    "call_id": call_id,
                    "action": "accept"
                }
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
