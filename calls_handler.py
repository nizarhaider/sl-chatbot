import os
import asyncio
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
                await asyncio.sleep(10) # Reduced sleep for better response
                return None # The track should ideally yield frames if it was real

        pc.addTrack(DummyAudioTrack())

        # 2. Set Remote Description
        offer = RTCSessionDescription(sdp=sdp_offer, type="offer")
        await pc.setRemoteDescription(offer)

        # 3. Create Answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # 4. Refine SDP for WhatsApp compatibility
        sdp = pc.localDescription.sdp
        
        # Meta requirements for SDP validation:
        # 1. Fingerprint must be uppercase
        sdp = sdp.replace("a=fingerprint:sha-256", "a=fingerprint:SHA-256")
        
        # 2. MID MUST match the offer's MID ('audio')
        # aiortc usually uses '0' or '1'
        lines = sdp.splitlines()
        refined_lines = []
        for line in lines:
            # Force 'audio' as the mid
            if line.startswith("a=mid:"):
                refined_lines.append("a=mid:audio")
                continue
            
            # Use 'active' setup mode
            if line.startswith("a=setup:"):
                refined_lines.append("a=setup:active")
                continue

            # Remove problematic/unsupported lines
            if any(x in line for x in [
                "a=extmap:", 
                "a=msid-semantic:", 
                "a=msid:", 
                "a=ssrc:", 
                "a=rtcp:",
                "c=IN IP4 0.0.0.0" # Some gateways hate this
            ]):
                continue
            
            # Ensure the BUNDLE line matches the mid
            if line.startswith("a=group:BUNDLE"):
                refined_lines.append("a=group:BUNDLE audio")
                continue

            refined_lines.append(line)

        # Ensure c= line is present and not 0.0.0.0
        # For simplicity in signaling, we can use 127.0.0.1 or a placeholder
        # if aiortc didn't provide a host candidate. Meta usually ignores this
        # if ICE candidates are present, but the validator might check it.
        has_c_line = any(line.startswith("c=") for line in refined_lines)
        if not has_c_line:
            refined_lines.insert(3, "c=IN IP4 127.0.0.1")

        final_sdp = "\r\n".join(refined_lines) + "\r\n"
        
        # Log the full SDP for debugging
        print(f"--- GENERATED SDP ANSWER FOR {call_id} ---")
        print(final_sdp)
        print("------------------------------------------")

        # 5. Send Answer back via WhatsApp Calls API
        success = await self.send_sdp_answer(call_id, final_sdp)
        
        if success:
            logger.info(f"Successfully connected call {call_id}")
        else:
            logger.error(f"Failed to connect call {call_id}")
            await pc.close()
            self.pcs.discard(pc)

    async def send_sdp_answer(self, call_id: str, sdp_answer: str):
        """
        Sends the generated SDP answer back to Meta using the WhatsApp Calls API.
        Reference: https://developers.facebook.com/documentation/business-messaging/whatsapp/calling/user-initiated-calls
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
                # Step 1: Pre-accept (Combining with SDP Answer as allowed by Meta)
                # Note: Meta uses underscores for actions: 'pre_accept', 'accept'
                payload = {
                    "messaging_product": "whatsapp",
                    "call_id": call_id,
                    "action": "pre_accept",
                    "session": {
                        "sdp": sdp_answer,
                        "sdp_type": "answer"
                    }
                }
                
                logger.info(f"Sending pre_accept + SDP answer for {call_id}")
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    logger.error(f"Error in pre_accept step: {response.text}")
                    response.raise_for_status()

                # Step 2: Accept the call to finalize
                accept_payload = {
                    "messaging_product": "whatsapp",
                    "call_id": call_id,
                    "action": "accept"
                }
                
                logger.info(f"Sending accept for {call_id}")
                response = await client.post(url, headers=headers, json=accept_payload)
                if response.status_code != 200:
                    logger.error(f"Error in accept step: {response.text}")
                    response.raise_for_status()
                
                return True
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error in WhatsApp Calling API: {e.response.text}")
                return False
            except Exception as e:
                logger.error(f"Error in WhatsApp Calling API: {e}")
                return False

# Global handler instance
calls_handler = CallsHandler()
