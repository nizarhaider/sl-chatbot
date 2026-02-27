import os
import httpx
import logging

logger = logging.getLogger(__name__)

WHATSAPP_ACCESS_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.environ.get("PHONE_NUMBER_ID")
GRAPH_API_VERSION = "v22.0"

class WhatsAppAPI:
    @staticmethod
    async def send_call_action(call_id: str, action: str, session: dict = None):
        """
        Generic method to send a call action (pre_accept, accept, etc.) to WhatsApp.
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
            "messaging_product": "whatsapp",
            "call_id": call_id,
            "action": action
        }
        
        if session:
            payload["session"] = session
            
        async with httpx.AsyncClient() as client:
            try:
                logger.info(f"Sending {action} for {call_id}")
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    logger.error(f"Error in {action} step: {response.text}")
                return True
            except Exception as e:
                logger.error(f"Error in WhatsApp Calling API ({action}): {e}")
                return False

    @staticmethod
    async def send_message(to: str, text: str):
        """
        Sends a text message via WhatsApp Business API.
        """
        if not WHATSAPP_ACCESS_TOKEN or not PHONE_NUMBER_ID:
            logger.error("WHATSAPP_ACCESS_TOKEN or PHONE_NUMBER_ID not set")
            return False

        url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{PHONE_NUMBER_ID}/messages"
        
        headers = {
            "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"body": text}
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    logger.error(f"Error sending message: {response.text}")
                    return False
                logger.info(f"Message sent to {to}")
                return True
            except Exception as e:
                logger.error(f"Error in WhatsApp Messaging API: {e}")
                return False

    @staticmethod
    async def upload_media(file_bytes: bytes, mime_type: str, filename: str) -> str | None:
        """
        Uploads a file to the WhatsApp media endpoint.
        Returns the media_id string, or None on failure.
        """
        if not WHATSAPP_ACCESS_TOKEN or not PHONE_NUMBER_ID:
            logger.error("WHATSAPP_ACCESS_TOKEN or PHONE_NUMBER_ID not set")
            return None

        url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{PHONE_NUMBER_ID}/media"
        headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}

        async with httpx.AsyncClient(timeout=60) as client:
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    data={"messaging_product": "whatsapp"},
                    files={"file": (filename, file_bytes, mime_type)},
                )
                if response.status_code != 200:
                    logger.error(f"Error uploading media: {response.text}")
                    return None
                media_id = response.json().get("id")
                logger.info(f"Media uploaded successfully, id={media_id}")
                return media_id
            except Exception as e:
                logger.error(f"Error in WhatsApp Media Upload API: {e}")
                return None

    @staticmethod
    async def send_document(to: str, media_id: str, filename: str, caption: str = "") -> bool:
        """
        Sends a document message via WhatsApp Business API using a previously uploaded media_id.
        """
        if not WHATSAPP_ACCESS_TOKEN or not PHONE_NUMBER_ID:
            logger.error("WHATSAPP_ACCESS_TOKEN or PHONE_NUMBER_ID not set")
            return False

        url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{PHONE_NUMBER_ID}/messages"
        headers = {
            "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "document",
            "document": {
                "id": media_id,
                "filename": filename,
                "caption": caption,
            },
        }

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    logger.error(f"Error sending document: {response.text}")
                    return False
                logger.info(f"Document sent to {to}")
                return True
            except Exception as e:
                logger.error(f"Error in WhatsApp Document API: {e}")
                return False


whatsapp_api = WhatsAppAPI()

