import logging
import os

import httpx

logger = logging.getLogger(__name__)

GRAPH_API_VERSION = os.environ.get("GRAPH_API_VERSION", "v22.0")


def _whatsapp_access_token() -> str | None:
    return os.environ.get("WHATSAPP_ACCESS_TOKEN") or os.environ.get("WHATSAPP_TOKEN")


def _phone_number_id() -> str | None:
    return os.environ.get("PHONE_NUMBER_ID")


class WhatsAppAPI:
    @staticmethod
    async def send_call_action(call_id: str, action: str, session: dict | None = None) -> bool:
        access_token = _whatsapp_access_token()
        phone_number_id = _phone_number_id()
        if not access_token or not phone_number_id:
            logger.error("WHATSAPP_ACCESS_TOKEN/WHATSAPP_TOKEN or PHONE_NUMBER_ID not set")
            return False

        url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{phone_number_id}/calls"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "call_id": call_id,
            "action": action,
        }
        if session:
            payload["session"] = session

        async with httpx.AsyncClient() as client:
            try:
                logger.info("Sending %s for %s", action, call_id)
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    logger.error("Error in %s step: %s", action, response.text)
                return response.status_code == 200
            except Exception as exc:
                logger.error("Error in WhatsApp Calling API (%s): %s", action, exc)
                return False

    @staticmethod
    async def send_message(to: str, text: str) -> bool:
        access_token = _whatsapp_access_token()
        phone_number_id = _phone_number_id()
        if not access_token or not phone_number_id:
            logger.error("WHATSAPP_ACCESS_TOKEN/WHATSAPP_TOKEN or PHONE_NUMBER_ID not set")
            return False

        url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"body": text},
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    logger.error("Error sending message: %s", response.text)
                    return False
                logger.info("Message sent to %s", to)
                return True
            except Exception as exc:
                logger.error("Error in WhatsApp Messaging API: %s", exc)
                return False

    @staticmethod
    async def send_image(to: str, image_url: str, caption: str = "") -> bool:
        access_token = _whatsapp_access_token()
        phone_number_id = _phone_number_id()
        if not access_token or not phone_number_id:
            logger.error("WHATSAPP_ACCESS_TOKEN/WHATSAPP_TOKEN or PHONE_NUMBER_ID not set")
            return False

        url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        image_payload = {"link": image_url}
        if caption:
            image_payload["caption"] = caption[:1024]
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "image",
            "image": image_payload,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    logger.error("Error sending image: %s", response.text)
                    return False
                logger.info("Image sent to %s: %s", to, image_url)
                return True
            except Exception as exc:
                logger.error("Error in WhatsApp Image API: %s", exc)
                return False


whatsapp_api = WhatsAppAPI()
