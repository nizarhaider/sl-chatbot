import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

GRAPH_API_VERSION = "v22.0"


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

        payload = {
            "messaging_product": "whatsapp",
            "call_id": call_id,
            "action": action,
        }
        if session:
            payload["session"] = session

        url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{phone_number_id}/calls"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        # Vast instances in this deployment have no usable IPv6 route to Meta.
        # Binding the client to IPv4 avoids the default resolver's failed IPv6
        # attempt consuming the time available to pre-accept an incoming call.
        transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0", retries=1)
        timeout = httpx.Timeout(10.0, connect=3.0)
        async with httpx.AsyncClient(transport=transport, timeout=timeout) as client:
            try:
                logger.info("Sending %s for %s", action, call_id)
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    logger.error("Error in %s step: %s", action, response.text)
                return response.status_code == 200
            except Exception as exc:
                logger.error(
                    "Error in WhatsApp Calling API (%s): %s: %s",
                    action, type(exc).__name__, exc,
                )
                return False

    @staticmethod
    async def send_text_message(to_phone: str, body: str) -> bool:
        access_token = _whatsapp_access_token()
        phone_number_id = _phone_number_id()
        recipient = _normalize_phone_number(to_phone)
        if not access_token or not phone_number_id:
            logger.error("WhatsApp text message credentials are not set")
            return False
        if not recipient:
            logger.error("Cannot send WhatsApp confirmation without a valid caller number")
            return False

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "text",
            "text": {"preview_url": False, "body": body},
        }
        url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code not in (200, 201):
                    logger.error("WhatsApp confirmation failed with status %s", response.status_code)
                return response.status_code in (200, 201)
            except Exception as exc:
                logger.error("Error sending WhatsApp confirmation: %s", exc)
                return False


def _normalize_phone_number(value: str) -> str:
    digits = re.sub(r"\D", "", value or "")
    if digits.startswith("00"):
        digits = digits[2:]
    if digits.startswith("0") and len(digits) == 10:
        digits = "94" + digits[1:]
    return digits if len(digits) >= 10 else ""


whatsapp_api = WhatsAppAPI()
