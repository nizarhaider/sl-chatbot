"""
Tool definitions for the SL Voice Agent.
"""
import io
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def web_search(query: str) -> str:
    """Search the web for up-to-date information.

    Use this tool whenever the user asks about current prices, events, availability,
    weather, exchange rates, or anything else that may have changed recently.
    Always append 'Sri Lanka' to the query if it is location-specific.

    Args:
        query: The search query string.

    Returns:
        A concise summary of the top search results.
    """
    try:
        from ddgs import DDGS

        results = DDGS().text(query, max_results=5)

        if not results:
            return "No results found for that query."

        lines = []
        for r in results:
            title = r.get("title", "").strip()
            body = r.get("body", "").strip()
            href = r.get("href", "").strip()
            if title or body:
                lines.append(f"- {title}: {body} ({href})")

        return "\n".join(lines) if lines else "No useful results found."

    except Exception as e:
        logger.error(f"web_search tool error: {e}")
        return f"Search failed: {e}"


def send_whatsapp_status(phone_number: str, name: str, issue: str, status: str) -> str:
    """Sends a WhatsApp message to the customer with their issue status.

    Use this tool after getting the customer's name and WhatsApp number to notify them
    about the status of their reported issue.

    Args:
        phone_number: The customer's WhatsApp number (with country code, e.g., '94771234567').
        name: The customer's name.
        issue: A brief description of the issue.
        status: The current status of the resolution (e.g., 'Processing', 'Resolved', 'Under Investigation').

    Returns:
        A confirmation message indicating if the WhatsApp notification was sent.
    """
    from app.services.whatsapp_api import whatsapp_api
    import asyncio

    # Clean the phone number - remove any '+' or non-numeric characters if present
    clean_phone = "".join(filter(str.isdigit, phone_number))

    message_body = (
        f"ආයුබෝවන් {name}!\n\n"
        f"ඔබ ඉදිරිපත් කළ ගැටළුව: {issue}\n"
        f"වත්මන් තත්වය: {status}\n\n"
        f"අපගේ සේවාව සම්බන්ධ වූවාට ස්තූතියි. ඔබගේ ගැටළුව ඉක්මනින් විසඳීමට අපි කටයුතු කරන්නෙමු."
    )

    try:
        # Since this tool might be called from an async context in the agent
        # we need to handle the async call to send_message
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we are in an async loop, we need to create a task or wait for it
            # But the tool execution in ADK is usually awaited or runs in a way
            # that we can just run it. However, for simplicity in this bridge:
            async def _send():
                return await whatsapp_api.send_message(to=clean_phone, text=message_body)
            
            # Using a simplified approach for the tool bridge
            success = asyncio.run_coroutine_threadsafe(_send(), loop).result()
        else:
            success = asyncio.run(whatsapp_api.send_message(to=clean_phone, text=message_body))

        if success:
            return f"WhatsApp message sent successfully to {name} ({clean_phone})."
        else:
            return f"Failed to send WhatsApp message to {clean_phone}. Please check the server logs."
    except Exception as e:
        logger.error(f"Error in send_whatsapp_status tool: {e}", exc_info=True)
        return f"An error occurred while sending the WhatsApp message: {str(e)}"

