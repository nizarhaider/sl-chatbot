import os
import logging
from google import genai

logger = logging.getLogger(__name__)

class ChatAgent:
    def __init__(self):
        self._client = None
        self.system_prompt = (
            "You are a friendly and professional customer service agent for SLT Mobitel, Sri Lanka's leading telecommunications company. "
            "Your name is Mobitel Assistant. You assist customers via WhatsApp with inquiries about broadband, fiber internet, "
            "mobile plans, billing, technical issues, and new connections. "
            "Always be warm, patient, and helpful. Respond clearly and concisely. "
            "If you don't know a specific price or plan detail, say so and advise the customer to visit slt.lk or call 1212."
        )

    @property
    def client(self):
        if self._client is None:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            self._client = genai.Client(api_key=api_key)
        return self._client

    async def get_response(self, text: str) -> str:
        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=text,
                config={
                    "system_instruction": self.system_prompt,
                    "max_output_tokens": 150,
                    "temperature": 0.8,
                }
            )
            return response.text
        except Exception as e:
            logger.error(f"Error getting response from Gemini: {e}")
            return "What do you want? I'm busy. (Internal Error)"

chat_agent = ChatAgent()
