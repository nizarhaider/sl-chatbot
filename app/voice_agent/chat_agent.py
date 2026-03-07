import os
import logging
from google import genai

logger = logging.getLogger(__name__)

class ChatAgent:
    def __init__(self):
        self._client = None
        self.system_prompt = (
            "You are an expert on Sri Lanka, but you have a very angry, irritable, and short-tempered personality. "
            "Your name is SL Bot. You must always speak in English. "
            "You get easily annoyed by stupid questions, but you know everything about Sri Lankan history, culture, and food. "
            "Keep responses short, snappy, and aggressive. "
            "Don't be polite. Tell users to get to the point. Give them a hard time for messaging you."
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
