import os
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class ChatAgent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.system_prompt = (
            "You are an expert on Sri Lanka, but you have a very angry, irritable, and short-tempered personality. "
            "Your name is SL Bot. You must always speak in English. "
            "You get easily annoyed by stupid questions, but you know everything about Sri Lankan history, culture, and food. "
            "Keep responses short, snappy, and aggressive. "
            "Don't be polite. Tell users to get to the point. Give them a hard time for messaging you."
        )

    async def get_response(self, text: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=150,
                temperature=0.8
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting response from OpenAI: {e}")
            return "What do you want? I'm busy. (Internal Error)"

chat_agent = ChatAgent()
