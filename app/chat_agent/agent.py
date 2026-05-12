import logging
import os

from google import genai

logger = logging.getLogger(__name__)

DEFAULT_BUSINESS_NAME = "SLT Mobitel"
DEFAULT_BUSINESS_DESCRIPTION = (
    "Sri Lanka's leading telecommunications company. You assist customers via "
    "WhatsApp with inquiries about broadband, fiber internet, mobile plans, "
    "billing, technical issues, and new connections."
)
DEFAULT_ESCALATION_MESSAGE = "visit slt.lk or call 1212"


class ChatAgent:
    def __init__(self):
        self._client = None
        self.model = os.environ.get("CHAT_AGENT_MODEL", "gemini-2.5-flash-lite")
        self.system_prompt = os.environ.get(
            "CHAT_AGENT_SYSTEM_PROMPT",
            self._build_system_prompt(),
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
                model=self.model,
                contents=text,
                config={
                    "system_instruction": self.system_prompt,
                    "max_output_tokens": int(os.environ.get("CHAT_AGENT_MAX_OUTPUT_TOKENS", "150")),
                    "temperature": float(os.environ.get("CHAT_AGENT_TEMPERATURE", "0.8")),
                },
            )
            return response.text
        except Exception as e:
            logger.error("Error getting chat response from Gemini: %s", e)
            return "Sorry, I could not process that right now. Please try again shortly."

    def _build_system_prompt(self) -> str:
        business_name = os.environ.get("CHAT_AGENT_BUSINESS_NAME", DEFAULT_BUSINESS_NAME)
        business_description = os.environ.get(
            "CHAT_AGENT_BUSINESS_DESCRIPTION",
            DEFAULT_BUSINESS_DESCRIPTION,
        )
        escalation_message = os.environ.get(
            "CHAT_AGENT_ESCALATION_MESSAGE",
            DEFAULT_ESCALATION_MESSAGE,
        )

        return (
            f"You are a friendly and professional customer service agent for {business_name}. "
            f"{business_description} "
            "Always be warm, patient, and helpful. Respond clearly and concisely. "
            f"If you do not know a specific price, policy, or service detail, say so and advise "
            f"the customer to {escalation_message}."
        )


chat_agent = ChatAgent()
