import logging
import os
import re
from dataclasses import dataclass

from google import genai

from app.services.business_info import BusinessInfoProvider
from app.services.order_sheet import CustomerOrder, OrderLine, OrderSheetClient
from app.services.product_catalog import Product, ProductCatalog

logger = logging.getLogger(__name__)

DEFAULT_BUSINESS_NAME = "Ayidaah Beauty"
DEFAULT_BUSINESS_DESCRIPTION = (
    "Ayidaah Beauty is a Sri Lankan skincare and beauty brand founded in 2019. "
    "The store is at 119, Kurunduwatta Mosque Lane, Nawala Road, Rajagiriya, "
    "Sri Jayawardenepura Kotte 10100. Opening hours are Monday to Saturday, "
    "9:00 am to 8:00 pm, and closed on Sunday. Delivery is available. "
    "Payment options include cash, credit cards, and debit cards. "
    "The public contact number is +94 77 167 9595."
)
DEFAULT_ESCALATION_MESSAGE = "call or WhatsApp +94 77 167 9595"


@dataclass(frozen=True)
class ChatAgentResult:
    reply: str
    manager_message: str | None = None


class ChatAgent:
    def __init__(self):
        self._client = None
        self.model = os.environ.get("CHAT_AGENT_MODEL", "gemini-2.5-flash-lite")
        self.product_catalog = ProductCatalog()
        self.business_info_provider = BusinessInfoProvider()
        self.order_sheet = OrderSheetClient()
        self.pending_orders: dict[str, CustomerOrder] = {}
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

    async def get_response(self, text: str, sender_id: str | None = None) -> str:
        result = await self.process_message(text, sender_id=sender_id)
        return result.reply

    async def process_message(self, text: str, sender_id: str | None = None) -> ChatAgentResult:
        commerce_result = self._try_handle_commerce_message(text, sender_id=sender_id)
        if commerce_result:
            return commerce_result

        try:
            business_info = self.business_info_provider.get_text(
                os.environ.get("CHAT_AGENT_BUSINESS_DESCRIPTION", DEFAULT_BUSINESS_DESCRIPTION)
            )
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=(
                    f"Current store information from the business document:\n{business_info}\n\n"
                    f"Customer message:\n{text}"
                ),
                config={
                    "system_instruction": self.system_prompt,
                    "max_output_tokens": int(os.environ.get("CHAT_AGENT_MAX_OUTPUT_TOKENS", "150")),
                    "temperature": float(os.environ.get("CHAT_AGENT_TEMPERATURE", "0.8")),
                },
            )
            return ChatAgentResult(
                response.text
                or "Sorry, I could not process that right now. Please try again shortly."
            )
        except Exception as e:
            logger.error("Error getting chat response from Gemini: %s", e)
            return ChatAgentResult("Sorry, I could not process that right now. Please try again shortly.")

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

        business_info_source = (
            "Use the current store information provided in each user message as the source "
            "of truth for opening hours, address, notices, delivery information, and policies. "
        )

        return (
            f"You are a friendly and professional customer service agent for {business_name}. "
            f"{business_description} "
            f"{business_info_source}"
            "Always be warm, patient, and helpful. Respond clearly and concisely. "
            "Customers may ask about opening hours, location, delivery, payment methods, "
            "skincare products, beauty items, prices, and whether specific products are in stock. "
            "You do not have access to live inventory beyond the product catalog context. "
            "Answer product, price, and stock questions only from the product catalog context "
            "when it is provided by the application. If you do not know a specific price, policy, ingredient, suitability, "
            f"or service detail, say so and advise the customer to {escalation_message}."
        )

    def _try_handle_commerce_message(
        self,
        text: str,
        sender_id: str | None = None,
    ) -> ChatAgentResult | None:
        sender_key = sender_id or "unknown"
        normalized = text.strip().lower()

        if sender_key in self.pending_orders and _is_cancel_message(normalized):
            self.pending_orders.pop(sender_key, None)
            return ChatAgentResult("No problem, I have cancelled that pending order.")

        if sender_key in self.pending_orders and _is_confirm_message(normalized):
            order = self.pending_orders.pop(sender_key)
            order_written = self.order_sheet.append_order(order)
            manager_message = _manager_order_message(order, order_written)
            customer_reply = (
                "Thank you, your order has been confirmed and sent to our team. "
                "We will contact you shortly if anything else is needed."
            )
            if not order_written:
                customer_reply = (
                    "Thank you, your order has been confirmed and sent to our team. "
                    "Our order sheet could not be updated automatically, but the manager has been notified."
                )
            return ChatAgentResult(reply=customer_reply, manager_message=manager_message)

        product_matches = self.product_catalog.search(text)
        if not product_matches:
            return None

        if _is_order_request(normalized):
            order = CustomerOrder(
                customer_phone=sender_key,
                customer_message=text,
                lines=[_order_line_from_product(product, text) for product in product_matches[:3]],
            )
            self.pending_orders[sender_key] = order
            return ChatAgentResult(
                reply=(
                    "I found these items for your order:\n"
                    f"{order.summary()}\n\n"
                    "Reply CONFIRM to place this order, or CANCEL to stop. "
                    "If you need a different quantity or shade, send it before confirming."
                )
            )

        if _is_product_query(normalized):
            return ChatAgentResult(reply=_format_product_matches(product_matches))

        return None


def _is_confirm_message(text: str) -> bool:
    return bool(re.search(r"\b(confirm|confirmed|yes|ok|okay|place order|go ahead)\b", text))


def _is_cancel_message(text: str) -> bool:
    return bool(re.search(r"\b(cancel|stop|no|never mind|nevermind)\b", text))


def _is_order_request(text: str) -> bool:
    return bool(
        re.search(
            r"\b(order|buy|purchase|place an order|place order|i want|i need|can i get|send me)\b",
            text,
        )
    )


def _is_product_query(text: str) -> bool:
    return bool(
        re.search(
            r"\b(price|stock|available|availability|have|product|shade|size|cost|how much)\b",
            text,
        )
    )


def _order_line_from_product(product: Product, text: str) -> OrderLine:
    quantity = _extract_quantity(text) or 1
    return OrderLine(
        name=product.display_name(),
        quantity=quantity,
        sku=product.sku,
        price=product.price,
    )


def _extract_quantity(text: str) -> int | None:
    quantity_match = re.search(r"\b(?:qty|quantity|x)?\s*(\d{1,3})\b", text.lower())
    if not quantity_match:
        return None
    quantity = int(quantity_match.group(1))
    if quantity <= 0:
        return None
    return min(quantity, 999)


def _format_product_matches(products: list[Product]) -> str:
    lines = ["I found these matching products:"]
    for product in products:
        details = []
        if product.price:
            details.append(f"Price: {product.price}")
        if product.stock:
            details.append(f"Stock: {product.stock}")
        if product.sku:
            details.append(f"SKU: {product.sku}")
        suffix = f" ({', '.join(details)})" if details else ""
        lines.append(f"- {product.display_name()}{suffix}")
    lines.append("\nTo order, reply with: Order <product name> qty <number>.")
    return "\n".join(lines)


def _manager_order_message(order: CustomerOrder, order_written: bool) -> str:
    sheet_status = "Order sheet updated." if order_written else "Order sheet update failed."
    return (
        "New confirmed WhatsApp order\n"
        f"Customer: {order.customer_phone}\n"
        f"{order.summary()}\n"
        f"Original message: {order.customer_message}\n"
        f"{sheet_status}"
    )


chat_agent = ChatAgent()
