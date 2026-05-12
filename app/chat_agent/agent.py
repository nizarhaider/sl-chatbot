import logging
import os
import re
from dataclasses import dataclass

from google import genai

from app.services.business_info import BusinessInfoProvider
from app.services.order_sheet import CustomerOrder, OrderLine, OrderSheetClient
from app.services.product_catalog import Product, ProductCatalog

logger = logging.getLogger(__name__)

DEFAULT_CHAT_AGENT_MAX_OUTPUT_TOKENS = "512"

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
        self.model = os.environ.get("CHAT_AGENT_MODEL", "gemini-3-flash-preview")
        self.product_catalog = ProductCatalog()
        self.business_info_provider = BusinessInfoProvider()
        self.order_sheet = OrderSheetClient()
        self.pending_orders: dict[str, CustomerOrder] = {}
        self.last_product_searches: dict[str, list[Product]] = {}
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

        tool_result = await self._get_model_tool_response(text, sender_id=sender_id)
        if tool_result:
            return tool_result

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
                    "max_output_tokens": _max_output_tokens(),
                    "temperature": float(os.environ.get("CHAT_AGENT_TEMPERATURE", "0.8")),
                },
            )
            _log_model_response("chat", response)
            return ChatAgentResult(
                response.text
                or "Sorry, I could not process that right now. Please try again shortly."
            )
        except Exception as e:
            logger.error("Error getting chat response from Gemini: %s", e)
            return ChatAgentResult("Sorry, I could not process that right now. Please try again shortly.")

    async def _get_model_tool_response(
        self,
        text: str,
        sender_id: str | None = None,
    ) -> ChatAgentResult | None:
        sender_key = sender_id or "unknown"
        manager_message: str | None = None

        def search_products(query: str) -> dict:
            """Search the product catalog. Use this for product, price, stock, and order-item questions."""
            products = self.product_catalog.search(query)
            self.last_product_searches[sender_key] = products
            return {"products": [_product_payload(product, index) for index, product in enumerate(products, 1)]}

        def create_pending_order(sku: str, quantity: int = 1) -> dict:
            """Create a pending order for exactly one SKU after the customer has chosen the item."""
            product = self.product_catalog.get_by_sku(sku)
            if not product:
                return {"ok": False, "message": f"I could not find SKU {sku}. Please choose one listed product."}

            order = self._create_pending_order(sender_key, product, quantity, text)
            return {
                "ok": True,
                "message": "Pending order created. Ask the customer to reply CONFIRM to place it or CANCEL to stop.",
                "order": _order_payload(order),
            }

        def update_pending_order_quantity(quantity: int) -> dict:
            """Update the quantity on the customer's current pending order."""
            order = self.pending_orders.get(sender_key)
            if not order:
                return {"ok": False, "message": "There is no pending order to update."}
            safe_quantity = _safe_quantity(quantity)
            updated_order = CustomerOrder(
                customer_phone=order.customer_phone,
                customer_message=order.customer_message,
                lines=[
                    OrderLine(
                        name=line.name,
                        quantity=safe_quantity,
                        sku=line.sku,
                        price=line.price,
                    )
                    for line in order.lines
                ],
            )
            self.pending_orders[sender_key] = updated_order
            return {"ok": True, "order": _order_payload(updated_order)}

        def cancel_pending_order() -> dict:
            """Cancel the customer's current pending order."""
            self.pending_orders.pop(sender_key, None)
            return {"ok": True, "message": "Pending order cancelled."}

        def confirm_pending_order() -> dict:
            """Confirm the customer's pending order, append it to the order workbook, and prepare manager notification."""
            nonlocal manager_message
            result = self._confirm_pending_order(sender_key)
            manager_message = result.manager_message
            return {"ok": bool(result.manager_message), "message": result.reply}

        try:
            business_info = self.business_info_provider.get_text(
                os.environ.get("CHAT_AGENT_BUSINESS_DESCRIPTION", DEFAULT_BUSINESS_DESCRIPTION)
            )
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=(
                    f"Current store information:\n{business_info}\n\n"
                    f"Customer WhatsApp number: {sender_key}\n"
                    f"Customer message: {text}"
                ),
                config={
                    "system_instruction": self._build_tool_system_prompt(),
                    "max_output_tokens": _max_output_tokens(),
                    "temperature": float(os.environ.get("CHAT_AGENT_TEMPERATURE", "0.3")),
                    "tools": [
                        search_products,
                        create_pending_order,
                        update_pending_order_quantity,
                        cancel_pending_order,
                        confirm_pending_order,
                    ],
                },
            )
            _log_model_response("commerce_tools", response)
        except Exception as exc:
            logger.error("Error getting model tool response from Gemini: %s", exc)
            return None

        if not response.text:
            return None
        return ChatAgentResult(reply=response.text, manager_message=manager_message)

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

    def _build_tool_system_prompt(self) -> str:
        return (
            f"{self.system_prompt} "
            "You have commerce tools. Use search_products before answering product availability, price, or product-list questions. "
            "Never invent products, prices, SKUs, or stock. "
            "When the customer wants to order, create_pending_order only after one exact SKU is clear. "
            "If multiple products match, show a numbered list with SKU and ask the customer to pick one. "
            "Do not create an order containing multiple products unless the customer explicitly asks for multiple specific SKUs. "
            "When a pending order exists, ask the customer to reply CONFIRM to place it or CANCEL to stop. "
            "Use confirm_pending_order only when the customer clearly confirms."
        )

    def _create_pending_order(
        self,
        sender_key: str,
        product: Product,
        quantity: int,
        customer_message: str,
    ) -> CustomerOrder:
        order = CustomerOrder(
            customer_phone=sender_key,
            customer_message=customer_message,
            lines=[
                OrderLine(
                    name=product.display_name(),
                    quantity=_safe_quantity(quantity),
                    sku=product.sku,
                    price=product.price,
                )
            ],
        )
        self.pending_orders[sender_key] = order
        return order

    def _confirm_pending_order(self, sender_key: str) -> ChatAgentResult:
        order = self.pending_orders.pop(sender_key, None)
        if not order:
            return ChatAgentResult("I do not have a pending order to confirm yet. Please choose a product first.")

        order_written = self.order_sheet.append_order(order)
        logger.info(
            "Confirmed order for %s with %s line(s); order_written=%s",
            sender_key,
            len(order.lines),
            order_written,
        )
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
            return self._confirm_pending_order(sender_key)

        selected_product = self._product_from_selection(sender_key, normalized)
        if selected_product:
            quantity = _extract_quantity(text) or 1
            order = self._create_pending_order(sender_key, selected_product, quantity, text)
            return ChatAgentResult(reply=_format_pending_order(order))

        product_matches = self.product_catalog.search(text)
        if not product_matches:
            return None
        self.last_product_searches[sender_key] = product_matches

        if _is_order_request(normalized):
            exact_product = _best_order_product(text, product_matches)
            if exact_product:
                quantity = _extract_quantity(text) or 1
                order = self._create_pending_order(sender_key, exact_product, quantity, text)
                return ChatAgentResult(reply=_format_pending_order(order))
            return ChatAgentResult(reply=_format_product_choices(product_matches))

        if _is_product_query(normalized):
            return ChatAgentResult(reply=_format_product_matches(product_matches))

        return None

    def _product_from_selection(self, sender_key: str, normalized_text: str) -> Product | None:
        products = self.last_product_searches.get(sender_key, [])
        if not products:
            return None

        selection_match = re.fullmatch(r"(?:option\s*)?(\d{1,2})", normalized_text)
        if selection_match:
            index = int(selection_match.group(1)) - 1
            if 0 <= index < len(products):
                return products[index]

        sku_match = re.search(r"\b([a-z]{2,}[-_][a-z0-9_-]+)\b", normalized_text, re.IGNORECASE)
        if sku_match:
            return self.product_catalog.get_by_sku(sku_match.group(1))

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


def _extract_quantity(text: str) -> int | None:
    quantity_match = re.search(r"\b(?:qty|quantity|x)?\s*(\d{1,3})\b", text.lower())
    if not quantity_match:
        return None
    quantity = int(quantity_match.group(1))
    if quantity <= 0:
        return None
    return min(quantity, 999)


def _safe_quantity(quantity: int | str | None) -> int:
    try:
        parsed = int(quantity or 1)
    except (TypeError, ValueError):
        parsed = 1
    if parsed <= 0:
        return 1
    return min(parsed, 999)


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


def _format_product_choices(products: list[Product]) -> str:
    lines = ["I found a few nail options. Please reply with the number or SKU for the one you want:"]
    for index, product in enumerate(products, 1):
        details = []
        if product.sku:
            details.append(f"SKU: {product.sku}")
        if product.price:
            details.append(f"Price: {product.price}")
        lines.append(f"{index}. {product.display_name()} ({', '.join(details)})")
    return "\n".join(lines)


def _format_pending_order(order: CustomerOrder) -> str:
    return (
        "I have this pending order:\n"
        f"{order.summary()}\n\n"
        "Reply CONFIRM to place this order, or CANCEL to stop."
    )


def _best_order_product(text: str, products: list[Product]) -> Product | None:
    query_tokens = set(_important_tokens(text))
    if not query_tokens:
        return None

    best_product: Product | None = None
    best_score = 0
    second_score = 0
    for product in products:
        product_tokens = set(_important_tokens(product.display_name()))
        score = len(query_tokens & product_tokens)
        if score > best_score:
            second_score = best_score
            best_score = score
            best_product = product
        elif score > second_score:
            second_score = score

    if best_product and best_score >= 2 and best_score > second_score:
        return best_product
    return None


def _important_tokens(text: str) -> list[str]:
    stop_words = {
        "the",
        "and",
        "for",
        "set",
        "press",
        "nail",
        "nails",
        "order",
        "want",
        "need",
        "get",
        "qty",
        "quantity",
        "on",
    }
    return [
        token
        for token in "".join(char.lower() if char.isalnum() else " " for char in text).split()
        if len(token) > 2 and token not in stop_words
    ]


def _product_payload(product: Product, index: int) -> dict:
    return {
        "index": index,
        "name": product.display_name(),
        "sku": product.sku,
        "price": product.price,
        "stock": product.stock,
        "description": product.description,
    }


def _order_payload(order: CustomerOrder) -> dict:
    return {
        "customer_phone": order.customer_phone,
        "lines": [
            {
                "name": line.name,
                "quantity": line.quantity,
                "sku": line.sku,
                "price": line.price,
            }
            for line in order.lines
        ],
    }


def _manager_order_message(order: CustomerOrder, order_written: bool) -> str:
    sheet_status = "Order sheet updated." if order_written else "Order sheet update failed."
    return (
        "New confirmed WhatsApp order\n"
        f"Customer: {order.customer_phone}\n"
        f"{order.summary()}\n"
        f"Original message: {order.customer_message}\n"
        f"{sheet_status}"
    )


def _max_output_tokens() -> int:
    return int(os.environ.get("CHAT_AGENT_MAX_OUTPUT_TOKENS", DEFAULT_CHAT_AGENT_MAX_OUTPUT_TOKENS))


def _log_model_response(flow: str, response) -> None:
    finish_reason = _response_finish_reason(response)
    text_length = len(response.text or "")
    if finish_reason:
        logger.info(
            "Gemini %s response finish_reason=%s text_chars=%s",
            flow,
            finish_reason,
            text_length,
        )
    if str(finish_reason).upper().endswith("MAX_TOKENS"):
        logger.warning(
            "Gemini %s response may be partial because max output tokens were reached; text_chars=%s",
            flow,
            text_length,
        )


def _response_finish_reason(response) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    finish_reason = getattr(candidates[0], "finish_reason", None)
    if finish_reason is None:
        finish_reason = getattr(candidates[0], "finishReason", None)
    return str(finish_reason) if finish_reason is not None else None


chat_agent = ChatAgent()
