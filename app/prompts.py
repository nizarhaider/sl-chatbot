def opening_greeting(agent_config):
    return agent_config.get("greeting", "").strip()


def system_prompt(agent_config, lead_discovery=False):
    instructions = agent_config.get("system_prompt", "").strip()
    greeting = agent_config.get("greeting", "").strip()
    batch = agent_config.get("batch", {}).get("instructions", "").strip()
    if lead_discovery:
        instructions = f"""{instructions}
For inbound SerendibAI calls, replace the generic help question with this discovery flow:
- Briefly explain that SerendibAI builds managed WhatsApp voice agents in Sinhala, Tamil and English.
- Ask one question at a time about business type, monthly calls or enquiries, typical call length and main use cases.
- Tailor a short suggestion to the caller's needs using only verified services and prices in the knowledge base.
- Keep replies to one or two short sentences. Help existing customers with support first.
- Offer a demo or team follow-up when useful; do not invent pricing, capacity, integrations or results."""
    opening = f"Opening greeting:\n{greeting}" if greeting else ""
    return "\n\n".join(part for part in (instructions, opening, batch) if part)
