def opening_greeting(agent_config):
    return agent_config.get("greeting", "").strip()


def system_prompt(agent_config):
    instructions = agent_config.get("system_prompt", "").strip()
    greeting = agent_config.get("greeting", "").strip()
    batch = agent_config.get("batch", {}).get("instructions", "").strip()
    opening = f"Opening greeting:\n{greeting}" if greeting else ""
    return "\n\n".join(part for part in (instructions, opening, batch) if part)
