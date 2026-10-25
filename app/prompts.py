from app.case_prompt import CASE_PROMPTS, DEFAULT_CASE, OUTBOUND_CASES
from app.solution_prompts import solution_prompt


def case_profile(agent_config):
    key = (
        OUTBOUND_CASES[agent_config["case_slug"]]
        if agent_config.get("outbound")
        else DEFAULT_CASE
    )
    return CASE_PROMPTS[key]


def opening_greeting(agent_config):
    case = case_profile(agent_config)
    return case["greeting"]


def system_prompt(agent_config):
    case = case_profile(agent_config)
    return (
        solution_prompt(case["solution"], agent_config.get("outbound", False))
        + " "
        + case["instructions"]
    )


START_CALL = "Begin this phone call with exactly this greeting: {greeting}"
AUDITION_PROMPT = "Listen to the caller and respond to their recording."
