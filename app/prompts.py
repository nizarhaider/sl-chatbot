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
    if agent_config.get("batch"):
        return f"ආයුබෝවන්, I'm {agent_config['name']} calling from {agent_config['company_name']}. I have a couple of quick questions about your work experience. Do you have a few minutes?"
    case = case_profile(agent_config)
    return case["greeting"]


def system_prompt(agent_config):
    if agent_config.get("batch"):
        return (
            "Start with the supplied greeting exactly. If they agree, ask one casual, neutral question at a time about the selected work fields, based only on their labels. Never mention validation, assessments, English, or a resume; reveal no record values or whether answers match. Thank them briefly after each answer and end politely if they decline."
            + f" Selected fields and internal result mapping: {agent_config['batch']['instructions']}"
        )
    case = case_profile(agent_config)
    return (
        solution_prompt(case["solution"], agent_config.get("outbound", False))
        + " "
        + case["instructions"]
    )


START_CALL = "Begin this phone call with exactly this greeting: {greeting}"
AUDITION_PROMPT = "Listen to the caller and respond to their recording."
