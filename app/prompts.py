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
    if agent_config.get("outbound"):
        return case["greeting"]
    return (
        f"ආයුබෝවන්!. මම {case['name']}, {case['company']} වෙතින් කතා කරන්නේ. "
        "ඔබට සිංහලෙන් කතා කිරීමට ‘සිංහල’ කියන්න. தமிழில் பேச, ‘தமிழ்’ என்று சொல்லுங்கள். "
        "To speak in English, say 'English'."
    )


def system_prompt(agent_config):
    case = case_profile(agent_config)
    return (
        solution_prompt(case["solution"], agent_config.get("outbound", False))
        + " "
        + case["instructions"]
    )


START_CALL = "Begin this phone call with exactly this greeting: {greeting}"
AUDITION_PROMPT = "Listen to the caller and respond to their recording."
