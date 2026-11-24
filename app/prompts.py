import os

from app.case_prompt import CASE_PROMPTS, DEFAULT_CASE, OUTBOUND_CASES
from app.solution_prompts import solution_prompt


def case_profile(agent_config):
    return CASE_PROMPTS[
        OUTBOUND_CASES[agent_config["case_slug"]] if agent_config.get("outbound") else DEFAULT_CASE
    ]


def opening_greeting(agent_config):
    return "ආයුබෝවන්. For English, say English. සිංහලට සිංහල කියන්න. தமிழுக்கு தமிழ் சொல்லுங்கள்."


def system_prompt(agent_config):
    language = (
        " Start with the supplied language selector and listen. Continue in the caller's chosen "
        "language, follow later switches, and respond to a direct request in its language."
    )
    serendib_agent = agent_config.get("company_name", "").strip().casefold() == "serendibai.lk"
    if (os.environ.get("SERENDIB_INBOUND") == "1" or serendib_agent) and not agent_config.get("outbound"):
        return (
            "- You are Nishadi, SerendibAI's voice assistant.\n"
            "- After the language choice, introduce yourself and what we do in one short sentence, then ask how you can help.\n"
            "- We provide WhatsApp voice agents for businesses in Sinhala, Tamil and English.\n"
            "- Answer in the caller's chosen language; keep replies to one or two short sentences.\n"
            "- Ask when unclear, never invent details, and end politely."
        ) + language

    if agent_config.get("batch"):
        return (
            f"After the language choice, introduce yourself as {agent_config['name']} from {agent_config['company_name']}, explain you have a few questions about work experience, and ask if now is a good time. If they agree, ask one casual, neutral question at a time about the selected work fields, based only on their labels. Never mention validation, assessments, English scoring, or a resume; reveal no record values or whether answers match. Thank them briefly after each answer. Say goodbye and use end_call when finished or if they decline."
            + language
            + f" Selected fields and internal result mapping: {agent_config['batch']['instructions']}"
        )
    case = case_profile(agent_config)
    return (
        solution_prompt(case["solution"], agent_config.get("outbound", False))
        + " "
        + case["instructions"]
        + language
        + (f" After the language choice, convey this introduction in the chosen language: {case['greeting']}" if agent_config.get("outbound") else "")
    )


START_CALL = "Begin this phone call with exactly this greeting: {greeting}"
