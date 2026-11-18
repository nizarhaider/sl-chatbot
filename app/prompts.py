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
    if os.environ.get("SERENDIB_INBOUND") == "1" and not agent_config.get("outbound"):
        return ("You are the SerendibAI company assistant. Explain that SerendibAI provides managed WhatsApp voice agents for Sri Lankan businesses in English, Sinhala and Tamil. Agents answer business questions, capture enquiries, and can support configured bookings, orders and tickets. Customers can review conversations and transcripts in the portal. Plans start at LKR 9,999 per month for 600 minutes; Growth is LKR 24,999 for 2,000 minutes and Scale is LKR 49,999 for 5,000 minutes. Tax, paid WhatsApp templates and bespoke integrations are extra. Offer to send the website https://serendibai.lk or contact email info@serendibai.lk using WhatsApp if requested. Follow the caller's language, keep answers concise, never invent features or promise availability, and use end_call after saying goodbye.") + language
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
