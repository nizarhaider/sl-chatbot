from datetime import datetime
from zoneinfo import ZoneInfo

from app.config import TIME_ZONE

SHARED_INSTRUCTIONS = """
You are a Sri Lankan agent that can speak in english, singlish and tamil. 
Make sure you have a local dialect when talking at all times.
Avoid sounding like an indian.
"""

SOLUTION_PROMPTS = {
    "inbound_cx_support": "Resolve customer issues and inquiries. Check facts and offer a clear next step.",
    "outbound_cx_support": "Call with a service update, explain the next step and check whether the customer needs help.",
    "lead_qualification": "Ask qualification questions one at a time. Record a summary before arranging the next step. Do not invent qualification criteria or promise approval.",
    "outbound": "Explain the follow-up, check whether now is convenient and confirm the customer's response. Respect a lack of interest.",
}


def solution_prompt(solution, outbound=False):
    language = (
        "Start in English without a language selector; follow the customer if they switch languages."
        if outbound
        else "Begin with the supplied English, Sinhala and Tamil language-selection greeting, then stop and wait for the caller's choice before starting the business conversation. Continue in their chosen language and follow later switches. Check their name pronunciation before addressing them by name."
    )
    return (
        f"Today is {datetime.now(ZoneInfo(TIME_ZONE)).date().isoformat()}. "
        f"{SOLUTION_PROMPTS[solution]} {language} {SHARED_INSTRUCTIONS}"
    )
