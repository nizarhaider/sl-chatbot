from pathlib import Path


VOICE_AGENT_PROMPT_PATH = Path(__file__).with_name("prompts") / "slt_agent.md"
VOICE_AGENT_PROMPT = VOICE_AGENT_PROMPT_PATH.read_text(encoding="utf-8").strip()
