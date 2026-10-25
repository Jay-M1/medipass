"""A short plain-language explanation of a verdict. The model explains the findings and never decides them.

If there is no Gemini key or the call fails, the check still works and the explanation is simply missing.
"""

import json
import logging

from agent.ai.llm import ask_llm
from agent.ai.resilient import ask_resilient

from .engine import CheckResult

logger = logging.getLogger("agent")

LANGUAGES = {"en": "English", "de": "German"}


async def explain(result: CheckResult, lang: str = "en") -> str | None:
    if result.verdict == "unknown" or lang not in LANGUAGES:
        return None
    facts = {"medicine": result.medicine, "traffic_light": result.verdict, "findings": [{"title": f.title, "detail": f.detail} for f in result.findings]}
    prompt = (
        f"Explain this medicine check to a patient in {LANGUAGES[lang]}, in at most three short sentences, in a calm and clear tone.\n"
        "Use ONLY the facts below. Do not add medical facts, doses or advice that is not in them, and never change the traffic light.\n"
        "If the light is green, say that nothing was found in the checked data and that this is not a guarantee.\n"
        "End by suggesting to ask a pharmacist or doctor when the light is yellow or red.\n\n"
        f"Facts\n{json.dumps(facts)}"
    )
    try:
        text = (await ask_resilient(ask_llm, prompt)).strip()
    except Exception as err:
        logger.warning("MediPass explanation skipped (%s)", type(err).__name__)
        return None
    return text or None
