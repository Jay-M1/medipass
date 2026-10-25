"""Lets the model read a free sentence in any language, so people do not have to use keywords.

The model only suggests. It may add a symptom, name a medicine we know, or raise a red flag. It can never remove a
red flag the rules found, and everything it returns is checked against fixed lists before the agent uses it.
If the model is unavailable the agent falls back to the keyword rules alone.
"""

import asyncio
import json
import logging
import re
from typing import Any

from agent.ai.llm import ask_llm
from agent.ai.resilient import ask_resilient

from .catalog import INTENTS

logger = logging.getLogger("agent")

TIMEOUT_SECONDS = 10
MAX_FLAGS = 5
MAX_FLAG_CHARS = 80
INTENT_NAMES = {name for name, _ in INTENTS}


def _prompt(text: str) -> str:
    return (
        "You read one message from a patient to a medicine ordering assistant. The message can be in any language.\n"
        'Reply with JSON only, shaped like {"intent": <"infection"|"headache"|"fever"|"heartburn"|"pain"|null>, "medicine": <the medicine the person names, or null>, '
        '"red_flags": [<short English phrases>]}.\n'
        "intent is the main complaint, or null if none of those fits. medicine is only a medicine the person explicitly names. Do not suggest one.\n"
        "red_flags lists every possible emergency sign the message mentions or hints at, for example a sudden or worst ever pain, chest pain, trouble breathing, "
        "confusion, trouble speaking, weakness or numbness, vision loss, fainting or seizure, a head injury, stiff neck, pregnancy. If unsure, include it. Never diagnose.\n\n"
        f"Message\n{text}"
    )


def clean(raw: Any) -> dict[str, Any] | None:
    """Keeps only what fits the fixed lists. Returns None when the reply is unusable."""
    if not isinstance(raw, dict):
        return None
    intent = raw.get("intent")
    medicine = raw.get("medicine")
    flags = raw.get("red_flags")
    return {
        "intent": intent if intent in INTENT_NAMES else None,
        "medicine": medicine.strip()[:60] if isinstance(medicine, str) and medicine.strip() else None,
        "red_flags": [f.strip()[:MAX_FLAG_CHARS] for f in flags if isinstance(f, str) and f.strip()][:MAX_FLAGS] if isinstance(flags, list) else [],
    }


async def understand_prompt(text: str) -> dict[str, Any] | None:
    """The model's reading of the message, or None if it could not be used."""
    try:
        reply = await asyncio.wait_for(ask_resilient(ask_llm, _prompt(text)), TIMEOUT_SECONDS)
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip())
        return clean(json.loads(cleaned))
    except Exception as err:
        logger.warning("MediPass understanding skipped (%s)", type(err).__name__)
        return None
