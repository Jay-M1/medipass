"""One place that decides which model answers. Claude when ANTHROPIC_API_KEY is set, otherwise Gemini.

LLM_PROVIDER=claude or LLM_PROVIDER=gemini forces a choice. The rest of the app only calls ask_llm.
"""

import os

from .claude import MissingClaudeKeyError, ask_claude
from .gemini import MissingGeminiKeyError, ask_gemini

FALLBACK_MODELS = {"gemini": "gemini-flash-lite-latest", "claude": "claude-haiku-5-5"}


class MissingLlmKeyError(Exception):
    pass


def provider() -> str:
    chosen = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if chosen in FALLBACK_MODELS:
        return chosen
    return "claude" if os.environ.get("ANTHROPIC_API_KEY") else "gemini"


def fallback_model() -> str:
    """A lighter model of the same provider, used when the main one is busy."""
    return FALLBACK_MODELS[provider()]


async def ask_llm(prompt: str, model: str | None = None) -> str:
    try:
        return await (ask_claude(prompt, model) if provider() == "claude" else ask_gemini(prompt, model))
    except (MissingClaudeKeyError, MissingGeminiKeyError) as err:
        raise MissingLlmKeyError(str(err)) from err
