"""One place that decides which model answers. Claude when ANTHROPIC_API_KEY is set, otherwise Gemini.

LLM_PROVIDER=claude or LLM_PROVIDER=gemini forces a choice. The rest of the app only calls ask_llm.
"""

import base64
import os

from .claude import MissingClaudeKeyError, ask_claude
from .gemini import MissingGeminiKeyError, ask_gemini, ask_gemini_image

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


async def ask_llm(prompt: str, model: str | None = None, image: tuple[str, str] | None = None) -> str:
    """`image` is (mime type, base64 data). Without it the call is text only, as before."""
    try:
        if provider() == "claude":
            return await (ask_claude(prompt, model, image=image) if image else ask_claude(prompt, model))
        return await (ask_gemini(prompt, model, image) if image else ask_gemini(prompt, model))
    except (MissingClaudeKeyError, MissingGeminiKeyError) as err:
        raise MissingLlmKeyError(str(err)) from err


async def ask_llm_image(prompt: str, image: bytes, media_type: str, model: str | None = None) -> str:
    """A question about one picture, answered by whichever provider is chosen."""
    try:
        if provider() == "claude":
            return await ask_claude(prompt, model, image=(media_type, base64.b64encode(image).decode()))
        return await ask_gemini_image(prompt, image, media_type, model)
    except (MissingClaudeKeyError, MissingGeminiKeyError) as err:
        raise MissingLlmKeyError(str(err)) from err
