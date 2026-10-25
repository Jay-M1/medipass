"""Asks the model, and copes with a busy one. 429 and 5xx are retried with a short pause, then a lighter model steps in."""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from .llm import fallback_model

logger = logging.getLogger("agent")

BUSY_CODES = {429, 500, 503, 504, 529}
RETRIES = 2

Asker = Callable[..., Awaitable[str]]


async def ask_resilient(ask: Asker, prompt: str) -> str:
    """`ask(prompt)` first. A busy model is retried, and the last attempt uses `ask(prompt, model=<lighter model>)`."""
    last: Exception | None = None
    for attempt in range(RETRIES + 1):
        try:
            return await (ask(prompt) if attempt < RETRIES else ask(prompt, model=fallback_model()))
        except Exception as err:
            if getattr(err, "code", None) not in BUSY_CODES:
                raise
            last = err
            logger.warning("Model busy (%s), attempt %d", getattr(err, "code", "?"), attempt + 1)
            await asyncio.sleep(0.5 * (attempt + 1))
    assert last is not None
    raise last
