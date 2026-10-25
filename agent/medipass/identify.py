"""Finds out which medicine a package or a scanned code stands for.

Two inputs feed it. Text from a barcode or from the model reading a package photo. Both end in find_medicine, which
returns an ingredient the rules know, or None. Nothing here decides safety. That stays with the rules.
"""

import asyncio
import base64
import binascii
import json
import logging
import re
import time
from collections import defaultdict, deque
from typing import Any

from agent.ai.llm import ask_llm
from agent.ai.resilient import ask_resilient

from .engine import find_medicine

logger = logging.getLogger("agent")

MAX_IMAGE_BYTES = 6 * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
TIMEOUT_SECONDS = 40
RATE_LIMIT = 20
RATE_WINDOW_SECONDS = 600

PROMPT = (
    "This is a photo of a medicine package. Return the product name, the active substance, the strength and the 8-digit PZN if printed. "
    "Use null for anything you can't read. Don't guess.\n"
    'Reply with JSON only, shaped like {"name": string|null, "substance": string|null, "strength": string|null, "pzn": string|null}.'
)

_hits: dict[str, deque[float]] = defaultdict(deque)


class IdentifyError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def rate_limited(client: str, now: float | None = None) -> bool:
    """A small per client limit, because this route spends model credit."""
    now = time.monotonic() if now is None else now
    window = _hits[client]
    while window and now - window[0] > RATE_WINDOW_SECONDS:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        return True
    window.append(now)
    return False


def decode_image(raw: Any, mime: Any) -> tuple[str, str]:
    """Returns (mime type, base64) for a data URL or plain base64. Raises IdentifyError 400 or 413."""
    if not isinstance(raw, str) or not raw.strip():
        raise IdentifyError(400, 'Send JSON like { "image": "<base64 or data URL>" }.')
    text = raw.strip()
    found = re.match(r"^data:([\w/+.-]+);base64,(.*)$", text, re.S)
    if found:
        mime, text = found[1], found[2]
    mime = mime if isinstance(mime, str) and mime else "image/jpeg"
    if mime not in ALLOWED_MIME:
        raise IdentifyError(400, "Use a JPEG, PNG or WebP image.")
    if len(text) > MAX_IMAGE_BYTES * 4 // 3 + 16:
        raise IdentifyError(413, "The image is too large. Please use a smaller photo.")
    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        raise IdentifyError(400, "The image is not valid base64.") from None
    if not data:
        raise IdentifyError(400, 'Send JSON like { "image": "<base64 or data URL>" }.')
    if len(data) > MAX_IMAGE_BYTES:
        raise IdentifyError(413, "The image is too large. Please use a smaller photo.")
    return mime, base64.b64encode(data).decode()


def clean(raw: Any) -> dict[str, str | None]:
    """Keeps only short strings, and a PZN only if it is 7 or 8 digits."""
    fields = raw if isinstance(raw, dict) else {}

    def text(key: str) -> str | None:
        value = fields.get(key)
        return value.strip()[:120] if isinstance(value, str) and value.strip() else None

    pzn = fields.get("pzn")
    digits = re.sub(r"\D", "", pzn) if isinstance(pzn, str) else ""
    return {"name": text("name"), "substance": text("substance"), "strength": text("strength"), "pzn": digits if len(digits) in (7, 8) else None}


async def identify_package(image: tuple[str, str]) -> dict[str, Any]:
    async def ask(prompt: str, model: str | None = None) -> str:
        return await ask_llm(prompt, model, image=image)

    try:
        reply = await asyncio.wait_for(ask_resilient(ask, PROMPT), TIMEOUT_SECONDS)
        found = clean(json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip())))
    except Exception as err:
        logger.warning("MediPass package reading failed (%s)", type(err).__name__)
        raise IdentifyError(502, "The package could not be read right now. Please try again or type the name.") from err
    text = " ".join(v for v in (found["pzn"], found["name"], found["substance"]) if v)
    return {**found, "medicine": find_medicine(text) if text else None}
