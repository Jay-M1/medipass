"""Reads a photo of a medicine package and says which medicine it is.

The model only reads what is printed on the box and never guesses from shape or colour. What it reads is then matched
against the medicines the rules know. The person always confirms the result before a traffic light is shown.
Photos are sent to the model provider for this one question and are not stored here.
"""

import json
import logging
import re
from typing import Any

from agent.ai.resilient import ask_resilient

from .engine import normalise_medicine

logger = logging.getLogger("agent")

MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_IMAGE_BYTES = 3_000_000
CONFIDENCE = {"high", "medium", "low"}

PROMPT = (
    "Look at this photo of a medicine package. Read only what is printed on it. Do not guess from the shape or colour of the pack or tablets.\\n"
    'Reply with JSON only, shaped like {"is_medicine_package": <true or false>, "brand": <brand name or null>, "active_ingredient": <the printed active ingredient in English, lowercase, or null>, '
    '"strength": <for example "400 mg" or null>, "confidence": <"high", "medium" or "low">}.\\n'
    'Use "high" only when the name or ingredient is clearly readable. If you cannot read a medicine name, set both brand and active_ingredient to null and confidence to "low". '
    "If the photo does not show a medicine package, set is_medicine_package to false."
)


class ScanError(Exception):
    pass


def clean(raw: Any) -> dict[str, Any]:
    """Keeps only expected fields and matches the ingredient against the medicines the rules know."""
    if not isinstance(raw, dict):
        raise ScanError("The answer could not be read.")

    def text(key: str, limit: int = 80) -> str | None:
        value = raw.get(key)
        return value.strip()[:limit] if isinstance(value, str) and value.strip() else None

    brand, ingredient, strength = text("brand"), text("active_ingredient"), text("strength", 30)
    confidence = raw.get("confidence") if raw.get("confidence") in CONFIDENCE else "low"
    known = next((found for found in (normalise_medicine(ingredient) if ingredient else None, normalise_medicine(brand) if brand else None) if found), None)
    return {"isMedicinePackage": raw.get("is_medicine_package") is True, "brand": brand, "ingredient": ingredient, "strength": strength, "confidence": confidence, "known": known}


async def identify(image: bytes, media_type: str, ask_image: Any) -> dict[str, Any]:
    if media_type not in MEDIA_TYPES:
        raise ScanError("Send a JPEG, PNG, WebP or GIF image.")
    if not image or len(image) > MAX_IMAGE_BYTES:
        raise ScanError("The image is empty or larger than 3 MB.")
    reply = await ask_resilient(lambda prompt, model=None: ask_image(prompt, image, media_type, model), PROMPT)
    try:
        return clean(json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip())))
    except ValueError as err:
        logger.warning("MediPass scan reply was not JSON")
        raise ScanError("The answer could not be read.") from err
