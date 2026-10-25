"""Asks Google Gemini a question with the official google-genai SDK.

GEMINI_API_KEY is read from the environment only. Never return it in a response or log it.
"""

import base64
import os

from google import genai
from google.genai import types
from google.genai import types

DEFAULT_MODEL = "gemini-flash-latest"

_client: genai.Client | None = None


class MissingGeminiKeyError(Exception):
    def __init__(self) -> None:
        super().__init__("GEMINI_API_KEY is not set. Add it to .env and restart the agent.")


def _get_client() -> genai.Client:
    global _client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise MissingGeminiKeyError()
    if _client is None:
        _client = genai.Client(api_key=api_key)
    return _client


async def ask_gemini(prompt: str, model: str | None = None, image: tuple[str, str] | None = None) -> str:
    """`image` is (mime type, base64 data) and is sent together with the prompt."""
    contents: object = prompt
    if image:
        contents = [types.Part.from_bytes(data=base64.b64decode(image[1]), mime_type=image[0]), prompt]
    response = await _get_client().aio.models.generate_content(
        model=model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL,
        contents=contents,
    )
    return response.text or ""


async def ask_gemini_image(prompt: str, image: bytes, media_type: str, model: str | None = None) -> str:
    response = await _get_client().aio.models.generate_content(
        model=model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL,
        contents=[types.Part.from_bytes(data=image, mime_type=media_type), prompt],
    )
    return response.text or ""
