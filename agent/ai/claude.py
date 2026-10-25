"""Asks Anthropic's Claude through the Messages API with httpx, so no extra package is needed.

ANTHROPIC_API_KEY is read from the environment only. Never return it in a response or log it.
"""

import os

import httpx

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 1024
TIMEOUT_SECONDS = 30


class MissingClaudeKeyError(Exception):
    def __init__(self) -> None:
        super().__init__("ANTHROPIC_API_KEY is not set. Add it to .env and restart the agent.")


class ClaudeError(Exception):
    """The API answered with an error. `code` is the HTTP status, so callers can tell a busy model (429, 529) from a real error."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"Claude request failed with HTTP {code}. {message}")
        self.code = code


async def ask_claude(prompt: str, model: str | None = None, transport: httpx.AsyncBaseTransport | None = None) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise MissingClaudeKeyError()
    body = {"model": model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL, "max_tokens": MAX_TOKENS, "messages": [{"role": "user", "content": prompt}]}
    headers = {"x-api-key": api_key, "anthropic-version": API_VERSION, "content-type": "application/json"}
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, transport=transport) as client:
        res = await client.post(API_URL, json=body, headers=headers)
    if res.status_code >= 400:
        try:
            detail = str(res.json().get("error", {}).get("message", ""))[:200]
        except ValueError:
            detail = ""
        raise ClaudeError(res.status_code, detail)
    data = res.json()
    return "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
