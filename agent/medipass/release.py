"""A time-limited pharmacy code that shows warnings only.

The code carries the medicine, the traffic light and the warning titles. It never carries the profile
(other medicines, allergies, conditions or genes). It is signed, so it cannot be changed, and it expires.
Nothing is stored on the server.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any

from .engine import CheckResult

TTL_SECONDS = 15 * 60
_fallback_secret = secrets.token_bytes(32)


def _secret() -> bytes:
    configured = os.environ.get("MEDIPASS_SECRET") or os.environ.get("AGENT_STUDIO_VERIFIER_TOKEN")
    return configured.encode() if configured else _fallback_secret


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(payload: str) -> str:
    return _b64(hmac.new(_secret(), payload.encode(), hashlib.sha256).digest())


def issue(result: CheckResult, now: float | None = None) -> str:
    body = {
        "m": result.medicine,
        "v": result.verdict,
        "w": [{"l": f.light, "t": f.title} for f in result.findings],
        "e": int((now if now is not None else time.time()) + TTL_SECONDS),
    }
    payload = _b64(json.dumps(body, separators=(",", ":")).encode())
    return f"{payload}.{_sign(payload)}"


def read(code: str, now: float | None = None) -> dict[str, Any] | None:
    """The warnings in a code, or None if it is forged, damaged or expired."""
    payload, _, signature = code.partition(".")
    if not payload or not hmac.compare_digest(signature, _sign(payload)):
        return None
    try:
        body = json.loads(_unb64(payload))
    except ValueError:
        return None
    if not isinstance(body, dict) or body.get("e", 0) < (now if now is not None else time.time()):
        return None
    return {"medicine": body.get("m"), "verdict": body.get("v"), "warnings": [{"light": w.get("l"), "title": w.get("t")} for w in body.get("w", [])], "expiresAt": body["e"]}
