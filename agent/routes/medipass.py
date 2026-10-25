"""POST /api/medipass/check is free (the app demo). POST /v1/interaction-check is the same check, paid per call.

Body for both  { "medicine": "Nurofen", "profile": { "medications": ["Clopidogrel"], "allergies": [],
"conditions": [], "genotype": { "CYP2C19": "poor" } }, "explain": true, "lang": "en" }

POST /api/medipass/review checks every medicine in the profile (the medication review).
POST /api/medipass/release takes the same body and returns a signed pharmacy code (warnings only, 15 minutes).
GET /api/medipass/release/{code} reads a code. The pharmacy page for it is /r/{code}.
"""

import base64
import binascii
import json
import logging
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent.ai.llm import MissingLlmKeyError, ask_llm_image
from agent.medipass import release
from agent.medipass.engine import check, known_medicines, parse_profile, review
from agent.medipass.explain import explain
from agent.medipass.scan import MAX_IMAGE_BYTES, ScanError, identify
from agent.payments.server import PaidRouteOptions, paid_route

MAX_NAME_CHARS = 100
logger = logging.getLogger("agent")
MAX_ITEMS = 50


async def _parse(request: Request) -> tuple[Any, JSONResponse | None]:
    """Returns ((body, medicine, profile), None) or (None, error response)."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        body = None
    medicine = body.get("medicine") if isinstance(body, dict) else None
    if not isinstance(medicine, str) or not medicine.strip() or len(medicine) > MAX_NAME_CHARS:
        return None, JSONResponse({"error": 'Send JSON like { "medicine": "Ibuprofen", "profile": { ... } }'}, status_code=400)
    raw_profile: Any = body.get("profile", {})
    if raw_profile is None:
        raw_profile = {}
    try:
        if not isinstance(raw_profile, dict):
            raise ValueError('"profile" must be an object.')
        profile = parse_profile(raw_profile)
        if any(len(v) > MAX_ITEMS for v in (profile.medications, profile.allergies, profile.conditions)):
            raise ValueError(f"At most {MAX_ITEMS} entries per list.")
    except ValueError as err:
        return None, JSONResponse({"error": str(err)}, status_code=400)
    return (body, medicine, profile), None


async def _run_check(request: Request) -> JSONResponse:
    parsed, error = await _parse(request)
    if error or parsed is None:
        return error or JSONResponse({"error": "Bad request"}, status_code=400)
    body, medicine, profile = parsed
    result = check(medicine, profile)
    payload = result.to_dict()
    if body.get("explain") is True:
        payload["explanation"] = await explain(result, str(body.get("lang") or "en"))
    return JSONResponse(payload)


def price() -> PaidRouteOptions:
    return PaidRouteOptions(price=f"${os.environ.get('MEDIPASS_PRICE_USD') or '0.01'}", description="One MediPass interaction and pharmacogenetics check")


def register(app: FastAPI) -> None:
    @app.get("/api/medipass/medicines")
    async def medicines() -> dict[str, list[str]]:
        return {"medicines": known_medicines()}

    @app.post("/api/medipass/check")
    async def free_check(request: Request) -> JSONResponse:
        return await _run_check(request)

    @app.post("/api/medipass/scan")
    async def scan(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = None
        image = body.get("image") if isinstance(body, dict) else None
        if not isinstance(image, str) or not image:
            return JSONResponse({"error": 'Send JSON like { "image": "<base64 or data URL>", "mediaType": "image/jpeg" }'}, status_code=400)
        media_type = body.get("mediaType") if isinstance(body.get("mediaType"), str) else "image/jpeg"
        if image.startswith("data:"):
            header, _, image = image.partition(",")
            media_type = header[5:].split(";")[0] or media_type
        if len(image) > MAX_IMAGE_BYTES * 4 // 3 + 16:
            return JSONResponse({"error": "The image is larger than 3 MB."}, status_code=413)
        try:
            raw = base64.b64decode(image, validate=True)
        except (binascii.Error, ValueError):
            return JSONResponse({"error": "The image is not valid base64."}, status_code=400)
        try:
            return JSONResponse(await identify(raw, media_type, ask_llm_image))
        except ScanError as err:
            return JSONResponse({"error": str(err)}, status_code=400)
        except MissingLlmKeyError as err:
            return JSONResponse({"error": str(err)}, status_code=503)
        except Exception as err:
            logger.error("Scan failed (%s)", type(err).__name__)
            return JSONResponse({"error": "The photo could not be read right now. Try again or type the name."}, status_code=502)

    @app.post("/api/medipass/review")
    async def review_medicines(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = None
        try:
            if not isinstance(body, dict) or not isinstance(body.get("profile"), dict):
                raise ValueError('Send JSON like { "profile": { "medications": [...] } }')
            profile = parse_profile(body["profile"])
            if len(profile.medications) > MAX_ITEMS:
                raise ValueError(f"At most {MAX_ITEMS} medicines.")
        except ValueError as err:
            return JSONResponse({"error": str(err)}, status_code=400)
        return JSONResponse(review(profile))

    @app.post("/api/medipass/release")
    async def issue_release(request: Request) -> JSONResponse:
        parsed, error = await _parse(request)
        if error or parsed is None:
            return error or JSONResponse({"error": "Bad request"}, status_code=400)
        _body, medicine, profile = parsed
        result = check(medicine, profile)
        if result.verdict == "unknown":
            return JSONResponse({"error": "No code for a medicine the rules do not know."}, status_code=422)
        code = release.issue(result)
        return JSONResponse({"code": code, "path": f"/r/{code}", "expiresInSeconds": release.TTL_SECONDS})

    @app.get("/api/medipass/release/{code}")
    async def read_release(code: str) -> JSONResponse:
        content = release.read(code)
        if content is None:
            return JSONResponse({"error": "This code is invalid or has expired."}, status_code=404)
        return JSONResponse(content)

    @paid_route(app, "POST", "/v1/interaction-check", price)
    async def paid_check(request: Request) -> JSONResponse:
        return await _run_check(request)
