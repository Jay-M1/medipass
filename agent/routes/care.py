"""The care agent API.

POST /api/care/agent                 { "prompt": "I have a bad headache, get me something", "profile": {...}, "wearable": {...} }
GET  /api/care/requests              the doctor's queue (demo, no login)
GET  /api/care/requests/{id}         one request, the person polls this while waiting
POST /api/care/requests/{id}/decision  { "decision": "approve" | "reject", "note": "..." }
"""

import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent.medipass.care import agent as care
from agent.medipass.care.store import store
from agent.medipass.care.understand import understand_prompt
from agent.medipass.engine import parse_profile

MAX_PROMPT_CHARS = 1000


async def _body(request: Request) -> dict:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return body if isinstance(body, dict) else {}


def register(app: FastAPI) -> None:
    @app.post("/api/care/agent")
    async def run_agent(request: Request) -> JSONResponse:
        body = await _body(request)
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
            return JSONResponse({"error": f'Send JSON like {{ "prompt": "I have a headache, get me something" }} (up to {MAX_PROMPT_CHARS} characters).'}, status_code=400)
        raw_profile = body.get("profile", {})
        try:
            if not isinstance(raw_profile, dict):
                raise ValueError('"profile" must be an object.')
            profile = parse_profile(raw_profile)
        except ValueError as err:
            return JSONResponse({"error": str(err)}, status_code=400)
        return JSONResponse(await care.run(prompt.strip(), profile, body.get("wearable"), understand_prompt, auto_buy=body.get("autoBuy") is not False))

    @app.post("/api/care/buy")
    async def buy(request: Request) -> JSONResponse:
        body = await _body(request)
        ingredient, raw_profile = body.get("ingredient"), body.get("profile", {})
        try:
            if not isinstance(ingredient, str) or not isinstance(raw_profile, dict):
                raise ValueError('Send { "ingredient": "paracetamol", "profile": { ... } }')
            return JSONResponse(await care.buy(ingredient, parse_profile(raw_profile)))
        except ValueError as err:
            return JSONResponse({"error": str(err)}, status_code=409 if str(err).startswith(("Not ordered", "Only over")) else 400)

    @app.get("/api/care/requests")
    async def queue() -> JSONResponse:
        return JSONResponse({"requests": await store.all(), "persistent": store.persistent})

    @app.get("/api/care/requests/{request_id}")
    async def one(request_id: str) -> JSONResponse:
        record = await store.get(request_id)
        return JSONResponse(record) if record else JSONResponse({"error": "Not found."}, status_code=404)

    @app.post("/api/care/requests/{request_id}/decision")
    async def decision(request_id: str, request: Request) -> JSONResponse:
        body = await _body(request)
        verdict, note = body.get("decision"), body.get("note", "")
        if verdict not in ("approve", "reject") or not isinstance(note, str):
            return JSONResponse({"error": 'Send { "decision": "approve" or "reject", "note": "..." }'}, status_code=400)
        record = await care.decide(request_id, verdict, note)
        return JSONResponse(record) if record else JSONResponse({"error": "Not found."}, status_code=404)
