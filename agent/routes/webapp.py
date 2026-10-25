"""GET /app, /agent, /doctor and /vision are the MediPass pages. GET /r/{code} is what a pharmacist sees after scanning a release code."""

import html
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response

from agent.medipass import release

WEB = Path(__file__).resolve().parent.parent / "medipass" / "web"
PAGES = {"/app": "app.html", "/agent": "agent.html", "/doctor": "doctor.html", "/vision": "vision.html"}
COLOURS = {"green": "#1f9d55", "yellow": "#e0a100", "red": "#d6382f"}


def _pharmacy_page(content: dict) -> str:
    verdict = str(content.get("verdict"))
    colour = COLOURS.get(verdict, "#5b6472")
    minutes = max(0, int((content["expiresAt"] - time.time()) // 60))
    warnings = "".join(f'<li style="border-left:4px solid {COLOURS.get(w["light"], "#5b6472")};padding:4px 10px;margin:8px 0;list-style:none">{html.escape(str(w["title"]))}</li>' for w in content["warnings"])
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>MediPass for pharmacy</title></head>'
        '<body style="font:16px/1.5 system-ui,sans-serif;max-width:520px;margin:0 auto;padding:24px 16px">'
        f'<h1 style="margin:0">MediPass</h1><p style="color:#5b6472">Warnings only. No other patient data is shared.</p>'
        f'<div style="display:flex;align-items:center;gap:14px"><div style="width:56px;height:56px;border-radius:50%;background:{colour}"></div>'
        f'<div><b style="text-transform:capitalize;font-size:22px">{html.escape(verdict)}</b><br>{html.escape(str(content.get("medicine")))}</div></div>'
        f'<ul style="padding:0">{warnings or "<li style=list-style:none>No warnings in the checked data.</li>"}</ul>'
        f'<p style="color:#5b6472;font-size:13px">Valid for about {minutes} more minutes. Please ask the patient and use your own judgement. MediPass does not replace advice from a pharmacist.</p>'
        "</body></html>"
    )


def register(app: FastAPI) -> None:
    def page(path: str, file: str) -> None:
        @app.get(path, response_class=HTMLResponse, name=f"page{path}")
        async def serve() -> HTMLResponse:
            return HTMLResponse((WEB / file).read_text(encoding="utf-8"))

    for path, file in PAGES.items():
        page(path, file)

    @app.get("/assets/medipass.css")
    async def stylesheet() -> Response:
        return Response((WEB / "shared.css").read_text(encoding="utf-8"), media_type="text/css", headers={"cache-control": "public, max-age=300"})

    @app.get("/r/{code}", response_class=HTMLResponse)
    async def pharmacy(code: str) -> HTMLResponse:
        content = release.read(code)
        if content is None:
            return HTMLResponse("<!doctype html><meta charset=utf-8><body style='font:16px system-ui;padding:24px'><h1>MediPass</h1><p>This code is invalid or has expired.</p>", status_code=404)
        return HTMLResponse(_pharmacy_page(content))
