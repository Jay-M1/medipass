"""Offline tests for reading a medicine package from a photo. Run: python -m unittest discover -s tests -v"""

import asyncio
import base64
import json
import unittest
from typing import Any
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from agent.ai.claude import ask_claude
from agent.ai.llm import MissingLlmKeyError
from agent.medipass.scan import MAX_IMAGE_BYTES, ScanError, clean, identify
from agent.server import app

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def reading(**fields: Any) -> str:
    return json.dumps({"is_medicine_package": True, "brand": "Nurofen", "active_ingredient": "ibuprofen", "strength": "400 mg", "confidence": "high", **fields})


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class CleanTest(unittest.TestCase):
    def test_matches_the_ingredient_against_known_medicines(self) -> None:
        out = clean(json.loads(reading()))
        self.assertEqual((out["known"], out["confidence"], out["brand"]), ("ibuprofen", "high", "Nurofen"))

    def test_a_brand_alone_is_enough_when_the_rules_know_it(self) -> None:
        self.assertEqual(clean(json.loads(reading(active_ingredient=None)))["known"], "ibuprofen")

    def test_an_unknown_medicine_stays_unknown(self) -> None:
        out = clean(json.loads(reading(brand="Wonderpill", active_ingredient="wonderium")))
        self.assertIsNone(out["known"])

    def test_junk_in_the_reply_is_cleaned(self) -> None:
        out = clean({"is_medicine_package": "yes", "brand": 7, "active_ingredient": "x" * 500, "confidence": "certain"})
        self.assertEqual((out["isMedicinePackage"], out["brand"], len(out["ingredient"]), out["confidence"]), (False, None, 80, "low"))

    def test_a_reply_that_is_not_an_object_is_rejected(self) -> None:
        with self.assertRaises(ScanError):
            clean(["nope"])


class IdentifyTest(unittest.TestCase):
    def test_reads_a_package(self) -> None:
        async def ask(_prompt: str, _image: bytes, _type: str, _model: Any = None) -> str:
            return "```json\n" + reading() + "\n```"

        out = run(identify(PNG, "image/png", ask))
        self.assertEqual(out["known"], "ibuprofen")

    def test_rejects_other_file_types_and_empty_or_huge_images(self) -> None:
        async def ask(*_a: Any, **_k: Any) -> str:
            return reading()

        for data, kind in ((PNG, "application/pdf"), (b"", "image/png"), (b"x" * (MAX_IMAGE_BYTES + 1), "image/png")):
            with self.assertRaises(ScanError):
                run(identify(data, kind, ask))

    def test_an_unreadable_reply_is_a_scan_error(self) -> None:
        async def ask(*_a: Any, **_k: Any) -> str:
            return "I cannot tell"

        with self.assertRaises(ScanError):
            run(identify(PNG, "image/png", ask))


class RouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def post(self, body: Any) -> Any:
        return self.client.post("/api/medipass/scan", json=body)

    def test_scan_route_with_a_data_url(self) -> None:
        async def fake(_prompt: str, _image: bytes, _type: str, _model: Any = None) -> str:
            return reading()

        with patch("agent.routes.medipass.ask_llm_image", fake):
            res = self.post({"image": "data:image/png;base64," + base64.b64encode(PNG).decode()})
        self.assertEqual((res.status_code, res.json()["known"]), (200, "ibuprofen"))

    def test_bad_requests(self) -> None:
        self.assertEqual(self.post({}).status_code, 400)
        self.assertEqual(self.post({"image": "not base64!!"}).status_code, 400)
        self.assertEqual(self.post({"image": base64.b64encode(PNG).decode(), "mediaType": "text/html"}).status_code, 400)
        self.assertEqual(self.post({"image": "A" * (MAX_IMAGE_BYTES * 2)}).status_code, 413)

    def test_a_missing_model_key_is_a_503_not_a_crash(self) -> None:
        async def nokey(*_a: Any, **_k: Any) -> str:
            raise MissingLlmKeyError("ANTHROPIC_API_KEY is not set.")

        with patch("agent.routes.medipass.ask_llm_image", nokey):
            res = self.post({"image": base64.b64encode(PNG).decode()})
        self.assertEqual(res.status_code, 503)

    def test_a_model_failure_never_leaks_details(self) -> None:
        async def boom(*_a: Any, **_k: Any) -> str:
            raise RuntimeError("secret detail sk-ant-123")

        with patch("agent.routes.medipass.ask_llm_image", boom):
            res = self.post({"image": base64.b64encode(PNG).decode()})
        self.assertEqual(res.status_code, 502)
        self.assertNotIn("sk-ant", res.text)

    def test_brand_assets_are_served(self) -> None:
        for name, kind in (("logo.png", "image/png"), ("favicon.png", "image/png"), ("medipass.css", "text/css")):
            res = self.client.get(f"/assets/{name}")
            self.assertEqual((res.status_code, res.headers["content-type"].split(";")[0]), (200, kind), name)
        self.assertEqual(self.client.get("/assets/secrets.env").status_code, 404)
        self.assertEqual(self.client.get("/assets/..%2F.env").status_code, 404)

    def test_pages_show_the_logo_and_the_scanner(self) -> None:
        page = self.client.get("/app").text
        self.assertIn('src="/assets/logo.png"', page)
        self.assertIn('id="photoBtn"', page)


class ClaudeImageTest(unittest.TestCase):
    def test_the_image_goes_first_as_a_content_block(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k"}, clear=True):
            run(ask_claude("What is this?", transport=httpx.MockTransport(handler), image=("image/png", "QUJD")))
        content = seen["body"]["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}})
        self.assertEqual(content[1], {"type": "text", "text": "What is this?"})


if __name__ == "__main__":
    unittest.main()
