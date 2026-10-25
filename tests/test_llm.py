"""Offline tests for choosing and calling the model. Run: python -m unittest discover -s tests -v"""

import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

from agent.ai import llm
from agent.ai.claude import ClaudeError, MissingClaudeKeyError, ask_claude
from agent.ai.resilient import BUSY_CODES


def run(coro):
    return asyncio.run(coro)


class ProviderTest(unittest.TestCase):
    def test_claude_when_its_key_is_set(self) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k"}, clear=True):
            self.assertEqual((llm.provider(), llm.fallback_model()), ("claude", "claude-haiku-5-5"))

    def test_gemini_by_default(self) -> None:
        with patch.dict("os.environ", {"GEMINI_API_KEY": "k"}, clear=True):
            self.assertEqual((llm.provider(), llm.fallback_model()), ("gemini", "gemini-flash-lite-latest"))

    def test_a_forced_choice_wins(self) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k", "LLM_PROVIDER": "gemini"}, clear=True):
            self.assertEqual(llm.provider(), "gemini")
        with patch.dict("os.environ", {"LLM_PROVIDER": "claude"}, clear=True):
            self.assertEqual(llm.provider(), "claude")

    def test_a_missing_key_is_one_error_for_both_providers(self) -> None:
        for provider in ("claude", "gemini"):
            with patch.dict("os.environ", {"LLM_PROVIDER": provider}, clear=True), self.assertRaises(llm.MissingLlmKeyError):
                run(llm.ask_llm("hi"))

    def test_ask_llm_routes_to_claude(self) -> None:
        async def fake(prompt: str, model: str | None = None) -> str:
            return f"claude says {prompt} {model}"

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k"}, clear=True), patch.object(llm, "ask_claude", fake):
            self.assertEqual(run(llm.ask_llm("hi", "m")), "claude says hi m")


class ClaudeClientTest(unittest.TestCase):
    def call(self, handler, **env):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "secret-key", **env}, clear=True):
            return run(ask_claude("Hello", transport=httpx.MockTransport(handler)))

    def test_sends_the_request_and_joins_the_text(self) -> None:
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["headers"], seen["body"] = request.headers, json.loads(request.content)
            return httpx.Response(200, json={"content": [{"type": "text", "text": "Hi "}, {"type": "tool_use"}, {"type": "text", "text": "there"}]})

        self.assertEqual(self.call(handler), "Hi there")
        self.assertEqual(seen["headers"]["x-api-key"], "secret-key")
        self.assertEqual(seen["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(seen["body"]["model"], "claude-sonnet-5-5")
        self.assertEqual(seen["body"]["messages"], [{"role": "user", "content": "Hello"}])

    def test_the_model_can_be_chosen_in_the_environment(self) -> None:
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["model"] = json.loads(request.content)["model"]
            return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

        self.call(handler, ANTHROPIC_MODEL="claude-opus-5-5")
        self.assertEqual(seen["model"], "claude-opus-5-5")

    def test_an_overloaded_model_is_a_retryable_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(529, json={"error": {"message": "Overloaded"}})

        with self.assertRaises(ClaudeError) as ctx:
            self.call(handler)
        self.assertIn(ctx.exception.code, BUSY_CODES)

    def test_a_bad_key_is_not_retryable_and_never_echoed(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": {"message": "invalid x-api-key"}})

        with self.assertRaises(ClaudeError) as ctx:
            self.call(handler)
        self.assertNotIn(ctx.exception.code, BUSY_CODES)
        self.assertNotIn("secret-key", str(ctx.exception))

    def test_missing_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(MissingClaudeKeyError):
            run(ask_claude("hi"))


if __name__ == "__main__":
    unittest.main()
