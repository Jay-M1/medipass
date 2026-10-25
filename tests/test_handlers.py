"""Offline tests for the race handlers with a fake model and fake tools. Run: python -m unittest discover -s tests -v"""

import asyncio
import json
import unittest
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

from agent.ai.llm import fallback_model
from agent.race import handlers as h
from agent.race.memory import Memory
from agent.race.protocol import TaskError
from agent.race.tools import ToolCallError

ADDRESS = "31NmqqGmupVi3PaQ9KvH71U6VnMUDBmgDEYtSLzFqX9z"


def run(coro: Any) -> Any:
    return asyncio.run(coro)


async def no_sleep(_seconds: float) -> None:
    return None


def model(*replies: str) -> Callable[[str], Any]:
    """A fake Gemini that answers with the given replies in order (the last one repeats)."""
    queue = list(replies)

    async def ask(_prompt: str) -> str:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return ask


TOOLS = [{"name": "lookup_record", "description": "Look up a record", "url": "https://x.example/t/lookup_record", "inputSchema": {}}]


class BusyModelTest(unittest.TestCase):
    class Busy(Exception):
        code = 503

    def test_retries_a_busy_model_then_succeeds(self) -> None:
        calls: list[str | None] = []

        async def ask(_prompt: str, model: str | None = None) -> str:
            calls.append(model)
            if len(calls) < 2:
                raise self.Busy()
            return "fine"

        with patch.object(h, "ask_llm", ask), patch.object(h.asyncio, "sleep", no_sleep):
            self.assertEqual(run(h._ask("hi")), "fine")
        self.assertEqual(calls, [None, None])

    def test_falls_back_to_the_lighter_model(self) -> None:
        calls: list[str | None] = []

        async def ask(_prompt: str, model: str | None = None) -> str:
            calls.append(model)
            if model is None:
                raise self.Busy()
            return "from the fallback"

        with patch.object(h, "ask_llm", ask), patch.object(h.asyncio, "sleep", no_sleep):
            self.assertEqual(run(h._ask("hi")), "from the fallback")
        self.assertEqual(calls[-1], fallback_model())

    def test_a_real_error_is_not_retried(self) -> None:
        calls = {"n": 0}

        class Denied(Exception):
            code = 403

        async def ask(_prompt: str, model: str | None = None) -> str:
            calls["n"] += 1
            raise Denied()

        with patch.object(h, "ask_llm", ask), self.assertRaises(TaskError) as ctx:
            run(h._ask("hi"))
        self.assertEqual((ctx.exception.code, calls["n"]), ("upstream_unavailable", 1))


class SchemaTest(unittest.TestCase):
    def test_structural_checks(self) -> None:
        schema = {"type": "object", "required": ["name", "age"], "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
        self.assertIsNone(h._schema_issue({"name": "A", "age": 3}, schema))
        self.assertIn("age", h._schema_issue({"name": "A"}, schema) or "")
        self.assertIn("integer", h._schema_issue({"name": "A", "age": "3"}, schema) or "")
        self.assertIn("integer", h._schema_issue({"name": "A", "age": True}, schema) or "")


class SimpleHandlersTest(unittest.TestCase):
    def test_respond_returns_text(self) -> None:
        with patch.object(h, "ask_llm", model("  Paris  ")):
            self.assertEqual(run(h.respond({"type": "respond", "input": "Capital of France?"}, {})), {"text": "Paris"})

    def test_respond_without_input_is_invalid(self) -> None:
        with self.assertRaises(TaskError) as ctx:
            run(h.respond({"type": "respond"}, {}))
        self.assertEqual(ctx.exception.code, "invalid_input")

    def test_extract_requires_a_schema(self) -> None:
        with self.assertRaises(TaskError) as ctx:
            run(h.extract({"type": "extract", "input": "x"}, {}))
        self.assertEqual(ctx.exception.code, "invalid_input")

    def test_extract_retries_once_when_the_schema_does_not_match(self) -> None:
        schema = {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}}
        replies = ['{"n": "two"}', '```json\n{"n": 2}\n```']
        with patch.object(h, "ask_llm", model(*replies)):
            out = run(h.extract({"type": "extract", "input": "two", "schema": schema}, {}))
        self.assertEqual(out["data"], {"n": 2})

    def test_unreadable_model_output_is_an_internal_error(self) -> None:
        with patch.object(h, "ask_llm", model("not json")), self.assertRaises(TaskError) as ctx:
            run(h.extract({"type": "extract", "input": "x", "schema": {"type": "object"}}, {}))
        self.assertEqual(ctx.exception.code, "internal")


class ToolsTest(unittest.TestCase):
    PLAN = json.dumps({"calls": [{"tool": "lookup_record", "input": {"id": 7}}]})

    def task(self) -> dict[str, Any]:
        return {"type": "use_tools", "input": "Who is record 7?", "tools": TOOLS, "challengeToken": "t", "limits": {"timeoutMs": 500}}

    def test_requires_tools(self) -> None:
        with self.assertRaises(TaskError) as ctx:
            run(h.use_tools({"type": "use_tools", "input": "x"}, {}))
        self.assertEqual(ctx.exception.code, "invalid_input")

    def test_uses_the_tool_and_reports_its_result(self) -> None:
        async def tool(*_a: Any, **_k: Any) -> dict[str, str]:
            return {"name": "Ada"}

        with patch.object(h, "ask_llm", model(self.PLAN, "It is Ada.")), patch.object(h, "call_tool", tool):
            out = run(h.use_tools(self.task(), {}))
        self.assertEqual(out["toolResults"], [{"tool": "lookup_record", "result": {"name": "Ada"}}])
        self.assertEqual(out["text"], "It is Ada.")

    def test_recovers_from_transient_503(self) -> None:
        calls = {"n": 0}

        async def flaky(*_a: Any, **_k: Any) -> dict[str, str]:
            calls["n"] += 1
            if calls["n"] < 3:
                raise ToolCallError("lookup_record", 503, "busy")
            return {"name": "Ada"}

        with patch.object(h, "ask_llm", model(self.PLAN, "Ada")), patch.object(h, "call_tool", flaky), patch.object(h.asyncio, "sleep", no_sleep):
            out = run(h.use_tools(self.task(), {}))
        self.assertEqual(calls["n"], 3)
        self.assertEqual(out["toolResults"][0]["result"], {"name": "Ada"})

    def test_a_tool_that_stays_down_is_upstream_unavailable(self) -> None:
        async def down(*_a: Any, **_k: Any) -> None:
            raise ToolCallError("lookup_record", 503, "busy")

        with patch.object(h, "ask_llm", model(self.PLAN)), patch.object(h, "call_tool", down), patch.object(h.asyncio, "sleep", no_sleep), self.assertRaises(TaskError) as ctx:
            run(h.use_tools(self.task(), {}))
        self.assertEqual(ctx.exception.code, "upstream_unavailable")

    def test_a_silent_tool_is_upstream_timeout_without_retrying(self) -> None:
        calls = {"n": 0}

        async def silent(*_a: Any, **_k: Any) -> None:
            calls["n"] += 1
            raise ToolCallError("lookup_record", None, "lookup_record did not answer within 500 ms")

        with patch.object(h, "ask_llm", model(self.PLAN)), patch.object(h, "call_tool", silent), self.assertRaises(TaskError) as ctx:
            run(h.use_tools(self.task(), {}))
        self.assertEqual((ctx.exception.code, calls["n"]), ("upstream_timeout", 1))

    def test_a_plan_with_an_unknown_tool_is_rejected(self) -> None:
        with patch.object(h, "ask_llm", model('{"calls": [{"tool": "delete_everything", "input": {}}]}')), self.assertRaises(TaskError):
            run(h.use_tools(self.task(), {}))


class SourceTest(unittest.TestCase):
    def task(self) -> dict[str, Any]:
        return {"type": "answer_from_source", "input": "What is X?", "source": {"url": "https://x.example/s"}, "challengeToken": "t"}

    def test_cites_the_documents_it_used(self) -> None:
        async def source(*_a: Any, **_k: Any) -> dict[str, Any]:
            return {"documents": [{"id": "d1", "text": "X is 4."}]}

        with patch.object(h, "ask_llm", model('{"answer": "X is 4.", "sourceIds": ["d1"]}')), patch.object(h, "search_source", source):
            out = run(h.answer_from_source(self.task(), {}))
        self.assertEqual(out["data"], {"answer": "X is 4.", "sourceIds": ["d1"]})

    def test_null_answer_has_no_citations(self) -> None:
        async def source(*_a: Any, **_k: Any) -> dict[str, Any]:
            return {"documents": []}

        with patch.object(h, "ask_llm", model('{"answer": null, "sourceIds": ["d9"]}')), patch.object(h, "search_source", source):
            out = run(h.answer_from_source(self.task(), {}))
        self.assertEqual(out["data"], {"answer": None, "sourceIds": []})


class ConverseTest(unittest.TestCase):
    def test_sessions_are_kept_apart(self) -> None:
        prompts: list[str] = []

        async def ask(prompt: str) -> str:
            prompts.append(prompt)
            return "ok"

        fresh = Memory(env={})
        with patch.object(h, "ask_llm", ask), patch.object(h, "memory", fresh):
            run(h.converse({"type": "converse", "input": "My dog is called Rex", "sessionId": "a"}, {}))
            run(h.converse({"type": "converse", "input": "What is my dog called?", "sessionId": "a"}, {}))
            run(h.converse({"type": "converse", "input": "What is my dog called?", "sessionId": "b"}, {}))
        self.assertIn("Rex", prompts[1])
        self.assertNotIn("Rex", prompts[2])


class WorkflowTest(unittest.TestCase):
    def test_parses_the_on_chain_payment_task(self) -> None:
        self.assertEqual(h._parse_payment(f"Pay 0.01 devnet USDC to {ADDRESS} with the memo agent-studio:ab12cd."), (0.01, ADDRESS, "agent-studio:ab12cd"))
        self.assertEqual(h._parse_payment(f'Pay 0.5 devnet USDC to {ADDRESS} with the memo "hello world".'), (0.5, ADDRESS, "hello world"))
        self.assertIsNone(h._parse_payment("Order three widgets"))

    def test_runs_tools_in_a_loop_until_final(self) -> None:
        steps = [
            '{"action": "call", "tool": "lookup_record", "input": {}}',
            '{"action": "final", "data": {"orders": 1}}',
        ]

        async def tool(*_a: Any, **_k: Any) -> dict[str, int]:
            return {"price": 3}

        task = {"type": "run_workflow", "input": "Buy a widget", "tools": TOOLS, "challengeToken": "t", "limits": {"maxSpendUsd": 10, "maxToolCalls": 3}, "schema": {"type": "object", "required": ["orders"]}}
        with patch.object(h, "ask_llm", model(*steps)), patch.object(h, "call_tool", tool):
            out = run(h.run_workflow(task, {}))
        self.assertEqual(out["data"], {"orders": 1})
        self.assertEqual(len(out["toolResults"]), 1)

    def test_stops_at_the_tool_call_limit(self) -> None:
        async def tool(*_a: Any, **_k: Any) -> dict[str, int]:
            return {}

        task = {"type": "run_workflow", "input": "Loop", "tools": TOOLS, "challengeToken": "t", "limits": {"maxToolCalls": 2}}
        with patch.object(h, "ask_llm", model('{"action": "call", "tool": "lookup_record", "input": {}}')), patch.object(h, "call_tool", tool), self.assertRaises(TaskError) as ctx:
            run(h.run_workflow(task, {}))
        self.assertEqual(ctx.exception.code, "limit_exceeded")


if __name__ == "__main__":
    unittest.main()
