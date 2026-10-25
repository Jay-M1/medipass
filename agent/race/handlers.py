"""Your race logic. The verifier sends one task at a time; each function answers one task type.

Return the output (text, data, toolResults) or raise a TaskError. adapter.py does the rest: checking
the verifier token, echoing the nonce and wrapping your answer in the ASRP/1 envelope.

Each race layer tests a capability, and building it is the challenge. The race page shows each
layer's instructions and what must work. Use any AI model you like.

MediPass answers these tasks as a careful assistant. Gemini writes the answers and plans the tool
calls, but the tools and the source decide the facts, so nothing is made up.
"""

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from agent.ai.llm import MissingLlmKeyError, ask_llm, fallback_model
from agent.ai.resilient import ask_resilient

from .memory import memory
from .protocol import Task, TaskError, TaskOutput, TaskType
from .tools import ToolCallError, call_tool, search_source

logger = logging.getLogger("agent")

Handler = Callable[[Task, dict[str, Any]], Awaitable[TaskOutput]]
_NOT_BUILT: set[Handler] = set()

DEFAULT_TIMEOUT_MS = 5_000
DEFAULT_MAX_TOOL_CALLS = 12
TRANSIENT_STATUSES = {429, 502, 503}
TOOL_RETRIES = 3
SYSTEM = "You are MediPass, a careful assistant. Answer exactly what is asked. Never invent facts, tool results or sources."


def not_yet(task_type: TaskType) -> Handler:
    """Marks a task type you haven't built yet. The verifier gets a clear error instead of a wrong answer."""

    async def handler(_task: Task, _request: dict[str, Any]) -> TaskOutput:
        raise TaskError("internal", f'This agent doesn\'t handle "{task_type}" tasks yet (HANDLERS["{task_type}"] in agent/race/handlers.py).')

    _NOT_BUILT.add(handler)
    return handler


def is_built(task_type: TaskType) -> bool:
    return HANDLERS[task_type] not in _NOT_BUILT


async def _ask(prompt: str) -> str:
    try:
        return await ask_resilient(ask_llm, f"{SYSTEM}\n\n{prompt}")
    except MissingLlmKeyError as err:
        raise TaskError("internal", str(err)) from err
    except Exception as err:
        logger.error("Model request failed (%s)", type(err).__name__)
        raise TaskError("upstream_unavailable", "The AI model did not answer.") from err


def _parse_json(text: str) -> Any:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(cleaned)


async def _ask_json(prompt: str, valid: Callable[[Any], str | None] = lambda _v: None) -> Any:
    """Asks for JSON and tries once more with the problem named if it is unreadable or fails `valid`."""
    problem = ""
    for _ in range(2):
        text = await _ask(f"{prompt}\n\nReply with JSON only, no prose and no code fences.{problem}")
        try:
            value = _parse_json(text)
        except ValueError:
            problem = "\nYour last reply was not valid JSON."
            continue
        issue = valid(value)
        if issue is None:
            return value
        problem = f"\nYour last reply was rejected, {issue}"
    raise TaskError("internal", "The AI model did not return usable JSON.")


def _schema_issue(value: Any, schema: Any, path: str = "data") -> str | None:
    """A small structural check (type, required, properties, items, enum). Gemini does the real matching."""
    if not isinstance(schema, dict):
        return None
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}
    expected = schema.get("type")
    for t in [expected] if isinstance(expected, str) else (expected or []):
        if t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            break
        if t == "integer" and isinstance(value, int) and not isinstance(value, bool):
            break
        if t in types and isinstance(value, types[t]):
            break
    else:
        if expected:
            return f"{path} must be of type {expected}."
    if "enum" in schema and value not in schema["enum"]:
        return f"{path} must be one of {schema['enum']}."
    if isinstance(value, dict):
        for key in schema.get("required") or []:
            if key not in value:
                return f'{path} is missing "{key}".'
        for key, sub in (schema.get("properties") or {}).items():
            if key in value and (issue := _schema_issue(value[key], sub, f"{path}.{key}")):
                return issue
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            if issue := _schema_issue(item, schema["items"], f"{path}[{i}]"):
                return issue
    return None


def _limits(task: Task) -> dict[str, Any]:
    limits = task.get("limits")
    return limits if isinstance(limits, dict) else {}


def _timeout_ms(task: Task) -> int:
    return int(_limits(task).get("timeoutMs") or DEFAULT_TIMEOUT_MS)


async def _call_tool_reliably(tool: dict[str, Any], task: Task, tool_input: dict[str, Any]) -> Any:
    """Retries transient failures with a short backoff and maps what is left to a protocol error."""
    name = str(tool.get("name", "tool"))
    last: ToolCallError | None = None
    for attempt in range(TOOL_RETRIES):
        try:
            return await call_tool(tool, task, tool_input, _timeout_ms(task))
        except ToolCallError as err:
            last = err
            if err.http_status is None:
                # No answer within the time limit (or no network): retrying would break the limit.
                raise TaskError("upstream_timeout", str(err)) from err
            if err.http_status not in TRANSIENT_STATUSES:
                raise TaskError("upstream_unavailable", str(err)) from err
            await asyncio.sleep(0.4 * 2**attempt)
    raise TaskError("upstream_unavailable", f"{name} kept failing after {TOOL_RETRIES} tries ({last}).")


def _tools(task: Task) -> list[dict[str, Any]]:
    tools = task.get("tools")
    if not isinstance(tools, list) or not tools or not all(isinstance(t, dict) and t.get("name") and t.get("url") for t in tools):
        raise TaskError("invalid_input", "This task needs a non-empty list of tools, each with a name and a url.")
    return tools


def _describe(tools: list[dict[str, Any]]) -> str:
    return json.dumps([{k: t.get(k) for k in ("name", "description", "inputSchema")} for t in tools])


def _goal(task: Task) -> str:
    value = task.get("input")
    if value in (None, "", {}, []):
        raise TaskError("invalid_input", "This task needs an input.")
    return value if isinstance(value, str) else json.dumps(value)


async def respond(task: Task, _request: dict[str, Any]) -> TaskOutput:
    return {"text": (await _ask(_goal(task))).strip()}


async def extract(task: Task, _request: dict[str, Any]) -> TaskOutput:
    text, schema = _goal(task), task.get("schema")
    if not isinstance(schema, dict) or not schema:
        raise TaskError("invalid_input", "An extract task needs a JSON Schema in task.schema.")
    prompt = f"Extract data from the text below so it matches this JSON Schema exactly.\nUse only what the text says. Do not add keys the schema does not allow.\n\nSchema\n{json.dumps(schema)}\n\nText\n{text}"
    data = await _ask_json(prompt, lambda v: _schema_issue(v, schema))
    return {"data": data, "text": json.dumps(data)}


async def _gather(task: Task, goal: str, tools: list[dict[str, Any]], max_calls: int) -> list[dict[str, Any]]:
    """Calls tools one at a time. The model sees every result so far, so a later call can use an earlier answer."""
    by_name = {str(t["name"]): t for t in tools}
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    def valid_step(step: Any) -> str | None:
        if isinstance(step, dict) and step.get("action") == "done":
            return None
        if isinstance(step, dict) and step.get("action") == "call" and step.get("tool") in by_name and isinstance(step.get("input", {}), dict):
            return None
        return 'it must be {"action": "call", "tool": <a listed tool name>, "input": {...}} or {"action": "done"}.'

    for _ in range(max_calls + 1):
        prompt = (
            "You answer a request by calling tools. Decide the next step.\n"
            "Call every tool the request needs. If the results are in different currencies or units, call the tool that converts them, once for each one you need.\n"
            "Use values from earlier results as inputs. Never invent a value. Do not repeat a call you already made. Reply with done when you have everything you need.\n\n"
            f"Request\n{goal}\n\nTools\n{_describe(tools)}\n\nResults so far\n{json.dumps(results)}"
        )
        step = await _ask_json(prompt, valid_step)
        if step["action"] == "done" or len(results) >= max_calls:
            break
        key = json.dumps([step["tool"], step.get("input") or {}], sort_keys=True)
        if key in seen:
            break
        seen.add(key)
        results.append({"tool": step["tool"], "result": await _call_tool_reliably(by_name[step["tool"]], task, step.get("input") or {})})
    return results


async def use_tools(task: Task, _request: dict[str, Any]) -> TaskOutput:
    goal, tools, schema = _goal(task), _tools(task), task.get("schema")
    max_calls = int(_limits(task).get("maxToolCalls") or DEFAULT_MAX_TOOL_CALLS)
    results = await _gather(task, goal, tools, max_calls)

    def valid_answer(value: Any) -> str | None:
        if not isinstance(value, dict) or not isinstance(value.get("text"), str) or not isinstance(value.get("data"), dict):
            return 'it must be {"text": <string>, "data": {...}}.'
        return _schema_issue(value["data"], schema, "data") if isinstance(schema, dict) else None

    prompt = (
        "Answer the request using only these tool results. Do the arithmetic carefully and state values exactly as the tools returned them.\n"
        'Reply as {"text": <a short answer that shows the working>, "data": {...}}.\n'
        "If the request or the schema names output fields (for example totalGbp), put them in data with exactly those names and plain numbers, with no currency symbol and no thousands separator. Round money to 2 decimals unless the request says otherwise. If no fields are named, data is an empty object.\n"
        + (f"data must match this JSON Schema {json.dumps(schema)}\n" if isinstance(schema, dict) else "")
        + f"\nRequest\n{goal}\n\nTool results\n{json.dumps(results)}"
    )
    answer = await _ask_json(prompt, valid_answer)
    return {"text": answer["text"].strip(), "data": answer["data"], "toolResults": results}


async def answer_from_source(task: Task, _request: dict[str, Any]) -> TaskOutput:
    question = _goal(task)
    try:
        found = await search_source(task, question, _timeout_ms(task))
    except ToolCallError as err:
        raise TaskError("upstream_timeout" if err.http_status is None else "upstream_unavailable", str(err)) from err
    prompt = (
        "Answer the question using ONLY the source documents below. Cite the id of every document you used.\n"
        'If the documents do not contain the answer, use null. Reply as {"answer": <string or null>, "sourceIds": [<ids>]}.\n\n'
        f"Question\n{question}\n\nSource documents\n{json.dumps(found)}"
    )

    def valid(v: Any) -> str | None:
        if not isinstance(v, dict) or "answer" not in v or not isinstance(v.get("sourceIds"), list):
            return 'it must be {"answer": ..., "sourceIds": [...]}.'
        return None

    data = await _ask_json(prompt, valid)
    if data["answer"] is None:
        data["sourceIds"] = []
    return {"text": data["answer"] or "The source does not contain an answer.", "data": data}


async def converse(task: Task, _request: dict[str, Any]) -> TaskOutput:
    message, session = _goal(task), task.get("sessionId")
    turns = await memory.history(session)
    transcript = "\n".join(f"{t['role']}: {t['text']}" for t in turns)
    answer = (await _ask(f"This is one conversation. Use only what this conversation said earlier, never anything from elsewhere.\n\nEarlier turns\n{transcript or '(none)'}\n\nuser: {message}\nassistant:")).strip()
    await memory.append(session, {"role": "user", "text": message}, {"role": "assistant", "text": answer})
    return {"text": answer}


# "Pay 0.01 devnet USDC to <address> with the memo <memo>"
PAYMENT_TASK = re.compile(r"pay\s+(?P<amount>\d*\.?\d+)\s+devnet\s+usdc\s+to\s+(?P<to>[1-9A-HJ-NP-Za-km-z]{32,44})\s+with\s+the\s+memo\s*:?\s*(?P<memo>.+)", re.I | re.S)


def _parse_payment(goal: str) -> tuple[float, str, str] | None:
    match = PAYMENT_TASK.search(goal)
    if not match:
        return None
    memo = match["memo"].strip()
    quoted = re.match(r"""^(["'`])(.*?)\1""", memo, re.S)
    memo = quoted[2] if quoted else memo.rstrip(".").strip()
    return float(match["amount"]), match["to"], memo


async def _pay(amount: float, to: str, memo: str) -> TaskOutput:
    from agent.paidapi.buyer import get_spending_tracker
    from agent.payments.config import PaymentsConfigError
    from agent.payments.transfer import TransferError, load_agent_keypair, pay_usdc

    try:
        signature = await pay_usdc(load_agent_keypair(), to=to, amount_usdc=amount, memo=memo, guard=get_spending_tracker())
    except PaymentsConfigError as err:
        raise TaskError("internal", str(err)) from err
    except TransferError as err:
        raise TaskError("limit_exceeded" if "spending limits" in str(err) else "upstream_unavailable", str(err)) from err
    return {"data": {"signature": signature}, "text": f"Paid {amount} devnet USDC. Signature {signature}"}


async def run_workflow(task: Task, _request: dict[str, Any]) -> TaskOutput:
    goal = _goal(task)
    if payment := _parse_payment(goal):
        return await _pay(*payment)

    tools, schema, limits = _tools(task), task.get("schema"), _limits(task)
    by_name = {str(t["name"]): t for t in tools}
    max_calls = int(limits.get("maxToolCalls") or DEFAULT_MAX_TOOL_CALLS)
    budget = limits.get("maxSpendUsd")
    results: list[dict[str, Any]] = []

    def valid_step(step: Any) -> str | None:
        if not isinstance(step, dict):
            return "it must be a JSON object."
        if step.get("action") == "call" and step.get("tool") in by_name and isinstance(step.get("input", {}), dict):
            return None
        if step.get("action") == "final" and isinstance(step.get("data"), dict):
            return _schema_issue(step["data"], schema) if isinstance(schema, dict) else None
        return 'it must be {"action": "call", "tool": <listed name>, "input": {...}} or {"action": "final", "data": {...}}.'

    for _ in range(max_calls + 1):
        prompt = (
            "Work toward the goal one step at a time. Call the tools in the order their descriptions imply, and never repeat a call you already made.\n"
            f"Never spend more than {budget} USD in total, and do not place an order whose quoted cost would go over it. "
            'When the goal is done, finish with {"action": "final", "data": {...}}'
            + (f" where data matches this schema {json.dumps(schema)}" if isinstance(schema, dict) else "")
            + f".\n\nGoal\n{goal}\n\nTools\n{_describe(tools)}\n\nCalls so far\n{json.dumps(results)}"
        )
        step = await _ask_json(prompt, valid_step)
        if step["action"] == "final":
            return {"data": step["data"], "text": json.dumps(step["data"]), "toolResults": results}
        if len(results) >= max_calls:
            break
        results.append({"tool": step["tool"], "result": await _call_tool_reliably(by_name[step["tool"]], task, step.get("input") or {})})
    raise TaskError("limit_exceeded", f"The goal was not reached within {max_calls} tool calls.")


HANDLERS: dict[TaskType, Handler] = {
    # First Intelligence: answer task["input"] (a question) with your AI model, in "text".
    "respond": respond,
    # Structured Outputs: turn task["input"] into JSON matching task["schema"] exactly, in "data".
    "extract": extract,
    # Tool Calling and Reliability: plan which of task["tools"] to call, use their results, report them in "toolResults".
    "use_tools": use_tools,
    # External Knowledge: answer from task["source"] and cite the document ids, or answer null.
    "answer_from_source": answer_from_source,
    # Memory and Context: remember earlier turns for task["sessionId"], and only for that session.
    "converse": converse,
    # Autonomous Workflow and On-chain payment: plan tool calls within the limits, or send the devnet USDC.
    "run_workflow": run_workflow,
}
