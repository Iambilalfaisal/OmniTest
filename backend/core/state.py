"""QAState: the shared LangGraph state. `test_results` uses `operator.add` so every
parallel `worker_node` branch appends its own TestResult instead of overwriting
the others.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from .models import TestCase, TestResult


class QAState(TypedDict):
    target_url: str
    instruction: str
    test_cases: list[TestCase]
    test_results: Annotated[list[TestResult], operator.add]
    summary: dict
    plan_approved: bool


class WorkerState(TypedDict):
    """Local state for the `worker_node` subgraph — one instance per TestCase, spawned
    via `Send` with only `target_url`/`test_case` populated; the rest are absent on
    entry and filled in as `agent_node`/`tool_node`/`verdict_node` run. `test_results`
    has no reducer here (only `verdict_node` ever writes it, once) — it's named to
    match `QAState.test_results` so the parent's `operator.add` reducer merges each
    branch's one-element list when this subgraph-as-node returns.
    """

    target_url: str
    test_case: TestCase
    messages: Annotated[list[AnyMessage], add_messages]
    pending_tool_calls: list[dict]
    turn_count: int
    test_results: list[TestResult]
