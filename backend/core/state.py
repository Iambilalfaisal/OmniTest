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


def _keep_latest(_current: str, new: str) -> str:
    """Reducer for QAState.target_url only — required because `worker_node` is added as
    a compiled subgraph directly (graph/builder.py), and WorkerState.target_url shares
    its field name with QAState.target_url. Every parallel Send-spawned worker branch
    carries its own (identical) copy of target_url through to its final state, and a
    subgraph-as-node merges its ENTIRE final state back into the parent by matching
    field names — so N parallel test cases all "write" target_url in the same parent
    superstep. Without a reducer here, LangGraph's default single-value channel raises
    `InvalidUpdateError: Can receive only one value per step` as soon as 2+ workers are
    active/paused in the same step (confirmed by reproducing it directly against a real
    run with multiple test cases). All branches write the same value, so this reducer
    just needs to accept more than one write per step — which value wins is irrelevant.
    """
    return new


def _keep_latest_auth_state(_current: str | None, new: str | None) -> str | None:
    """Same rationale as `_keep_latest` above, generalized for `auth_storage_state`
    (Stage 3): `auth_setup_node` sets this ONCE, before the fan-out to `worker_node`, and
    every parallel branch then carries that same (unmodified) value through to its own
    final state. Without a reducer here, N branches "writing" it back in the same
    superstep hits the identical `InvalidUpdateError` `_keep_latest` exists to avoid —
    which value wins is irrelevant since every branch's copy is identical.
    """
    return new


class QAState(TypedDict):
    target_url: Annotated[str, _keep_latest]
    instruction: str
    discovery_context: str  # freeform context from a prior chat/HITL discovery phase
                             # (credentials, user preferences/clarifications); "" if none.
    run_token: str  # set by planner_node — a run-unique, non-LLM value injected into
                     # PLANNER_PROMPT for unique generated test data (see core/run_planning.py).
    test_cases: list[TestCase]
    test_results: Annotated[list[TestResult], operator.add]
    summary: dict
    plan_approved: bool
    # Stage 3 — set once by the "auth_setup_node" subgraph (nodes/auth/nodes.py) before
    # the worker fan-out, from a single shared login/signup; None if no test_case in this run has
    # requires_auth=True, or if establishing it failed. Holds the ABSOLUTE PATH to the
    # storage-state file auth_setup_node had browser_storage_state write to (confirmed
    # against the installed @playwright/mcp: both browser_storage_state and
    # browser_set_storage_state are file-based, taking a `filename` param, not an inline
    # blob) — agent_node passes this same path straight back as browser_set_storage_state's
    # `filename`. Still just a plain, JSON-checkpointable string.
    auth_storage_state: Annotated[str | None, _keep_latest_auth_state]


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
    # Earned-extension budget (backend/nodes/worker/nodes.py's MAX_TOOL_TURNS_CEILING /
    # TURN_BUDGET_BONUS): starts unset (agent_node seeds it to MAX_TOOL_TURNS on turn 1),
    # then grows when a deviation is handled or an ask_human answer lands, capped at the
    # ceiling. No reducer needed — like turn_count, only this one worker's own sequential
    # nodes ever write it, and it isn't a QAState field name so a subgraph-as-node merge
    # never has to reconcile it across parallel branches (see QAState._keep_latest's
    # docstring for which field names DO need one and why).
    turn_budget: int
    test_results: list[TestResult]
    sensitive_answers: list[str]  # ask_human answers marked sensitive — redacted out of
                                   # the final verdict reason before it leaves this subgraph.
    # One relative path per mutating tool call, appended by tool_node — needs
    # operator.add (unlike test_results, sensitive_answers) because, unlike those,
    # MULTIPLE separate tool_node invocations each contribute their own item to this
    # SAME list across one test case's loop, not just once at the end. See
    # nodes/worker/evidence.py's capture_mutation_clip for why each clip covers only
    # one action (plus a few seconds of padding) instead of one video for the whole
    # test case — a continuous recording captures however long the LLM took to decide
    # between actions too, which dominates a video's length and shows nothing.
    video_clips: Annotated[list[str], operator.add]
    # Stage 3 — populated in the Send() payload (graph/builder.py's route_to_workers)
    # only when test_case.requires_auth is True; None otherwise. No reducer needed here:
    # unlike QAState's copy, nothing inside this subgraph ever rewrites it.
    auth_storage_state: str | None
