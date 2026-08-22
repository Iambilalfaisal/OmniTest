"""QAState: the shared LangGraph state. `test_results` uses `operator.add` so every
parallel `worker_node` branch appends its own TestResult instead of overwriting
the others.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from .models import Feature, FlowReport, TestCase, TestResult
from .run_planning import ensure_unique_test_ids


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


def _merge_test_cases(current: list[TestCase], new: list[TestCase]) -> list[TestCase]:
    """Reducer for QAState.test_cases — required once recon (nodes/recon/) can append
    newly-discovered scenarios to this SAME field from parallel Send-spawned branches,
    alongside auth_setup_node's baseline write, in the same superstep (see
    graph/builder.py's route_to_recon/route_to_workers). `test_cases` had NO reducer at
    all before this, which was safe only because planner_node was its one and only
    writer, always alone, before any fan-out existed — N branches writing it in the same
    superstep now would otherwise hit the identical `InvalidUpdateError` `_keep_latest`
    above exists to avoid.

    Concatenates, then re-applies `ensure_unique_test_ids` (core/run_planning.py) over
    the COMBINED list — plain `operator.add` only concatenates: a parallel recon branch
    cannot see what its SIBLING branches are generating in the same superstep, so id
    uniqueness (session_key/evidence-dir collisions ride on it — see that function's own
    docstring) is unachievable at production time and must be enforced here, at merge
    time, instead. Deterministic and idempotent by construction: `ensure_unique_test_ids`
    walks the list in a fixed order and only suffixes an id that's ALREADY taken, so a
    LangGraph node replay re-running this reducer over the same current+new never
    renumbers an id it already assigned on an earlier pass.
    """
    return ensure_unique_test_ids([*current, *new])


class QAState(TypedDict):
    target_url: Annotated[str, _keep_latest]
    instruction: str
    discovery_context: str  # freeform context from a prior chat/HITL discovery phase
                             # (credentials, user preferences/clarifications); "" if none.
    run_token: str  # set by planner_node — a run-unique, non-LLM value injected into
                     # PLANNER_PROMPT for unique generated test data (see core/run_planning.py).
    test_cases: Annotated[list[TestCase], _merge_test_cases]
    # Features the planner/discovery chat identified from the user's instruction (e.g.
    # "Sign-Up", "Create Agent") — the top level of the Feature -> Flow -> Scenario
    # hierarchy (core/models.py's Feature). Written once, before any fan-out (same
    # lifecycle as test_cases' own planner_node write used to be alone), so no reducer
    # is needed yet — nothing downstream of the planner currently adds a NEW Feature
    # mid-run, only new Scenarios (TestCases) under an already-declared one.
    features: list[Feature]
    # Grounded evidence recon gathered per Flow (core/models.py's FlowReport) — one
    # entry per Flow across however many Features had one recon'd. `operator.add`, not
    # `_merge_test_cases`'s dedup-by-id merge: FlowReport has no id collision risk
    # analogous to TestCase.test_id (nothing downstream keys a browser session or
    # evidence directory off flow_id the way session_key does off test_id), and unlike
    # test_cases, nothing PRE-fan-out ever writes this — every write is a parallel
    # recon branch appending its own one-element contribution, the same shape
    # test_results' own operator.add reducer already handles for worker_node.
    flow_reports: Annotated[list[FlowReport], operator.add]
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
    # Wall-clock backstop (nodes/agent_loop.py's SCENARIO_DEADLINE_SECONDS) — a
    # time.monotonic() value agent_node sets on turn 1 and checks on every later turn,
    # short-circuiting to verdict_node as Blocked if exceeded. Pushed forward (not
    # accumulated) by tool_node after a genuine human-in-the-loop pause resumes, so a
    # slow human answer never itself trips the deadline. No reducer needed — same class
    # as turn_budget above: only this one worker's own sequential nodes ever write it.
    deadline_at: float | None
    # Set by agent_node/tool_node when this leaf can't make progress through no fault of
    # the test case itself — a session-open timeout, or an exceeded deadline_at above —
    # instead of letting the underlying exception propagate uncaught and crash the WHOLE
    # run (see mcp/client.py's invoke_tool docstring for why a raise here is unsafe).
    # verdict_node checks this FIRST and returns a deterministic Blocked TestResult,
    # skipping the LLM grading call and the final browser_snapshot fetch entirely, since
    # grading against a browser that may itself be the thing that's wedged is meaningless
    # and risks the exact same hang this mechanism exists to bound.
    abort_reason: str | None
