"""ReconState: state for the recon subgraph (nodes/recon/nodes.py) — one instance per
Feature, spawned via Send with only `feature` (plus the shared target_url/run_token/
discovery_context) populated. Mirrors nodes/auth/state.py's AuthState exactly: field
names match QAState's wherever the value is genuinely shared with the parent
(target_url, run_token, discovery_context, flow_reports) so LangGraph's subgraph-as-node
mechanism inherits/merges them automatically by name, the same pattern WorkerState/
AuthState already rely on.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from ...core.models import Feature, FlowReport, TestCase


class ReconState(TypedDict):
    target_url: str
    run_token: str
    discovery_context: str  # credentials/preferences from a prior discovery chat; "" if none.
    feature: Feature  # which Feature this instance explores — set once in the Send payload.
    # The baseline planner's/discovery chat's test cases already covering this SAME
    # feature_id (graph/builder.py's route_to_recon filters QAState.test_cases down to
    # just this feature before the Send) — set once in the Send payload, read only by
    # recon_plan_node. Without this, recon's synthesis call has no way to know a scenario
    # it's about to propose already exists under the baseline plan, since nothing else in
    # this subgraph's context (the exploration transcript) mentions the baseline at all.
    existing_test_cases: list[TestCase]
    # Path to the shared login's storage-state file (set once by auth_setup_node,
    # nodes/auth/nodes.py), or None if no baseline test case in this run required auth —
    # set once in the Send payload. Without this, recon explores every feature LOGGED
    # OUT even when a shared login already exists, so a feature that requires being
    # logged in to reach (e.g. an account/agent-management area) hits a login wall
    # immediately and recon reports the whole feature blocked, discovering nothing.
    auth_storage_state: str | None
    messages: Annotated[list[AnyMessage], add_messages]
    pending_tool_calls: list[dict]
    turn_count: int
    # Mirrors WorkerState/AuthState's fields of the same name — same mechanism, same
    # rationale: a wall-clock deadline checked at the top of every recon_agent_node
    # turn, and an abort_reason set on a session-open timeout or exceeded deadline so
    # recon_plan_node can produce a partial/empty result cleanly instead of an uncaught
    # exception crashing the whole run.
    deadline_at: float | None
    abort_reason: str | None
    # Written ONCE by recon_plan_node, at the end — potentially more than one FlowReport
    # per Feature (a Feature can have several distinct Flows: Email+Password, Google
    # OAuth, ...). No reducer needed here: unlike QAState's copy, nothing inside this
    # subgraph ever rewrites it. Field name matches QAState.flow_reports so the parent's
    # operator.add reducer appends this branch's list when this subgraph-as-node returns.
    flow_reports: list[FlowReport]
