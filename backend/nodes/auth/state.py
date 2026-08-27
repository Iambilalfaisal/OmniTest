"""AuthState: state for the auth subgraph (nodes/auth/nodes.py) — establishes ONE
shared logged-in session before the worker fan-out. Field names deliberately match
QAState's (core/state.py) wherever the value is genuinely shared with the parent —
target_url, run_token, discovery_context, test_cases, auth_storage_state — so
LangGraph's subgraph-as-node mechanism inherits/merges them automatically by name, the
same pattern WorkerState (core/state.py) already relies on for the worker subgraph.
messages/pending_tool_calls/turn_count have no QAState counterpart and are simply
dropped on merge back to the parent, again mirroring WorkerState.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from ...core.models import TestCase


class AuthState(TypedDict):
    target_url: str
    run_token: str
    discovery_context: str  # credentials/preferences from a prior discovery chat; "" if none.
    test_cases: list[TestCase]  # read-only here — only to check whether any requires_auth
    messages: Annotated[list[AnyMessage], add_messages]
    pending_tool_calls: list[dict]
    turn_count: int
    auth_storage_state: str | None
    # Set alongside auth_storage_state on a genuine login success — the URL auth_save_node
    # found the browser sitting on right after login (core/state.py's QAState copy has the
    # full rationale). None whenever auth_storage_state is None.
    authenticated_landing_url: str | None
    # Mirrors WorkerState's fields of the same name (core/state.py) — same mechanism,
    # same rationale: a wall-clock deadline checked at the top of every auth_agent_node
    # turn, and an abort_reason set on a session-open timeout or exceeded deadline so
    # auth_save_node can degrade to unauthenticated cleanly instead of an uncaught
    # exception crashing the whole run before any worker branch even starts.
    deadline_at: float | None
    abort_reason: str | None
