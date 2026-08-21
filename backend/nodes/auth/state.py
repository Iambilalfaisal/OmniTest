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
