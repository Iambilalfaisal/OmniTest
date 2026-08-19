"""DiscoveryState: the chat-first discovery conversation's LangGraph state. Kept
separate from QAState (core/state.py) — discovery is a different lifecycle (may be
abandoned/revisited, holds no live browser session) from an execution run, and folding
it into QAState would permanently carry chat-only fields through every checkpoint of
the execution phase.
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from .models import SiteMap, TestPlan


class DiscoveryState(TypedDict):
    target_url: str
    starting_idea: str
    messages: Annotated[list[AnyMessage], add_messages]
    site_context: SiteMap
    extra_dives_used: int
    candidate_plan: TestPlan | None
    run_token: str  # generated once on the first turn, reused for the whole conversation
                    # (see core/run_planning.py) so generated test data stays consistent
                    # across revisions instead of changing every turn.
    turn_count: int
    status: Literal["in_progress", "approved", "cancelled"]
