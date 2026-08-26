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
    # "explore" (default): multi-turn conversation, discovery_agent_node asks clarifying
    # questions across turns. "quick": single-turn — DISCOVERY_QUICK_ADDENDUM
    # (nodes/discovery.py) tells the model to propose a complete plan and set
    # ready_to_run=true in this one turn rather than converse, and route_to_recon
    # (graph/builder.py) skips the recon subgraph for a run that came from this mode —
    # both exist to keep "quick" living up to its name rather than just changing the
    # first prompt. Threaded into QAState.discovery_mode at approval time (api.py).
    mode: Literal["explore", "quick"]
    messages: Annotated[list[AnyMessage], add_messages]
    site_context: SiteMap
    extra_dives_used: int
    candidate_plan: TestPlan | None
    run_token: str  # generated once on the first turn, reused for the whole conversation
                    # (see core/run_planning.py) so generated test data stays consistent
                    # across revisions instead of changing every turn.
    turn_count: int
    status: Literal["in_progress", "approved", "cancelled"]
    # A URL discovery_agent_node requested a closer look at (via explore_more) but has
    # not yet crawled — set at the end of the turn that requested it, consumed at the
    # START of the NEXT turn, rather than crawled inline in the same turn. Crawling
    # inline used to cost a second, largely redundant LLM call every time the model
    # asked for a dive (nodes/discovery.py's module docstring has the full rationale).
    # None when no dive is pending.
    pending_dive: str | None
