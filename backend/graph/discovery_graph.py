"""Discovery graph assembly: discovery_agent_node (crawl + LLM turn, does all the real
work) -> discovery_wait_node (the ONLY interrupt() call) -> loops back to
discovery_agent_node while still in_progress, else END. Two nodes, not one — see
nodes/discovery.py's module docstring for why (LangGraph replays a node function from
the top on resume; a single node combining the crawl/LLM work with interrupt() would
redo that work on every chat reply).

Compiled with the SAME checkpointer/store the main run graph uses (graph/builder.py) —
two compiled graph objects sharing one Postgres-backed checkpointer instance, the same
pattern already used for the worker subgraph.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, interrupt

from ..core.discovery_state import DiscoveryState
from ..nodes.discovery import discovery_agent_node


async def discovery_wait_node(state: DiscoveryState) -> dict:
    """Pauses for the user's next chat action. Resume payload:
    {"action": "reply", "text": "..."} | {"action": "approve"} | {"action": "cancel"}.
    """
    plan = state.get("candidate_plan")
    messages = state.get("messages") or []
    decision = interrupt(
        {
            "type": "discovery_turn",
            "assistant_message": messages[-1].content if messages else "",
            "candidate_plan": [tc.model_dump() for tc in plan.test_cases] if plan else [],
            "turn_count": state.get("turn_count", 0),
        }
    )
    action = decision.get("action")
    if action == "approve":
        return {"status": "approved"}
    if action == "cancel":
        return {"status": "cancelled"}
    return {"messages": [HumanMessage(decision.get("text", ""))], "status": "in_progress"}


def route_after_discovery(state: DiscoveryState) -> str:
    return "discovery_agent_node" if state["status"] == "in_progress" else END


def build_discovery_graph(checkpointer, store=None):
    graph = StateGraph(DiscoveryState)

    graph.add_node("discovery_agent_node", discovery_agent_node, retry_policy=RetryPolicy(max_attempts=3))
    # No retry on discovery_wait_node — nothing precedes its interrupt() that's worth
    # retrying, same reasoning as nodes/worker/nodes.py's tool_node.
    graph.add_node("discovery_wait_node", discovery_wait_node)

    graph.add_edge(START, "discovery_agent_node")
    graph.add_edge("discovery_agent_node", "discovery_wait_node")
    graph.add_conditional_edges("discovery_wait_node", route_after_discovery, ["discovery_agent_node", END])

    return graph.compile(checkpointer=checkpointer, store=store)
