"""DAG assembly: planner_node -> plan_review_node (human-in-the-loop) ->
Send-based fan-out over worker_node -> reporter_node -> memory_node."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send, interrupt

from ..core.models import TestCase
from ..core.state import QAState
from ..nodes.memory import memory_node
from ..nodes.planner import planner_node
from ..nodes.reporter import reporter_node
from ..nodes.worker import build_worker_subgraph


async def plan_review_node(state: QAState) -> dict:
    """Pauses for a human to approve or edit the generated test plan before any
    worker touches the browser. Portable across instances/restarts — unlike the
    risky-action interrupt in nodes/worker.py, no live external resource is
    involved here, so this checkpoint can be resumed from anywhere.
    """
    decision = interrupt(
        {"type": "plan_review", "test_cases": [tc.model_dump() for tc in state["test_cases"]]}
    )
    test_cases = (
        [TestCase(**tc) for tc in decision["test_cases"]]
        if decision.get("test_cases")
        else state["test_cases"]
    )
    return {"test_cases": test_cases, "plan_approved": bool(decision.get("approved", False))}


def route_to_workers(state: QAState):
    if not state["plan_approved"]:
        return "reporter_node"
    return [
        Send("worker_node", {"target_url": state["target_url"], "test_case": test})
        for test in state["test_cases"]
    ]


def route_entry(state: QAState):
    """Chat-approved runs (backend/api.py's POST /discover/{id}/message approve handler)
    arrive with test_cases/plan_approved already set by the discovery conversation —
    skip planner_node's LLM call and plan_review_node's interrupt (already approved
    conversationally) straight to the worker fan-out. Runs started the old way
    (POST /runs, or any future programmatic caller) arrive with plan_approved=False and
    take the original planner_node -> plan_review_node path, unchanged.
    """
    if state.get("plan_approved") and state.get("test_cases"):
        return route_to_workers(state)
    return "planner_node"


def build_graph(checkpointer, store=None):
    graph = StateGraph(QAState)

    graph.add_node("planner_node", planner_node, retry_policy=RetryPolicy(max_attempts=3))
    graph.add_node("plan_review_node", plan_review_node)
    graph.add_node("worker_node", build_worker_subgraph())
    graph.add_node("reporter_node", reporter_node)
    graph.add_node("memory_node", memory_node)

    graph.add_conditional_edges(START, route_entry, ["planner_node", "worker_node", "reporter_node"])
    graph.add_edge("planner_node", "plan_review_node")
    graph.add_conditional_edges("plan_review_node", route_to_workers, ["worker_node", "reporter_node"])
    graph.add_edge("worker_node", "reporter_node")
    graph.add_edge("reporter_node", "memory_node")
    graph.add_edge("memory_node", END)

    return graph.compile(checkpointer=checkpointer, store=store)
