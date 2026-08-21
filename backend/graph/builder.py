"""DAG assembly: planner_node -> plan_review_node (human-in-the-loop) ->
Send-based fan-out over worker_node -> reporter_node -> memory_node."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from ..core.llm import LLM_RETRY_POLICY
from ..core.models import TestCase
from ..core.state import QAState
from ..nodes.auth import build_auth_subgraph
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


def route_after_plan_review(state: QAState) -> str:
    return "auth_setup_node" if state["plan_approved"] else "reporter_node"


def route_to_workers(state: QAState):
    """auth_setup_node's outgoing edge — by construction (see route_entry and
    route_after_plan_review) this is only ever reached with plan_approved already True,
    so unlike the pre-Stage-3 version of this function there's no unapproved-plan branch
    to check here anymore. Stage 3: auth_storage_state (set once by auth_setup_node) is
    only threaded into a given worker's payload when that test case actually declared
    requires_auth — a case that didn't ask for a shared login shouldn't have one silently
    injected into its browser.
    """
    auth_state = state.get("auth_storage_state")
    return [
        Send(
            "worker_node",
            {
                "target_url": state["target_url"],
                "test_case": test,
                "auth_storage_state": auth_state if test.requires_auth else None,
            },
        )
        for test in state["test_cases"]
    ]


def route_entry(state: QAState):
    """Chat-approved runs (backend/api.py's POST /discover/{id}/message approve handler)
    arrive with test_cases/plan_approved already set by the discovery conversation —
    skip planner_node's LLM call and plan_review_node's interrupt (already approved
    conversationally) straight to auth_setup_node. Runs started the old way (POST /runs,
    or any future programmatic caller) arrive with plan_approved=False and take the
    original planner_node -> plan_review_node path, unchanged.
    """
    if state.get("plan_approved") and state.get("test_cases"):
        return "auth_setup_node"
    return "planner_node"


def build_graph(checkpointer, store=None):
    graph = StateGraph(QAState)

    graph.add_node("planner_node", planner_node, retry_policy=LLM_RETRY_POLICY)
    graph.add_node("plan_review_node", plan_review_node)
    # Subgraph-as-node, like worker_node below — no retry_policy at this level, since
    # each of its own nodes (nodes/auth/nodes.py) already carries its own.
    graph.add_node("auth_setup_node", build_auth_subgraph())
    graph.add_node("worker_node", build_worker_subgraph())
    graph.add_node("reporter_node", reporter_node)
    graph.add_node("memory_node", memory_node)

    graph.add_conditional_edges(START, route_entry, ["planner_node", "auth_setup_node"])
    graph.add_edge("planner_node", "plan_review_node")
    graph.add_conditional_edges("plan_review_node", route_after_plan_review, ["auth_setup_node", "reporter_node"])
    graph.add_conditional_edges("auth_setup_node", route_to_workers, ["worker_node"])
    graph.add_edge("worker_node", "reporter_node")
    graph.add_edge("reporter_node", "memory_node")
    graph.add_edge("memory_node", END)

    return graph.compile(checkpointer=checkpointer, store=store)
