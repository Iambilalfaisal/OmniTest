"""DAG assembly: planner_node -> Send-based fan-out over worker_node -> reporter_node."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..core.state import QAState
from ..nodes.planner import planner_node
from ..nodes.reporter import reporter_node
from ..nodes.worker import worker_node


def route_to_workers(state: QAState) -> list[Send]:
    return [
        Send("worker_node", {"target_url": state["target_url"], "test_case": test})
        for test in state["test_cases"]
    ]


def build_graph():
    graph = StateGraph(QAState)

    graph.add_node("planner_node", planner_node)
    graph.add_node("worker_node", worker_node)
    graph.add_node("reporter_node", reporter_node)

    graph.add_edge(START, "planner_node")
    graph.add_conditional_edges("planner_node", route_to_workers, ["worker_node"])
    graph.add_edge("worker_node", "reporter_node")
    graph.add_edge("reporter_node", END)

    return graph.compile()
