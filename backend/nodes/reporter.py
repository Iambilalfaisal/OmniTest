"""Reporter node: the reduce step — aggregates every worker's TestResult into
summary pass/fail metrics (overall and per test-case category) once all parallel
branches have rejoined the graph."""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from ..core import llm_metrics
from ..core.state import QAState


async def reporter_node(state: QAState, config: RunnableConfig) -> dict:
    results_by_id = {r.test_id: r for r in state["test_results"]}
    by_category: dict[str, dict[str, int]] = {}
    passed = failed = blocked = 0

    for test_case in state["test_cases"]:
        result = results_by_id.get(test_case.test_id)
        if result is None:
            continue  # e.g. filtered out at plan-review time
        bucket = by_category.setdefault(
            test_case.category, {"total": 0, "passed": 0, "failed": 0, "blocked": 0}
        )
        bucket["total"] += 1
        if result.status == "Pass":
            passed += 1
            bucket["passed"] += 1
        elif result.status == "Fail":
            failed += 1
            bucket["failed"] += 1
        # 'Blocked' (nodes/worker/nodes.py's Verdict) — a named external wall or an
        # exhausted extended budget, kept out of `failed` since it isn't evidence the
        # site itself misbehaved.
        elif result.status == "Blocked":
            blocked += 1
            bucket["blocked"] += 1

    # Folded in here (not read separately by api.py) so it rides along with the rest of
    # `summary` through the existing JSONB history column and SSE `done` payload with no
    # schema/wire-format change. NOTE: this snapshot is taken before memory_node (the
    # next node in graph/builder.py) makes its own extraction call, so that one call
    # isn't reflected here — acceptable per llm_metrics.py's own documented scope.
    thread_id = config["configurable"]["thread_id"]
    return {
        "summary": {
            "total": len(results_by_id),
            "passed": passed,
            "failed": failed,
            "blocked": blocked,
            "by_category": by_category,
            "llm": llm_metrics.snapshot(thread_id),
        }
    }
