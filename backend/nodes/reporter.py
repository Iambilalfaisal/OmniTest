"""Reporter node: the reduce step — aggregates every worker's TestResult into
summary pass/fail metrics (overall and per test-case category) once all parallel
branches have rejoined the graph."""
from __future__ import annotations

from ..core.state import QAState


async def reporter_node(state: QAState) -> dict:
    results_by_id = {r.test_id: r for r in state["test_results"]}
    by_category: dict[str, dict[str, int]] = {}
    passed = failed = 0

    for test_case in state["test_cases"]:
        result = results_by_id.get(test_case.test_id)
        if result is None:
            continue  # e.g. filtered out at plan-review time
        bucket = by_category.setdefault(test_case.category, {"total": 0, "passed": 0, "failed": 0})
        bucket["total"] += 1
        if result.status == "Pass":
            passed += 1
            bucket["passed"] += 1
        elif result.status == "Fail":
            failed += 1
            bucket["failed"] += 1

    return {
        "summary": {
            "total": len(results_by_id),
            "passed": passed,
            "failed": failed,
            "by_category": by_category,
        }
    }
