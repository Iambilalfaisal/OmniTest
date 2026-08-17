"""Reporter node: the reduce step — aggregates every worker's TestResult into
summary pass/fail metrics once all parallel branches have rejoined the graph."""
from __future__ import annotations

from ..core.state import QAState


async def reporter_node(state: QAState) -> dict:
    results = state["test_results"]
    passed = sum(1 for r in results if r.status == "Pass")
    failed = sum(1 for r in results if r.status == "Fail")

    return {"summary": {"total": len(results), "passed": passed, "failed": failed}}
