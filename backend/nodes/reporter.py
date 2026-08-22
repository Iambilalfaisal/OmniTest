"""Reporter node: the reduce step — aggregates every worker's TestResult into
summary pass/fail metrics (overall, per test-case category, and per Feature/Flow — see
by_feature below) once all parallel branches have rejoined the graph."""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from ..core import llm_metrics
from ..core.state import QAState


def _empty_counts() -> dict[str, int]:
    return {"total": 0, "passed": 0, "failed": 0, "blocked": 0}


def _bump(bucket: dict[str, int], status: str) -> None:
    bucket["total"] += 1
    if status == "Pass":
        bucket["passed"] += 1
    elif status == "Fail":
        bucket["failed"] += 1
    # 'Blocked' (nodes/worker/nodes.py's Verdict) — a named external wall or an
    # exhausted extended budget, kept out of `failed` since it isn't evidence the
    # site itself misbehaved.
    elif status == "Blocked":
        bucket["blocked"] += 1


async def reporter_node(state: QAState, config: RunnableConfig) -> dict:
    results_by_id = {r.test_id: r for r in state["test_results"]}
    flow_names = {fr.flow_id: fr.flow_name for fr in state.get("flow_reports", [])}
    by_category: dict[str, dict] = {}
    # Pre-seeded from state["features"] (not built lazily from test_cases alone) so a
    # Feature with zero graded scenarios so far — still running, or one recon left
    # empty — still appears in the rollup rather than being indistinguishable from a
    # Feature that was never planned at all.
    by_feature: dict[str, dict] = {
        f.feature_id: {"name": f.name, "description": f.description, **_empty_counts(), "by_flow": {}}
        for f in state.get("features", [])
    }
    passed = failed = blocked = 0

    for test_case in state["test_cases"]:
        result = results_by_id.get(test_case.test_id)
        if result is None:
            continue  # e.g. filtered out at plan-review time

        category_bucket = by_category.setdefault(test_case.category, _empty_counts())
        _bump(category_bucket, result.status)

        # feature_id/flow_id are optional (core/models.py) for a checkpoint written
        # before this hierarchy existed — grouped under a lazily-created "ungrouped"
        # bucket instead of dropped from the rollup. Every case produced by a planning
        # path in this codebase from this change forward always has a real feature_id
        # (core/run_planning.py's ensure_features guarantees it), so this is a
        # backward-compatibility path, not a normal one.
        fid = test_case.feature_id or "ungrouped"
        feature_bucket = by_feature.setdefault(
            fid, {"name": fid, "description": "", **_empty_counts(), "by_flow": {}}
        )
        _bump(feature_bucket, result.status)

        if test_case.flow_id:
            # test_ids, not a collapsed rationale/name string: several TestCases can
            # share one flow_id (one per ScenarioProposal recon generated for that
            # Flow), each with its OWN discovery_rationale/steps/origin already present
            # on that TestCase in the SAME payload (api.py's _model_dump) — the frontend
            # cross-references this list against the full test_cases/test_results
            # arrays it already has, rather than this rollup duplicating (and going
            # stale relative to) data that's already a single source of truth elsewhere.
            flow_bucket = feature_bucket["by_flow"].setdefault(
                test_case.flow_id,
                {"flow_name": flow_names.get(test_case.flow_id, test_case.flow_id), "test_ids": [], **_empty_counts()},
            )
            flow_bucket["test_ids"].append(test_case.test_id)
            _bump(flow_bucket, result.status)

        if result.status == "Pass":
            passed += 1
        elif result.status == "Fail":
            failed += 1
        elif result.status == "Blocked":
            blocked += 1

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
            "by_feature": by_feature,
            "llm": llm_metrics.snapshot(thread_id),
        }
    }
