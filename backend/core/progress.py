"""Per-run, per-test-case LIVE progress registry — feeds backend/api.py's SSE
`progress` event with step-level detail so the frontend can render a test case's card
while it is still running, not only once its `verdict_node` returns. Written from
inside nodes/worker/nodes.py's agent_node/tool_node/verdict_node (and graph/builder.py's
route_to_workers, to pre-register every case as `queued` before any of them starts).

Also tracks per-Feature recon progress (register_feature/update_feature/
feature_snapshot, below) — without this, a run with RECON_ENABLED would go visibly
dark for however long recon_node/recon_join_node take (nodes/recon/nodes.py's
RECON_MAX_TURNS times however many Features), since route_to_workers' own
pre-registration only happens AFTER that barrier — see graph/builder.py's route_to_recon
and nodes/recon/nodes.py for the write sites.

Same design precedent as core/llm_metrics.py and core/run_knowledge.py: a module-level
dict keyed by run_id (== LangGraph's `thread_id`), single-process/single-event-loop
(confirmed there — see those modules' own docstrings), so no new QAState channel/
reducer is needed. Every write here is a last-write-wins upsert, never an append, which
is what makes a LangGraph node replay (a retried node, or a resume re-entering a node
whose earlier attempt already wrote here) harmless — it just overwrites the same keys
with this attempt's values instead of double-counting anything, EXCEPT `step_index`,
which is deliberately kept as a monotonic high-water mark (see `update`) since a later
turn's parsed step is sometimes lower than an earlier one's (the model re-mentions a step
it already passed) and a progress bar visibly rewinding reads as broken.

Honest limits: process-local and best-effort, matching every other module in this
family — a resume landing on a different process starts this run's progress back at
nothing; api.py's SSE loop simply shows no `worker_progress` entry for a case until that
new process's route_to_workers (or its first node write) registers one, never a stale
or wrong entry.
"""
from __future__ import annotations

import time
from typing import Any, Literal, TypedDict

Phase = Literal["queued", "running", "awaiting_input", "grading", "done", "rediscovering", "replanning"]

# Type: one recorded mutation/adaptation event for a test case.  Streamed to the
# frontend via the SSE `progress` payload so the live WorkerCard can render a
# mutation timeline while the test is still running — not a LangGraph state field
# (same process-local, best-effort design as the rest of this module).
MutationEventType = Literal["deviation", "clarification", "risky_blocked"]


class MutationEvent(TypedDict, total=False):
    type: MutationEventType    # which kind of mutation/pause
    step: int                  # 1-based planned step index where it occurred
    description: str           # what the agent encountered / asked
    user_decision: str | None  # human answer, if any (set on resolve)
    sensitive: bool            # True → mask user_decision in the UI
    timestamp: float           # time.time() when the event was added
    resolved: bool             # True once the agent has resumed past this event


_PROGRESS: dict[str, dict[str, dict[str, Any]]] = {}


def _default_entry() -> dict[str, Any]:
    return {
        "phase": "queued",
        "step_index": 0,
        "total_steps": 0,
        "current_action": None,
        "turn": 0,
        "budget": None,
        "deviations": 0,
        "asks": 0,
        "mutation_events": [],
        # Adaptive replanning tracking — mirrors WorkerState's plan_version/plan_history;
        # streamed to the frontend so WorkerCard can render a plan-evolution timeline.
        "plan_version": 0,
        "plan_history": [],
    }


def register(run_id: str, test_id: str, *, total_steps: int) -> None:
    """Pre-registers `test_id` as `queued`, before its worker branch has actually
    started — called once per Send payload from graph/builder.py's route_to_workers, so
    every test case's card can render immediately on plan approval instead of only once
    its own agent_node first runs.
    """
    entry = _default_entry()
    entry["total_steps"] = total_steps
    entry["updated_at"] = time.time()
    _PROGRESS.setdefault(run_id, {})[test_id] = entry


def update(run_id: str, test_id: str, **fields: Any) -> None:
    """Last-write-wins partial update. `step_index`, if present in `fields`, is
    clamped to never go backwards (see this module's docstring) — every other field is
    a plain overwrite, since each represents "whatever this node's latest state is",
    not something that accumulates.
    """
    bucket = _PROGRESS.setdefault(run_id, {})
    entry = bucket.setdefault(test_id, _default_entry())
    if "step_index" in fields:
        fields["step_index"] = max(entry.get("step_index", 0), fields["step_index"])
    entry.update(fields)
    entry["updated_at"] = time.time()


def bump(run_id: str, test_id: str, field: str, *, by: int = 1) -> None:
    """Increments a counter field (`deviations`, `asks`) rather than overwriting it —
    separate from `update` since these accumulate across many tool_node/agent_node
    calls for the SAME test case, unlike every other field this module tracks.
    """
    bucket = _PROGRESS.setdefault(run_id, {})
    entry = bucket.setdefault(test_id, _default_entry())
    entry[field] = entry.get(field, 0) + by
    entry["updated_at"] = time.time()


def add_mutation_event(
    run_id: str,
    test_id: str,
    *,
    type: MutationEventType,
    step: int = 0,
    description: str,
    sensitive: bool = False,
) -> None:
    """Appends a new unresolved MutationEvent to this test case's list.
    Called by nodes/worker/nodes.py at the moment a deviation, clarification
    interrupt, or risky-action block is detected — BEFORE any LangGraph
    interrupt() call, so it's visible in the SSE stream immediately.
    """
    bucket = _PROGRESS.setdefault(run_id, {})
    entry = bucket.setdefault(test_id, _default_entry())
    entry.setdefault("mutation_events", []).append({
        "type": type,
        "step": step,
        "description": description,
        "sensitive": sensitive,
        "user_decision": None,
        "timestamp": time.time(),
        "resolved": False,
    })
    entry["updated_at"] = time.time()


def resolve_last_mutation(
    run_id: str,
    test_id: str,
    *,
    user_decision: str | None = None,
) -> None:
    """Marks the most recent unresolved MutationEvent as resolved and records
    the human's decision (if any).  Called in tool_node after ask_human_and_reply
    or a risky-action resume so the frontend knows the pause ended.
    """
    bucket = _PROGRESS.get(run_id, {})
    entry = bucket.get(test_id)
    if not entry:
        return
    events: list = entry.get("mutation_events", [])
    for evt in reversed(events):
        if not evt.get("resolved", False):
            evt["resolved"] = True
            if user_decision is not None:
                evt["user_decision"] = user_decision
            break
    entry["updated_at"] = time.time()


def snapshot(run_id: str) -> dict[str, dict[str, Any]]:
    return _PROGRESS.get(run_id, {})


FeaturePhase = Literal["exploring", "done"]

_FEATURE_PROGRESS: dict[str, dict[str, dict[str, Any]]] = {}


def register_feature(run_id: str, feature_id: str, *, name: str) -> None:
    """Pre-registers `feature_id` as `exploring`, before its recon_node branch has
    actually started — called once per Send payload from graph/builder.py's
    route_to_recon, mirroring `register` above for test cases.
    """
    _FEATURE_PROGRESS.setdefault(run_id, {})[feature_id] = {
        "name": name,
        "phase": "exploring",
        "scenario_count": 0,
        "updated_at": time.time(),
    }


def update_feature(run_id: str, feature_id: str, **fields: Any) -> None:
    bucket = _FEATURE_PROGRESS.setdefault(run_id, {})
    entry = bucket.setdefault(feature_id, {"name": feature_id, "phase": "exploring", "scenario_count": 0})
    entry.update(fields)
    entry["updated_at"] = time.time()


def feature_snapshot(run_id: str) -> dict[str, dict[str, Any]]:
    return _FEATURE_PROGRESS.get(run_id, {})


def discard(run_id: str) -> None:
    _PROGRESS.pop(run_id, None)
    _FEATURE_PROGRESS.pop(run_id, None)
