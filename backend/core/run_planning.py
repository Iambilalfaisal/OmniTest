"""Small planning helpers shared by both planning paths: the one-shot `planner_node`
(nodes/planner.py) and the chat-first discovery flow (nodes/discovery.py). Kept
LangGraph/state-free so either caller can use them directly.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from .models import TestCase


def generate_run_token() -> str:
    """A run-unique, non-LLM value injected into the planning prompt so a temperature=0
    LLM call still produces different "unique" generated test data (e.g. signup emails)
    across repeated runs of the same site — without something varying in the input, a
    temperature=0 call would deterministically regenerate the identical value every time.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def ensure_unique_test_ids(test_cases: list[TestCase]) -> list[TestCase]:
    """De-dupe `test_id` after an LLM call. `route_to_workers` (graph/builder.py) and
    `session_key` (nodes/worker/session.py) both key a test case's isolated browser
    session off `test_id` — a collision would silently share one session (and one
    TestResult slot) between two test cases instead of erroring.
    """
    seen: dict[str, int] = {}
    result = []
    for tc in test_cases:
        base = tc.test_id or "tc"
        seen[base] = seen.get(base, 0) + 1
        new_id = base if seen[base] == 1 else f"{base}-{seen[base]}"
        result.append(tc if new_id == tc.test_id else tc.model_copy(update={"test_id": new_id}))
    return result
