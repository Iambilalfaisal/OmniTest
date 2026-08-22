"""Small planning helpers shared by both planning paths: the one-shot `planner_node`
(nodes/planner.py) and the chat-first discovery flow (nodes/discovery.py). Kept
LangGraph/state-free so either caller can use them directly.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from .models import Feature, TestCase


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

    Checks the generated candidate against every id ALREADY EMITTED, not just a per-base
    occurrence count — a plain counter lets a generated suffix collide with an id that
    was already in the input: ["login", "login", "login-2"] used to produce
    ["login", "login-2", "login-2"], recreating the exact collision this function exists
    to prevent. Incrementing the suffix until the candidate is actually free closes that.
    """
    emitted: set[str] = set()
    result = []
    for tc in test_cases:
        base = tc.test_id or "tc"
        candidate = base
        suffix = 2
        while candidate in emitted:
            candidate = f"{base}-{suffix}"
            suffix += 1
        emitted.add(candidate)
        result.append(tc if candidate == tc.test_id else tc.model_copy(update={"test_id": candidate}))
    return result


# CONFIRMED against a live run (Gemini 2.x via langchain-google-genai's
# .with_structured_output): a Pydantic `Field(...)` with no default is NOT actually
# enforced across that structured-output round trip. When the model's function-call
# arguments omit a required key, the library still returns a TestCase — as verified
# directly against the installed version, `hasattr(tc, "expected_result")` is False and
# `tc.model_fields_set` doesn't include it, meaning it was built via something like
# `model_construct()` rather than full validation. `expected_result` is exactly the kind
# of field a weaker/free-tier model skips under load (a long list of test cases, each
# needing its own oracle) — and since it's the one field verdict_node's grading and the
# worker's stop condition depend on, a silent AttributeError several nodes downstream
# (or, worse, an ungraded test case) is a much worse failure mode than backfilling it
# here, once, right where every planning path already funnels through.
MISSING_EXPECTED_RESULT_NOTE = (
    "(the planner did not specify an expected result for this case — grading falls back "
    "to whether the goal below was concretely achieved)"
)


def ensure_expected_result(test_cases: list[TestCase]) -> list[TestCase]:
    """Guarantees every TestCase downstream (worker prompt, verdict grading, memory
    extraction, the discovery chat's own re-prompt, the frontend) can read
    `expected_result` as a real, non-empty string — never a missing attribute. Must run
    on EVERY test-case-producing LLM call (planner_node, each discovery_agent_node turn),
    not just once at final approval, since discovery.py's `_format_candidate_plan` reads
    `tc.expected_result` on the very next turn.
    """
    result = []
    for tc in test_cases:
        value = getattr(tc, "expected_result", None)
        if value and value.strip():
            result.append(tc)
        else:
            result.append(tc.model_copy(update={"expected_result": MISSING_EXPECTED_RESULT_NOTE}))
    return result


DEFAULT_FEATURE_ID = "general"
DEFAULT_FEATURE_NAME = "General"


def ensure_features(features: list[Feature], test_cases: list[TestCase]) -> tuple[list[Feature], list[TestCase]]:
    """Guarantees every TestCase.feature_id names a REAL Feature in the returned list —
    the same class of structured-output gap ensure_expected_result exists for
    (Field(...) with no enforced default across this codebase's Gemini structured-output
    path — see that function's docstring for the confirmed mechanism). Two ways this can
    drift: the model omits `features` entirely despite writing test cases fine, or it
    tags a TestCase with a feature_id that doesn't match any Feature it declared.
    graph/builder.py's route_to_recon Sends one recon instance per Feature and depends on
    every referenced feature_id actually resolving — an orphaned feature_id there would
    silently strand that TestCase out of any Feature/Flow grouping in the reporter and
    frontend, not raise, so it's corrected here rather than trusted.

    Synthesizes a single catch-all Feature (DEFAULT_FEATURE_ID) only when `features` is
    empty; an individual case with an unrecognized feature_id is reassigned to whichever
    Feature is first in the (real or synthesized) list, rather than inventing a Feature
    per orphaned id — a model that got this far wrong across many cases is better served
    by one predictable fallback grouping than several ad hoc ones.
    """
    if not features:
        features = [
            Feature(feature_id=DEFAULT_FEATURE_ID, name=DEFAULT_FEATURE_NAME, description="Ungrouped test cases.")
        ]
    known_ids = {f.feature_id for f in features}

    fixed_cases = []
    for tc in test_cases:
        if tc.feature_id in known_ids:
            fixed_cases.append(tc)
        else:
            fixed_cases.append(tc.model_copy(update={"feature_id": features[0].feature_id}))
    return features, fixed_cases
