"""Small planning helpers shared by both planning paths: the one-shot `planner_node`
(nodes/planner.py) and the chat-first discovery flow (nodes/discovery.py). Kept
LangGraph/state-free so either caller can use them directly.
"""
from __future__ import annotations

import re
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


def drop_duplicate_scenarios(test_cases: list[TestCase]) -> list[TestCase]:
    """Drops a TestCase whose (goal, steps) exactly matches one already kept, keeping the
    first occurrence.

    Backstop, not the primary fix: TEST_CASE_AUTHORING_GUIDELINES' self-check (core/models.py)
    already tells the model never to emit two cases with the same (goal, steps). That
    instruction alone was confirmed, live, to be unreliable on gemini-3.5-flash-lite — one
    run had a single case duplicated; a later run had its ENTIRE candidate plan emitted
    twice in one structured-output call (5 cases -> 10, exact goal/step matches, verified
    directly against the persisted checkpoint). Same class of gap as `expected_result` and
    `feature_id` elsewhere in this module: a cheap model's structured output not reliably
    honoring an instruction, needing a deterministic check in addition to the prompt fix,
    not instead of it.

    Also closes half of the plan doc's own D3 risk: recon's baseline-plus-top-up design
    means a recon-discovered scenario can restate a case the planner's baseline already
    covers. Applied at every point test_cases are finalized (api.py, planner.py, and the
    `_merge_test_cases` reducer in core/state.py), so it catches both within-turn LLM
    duplication and baseline/recon overlap the same way.

    Exact match only (normalized by trim + casefold) — not fuzzy — so two cases that are
    merely similar but prove different things are never mistaken for the same case.
    """
    seen: set[tuple[str, tuple[str, ...]]] = set()
    result = []
    for tc in test_cases:
        key = (tc.goal.strip().casefold(), tuple(s.strip().casefold() for s in tc.steps))
        if key in seen:
            continue
        seen.add(key)
        result.append(tc)
    return result


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


# Same class of gap as ensure_expected_result/ensure_features above: TEST_CASE_AUTHORING_
# GUIDELINES (core/models.py) already tells the model, with a worked example, that a case
# needing a logged-in start but NOT itself testing auth must set requires_auth=true and
# skip writing its own login steps — nodes/auth/nodes.py's auth_setup_node exists
# specifically to make that free. CONFIRMED live on gemini-3.5-flash-lite: a "create a
# project" case still came back requires_auth=false with its own inline
# navigate/click-sign-in/type-email/type-password/click-log-in prefix before the steps
# that actually tested project creation — burning 4-5 of its 10-turn budget on a login
# auth_setup_node would have handled for free, and running out of turns before ever
# clicking Save. A prompt instruction alone was not enough; this is the deterministic
# correction.
#
# Deliberately NOT applied to recon-discovered scenarios (graph/builder.py's
# _test_case_from_scenario hardcodes requires_auth=False there on purpose — "recon writes
# each scenario's OWN entry steps inline," a separate, intentional design choice for
# scenarios discovered independently of the baseline plan) — only planner/discovery-
# authored baseline cases go through this.
_STEP_NAVIGATE_RE = re.compile(r"^(navigate|go)\s+to\b", re.IGNORECASE)
_STEP_EMAIL_TYPE_RE = re.compile(
    r"^type\s+'.*?'\s+into the\s+'?[^']*\b(email|username|user\s*id)\b[^']*'?\s*field", re.IGNORECASE
)
_STEP_PASSWORD_TYPE_RE = re.compile(
    r"^type\s+'.*?'\s+into the\s+'?[^']*\bpassword\b[^']*'?\s*field", re.IGNORECASE
)
_STEP_LOGIN_CLICK_RE = re.compile(
    r"^click the\s+'?[^']*\b(log\s*in|sign\s*in|login|signin)\b[^']*'?", re.IGNORECASE
)
# Used only to prune a stale "starts from login page..." precondition line off a rewritten
# case below — NOT as a goal-based skip. A goal-text check ("does this goal mention
# login?") was tried and dropped: a case that's genuinely about something else, but whose
# goal is phrased "...after logging in..." (exactly the case this function exists to fix,
# confirmed against the live example), would match it and wrongly skip the rewrite. The
# real, sufficient protection against misfiring on a case that IS genuinely about auth is
# the "steps left over after the login prefix" check below: a pure login/signup test's
# steps consist entirely of that prefix, so prefix_len == len(steps) and nothing is
# rewritten no matter what the goal says.
_AUTH_MENTION_RE = re.compile(
    r"\b(log(ging)?\s*(in|out)|sign(ing)?\s*(in|up|out)|password\s*reset|forgot\s*password)\b", re.IGNORECASE
)


def ensure_requires_auth(test_cases: list[TestCase]) -> list[TestCase]:
    """Rewrites a planner/discovery-authored TestCase that (a) is not itself testing
    auth and (b) opens with its own inline login sequence (navigate + type email/username
    + type password + click a sign-in/log-in control) followed by MORE steps afterward —
    into requires_auth=true with that redundant login prefix stripped, so it starts from
    auth_setup_node's shared session instead of re-doing (and burning turn budget on) a
    login every other requires_auth case gets for free.

    Best-effort pattern match against TEST_CASE_AUTHORING_GUIDELINES' own mandated step
    phrasing ("Type 'X' into the 'Y' field", "Click the 'Z' button") — not a general NLP
    parse. A step written in some other shape simply isn't recognized and that case passes
    through unchanged (fails safe: worst case is the original wasteful-but-correct inline
    login survives, never a case incorrectly rewritten out of steps it still needs).
    """
    result = []
    for tc in test_cases:
        if tc.requires_auth:
            result.append(tc)
            continue

        prefix_len = 0
        has_email = has_password = has_submit = False
        for step in tc.steps:
            if _STEP_EMAIL_TYPE_RE.match(step):
                has_email = True
            elif _STEP_PASSWORD_TYPE_RE.match(step):
                has_password = True
            elif _STEP_LOGIN_CLICK_RE.match(step):
                has_submit = True
            elif _STEP_NAVIGATE_RE.match(step):
                pass  # navigate is compatible with the login prefix but not itself a signal
            else:
                break
            prefix_len += 1

        if has_email and has_password and has_submit and prefix_len < len(tc.steps):
            kept_preconditions = [p for p in tc.preconditions if not _AUTH_MENTION_RE.search(p)]
            result.append(
                tc.model_copy(
                    update={
                        "requires_auth": True,
                        "steps": tc.steps[prefix_len:],
                        "preconditions": [
                            *kept_preconditions,
                            "Runs already authenticated as the shared test account "
                            "(auto-corrected: the planner had written its own redundant login steps here).",
                        ],
                    }
                )
            )
        else:
            result.append(tc)
    return result


# A quoted value typed into a password field is the one place a TestCase's own `steps`
# necessarily contains a literal secret — the worker can only type a value written out
# verbatim (TEST_CASE_AUTHORING_GUIDELINES' own rule), so an explicit auth-test case
# (requires_auth=false, its own inline login) has no way to avoid it. That's fine for
# EXECUTION (the worker genuinely needs the real value), but every API response that
# returns TestCase data straight to the frontend was echoing it back in plain text —
# confirmed live: the shared test account's real password visible in the plan/report UI.
# Reuses the same quoted-value shape _STEP_PASSWORD_TYPE_RE already matches, but captures
# the value itself (group 2) rather than just testing for the step's presence.
_PASSWORD_STEP_VALUE_RE = re.compile(
    r"^type\s+'(.*?)'\s+into the\s+'?[^']*\bpassword\b[^']*'?\s*field", re.IGNORECASE
)


def redact_test_case_dict(tc: dict) -> dict:
    """Returns a copy of a (already `model_dump()`-ed) TestCase dict with any password
    value found in its own `steps` scrubbed from `steps` and `preconditions` — for API
    responses ONLY. Must never be applied to the TestCase object actually handed to
    agent_node (nodes/worker/nodes.py): that copy needs the real value to type it. Not
    applied to email/username values — those aren't secrets, and redacting them would
    make an auth-test case's own displayed steps unreadable for no security benefit.

    A no-op (returns `tc` unchanged) for the overwhelming majority of test cases, which
    have no password step at all.
    """
    steps = tc.get("steps")
    if not isinstance(steps, list):
        return tc
    secrets = {m.group(1) for step in steps if isinstance(step, str) for m in [_PASSWORD_STEP_VALUE_RE.match(step)] if m and m.group(1)}
    if not secrets:
        return tc

    def _scrub(text):
        if not isinstance(text, str):
            return text
        for secret in secrets:
            text = text.replace(secret, "[redacted]")
        return text

    redacted = dict(tc)
    redacted["steps"] = [_scrub(s) for s in steps]
    if isinstance(tc.get("preconditions"), list):
        redacted["preconditions"] = [_scrub(p) for p in tc["preconditions"]]
    return redacted
