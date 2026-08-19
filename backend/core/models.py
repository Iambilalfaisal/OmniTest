"""Pydantic schemas for the planner's/discovery chat's structured output and each
worker's result."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TestCategory = Literal["happy_path", "edge_case", "negative", "error_handling"]
TestPriority = Literal["high", "medium", "low"]

# Shared between PLANNER_PROMPT (nodes/planner.py) and DISCOVERY_SYSTEM_PROMPT
# (nodes/discovery.py) so the one-shot planner and the chat-first discovery flow can't
# silently drift apart on what "a good test case" means — both embed this constant
# verbatim into their own prompt text via an f-string, then `.format(run_token=...)` the
# combined string, which is why the `{run_token}` placeholder below is left unescaped.
TEST_CASE_AUTHORING_GUIDELINES = """Every test case is fully self-contained: it will run in its own fresh, isolated \
browser session in PARALLEL with every other test case, with no shared login state, no \
shared data, and no guaranteed ordering relative to any other test case. If a test case's \
goal requires being logged in, having an account, or any other prerequisite state, write \
that setup as the test case's OWN first steps (e.g. "Click 'Sign up'", "Type '<email>' \
into the Email field", ...) — never assume another test case already did it, and never \
reference another test case.

If existing test credentials are already known (see the context given to you) or reuse of \
an existing account is explicitly preferred over signing up, use those directly instead of \
generating a new one — only generate a new account when no usable existing credentials are \
available, or the goal specifically requires a fresh account (e.g. testing the sign-up \
flow itself).

Whenever a test case needs to create its own account (sign-up, registration, or any flow \
that stores a unique email/username), use this run token in the generated value so it is \
unique for this run: {run_token}
For example, an email could be "qa+{run_token}@example.com" or a username could be \
"qa_{run_token}". If MORE THAN ONE test case needs its own account, they MUST NOT reuse the \
same generated email/username as each other — further distinguish each one, e.g. by \
appending that test case's own test_id or an index ("qa+{run_token}-2@example.com"), so two \
test cases signing up in parallel never collide with a duplicate-registration error.

The worker executing your steps only has accessibility-tree-based Playwright tools, not CSS \
selectors — phrase every step in terms of visible roles/labels/text (e.g. "Click the 'Sign \
in' button", "Type 'demo@site.com' into the Email field"), never selectors or XPath.

For every test case, set:
- "category" to exactly one of: happy_path, edge_case, negative, error_handling.
  - happy_path: the normal, expected-to-succeed flow.
  - edge_case: boundary/unusual-but-valid input or flow (empty optional fields, very long
    input, special characters, minimum/maximum values, unusual but legitimate navigation
    order).
  - negative: input or actions that should be REJECTED (invalid email format, wrong
    password, missing required field, mismatched confirm-password) — this test case passes
    if the site correctly rejects it with a visible error, not if it succeeds.
  - error_handling: recovering from or observing the site's behavior after an error state
    (e.g. submitting twice, going back after an error, verifying an error message is
    specific and helpful).
- "priority" to "high", "medium", or "low" — how important this test case is given limited
  time.
- "preconditions" to a short list of plain-language notes describing what state this test
  case establishes for itself before its "real" assertion begins (e.g. ["Signs up a new,
  unique test account"]) — this is DISPLAY-ONLY metadata for a human reviewer; you must
  STILL write the actual setup actions inline as the first entries of "steps" too, since
  the worker only ever executes "steps" literally, in order.

Produce a set small enough that every test case is high-value — do not enumerate every
trivial permutation — but make sure the happy path AND the most important edge, negative,
and error-handling cases relevant to the goal are each represented."""


class TestCase(BaseModel):
    test_id: str
    goal: str
    category: TestCategory = Field(
        description="happy_path = normal expected-to-succeed flow. edge_case = boundary/unusual-but-valid "
        "input or ordering. negative = input/action that should be REJECTED by the site. "
        "error_handling = observing/recovering from an error state."
    )
    priority: TestPriority = Field(default="medium", description="Relative importance for a human skimming the plan.")
    preconditions: list[str] = Field(
        default_factory=list,
        description="DISPLAY-ONLY plain-language summary of prerequisite state this test case sets up for "
        "itself (e.g. 'Signs up a new unique test account'). The actual setup actions MUST ALSO appear "
        "inline as the first entries of `steps` — the worker only ever executes `steps`.",
    )
    steps: list[str] = Field(
        description="Ordered, plain-language instructions for the worker to execute — including any "
        "prerequisite/setup actions (sign-up, login, etc.) inline as the first steps, since each test case "
        "runs in full isolation with no shared session or ordering with any other case."
    )


class TestPlan(BaseModel):
    """Structured-output contract the planner LLM call is constrained to."""

    test_cases: list[TestCase]


class TestResult(BaseModel):
    test_id: str
    status: str = Field(description="Strictly 'Pass' or 'Fail'")
    screenshot_path: str
    trace_path: str | None
    video_path: str | None
    reason: str


class SiteMemory(BaseModel):
    """A single distilled fact about a target site, persisted to the long-term store."""

    kind: Literal["failure_pattern", "site_quirk"]
    summary: str = Field(description="One or two sentence distilled memory, phrased for reuse in a future planning prompt")
    related_goal: str | None = Field(default=None, description="The TestCase.goal this relates to, if applicable (failure_pattern only)")


class PageSummary(BaseModel):
    """One page discovered by a shallow site crawl (nodes/planner_explore.py) — a
    truncated, depth-limited digest, NOT a full accessibility tree (that would blow up
    the planning prompt across several pages)."""

    url: str
    title: str
    depth: int
    snapshot_digest: str


class SiteMap(BaseModel):
    """Output of a bounded, read-only, same-origin crawl. Cached per-domain (see
    core/memory.py's get_cached_site_map/save_site_map) so repeat runs against the same
    site skip re-crawling.
    """

    pages: list[PageSummary]
    truncated: bool = Field(default=False, description="True if max_pages was hit before the link frontier was exhausted.")
    crawled_at: str | None = Field(default=None, description="ISO timestamp, set when persisted to the long-term store.")


class ExploreRequest(BaseModel):
    """A discovery-chat LLM turn's request to crawl a narrower area on-demand, beyond
    the automatic upfront shallow crawl."""

    url: str
    reason: str


class DiscoveryTurn(BaseModel):
    """Structured-output contract for one discovery-chat LLM turn (nodes/discovery.py)."""

    assistant_message: str = Field(description="Chat-facing text: plan summary, follow-up question, or acknowledgement.")
    candidate_plan: TestPlan = Field(description="The full current best-guess test plan, revised each turn (not a diff).")
    explore_more: ExploreRequest | None = Field(
        default=None, description="Set only if answering well requires a page not yet covered by the site map."
    )
    ready_to_run: bool = Field(
        default=False,
        description="UI hint only (e.g. bolds the Approve button) — NEVER used for control flow. Approval is "
        "always an explicit, separate user action, never inferred from this field.",
    )
