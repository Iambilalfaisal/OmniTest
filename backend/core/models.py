"""Pydantic schemas for the planner's/discovery chat's structured output and each
worker's result."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TestCategory = Literal[
    "happy_path", "edge_case", "negative", "error_handling", "security", "state_interaction"
]
TestPriority = Literal["high", "medium", "low"]

# Shared between PLANNER_PROMPT (nodes/planner.py) and DISCOVERY_SYSTEM_PROMPT
# (nodes/discovery.py) so the one-shot planner and the chat-first discovery flow can't
# silently drift apart on what "a good test case" means — both embed this constant
# verbatim into their own prompt text via an f-string, then `.format(run_token=...)` the
# combined string, which is why the `{run_token}` placeholder below is left unescaped.
#
# IMPORTANT when editing: because of that `.format()` pass, `{run_token}` is the ONLY
# single-brace sequence allowed anywhere in this string — any other literal brace must be
# doubled (`{{`/`}}`) or `.format()` raises KeyError. The text below deliberately avoids
# braces entirely (examples are written as `field: value` lines, not JSON), so there is
# nothing to escape.
#
# The section order is deliberate: METHOD (how to choose cases) -> GROUNDING (only test
# what's observable) -> FIELDS (what each field must contain) -> COVERAGE CHECKLIST (per
# flow type, auth first, since auth is where a vague plan hurts most) -> EXECUTION
# CONSTRAINTS (the isolation/shared-auth/run-token mechanics) -> WORKED EXAMPLES ->
# SELF-CHECK. Written as short enumerated rules with concrete examples rather than prose:
# the cheap/free-tier models this runs on follow numbered rules and few-shot examples far
# more reliably than descriptive paragraphs, and the self-check at the end lands after
# the long evidence blocks the planner prompt appends, where a weak model's instruction
# adherence is otherwise weakest.
TEST_CASE_AUTHORING_GUIDELINES = """# How to author test cases

## 1. How to choose what to test

Work like a QA engineer signing off a release, not like someone listing features. For
each flow that is in scope:
1. Identify the flow's inputs (every field and control) and its success condition (what
   the site visibly does when the flow works).
2. For each input, ask what a real user could get wrong: left empty, wrong format, too
   short, too long, wrong type, a value already taken, a mismatched pair (password vs
   confirm password), leading or trailing spaces.
3. For each of those, decide what the site SHOULD do — accept it, or reject it with a
   visible error. That decision is the test case's expected_result.
4. Write the smallest test case that proves that one decision.

Every test case proves exactly ONE thing. If you find yourself writing "and also check
that ...", that is a second test case — split it.

## 2. Ground every test case in what you can actually see

You are given the target page's accessibility tree and a shallow site map of pages linked
from it. Only write test cases for pages, fields, buttons and links that appear there, or
that a visible link clearly implies exists (a "Sign in" link means a login page exists).
Never invent a feature, field, URL, or error message that nothing in your context
mentions.

Use the EXACT visible label text from the accessibility tree in your steps — "Click the
'Log in' button", not "Click the submit button". The worker that executes your steps
finds elements by their accessibility label, so a label you guessed wrong wastes that
entire test case. If a flow starts on a page other than the target URL, make your first
step navigate to it explicitly.

## 3. What each field must contain

- test_id: short, stable, lowercase-with-hyphens, derived from the behavior under test —
  "login-wrong-password", "signup-duplicate-email", "search-no-results". Unique across
  the whole plan.
- feature_id: before writing any test case, first name the small number of high-level
  features the instruction covers (1-4 is typical — e.g. "sign-up", "login",
  "create-agent") and list them in your separate features output. Every test case's
  feature_id must be one of those exact ids — never invent a feature_id that isn't in
  your own features list, and never leave a case's feature_id unset.
- goal: one sentence in the form "<action> should <expected outcome>", e.g. "Logging in
  with a wrong password should be rejected with an error". Never a bare feature name like
  "Test login" or "Check the signup page".
- expected_result: the single observable thing that decides Pass/Fail. This is the most
  important field you write: the run is graded strictly against it, by an agent that only
  sees the final page state and cannot guess your intent. Name the concrete signal — a
  visible message, the page or URL you end up on, an element that appears or disappears.
  Quote the real text if your context shows it ("an inline error under the Password field
  saying the credentials are invalid"); otherwise describe it precisely ("the logged-in
  dashboard, with the account menu visible in the header"). For a negative case, the
  expected_result IS the rejection.
  NEVER write an unverifiable oracle — "it works", "no errors occur", "the page loads",
  or "the user is logged in successfully" with nothing observable named. A vague
  expected_result makes the whole test case worthless, because it cannot be graded.
- steps: ordered, ONE action per step, phrased as a QA script a person could follow:
    Navigate to https://example.com/login
    Click the 'Sign up' link
    Type 'qa-user@example.com' into the 'Email' field
    Click the 'Create account' button
  Every value you want typed must appear literally, in quotes, inside the step — the
  worker types exactly what you wrote and cannot invent a value. Never write a
  placeholder like "a valid email", "some long text", or "the password". Never write CSS
  selectors or XPath; the worker has accessibility-tree tools only, not selectors.
  Do NOT add a final "Verify ..." or "Check ..." step: grading against expected_result
  happens automatically after your last step, so a verification step only burns part of
  the worker's limited turn budget.
- category: exactly one of happy_path, edge_case, negative, error_handling, security,
  state_interaction.
  - happy_path: the normal, expected-to-succeed flow.
  - edge_case: boundary or unusual-but-valid input or ordering (empty optional field,
    very long input, special characters, minimum/maximum values, unusual but legitimate
    navigation order).
  - negative: input or actions that SHOULD be rejected (invalid email format, wrong
    password, missing required field, mismatched confirm-password). This test case passes
    if the site correctly rejects it with a visible error, and FAILS if the site lets it
    through.
  - error_handling: recovering from or observing behavior after an error state
    (submitting twice, going back after an error, correcting a rejected input and
    resubmitting, checking that an error message is specific rather than generic).
  - security: ONLY for a free-text field you actually observed that gets echoed back or
    stored (a name, a bio, a comment) — typing a script-injection payload (e.g.
    `<script>alert(1)</script>`) should be sanitized/escaped/rejected, never executed or
    stored verbatim. Never invent this for a field with no evidence it accepts or displays
    free text.
  - state_interaction: ONLY for a control you actually observed to be stateful in itself —
    a password-visibility toggle, an expand/collapse section, a wizard's back/forward step.
    Passes if the control's own state changes as expected; this is not for flows that end
    in a page navigation, which belong under happy_path/edge_case instead.
- priority: high, medium, or low — how much it would matter if this behavior were broken.
  Auth, payment and data-loss paths are high; cosmetic or rarely-hit paths are low.
- preconditions: a short DISPLAY-ONLY list describing the state this case starts from,
  for a human reviewing the plan (e.g. "Runs already authenticated as the shared test
  account", "Signs up a new unique test account"). The worker never reads this — it only
  executes steps — so any setup a case genuinely needs must ALSO appear inline in steps.
- visual_assertion: true ONLY if expected_result can't be judged from element text/labels
  alone — layout/position, overlap, color, or whether a chart/canvas/image actually
  rendered. Leave false (the default) for anything a screen reader could also tell apart:
  a message, a URL, an element appearing or disappearing. Sparingly — only when a real
  visual claim is the whole point of the case.

## 4. Coverage checklist by flow type

Use the sections below for whichever flows this site actually has and the instruction
actually concerns. Do not include a case for a flow you can see no evidence of; do not
skip a listed case that clearly applies just to keep the plan short.

### Sign-up / registration
- happy_path: register a brand-new unique account; ends logged in, or on an explicit
  success / "check your email" state.
- negative: email in an invalid format (e.g. 'not-an-email').
- negative: a password that violates a stated rule (too short or too weak) — use the rule
  the form, or your prior-learnings context, actually states if there is one.
- negative: password and confirm-password do not match (only if a confirm field exists).
- negative: submit with a required field left empty.
- negative: register with an email that already exists; expect an "already registered"
  style error.
- edge_case: a very long value, or leading/trailing spaces, in a name or email field.

### Log in
- happy_path: correct credentials; ends in the authenticated state.
- negative: correct email with a wrong password; rejected, still on the login page.
- negative: an email that has no account; rejected.
- negative: submit with the email or the password left empty; validation error.
- edge_case: the email typed in a different letter case than it was registered with —
  this should still log in, since email addresses are not case-sensitive.
- error_handling: after one failed login, correct the password and submit again; the form
  recovers and logs in rather than staying stuck or erroring again.

### Log out, session and access control
- happy_path: log out; ends in a visibly logged-out state with the account menu gone.
- error_handling: while logged out, navigate directly to a URL that should require login;
  expect a redirect to login or an access-denied page, NOT the protected content.

### Password reset (only if the site offers it)
- happy_path: request a reset for a known address; expect a confirmation message.
- negative: request a reset for an address with no account; expect the same neutral
  confirmation or a generic message — the site should not reveal whether that account
  exists.

### Forms and data entry (contact, checkout, profile, "create X")
- happy_path: every field valid; expect the specific success confirmation.
- negative: a required field left empty — one case for each field that matters most.
- negative: a format-constrained field given a badly formatted value (email, phone, card
  number, or a numeric field given text).
- edge_case: boundary values — the minimum, the maximum, or exactly at a stated limit.

### Search, lists, filters
- happy_path: a term that certainly matches something; expect results.
- edge_case: a term that certainly matches nothing; expect an explicit empty state ("no
  results"), not a blank page, a stuck spinner, or an error.
- edge_case: an empty search submitted, and a term with special characters.

### Navigation and content
- happy_path: the primary nav links reach the pages they name.
- error_handling: a URL that does not exist; expect a real not-found page.

### Create, edit, delete a record
- happy_path: create it; expect it to appear in the list afterward.
- happy_path: edit it; expect the change to still be visible after a reload.
- negative: create it with a required field left blank.
- Deletion: only include if the instruction asks for it — it is destructive, and will
  pause the run for human approval.

## 5. Execution constraints you must design around

Every test case runs in its own fresh, isolated browser session in PARALLEL with every
other test case, with no shared data and no guaranteed ordering. Never reference another
test case. Keep each case small: at most 6 steps, and exactly one thing proven. If a goal
needs more than that, split it — a long case burns the worker's turns re-establishing
state instead of checking anything new, and one that runs out of turns produces no usable
result at all.

Login state is the one exception to "no shared state". If a case needs to START from an
already-logged-in session, and the case is NOT itself about signing up, logging in,
logging out or password reset, set requires_auth to true and write steps assuming you are
already authenticated when step 1 runs — do NOT write your own login steps, a valid
shared session is restored into your browser first. Only do this if one shared login can
be reused unmodified: if the case mutates account-level state in a way that could disrupt
other requires_auth cases running in parallel against the SAME shared account (changing
the password or the email address), leave requires_auth false and set its own state up
independently instead.

Any test case whose goal IS to exercise sign-up, log-in, log-out or password reset —
happy path or negative — MUST leave requires_auth false and write that flow's own steps
inline ("Click the 'Sign up' link", "Type '...' into the 'Email' field", ...). It is
testing that exact mechanism, so it must control it directly and assume no other state.

If existing test credentials are already known (see the context given to you), use those
rather than inventing an account. When a case genuinely needs its own fresh account
(because it is testing sign-up itself, or requires_auth does not apply and no existing
credentials work), put this run token into the generated value so it is unique to this
run: {run_token}
For example an email could be "qa+{run_token}@example.com" or a username "qa_{run_token}".
If MORE THAN ONE case needs its own account, they MUST NOT use the same generated
email/username as each other — further distinguish each one, e.g. by appending that
case's own test_id or an index ("qa+{run_token}-2@example.com"), so two cases signing up
in parallel never collide on a duplicate-registration error.

## 6. Worked examples (the level of detail expected)

Example A — an auth flow tested directly, so requires_auth stays false:
  test_id: login-wrong-password
  goal: Logging in with a valid email and a wrong password should be rejected
  category: negative
  priority: high
  requires_auth: false
  preconditions: Uses the known test account's email; no setup needed
  expected_result: Still on the login page, with a visible error message about invalid
    credentials shown near the form; no dashboard and no account menu appears.
  steps:
    1. Navigate to https://example.com/login
    2. Type 'demo@example.com' into the 'Email' field
    3. Type 'WrongPassword123' into the 'Password' field
    4. Click the 'Log in' button

Example B — needs to be logged in but is NOT testing auth, so requires_auth is true:
  test_id: profile-update-display-name
  goal: Updating the display name from the profile page should save and show the new name
  category: happy_path
  priority: medium
  requires_auth: true
  preconditions: Runs already authenticated as the shared test account
  expected_result: A visible saved/success confirmation appears, and the 'Display name'
    field shows 'QA Renamed' after saving.
  steps:
    1. Navigate to https://example.com/settings/profile
    2. Type 'QA Renamed' into the 'Display name' field
    3. Click the 'Save changes' button

Example C — an edge case with a concrete, checkable empty state:
  test_id: search-no-results
  goal: Searching for a term with no matches should show an explicit empty state
  category: edge_case
  priority: medium
  requires_auth: false
  preconditions: None
  expected_result: A visible "no results" style message, with an empty results list — not
    a blank page, a stuck spinner, or an error.
  steps:
    1. Navigate to https://example.com
    2. Type 'zzzqqxnomatch' into the 'Search' field
    3. Press Enter

## 7. Self-check before you answer

Go through every test case you wrote and confirm all of these. Fix any that fail.
- Does expected_result name something concretely observable on the final screen?
- Does the case prove exactly one thing, in 6 steps or fewer?
- Is every case's (goal, steps) pair genuinely distinct from every other case's? If two
  entries would read as the same scenario under different wording, or the same flow got
  written up once per feature it touches, that is one case duplicated — keep only one.
- Is every element label one you actually saw in the accessibility tree or site map?
- Does every value the worker must type appear literally, in quotes, in a step?
- Is requires_auth false for every case that is itself testing signup/login/logout/reset?
- For every OTHER case (not testing auth itself) that needs a logged-in starting point:
  is requires_auth TRUE, and did you leave its own login steps OUT of `steps` entirely?
  Writing "Navigate to the login page / Click Sign in / Type email / Type password /
  Click Log in" as a case's first steps when that case is really about something else
  (creating a project, updating a setting, ...) is wrong even if requires_auth also says
  true — it burns that case's limited turn budget re-doing a login the shared session
  already handles for free, often leaving too few turns to reach the thing being tested.
- Do two cases that each create an account use DIFFERENT generated emails?
- Is the happy path covered, plus the most important negative, edge-case and
  error-handling cases for the flows in scope?

Aim for the smallest set in which every case is high-value — typically around 5 to 12
cases for a normal instruction. Do not enumerate every trivial permutation, and do not
return only one or two vague cases either."""


class TestCase(BaseModel):
    test_id: str = Field(
        description="Short, stable, lowercase-with-hyphens id derived from the behavior under test, e.g. "
        "'login-wrong-password' or 'signup-duplicate-email'. Must be unique across the plan."
    )
    goal: str = Field(
        description="One sentence in the form '<action> should <expected outcome>' — e.g. 'Logging in with a "
        "wrong password should be rejected with an error'. Never a bare feature name like 'Test login'."
    )
    category: TestCategory = Field(
        description="happy_path = normal expected-to-succeed flow. edge_case = boundary/unusual-but-valid "
        "input or ordering. negative = input/action that should be REJECTED by the site. "
        "error_handling = observing/recovering from an error state. security = a free-text input field "
        "echoed back or stored should sanitize/reject a script-injection payload rather than execute or "
        "store it verbatim — only for a field actually observed to accept free text, never invented. "
        "state_interaction = a control's own UI state should change as expected (a visibility toggle, an "
        "expand/collapse, a wizard step back/forward) — only for a control actually observed to be stateful."
    )
    priority: TestPriority = Field(default="medium", description="Relative importance for a human skimming the plan.")
    requires_auth: bool = Field(
        default=False,
        description="True if this test case's goal needs an already-logged-in starting session AND is not "
        "itself testing auth (sign-up/login/logout/reset) — the run establishes one shared login once "
        "(nodes/auth/nodes.py) and restores it into this test case's browser before its first step, so "
        "`steps` must NOT include its own sign-up/login. See TEST_CASE_AUTHORING_GUIDELINES for when this "
        "is and isn't appropriate.",
    )
    preconditions: list[str] = Field(
        default_factory=list,
        description="DISPLAY-ONLY plain-language summary of the state this test case starts from (e.g. "
        "'Runs already authenticated as the shared test account' or 'Signs up a new unique test account'). "
        "Any setup actions a case DOES still need (per requires_auth) MUST ALSO appear inline as the first "
        "entries of `steps` — the worker only ever executes `steps`.",
    )
    # Required on purpose (no default): this is the oracle `verdict_node` grades against
    # (nodes/worker/nodes.py), and an optional field is exactly the one a model under
    # structured output quietly omits — which would put us straight back to grading
    # against nothing but the goal string. NOTE: a checkpoint written before this field
    # existed carries no value for it, so resuming a mid-flight run across this change
    # raises a validation error; that's the accepted trade for not shipping ungradeable
    # test cases.
    expected_result: str = Field(
        description="The ONE observable outcome that decides Pass/Fail, written for a grader who only sees "
        "the final page state and cannot guess your intent: name the visible message, the page/URL reached, "
        "or the element that appears/disappears — quoting the real text where it's known. For a `negative` "
        "case this is the rejection itself (the specific error shown). Never vague ('it works', 'no errors', "
        "'the page loads') — an unverifiable expected_result makes the test case ungradeable."
    )
    steps: list[str] = Field(
        description="Ordered, plain-language instructions for the worker to execute — ONE action per step, "
        "with every value to be typed written literally in quotes inside the step (the worker types exactly "
        "what's written and cannot invent values), phrased by visible accessibility label rather than any "
        "CSS/XPath selector. Include any prerequisite/setup actions (sign-up, login) inline as the first "
        "steps unless requires_auth is true, since each case runs in full isolation. Do not add a trailing "
        "'verify ...' step — grading against expected_result happens automatically after the last step."
    )
    # Feature -> Flow -> Scenario hierarchy (this TestCase IS one Scenario — the only
    # level that actually executes a browser; Feature/Flow are reporting-only groupings,
    # never graded themselves). All optional/defaulted so a planner-authored baseline
    # case — which doesn't know about any of this — still validates unchanged; a
    # currently-unset feature_id/flow_id just means "ungrouped" to the reporter/frontend
    # rather than a validation error. Not yet populated by any producing path as of this
    # change (nodes/planner.py, nodes/discovery.py) — that's nodes/recon/'s job.
    feature_id: str | None = Field(
        default=None,
        description="The Feature this scenario belongs to (e.g. 'sign-up') — set by the planner/discovery "
        "chat for a baseline case, or inherited from the Flow a recon pass discovered this scenario under.",
    )
    flow_id: str | None = Field(
        default=None,
        description="The specific Flow within feature_id this scenario exercises (e.g. 'sign-up-google-oauth' "
        "under feature_id 'sign-up') — None for a baseline case with no recon-discovered flow breakdown.",
    )
    origin: Literal["planner", "recon"] = Field(
        default="planner",
        description="'planner' for a case the one-shot planner or discovery chat authored directly from the "
        "target page's accessibility tree/site map. 'recon' for a case a recon pass generated from a "
        "FlowReport it gathered by actually interacting with that flow — see discovery_rationale for why.",
    )
    discovery_rationale: str | None = Field(
        default=None,
        description="Why THIS specific scenario was generated, grounded in what recon actually observed "
        "(e.g. \"recon found a Google OAuth button on the sign-up form\") — None for a planner-origin case, "
        "whose rationale is just TEST_CASE_AUTHORING_GUIDELINES' coverage checklist.",
    )
    depends_on_sibling: str | None = Field(
        default=None,
        description="test_id of another scenario in this SAME flow that must complete first (e.g. "
        "'signup-happy-path' for a 'signup-duplicate-email' case that needs that account to already exist) "
        "— None (the default, and every planner-origin case) means fully independent and safe to run in "
        "parallel like every other case. Only set this when the dependency is a genuinely observed one, "
        "never a guess — the ordinary rule remains 'no shared state between test cases'.",
    )
    # verdict_node (nodes/worker/nodes.py) grades every case off the accessibility-tree
    # text snapshot alone by default — fast, and sufficient for anything a screen reader
    # could also tell apart. Purely visual regressions (an element pushed off-screen by a
    # CSS change, a chart/canvas that silently failed to render, two elements overlapping)
    # produce no accessibility-tree difference at all, so a case whose expected_result
    # depends on one of those is ungradeable without actually looking at the page. Kept
    # opt-in, not default-on, for the same reason SCENARIOS_PER_FEATURE_MAX etc. are
    # budget-gated: an attached image costs meaningfully more tokens than the text
    # snapshot it doesn't replace, against this project's already-tight Gemini RPM/RPD
    # quota (core/llm.py).
    visual_assertion: bool = Field(
        default=False,
        description="True only if expected_result names something that can ONLY be judged by actually "
        "looking at the rendered page — layout/position, overlap, color, or whether a chart/canvas/image "
        "rendered — never set this for anything the accessibility tree text already shows (a message, a "
        "URL, an element's presence/absence). When true, verdict_node attaches a real screenshot to the "
        "grading call instead of relying on the text snapshot alone.",
    )


class Feature(BaseModel):
    """A high-level testing objective the planner/discovery chat identified from the
    user's instruction (e.g. "Sign-Up", "Create Agent") — the top level of the Feature
    -> Flow -> Scenario hierarchy every TestCase's feature_id/flow_id groups into. Never
    itself graded: reporter_node aggregates its scenarios' TestResults into a rollup
    (by_feature) instead of producing a Feature-level verdict of its own.
    """

    feature_id: str = Field(
        description="Short, stable, lowercase-with-hyphens id, e.g. 'sign-up'. Unique across the plan."
    )
    name: str = Field(description="Human-readable name shown in the UI, e.g. 'Sign-Up'.")
    description: str = Field(
        description="One sentence: what this feature covers and why it's in scope for this run's instruction."
    )


class TestPlan(BaseModel):
    """Structured-output contract the planner LLM call is constrained to."""

    features: list[Feature] = Field(
        description="The small number of high-level testing objectives this instruction covers (e.g. "
        "'Sign-Up', 'Create Agent') — typically 1-4. Every test_case below must set its own feature_id to "
        "one of these feature_id values."
    )
    test_cases: list[TestCase] = Field(
        description="The smallest set in which every case is high-value — typically ~5-12 for a normal "
        "instruction. Must cover the happy path plus the most important negative, edge-case and "
        "error-handling cases for the flows in scope."
    )


class ScenarioProposal(BaseModel):
    """One scenario a recon pass proposes for a Flow it just gathered evidence on —
    ranked so SCENARIOS_PER_FEATURE_MAX/SCENARIOS_PER_RUN_MAX (nodes/recon/) can
    truncate by dropping the lowest-ranked proposals rather than arbitrarily. Turned
    into a full TestCase (feature_id/flow_id/discovery_rationale/origin='recon' set from
    the enclosing FlowReport) once accepted — never executed directly itself.
    """

    goal: str = Field(description="Same contract as TestCase.goal — one sentence, '<action> should <outcome>'.")
    category: TestCategory
    priority: TestPriority = "medium"
    rationale: str = Field(
        description="Why this scenario matters for THIS application specifically, grounded in what recon "
        "actually observed (e.g. \"the sign-up form has a Google OAuth button\") — never a generic "
        "justification a static checklist could have produced for any site."
    )
    rank: int = Field(
        description="Recon's own priority ordering within this Flow — lower is more important. Used to "
        "truncate under a scenario budget without discarding arbitrarily."
    )
    # Without these two, this "scenario" would be a goal statement with nothing for a
    # worker to execute or grade against — a real gap in the first draft of this model,
    # caught before graph wiring made it load-bearing. Same contract as TestCase's own
    # fields of the same name (core/models.py above); recon_plan_node writes both from
    # the SAME synthesis call that produces this proposal, grounded in the exploration
    # transcript it just gathered rather than invented.
    expected_result: str = Field(
        description="Same contract as TestCase.expected_result — the ONE observable outcome that decides "
        "Pass/Fail, naming a visible message/page/element actually seen (or plausible given what was "
        "observed) during exploration. Never vague."
    )
    steps: list[str] = Field(
        description="Same contract as TestCase.steps — ordered, one action per step, every typed value "
        "literal and in quotes, phrased by the EXACT visible label observed during exploration. Include "
        "any prerequisite navigation (e.g. clicking to the flow's entry point) as the first steps, since "
        "each scenario runs in full isolation from a fresh browser, exactly like a planner-authored case."
    )


class FlowReport(BaseModel):
    """Grounded evidence a recon pass gathered for ONE Flow within a Feature (e.g.
    "Google OAuth" under the "Sign-Up" feature) by actually interacting with it — the
    artifact that lets scenario generation run TEST_CASE_AUTHORING_GUIDELINES' method
    against what THIS application actually has, instead of the static per-flow-type
    checklist. Cached the same way SiteMap already is (core/memory.py's
    SITE_MAP_KEY/SITE_MAP_TTL_HOURS pattern) so a second run against the same site skips
    recon entirely.
    """

    feature_id: str
    flow_id: str = Field(description="Short, stable, lowercase-with-hyphens id, e.g. 'sign-up-google-oauth'.")
    flow_name: str = Field(
        description="Human-readable name for this specific flow, e.g. 'Email + Password', 'Google OAuth'."
    )
    observed_fields: list[str] = Field(
        default_factory=list,
        description="Every input/control recon actually saw for this flow, by exact visible label.",
    )
    observed_validation: list[str] = Field(
        default_factory=list,
        description="Exact validation/error messages recon actually triggered or saw for this flow.",
    )
    blocked_by: str | None = Field(
        default=None,
        description="A named external wall (CAPTCHA, OTP, paywall) that stopped recon from completing this "
        "flow, if any — mirrors Verdict's own Blocked reasoning (nodes/worker/nodes.py).",
    )
    scenarios: list[ScenarioProposal] = Field(default_factory=list)
    crawled_at: str | None = Field(default=None, description="ISO timestamp, set when persisted to the long-term store.")


class TestResult(BaseModel):
    test_id: str
    # 'Blocked' (added alongside the adaptive worker — see nodes/agent_loop.py's
    # DEVIATION_POLICY and nodes/worker/nodes.py's Verdict) means the run couldn't reach
    # a verdict because of a named external wall (CAPTCHA, OTP, paywall, missing
    # credential) or ran out of its extended turn budget while genuinely still adapting
    # — NOT evidence the site itself is broken, so it's kept distinct from Fail.
    status: str = Field(description="Strictly 'Pass', 'Fail', or 'Blocked'")
    screenshot_path: str
    trace_path: str | None
    # One short clip per mutating action (browser_click, browser_type, ...), each
    # padded a few seconds before/after — NOT one video for the whole test case.
    # A continuous recording is dominated by however long the LLM took to decide
    # between actions, which shows nothing; see nodes/worker/evidence.py's
    # capture_mutation_clip. Empty list, never None, when nothing was captured (no
    # mutating action ran, or devtools capability unavailable) — always passed
    # explicitly by verdict_node, same convention as this model's other fields.
    video_clips: list[str]
    reason: str
    # Populated from Verdict's same-named fields (nodes/worker/nodes.py) — what the
    # worker had to work around this run, and the steps as actually executed when they
    # diverged from the written plan. default_factory (not required): a checkpoint
    # written before these fields existed carries no value for them, and unlike
    # expected_result (see its own comment above), an empty deviation list is a
    # perfectly valid, common outcome — not a sign something was skipped.
    deviations: list[str] = Field(default_factory=list)
    amended_steps: list[str] = Field(default_factory=list)
    last_step_reached: int = Field(default=0)


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
    candidate_plan: TestPlan = Field(
        description="The full current best-guess test plan, revised each turn (not a diff). Must always be "
        "COMPLETE — re-include every test case still believed in, not only the ones that changed this turn."
    )
    explore_more: ExploreRequest | None = Field(
        default=None, description="Set only if answering well requires a page not yet covered by the site map."
    )
    ready_to_run: bool = Field(
        default=False,
        description="UI hint only (e.g. bolds the Approve button) — NEVER used for control flow. Approval is "
        "always an explicit, separate user action, never inferred from this field.",
    )
