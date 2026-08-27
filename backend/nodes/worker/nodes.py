"""Worker subgraph nodes: `agent_node` (one LLM turn) -> `tool_node` (one tool call) ->
`verdict_node` (final Pass/Fail + evidence capture), looping between `agent_node` and
`tool_node` until the model stops calling tools or `MAX_TOOL_TURNS` is hit.

Restructured from a single-function loop into one node per unit of work so each is
independently checkpointed — required for `tool_node`'s risky-action `interrupt()` to
pause/resume without redoing prior real browser actions. LangGraph re-executes a node
function from the top on resume; splicing an interrupt into a single multi-turn loop
would replay every earlier turn's tool calls.
"""
from __future__ import annotations

import base64
import logging
import os
import re
import time
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ...core import progress, run_knowledge
from ...core.llm import LLM_RETRY_POLICY, ModelRole, with_fallback
from ...core.models import TestResult
from ...core.run_context import run_id_var, test_id_var
from ...core.state import WorkerState
from ...mcp.client import invoke_tool, invoke_tool_or_error_text
from ..agent_loop import (
    ASK_HUMAN_TOOL_NAME,
    DEVIATION_POLICY,
    EXCLUDED_TOOL_NAMES,
    SCENARIO_DEADLINE_SECONDS,
    TRIGGER_REDISCOVERY_TOOL_NAME,
    _risky_text_from_args,
    ask_human_and_reply,
    ask_human_tool,
    compact_history,
    new_deadline,
    review_if_risky,
    stale_snapshot_replacements,
    trigger_rediscovery_tool,
    truncate_tool_result,
)
from .evidence import (
    MUTATING_TOOL_NAMES,
    capture_mutation_clip,
    capture_screenshot,
    ensure_action_overlay,
    run_dir_for,
    stop_and_capture,
)
from .session import SessionGoneError, discard_session, get_session, session_key

# Stage 2: dropped from a hardcoded 20 to an env-tunable default of 8. 20 was set
# before two request-reducing fixes landed: multi-action turns (below, WORKER_SYSTEM_PROMPT)
# let one turn drive several tool calls instead of one, and Stage 3's requires_auth +
# shared storage-state injection deletes the ~15-turn signup/login detour that
# originally justified raising this from 15. At 20, one runaway case could still burn a
# large slice of MAX_CONCURRENT_WORKERS x 20 requests/day producing nothing; 8 caps that
# downside while multi-action turns keep a legitimate flow from actually needing more
# turns to fit the same steps. Override via env if a specific site's flows need more.
# Raised from 10: CONFIRMED live that the earned-extension system (TURN_BUDGET_BONUS on
# a self-reported `deviated` PROGRESS line, or rediscovery_node's replan bonus) requires
# the model to actually SIGNAL that it deviated — and for a multi-step wizard it
# discovers and adapts to gradually, turn by turn, in-context, it often never does
# either: no `deviated` PROGRESS line, no `trigger_rediscovery` call. It just quietly
# clicks through the extra tabs on its own initiative and silently burns the base budget
# with nothing to trigger a bonus. Proof: a real run's console log showed EXACTLY 10 LLM
# calls for a test case whose own final verdict narrated two separate "plan amendments"
# — the deviation was real, but only ever reported retrospectively in the one-shot
# verdict call's own `deviations`/`amended_steps` fields (nodes/worker/nodes.py's
# Verdict model), never live. The earned-extension mechanisms remain in place for the
# cases that DO signal (they still add on top of this base), but this base value is the
# only lever that isn't contingent on the model reliably telling the system it needs
# more room, so it's what actually protects a gradually-discovered multi-step flow.
MAX_TOOL_TURNS = int(os.getenv("MAX_TOOL_TURNS", "16"))

# Earned-extension budget (see agent_node/rediscovery_node/tool_node below): a worker
# starts at MAX_TOOL_TURNS but gains turns — up to this ceiling — each time it actually
# handles a deviation, replans after discovering unanticipated structure, or an
# ask_human answer lands — so the cost of adapting only rises for the specific case that
# needed it, rather than raising the flat cap (and therefore worst-case quota burn) for
# every case whether it deviates or not.
#
# Raised from 20: CONFIRMED live even after rediscovery_node's replan path started
# granting its own (step-count-scaled) bonus — a genuine multi-tab creation wizard
# (Project Info -> Add Members -> Roles and Tasks, each its own fill+Next) still ran
# the case right up against this ceiling twice in a row, reaching progressively further
# each time the budget mechanics were fixed but still falling short of the final submit
# + verification step. The bonus formula scaling with discovered step count was already
# working as intended; the ceiling itself was the remaining constraint capping how much
# of that earned bonus a legitimately complex flow could actually receive.
MAX_TOOL_TURNS_CEILING = int(os.getenv("MAX_TOOL_TURNS_CEILING", "30"))
TURN_BUDGET_BONUS = int(os.getenv("TURN_BUDGET_BONUS", "5"))  # 5 gives enough turns to complete login + resume original test after ask_human

# How many times in a row the SAME tool call (name + args) can repeat before tool_node
# appends a deterministic "you are stuck" nudge to its result — modeled on the proven
# browser_fill_form nudge below, which this codebase already confirmed lands far better
# as a contextual, fired-at-the-moment-it-matters message than a rule stated once at the
# top of a long system prompt.
STUCK_REPEAT_THRESHOLD = int(os.getenv("STUCK_REPEAT_THRESHOLD", "3"))

# Cap on how many of this run's core/run_knowledge.py facts get folded into a fresh
# worker's outbound context — bounded for the same reason MAX_TOOL_RESULT_CHARS exists
# (agent_loop.py): unbounded growth across a long-running multi-case plan would resend
# more and more tokens on every single turn of every later test case.
RUN_KNOWLEDGE_MAX_FACTS = int(os.getenv("RUN_KNOWLEDGE_MAX_FACTS", "8"))

WORKER_SYSTEM_PROMPT = """You are a QA test executor. You control a real browser through \
the tools available to you (Playwright, driven by the accessibility tree — call \
`browser_snapshot` whenever you need current element refs before acting).

Your job is to reach the state the test case's "Expected result" describes — that expected \
result is the only thing this test case is graded on. The numbered steps are the intended \
path to get there, written by someone who could not see every screen this real site might \
show. Follow them in order by default. When the real page doesn't match what a step \
assumes, do not just stop, and do not silently chase a different goal of your own either — \
take the shortest correct route back onto the intended path that still reaches the SAME \
expected result, and say what you changed in your final report.

## RULE #1 — Every step requires an actual tool call. Never stop early.

This is the most important rule. Read it before anything else.

Each numbered step is ONLY "done" when you have called the browser tool that performs \
the action. The mapping is exact:
- "Navigate to X" / "Go to X" → call `browser_navigate`
- "Click X" / "Select X" → call `browser_click` or `browser_select_option`
- "Type X" / "Enter X" / "Fill X" → call `browser_type` or `browser_fill_form`

**Seeing an element in a snapshot is NOT executing the step that targets it.** If step 2 \
says "Click the Submit button" and your last snapshot showed a Submit button, you have \
NOT completed step 2 — you must still call `browser_click` on it. The element appearing \
in the accessibility tree only means it is present on the page.

**Every response that still has unexecuted numbered steps MUST include at least one tool \
call.** A response with only text and no tool call is correct ONLY on your very last turn, \
after every step has been both executed AND its result observed. If you are writing \
sentences describing what you intend to do, add the tool call to that same response — \
do not describe an action and then stop without calling the tool.

**Never call `trigger_rediscovery` before you have navigated to the application.** That \
tool is exclusively for AFTER a login or similar gate WITHIN the app — never on a blank \
page, never before step 1.

**When you hit a login or signup wall, you MUST call `ask_human` — never stop silently.** \
If you are trying to complete a goal (e.g. create an agent, submit a form) and the site \
shows a login or signup screen you were not given credentials for, call `ask_human` \
immediately to ask the reviewer whether to log in or sign up and with what credentials. \
Do NOT stop calling tools, do NOT grade yourself Blocked, do NOT give up — ask first. \
Calling `ask_human` also extends your turn budget by 3 turns, so running low on turns is \
never a reason to skip it. Only use Blocked if you called `ask_human`, received an answer, \
and STILL could not proceed past the wall (e.g. the credentials were wrong, there was a \
CAPTCHA you could not bypass).

**Exception — do NOT call `ask_human` if your OWN numbered steps already typed a specific \
email/username and password into this same login/signup form.** That is not an unanticipated \
wall; it is the test case you were given, and the form still being there afterward (with or \
without a visible error) is data about the outcome, not a blocker. This is the normal shape \
of a negative/validation login test — see the negative-case rule below. Only treat a login/ \
signup screen as a wall requiring `ask_human` when your steps say nothing about submitting \
credentials into it at all (e.g. it interrupts an unrelated goal like creating a project).

## Page state is dynamic — NEVER treat a snapshot as a constraint

Every `browser_snapshot` is a frozen frame of the page at ONE moment. The page changes as \
you interact with it — filling a field, clicking a control, or scrolling can all change \
what other elements look like or whether they respond. You MUST reason about FUTURE state, \
not just the state you see right now, when deciding whether to execute a step.

The single most important pattern: a button or submit control that appears disabled when \
you first arrive at a page is disabled because a required field above it is still empty — \
NOT because the action is unavailable. Filling the field enables the button. This is \
standard web behaviour, and you will encounter it constantly. The correct response is \
always: execute the fill steps first, then attempt the click. Never skip a step or call \
`ask_human` because a downstream button looks disabled at the moment you arrive.

The same logic applies more broadly:
- A "Next" or "Continue" button that is greyed out → fill or select the required fields \
  on this screen first, then click it.
- A form that rejects your click with an error → read the error, correct the field it \
  names, then resubmit — do not treat one rejection as a permanent block.
- An input that looks read-only or inert → try typing into it anyway; the accessibility \
  tree sometimes marks an interactive field incorrectly.
- A click that produces no visible change → take a fresh `browser_snapshot`; the change \
  may have happened somewhere else on the page (a counter, a panel, a toast notification).

Execute your numbered steps in the order they are given. The current disabled/enabled \
state of a later step's target is irrelevant to whether you execute the earlier steps — \
always work through the sequence; the page will update itself as you go.

## Trigger rediscovery after a significant state transition

Sometimes a step in your plan completes a gate-like transition — the most common example \
is successfully logging in, which reveals the authenticated application that was \
completely inaccessible before your steps were written. When this happens, your original \
remaining steps may be wrong because they describe a page structure that nobody had \
seen yet when the plan was created.

When you complete such a transition, call `trigger_rediscovery` ONCE. You will receive \
an updated plan from the system, tailored to what the application actually looks like \
right now, toward the same original objective.

Call `trigger_rediscovery` ONLY when you have just:
- Successfully logged in and can see the authenticated dashboard or home screen.
- Completed an onboarding wizard or account-setup flow and can see the main application.
- Passed through any other authentication or access-control gate that opened up entirely \
  new application structure.

Do NOT call it for:
- Ordinary page navigation (clicking links, filling forms, going to a settings page).
- Any step that doesn't open new parts of the application that were previously gated.
- More than once per major transition — call it exactly once, then wait for the updated plan.

When you call `trigger_rediscovery`, make it the ONLY tool call in that turn.
""" + DEVIATION_POLICY + """
Begin every response with exactly one line, before anything else, in this literal format \
(this is for a progress display, not part of your reasoning):
PROGRESS: step=<n> status=<on_track|deviated> note=<a few words>
`n` is the step number (1-based) you are currently acting on or just finished. Use \
`deviated` only on a turn where you are handling something outside the deviation policy's \
"handle it yourself" bucket above.

CRITICAL for negative test cases (Category: negative). The step values are deliberately \
invalid — a wrong password, a malformed email, a blank required field. The site rejecting \
them with a visible error IS the successful outcome you are looking for. When you see that \
error, you are DONE: stop calling tools — including `ask_human`. Do NOT correct the value \
and retry, do NOT try a different value to "get it to work", and do NOT treat the error as \
a problem to solve or as something that needs a human's confirmation before you can stop. \
Recognizing that the rejection matches the test's own point IS your job, not a decision to \
hand off — asking a human to confirm what your own steps already told you is not real \
ambiguity. Doing so destroys the test case, because the final page state will no longer show \
the rejection that was being verified. The deviation policy above never overrides this: never \
"fix" a negative case's deliberately-invalid input into something the site accepts.

If a tool call fails or the page looks unexpectedly different (e.g. blank), take a fresh \
`browser_snapshot` and continue from your actual progress — never restart the test case \
from step 1, since that wastes your limited turns without giving you any new information. \
Before typing into a second field on the same form, stop and use `browser_fill_form` \
instead to fill every remaining empty field in that ONE call — never fill a multi-field \
form (e.g. a signup form) one `browser_type` call per field. You have a limited number of \
turns; this is usually the difference between finishing a multi-field form in budget or \
not. Call it exactly like this, one entry per field, using each field's own ref from your \
last snapshot as `target`:
browser_fill_form({"fields": [
  {"element": "Full Name input", "target": "e21", "name": "Full Name", "type": "textbox", "value": "QA Tester"},
  {"element": "Work Email input", "target": "e22", "name": "Work Email", "type": "textbox", "value": "qa+test@example.com"},
  {"element": "Password input", "target": "e23", "name": "Password", "type": "textbox", "value": "TestPass123!"}
]})
More generally: whenever you can see, from your last snapshot, that the next 2-3 actions
all target elements already visible on the CURRENT page and none of them is expected to
navigate or change the page's structure until the last one (e.g. click a menu item, then
click the option it reveals, then click Save) — call all of them in the SAME response
instead of one tool call at a time waiting for a fresh snapshot in between. Only take a
new browser_snapshot before continuing if a step actually navigates to a new page/state,
or if a tool result comes back looking wrong/unexpected — snapshotting after every single
click when nothing about the page changed just burns turns for no new information. Each
turn you spend costs one call against a shared quota; batching same-page actions is
usually the difference between finishing in budget or not.
Once every step has been attempted, stop calling tools and wait for the verdict question.

If you hit real ambiguity you cannot resolve yourself, call the `ask_human` tool rather \
than guessing, fabricating a value, or silently giving up. The single most common case: \
you land on a login or signup screen that is blocking your goal — even if nothing in your \
steps mentioned authentication, even if you are almost out of turns. Call `ask_human` \
IMMEDIATELY and ask "Should I log in or sign up to continue? If log in, what are the \
credentials?" Do NOT stop, do NOT write a final message saying you couldn't proceed, do \
NOT let the verdict be Blocked just because you hit an auth wall without asking. Calling \
`ask_human` pauses execution and waits for the reviewer — it is NOT "giving up"; it is the \
correct action. It also gives you 3 extra turns so you can complete the goal after the \
answer arrives. Other `ask_human` triggers: any required field or decision your steps \
don't cover. Never end your turns without either having completed the goal or having called \
`ask_human` — reaching the verdict with the goal unresolved and no `ask_human` call means \
you gave up silently, which is worse than asking. Never quote a sensitive value (a password \
or other secret a human gave you) verbatim in your final verdict — refer to it generically \
instead (e.g. "the provided password").

**Multi-step wizards with an unspecified required field**: some flows (e.g. a project- \
creation wizard's "Add Members" or "Roles and Tasks" tab) require you to search for, \
select, or type a SPECIFIC value — a person's name, a role, an ID — that your test's steps \
and preconditions do NOT give you a concrete value for. Do NOT guess a name, do NOT \
repeatedly search/scroll trying different values hoping one works, and do NOT keep \
clicking through hoping the field turns out to be optional. The moment you notice a \
required field with no value to give it, call `ask_human` immediately: ask exactly what \
value to use, or whether that step/field can be skipped or left at its default. This is the \
same rule as the login-wall case above, applied to any other required-but-unspecified \
field — silently trying to push past it, or getting stuck retrying, wastes turns on a \
decision only the reviewer can actually make.

## After ask_human returns — you MUST immediately act on the answer

When the tool result for ask_human arrives, you will see:
  "Human answer: <what the reviewer said>"
  "Current page state after answer: <snapshot>"

This is NOT the end of the test. You MUST use the answer to continue. Do NOT write a \
text-only response after receiving an ask_human answer — call the next required tool \
immediately, in the same response as any text you write.

The most common case — credentials for a login wall:
1. The page is showing a login/signup form.
2. Read the email/username and password from "Human answer".
3. In ONE turn, fill in ALL the form fields (email field, password field) and click the \
   login button. Use `browser_fill_form` or individual `browser_type`/`browser_click` \
   calls — batch them all into a single response with multiple tool calls.
4. After login succeeds, navigate back to the page or step you were on before the auth \
   wall appeared (the page your goal required) — do NOT assume the app will redirect you \
   there automatically.
5. Continue with the remaining original test steps from exactly where you left off.
6. Do not treat the login itself as the end goal unless the test was specifically about \
   logging in — the goal is whatever was in the original test case.

If the reviewer said "sign up" instead of "log in", do the signup flow with the provided \
details, then continue. If they said "skip", mark the test blocked and stop. If they gave \
a specific instruction, follow it, then continue with the test.\


Before you stop calling tools, make sure the browser is actually sitting on the state your \
expected result describes: if your last action submitted a form or navigated, take one \
final `browser_snapshot` so you have SEEN the real outcome instead of assuming it. Then, in \
your final message, state plainly what you actually observed — the real message text, page, \
or element you ended on, and which step you got to. Do not describe what you expected to \
happen or what "should" have happened; only what was on screen. A negative test case whose \
input was correctly rejected counts as completed — that needs no `ask_human` call.
"""

# What "Pass" means for each TestCase.category — without this, the worker/verdict LLMs
# have no way to tell that a `negative` case is SUPPOSED to be rejected by the site.
_CATEGORY_PASS_NOTES = {
    "happy_path": "Pass means the flow completed successfully as expected.",
    "edge_case": "Pass means the site handled this boundary/unusual-but-valid case correctly.",
    "negative": "Pass means the site correctly REJECTED this input/action with a visible "
    "error — if it incorrectly allowed it to succeed, that is a FAIL.",
    "error_handling": "Pass means the site's error state was handled or recovered from gracefully.",
    "security": "Pass means the injection payload was sanitized/escaped/rejected — if it was echoed back "
    "or stored unescaped (executable, or verbatim in a way that would render as markup), that is a FAIL, "
    "same polarity as a negative case.",
    "state_interaction": "Pass means the control's OWN state changed as expected (e.g. a toggle actually "
    "flipped) — not that the page navigated anywhere.",
}


def _category_note(category: str) -> str:
    return _CATEGORY_PASS_NOTES.get(category, "")


# CONFIRMED live (see core/run_planning.py's ensure_expected_result docstring):
# TestCase.expected_result being a required Pydantic field does NOT guarantee it survived
# langchain-google-genai's structured-output round trip — a real run produced a TestCase
# with the attribute entirely absent (not merely falsy), which planner_node/discovery.py
# now backfill on the way out of every LLM call. `getattr` (not `test_case.expected_result`)
# is still used here as a second line of defense: a checkpoint written before that backfill
# existed, or any future producing path that skips it, would otherwise raise a bare
# AttributeError deep in agent_node/verdict_node instead of degrading to this fallback.
_NO_EXPECTED_RESULT = (
    "(not specified for this test case — grade against the goal above, and Fail if the goal's "
    "outcome is not concretely visible)"
)


def _expected_result(test_case) -> str:
    return (getattr(test_case, "expected_result", None) or "").strip() or _NO_EXPECTED_RESULT


def _recent_consecutive_tool_calls(history: list, tool_name: str) -> int:
    """How many times `tool_name` was just called in a row, most recent first, before
    any other tool result breaks the streak."""
    count = 0
    for m in reversed(history):
        if isinstance(m, ToolMessage) and m.name == tool_name:
            count += 1
        elif isinstance(m, ToolMessage):
            break
    return count


def _recent_repeated_calls(messages: list, call: dict) -> int:
    """How many of the most recent tool_node turns, most recent first, were this EXACT
    tool+args before something different breaks the streak — the same-tool-different-
    page-state case (repeatedly calling `browser_snapshot`, say, with different/no args)
    is deliberately NOT what this counts, since only an identical call with no new
    information is evidence of being stuck. Walks `messages` (not `pending_tool_calls`,
    which only ever holds calls not yet executed) because a tool's own past *calls*
    aren't stored anywhere — only its results (ToolMessage) are — so a repeat is
    detected by comparing `call` against the name+args recorded in each prior AIMessage
    that produced one of those results. `ToolMessage`s themselves are skipped rather
    than treated as streak-breakers: they're each call's RESULT, not a different call,
    so the most recent one (the last thing in `messages` before this fresh `call` was
    even decided) must never end the walk before it reaches the AIMessage behind it.
    """
    target = (call["name"], call.get("args", {}))
    count = 0
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            continue
        if isinstance(m, AIMessage) and len(m.tool_calls) == 1:
            prior = m.tool_calls[0]
            if (prior["name"], prior.get("args", {})) == target:
                count += 1
                continue
        break
    return count


_PROGRESS_LINE = re.compile(
    r"^PROGRESS:\s*step=(\d+)\s+status=(on_track|deviated)(?:\s+note=(.*))?$", re.IGNORECASE
)


def _parse_progress_line(content) -> tuple[int | None, bool, str]:
    """Best-effort parse of WORKER_SYSTEM_PROMPT's requested `PROGRESS: step=<n>
    status=<...> note=<...>` line from a turn's raw response content. Returns
    (step_index or None, deviated, note). `content` may be a plain string or (some
    Gemini responses that also return tool calls) a list of content blocks — never
    raises on either shape, since this is a display/budget/knowledge-sharing nicety,
    not something correctness depends on: A5/A6 give the actual grading a deterministic
    backstop that doesn't need this line to be present at all.
    """
    if isinstance(content, list):
        text = " ".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
    else:
        text = str(content or "")
    for line in text.splitlines():
        m = _PROGRESS_LINE.match(line.strip())
        if m:
            return int(m.group(1)), m.group(2).lower() == "deviated", (m.group(3) or "").strip()
    return None, False, ""


class Verdict(BaseModel):
    status: Literal["Pass", "Fail", "Blocked"] = Field(
        description="'Pass' or 'Fail' as before. Use 'Blocked' ONLY when a specific, NAMED external wall "
        "stopped the run — a CAPTCHA, an OTP/email/SMS verification step, a paywall, or a credential nobody "
        "supplied — or the extended turn budget ran out while genuinely still working through a real "
        "deviation (see the budget note below if present). Simply being unsure, or the site not doing what "
        "was expected, is still a FAIL — 'Blocked' is not an escape hatch from a real Fail."
    )
    reason: str = Field(
        description="2-3 sentences citing the specific observed evidence this verdict rests on — the actual "
        "message text, page, or element seen. For a Fail, also say what was observed instead and at which "
        "step the run diverged. For a Blocked, name the exact wall that stopped you. No advice, no "
        "speculation about causes that weren't observed."
    )
    deviations: list[str] = Field(
        default_factory=list,
        description="Each unplanned thing you had to work around this run (a cookie banner, an extra "
        "required field, a renamed control) — one short entry each. Empty if the real page matched the "
        "written steps exactly.",
    )
    amended_steps: list[str] = Field(
        default_factory=list,
        description="The steps you ACTUALLY executed, in order, only if they differ from the given numbered "
        "steps (an inserted step, a changed target, an added field). Leave empty if you followed the given "
        "steps exactly with no deviation.",
    )
    last_step_reached: int = Field(
        default=0,
        description="The 1-based index, within the given numbered steps, of the last one you completed or "
        "were actively attempting when you stopped. 0 if you never started.",
    )


class RediscoveryPlan(BaseModel):
    """Structured output for the rediscovery_node LLM call — decides whether the
    remaining test steps need to be regenerated given the new application state seen
    after a significant transition (e.g. logging in reveals the authenticated app)."""

    should_replan: bool = Field(
        description=(
            "True if the new application state makes the original remaining steps wrong or "
            "impossible — different URLs, element names, or flow structure. False if they "
            "still make sense from the current position."
        )
    )
    reason: str = Field(
        description="1-2 sentences: why replanning is (or isn't) needed."
    )
    new_steps: list[str] = Field(
        default_factory=list,
        description=(
            "The updated steps to execute from the CURRENT position toward the original "
            "objective, if should_replan is True. Each step is a single, concrete, actionable "
            "instruction (e.g. 'Click the Create Agent button in the sidebar'). Empty list if "
            "should_replan is False."
        ),
    )
    updated_expected_result: str = Field(
        default="",
        description=(
            "The updated expected-result text, if replanning changes what observable success "
            "looks like. Empty string to keep the original."
        ),
    )


# Matches a plain "Navigate to X" / "Go to X" step and captures the URL token
# separately, so agent_node can substitute a different URL while leaving the step's
# own phrasing (and any trailing text) untouched — see the effective_steps rewrite in
# agent_node below for why.
_NAVIGATE_STEP_URL_RE = re.compile(r"^((?:navigate|go)\s+to\s+)(\S+)(.*)$", re.IGNORECASE)


async def agent_node(state: WorkerState, config: RunnableConfig) -> dict:
    test_case = state["test_case"]
    test_id_var.set(test_case.test_id)  # see core/run_context.py — set per-node, not
    # inherited, since LangGraph runs each node as its own separate asyncio task.
    run_id = run_id_var.get()

    # Wall-clock backstop (agent_loop.SCENARIO_DEADLINE_SECONDS), checked before doing
    # any work this turn. Set once on turn 1 (deadline_at is None), then just carried
    # forward every later turn — except tool_node pushes it forward after a genuine
    # ask_human/risky-action pause resumes, so a slow human answer never trips this.
    # MAX_TOOL_TURNS_CEILING already bounds turn COUNT; this bounds the sum of a case
    # that times out on nearly every tool call instead of ever converging.
    now = time.monotonic()
    deadline_at = state.get("deadline_at")
    if deadline_at is None:
        deadline_at = now + SCENARIO_DEADLINE_SECONDS
    elif now > deadline_at:
        return {"pending_tool_calls": [], "abort_reason": f"exceeded its {SCENARIO_DEADLINE_SECONDS:.0f}s execution deadline"}

    # Flips this card from `queued` (route_to_workers' pre-registration) to `running`
    # as soon as this test case's OWN first turn actually starts, rather than only once
    # the (potentially slow) model call below returns.
    progress.update(run_id, test_case.test_id, phase="running", total_steps=len(test_case.steps))
    try:
        _, tools, tool_map = await get_session(session_key(config, test_case.test_id))
    except Exception as exc:
        # A session that never opens (or whose tracing/video setup fails — see
        # session.py's get_session) would otherwise raise uncaught here, and this node
        # carries LLM_RETRY_POLICY — 3 attempts, then propagates and crashes the WHOLE
        # run (LangGraph's executor re-raises the first exception across every
        # concurrently-running Send-spawned branch on exit, not just this one). Abort
        # just this leaf instead; verdict_node turns abort_reason into a Blocked result.
        logging.exception("agent_node: failed to open a session for %s — aborting this test case", test_case.test_id)
        return {"pending_tool_calls": [], "abort_reason": f"could not open a browser session ({exc})"}
    offered_tools = [t for t in tools if t.name not in EXCLUDED_TOOL_NAMES]

    # `messages` uses the `add_messages` reducer (append-only) — the seed below must be
    # returned alongside the response on turn 1 so it's actually persisted into state.
    # Returning only `[response]` would drop it after this call, leaving turn 2+ with a
    # history that starts on an AIMessage with no preceding user turn — Gemini rejects
    # that ("function call turn [must come] immediately after a user turn or after a
    # function response turn").
    history = state.get("messages")
    seed: list = []
    # Adaptive planning fields to include in the turn-1 return dict — only non-empty on
    # the first turn (when seed is built), so later turns don't overwrite them needlessly.
    init_adaptive_fields: dict = {}
    if not history:
        # Stage 3: inject the shared login (nodes/auth/nodes.py) BEFORE the model's first
        # turn, not after — it must land before this test case's own browser_navigate,
        # since a storage-state restore only affects requests made after it, not
        # anything the browser already loaded. Best-effort: a failure here is logged and
        # the test case still proceeds, just unauthenticated — its steps were only
        # written assuming requires_auth in the first place, so this is a real (if
        # unlikely) partial-degradation path, not a silent one.
        auth_state = state.get("auth_storage_state")
        set_storage_state_tool = tool_map.get("browser_set_storage_state")
        auth_restored = False
        if test_case.requires_auth and auth_state and set_storage_state_tool is not None:
            try:
                # auth_state is a file path (see nodes/auth/nodes.py) — confirmed against
                # the installed @playwright/mcp's tool schema that browser_set_storage_state
                # restores from a file via `filename`, not an inline blob.
                await invoke_tool(set_storage_state_tool, {"filename": auth_state})
                auth_restored = True
            except Exception:
                logging.exception(
                    "agent_node: failed to inject shared auth storage state for %s — "
                    "steps will run unauthenticated",
                    test_case.test_id,
                )
        # Redirect this case's own first navigate step to the authenticated landing URL
        # auth_setup_node actually reached (nodes/auth/nodes.py), instead of leaving it
        # pointed at plain target_url — CONFIRMED live: restoring valid session cookies
        # does not guarantee target_url itself shows authenticated content (a site whose
        # root path is a public/marketing page regardless of session, only recognizing
        # auth under a deeper path like "/dashboard"), so the worker landed back on a
        # login page despite a genuinely successful shared login. Only ever touches step
        # 1, and only when it's actually a plain navigate step — every other step (which
        # names the real UI action to test) is untouched.
        effective_steps = test_case.steps
        landing_url = state.get("authenticated_landing_url") if auth_restored else None
        if landing_url and effective_steps:
            match = _NAVIGATE_STEP_URL_RE.match(effective_steps[0])
            if match:
                effective_steps = [
                    f"{match.group(1)}{landing_url}{match.group(3)}",
                    *effective_steps[1:],
                ]
        steps_block = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(effective_steps))
        # Gated on auth_restored, NOT test_case.requires_auth alone — the old version
        # unconditionally told the model "you are already logged in" whenever a test case
        # merely WANTED a shared session, even when auth_setup_node never established one
        # (no credentials reached it, it hit its own deadline, a tool error) or the restore
        # above failed. requires_auth cases are written with no login steps of their own —
        # so a model told it's authenticated when it isn't lands on a real login wall it
        # was explicitly instructed not to act on, with nothing in its own plan to fall
        # back to, and only ask_human as a legitimate way out. Confirmed live: this is what
        # produced the "test case keeps asking for credentials even though they were given
        # at the start" reports — auth_setup_node's own credentials were fine, but its
        # failure silently propagated as a false "already logged in" claim instead of the
        # documented unauthenticated degrade path.
        auth_note = (
            "\nStarting state: you are ALREADY logged in as the shared test account — a valid session was "
            "restored into this browser before your first turn. Do not log in or sign up again."
            if auth_restored
            else ""
        )
        seed = [
            SystemMessage(WORKER_SYSTEM_PROMPT),
            HumanMessage(
                f"Target URL: {state['target_url']}\nGoal: {test_case.goal}\n"
                f"Category: {test_case.category} ({_category_note(test_case.category)})\n"
                # The oracle the run is graded on (core/models.py's TestCase.expected_result),
                # given to the executor too and not only to verdict_node: a worker that knows
                # which observable outcome it's aiming for stops at the right moment — most
                # importantly on a `negative` case, where the error message IS the target state
                # and "helpfully" retrying with a valid value destroys the evidence.
                f"Expected result (what this test case is graded on): {_expected_result(test_case)}"
                f"{auth_note}\n\n"
                f"Steps:\n{steps_block}"
            ),
        ]
        history = seed
        # Seed the adaptive-planning state fields once, on turn 1.  These live in
        # WorkerState as last-write-wins channels — writing them here means
        # rediscovery_node and verdict_node can always read them via state.get(),
        # even before any rediscovery has happened.
        init_adaptive_fields = {
            "objective": test_case.goal,
            "working_steps": list(effective_steps),
            "plan_version": 0,
            "plan_history": [],
            "needs_rediscovery": False,
            "mutation_context": "",
            "current_expected_result": None,
        }

    # Computed once, before the outbound call, and returned below alongside this turn's
    # response — that's what makes the compaction land in the checkpoint (WorkerState's
    # add_messages reducer replaces each of these in place by id) rather than only
    # shrinking what gets sent to the model. See agent_loop.stale_snapshot_replacements
    # for why this is safe to persist and compact_history for the outbound-only
    # counterpart.
    replacements = stale_snapshot_replacements(history)

    # 0.3, not 0 — some Gemini variants (confirmed via a runtime warning against the
    # configured WORKER_MODEL) use fixed sampling defaults and ignore temperature=0
    # specifically; a non-zero value is more likely to actually take effect.
    # trigger_rediscovery_tool is offered alongside ask_human_tool — both are virtual
    # tools intercepted by tool_node before MCP dispatch, never reaching the MCP server.
    model = with_fallback(
        ModelRole.WORKER,
        lambda m: m.bind_tools([*offered_tools, ask_human_tool, trigger_rediscovery_tool]),
        temperature=0.3,
    )

    # core/run_knowledge.py facts recorded by OTHER test cases in this same run (a
    # deviation one of them already worked out) — appended fresh every turn, not
    # persisted into `history`/`seed`, so a fact a sibling records mid-run reaches this
    # worker's very next turn without this worker's own checkpoint ever storing it
    # (compact_history already does the analogous outbound-only trick for stale
    # snapshots, for the same "don't persist something that goes stale" reason).
    outbound = compact_history(history)
    facts_block = run_knowledge.context_block(run_id, max_facts=RUN_KNOWLEDGE_MAX_FACTS)
    if facts_block:
        outbound = [*outbound, HumanMessage(facts_block)]

    response = await model.ainvoke(outbound)

    if not response.tool_calls:
        # The single most useful line for diagnosing an early-stopping test case — logs
        # the model's own text the moment it produces a turn with NO tool call, whatever
        # the reason (thinks it's done, thinks it's blocked, genuinely finished). Full
        # content, not truncated: this is the one signal that explains why a case ended
        # turns short of its budget, and WORKER_SYSTEM_PROMPT's RULE #1 says this should
        # only ever happen on the true final turn — every other occurrence is the bug.
        logging.info(
            "agent_node: run %s test=%s turn %d produced NO tool call — content: %r",
            run_id, test_case.test_id, state.get("turn_count", 0), str(response.content or ""),
        )

    # Earned turn-budget extension: WORKER_SYSTEM_PROMPT asks every turn to open with a
    # `PROGRESS: step=<n> status=<on_track|deviated> note=<...>` line; a `deviated` turn
    # earns this test case TURN_BUDGET_BONUS more turns (capped at MAX_TOOL_TURNS_CEILING),
    # so the cost of adapting only rises for the case that actually needed it, and its
    # note (if any) is shared with sibling test cases via run_knowledge.record_fact.
    # Best-effort by design — _parse_progress_line never raises, so a model that omits
    # the line (or returns empty content alongside its tool calls, which Gemini
    # sometimes does) simply doesn't earn an extension or share a fact that turn,
    # rather than breaking the loop.
    turn_budget = state.get("turn_budget") or MAX_TOOL_TURNS
    parsed_step, deviated, note = _parse_progress_line(response.content)
    if deviated:
        turn_budget = min(turn_budget + TURN_BUDGET_BONUS, MAX_TOOL_TURNS_CEILING)
        if note:
            run_knowledge.record_fact(run_id, note)

    # core/progress.py: live, step-level detail for the SSE `progress` payload
    # (api.py) — step_index defaults to 0 (not persisted) on a turn that didn't parse
    # one; progress.update's own monotonic clamp means that never rewinds the bar.
    progress.update(
        run_id,
        test_case.test_id,
        phase="running",
        step_index=parsed_step or 0,
        total_steps=len(test_case.steps),
        turn=state.get("turn_count", 0) + 1,
        budget=turn_budget,
    )
    if deviated:
        progress.bump(run_id, test_case.test_id, "deviations")
        # Emit a structured mutation event so the frontend's live WorkerCard can render
        # a mutation timeline.  Best-effort: even if note is empty a deviation event is
        # still useful (shows the step and that SOMETHING changed); the agent will keep
        # going regardless of whether this emits cleanly.
        progress.add_mutation_event(
            run_id,
            test_case.test_id,
            type="deviation",
            step=parsed_step or 0,
            description=note or "Agent adapted to unexpected application state",
        )
        # Deviations are self-resolved immediately — the agent handled them without
        # pausing; there's no human interaction to wait for.
        progress.resolve_last_mutation(run_id, test_case.test_id)

    # ── No-tool-call reconsideration gate ─────────────────────────────────────
    # Fires when the model produced a text-only response (no tool calls) while turns
    # remain in the budget. WORKER_SYSTEM_PROMPT's own RULE #1 says this should only
    # ever happen on the true final turn — every other occurrence is the bug. Has an
    # auth-specific branch (below) for the original confirmed case, and a generic
    # fallback for the same failure shape on non-auth content: CONFIRMED live on a
    # `create-project` run — turn 6 of a 16-turn budget produced `PROGRESS: step=3
    # status=on_track note=Filling in project details form fields` with NO tool call,
    # silently ending the test case mid-form despite 10 turns still available and the
    # model's own note saying it was still working.
    #
    # Root cause (auth case): the model consistently hits a login modal, "knows" it lacks
    # credentials, and writes a text summary (going silently to verdict_node as
    # Blocked) instead of calling ask_human — even after multiple explicit prompt
    # instructions. This is a model-prior failure, not a prompt-following failure:
    # base models trained on typical QA automation strongly prefer "stop + report"
    # over "ask the human for credentials." The generic case looks like the same
    # underlying prior applied more broadly: narrate a step, then stop instead of
    # acting on it.
    #
    # Fix: one targeted extra model invocation with an unambiguous enforcement
    # message that leaves the model NO text-only exit path. Cost: one extra LLM
    # call, but only in this exact scenario (~once per auth-gated test case) — far
    # cheaper than silently marking a test Blocked without ever asking.
    # Negative/security cases are EXCLUDED outright: WORKER_SYSTEM_PROMPT's own "CRITICAL
    # for negative test cases" rule says a rejection (wrong password, invalid input, etc.)
    # IS the passing outcome and the model should stop — including a text-only final
    # response that happens to mention "password"/"credential"/"login" while describing
    # that rejection. Letting this gate force an ask_human call there directly
    # contradicts that rule and corrupts the test (the human's answer, and the login
    # attempt that follows it, overwrites the very rejection state being graded).
    _next_turn = state.get("turn_count", 0) + 1  # what turn_count becomes after return
    if not response.tool_calls and _next_turn <= turn_budget and test_case.category not in (
        "negative",
        "security",
    ):
        _resp_lower = str(response.content or "").lower()
        # Deliberately BLOCKING phrases only — NOT bare topical words like "login",
        # "credential", "password", "authentication", "sign in". Those alone match a
        # perfectly normal final report for a test that already SUCCEEDED (e.g.
        # "reached the dashboard using the provided credentials"), which used to force
        # a spurious ask_human demanding credentials again after login had already
        # completed — confirmed live: every auth-adjacent test case, not just the ones
        # genuinely stuck, was hitting this gate. Each phrase below only shows up when
        # the model is actually giving up, not when it's reporting an outcome.
        _AUTH_SIGNALS = (
            "blocked", "cannot proceed", "can't proceed",
            "cannot complete", "can't complete",
            "not provided", "not given",
            "need to log", "need to sign",
            "requires auth", "require auth",
        )
        _enforce = None
        if any(s in _resp_lower for s in _AUTH_SIGNALS):
            _enforce = HumanMessage(
                content=(
                    "[System enforcement — no tool call detected while turns remain.]\n\n"
                    "You described a blocker but made no tool call. This is not allowed. "
                    "You MUST call `ask_human` right now. Do NOT write more text.\n\n"
                    "Exact question to ask: "
                    "'A login or signup wall appeared and is blocking the goal. "
                    "Should I log in or sign up to continue? "
                    "If log in, what are the credentials (email/username and password)?'\n\n"
                    "Set sensitive=True because the answer will contain a password.\n\n"
                    "Calling `ask_human` also extends your turn budget by 3 turns, so "
                    "there will be plenty of turns to finish the task after the answer "
                    "arrives. The ONLY acceptable response right now is the `ask_human` "
                    "tool call — nothing else."
                )
            )
        elif parsed_step is not None and parsed_step < len(test_case.steps):
            # Generic fallback — deliberately gated on parsed_step (already extracted
            # above from this same turn's PROGRESS line), NOT on "any no-tool-call
            # response while budget remains": that broader condition was tried and
            # CONFIRMED live to regress a genuinely-finished test case — user-login had
            # already reached the dashboard and reported success at turn 7, but got
            # rejected and re-prompted every turn through turn 12, restating the same
            # completed outcome instead of finishing. A text-only response IS the
            # intended terminal report (WORKER_SYSTEM_PROMPT RULE #1); only treat it as
            # premature when the model's OWN step count says it isn't on the plan's
            # last step yet — e.g. create-project's "step=3 status=on_track note=Filling
            # in project details form fields" against a 4-step plan.
            _enforce = HumanMessage(
                content=(
                    "[System enforcement — no tool call detected while turns remain.]\n\n"
                    "Your own PROGRESS line says you are on step "
                    f"{parsed_step} of {len(test_case.steps)} — the task is not finished, and "
                    f"turns remain in your budget ({turn_budget - state.get('turn_count', 0)} left). "
                    "A turn with no tool call is only allowed once the test case is genuinely "
                    "complete. Call the next appropriate tool right now to continue — do NOT "
                    "write more text. If you are truly blocked and need information from the "
                    "reviewer, call `ask_human` instead. The ONLY acceptable response right now "
                    "is a tool call."
                )
            )
        # else: no auth signal, and either no PROGRESS line or its own step count says
        # this is the plan's last step — a genuine final report. Leave response as-is,
        # same as before this gate existed, so it proceeds straight to verdict_node.
        if _enforce is not None:
            _retry = await model.ainvoke([*outbound, response, _enforce])
            if _retry.tool_calls:
                logging.info(
                    "agent_node: reconsideration gate produced tool calls for %s "
                    "(first response had none)",
                    test_case.test_id,
                )
                response = _retry
            else:
                logging.warning(
                    "agent_node: reconsideration gate FAILED for %s — model returned "
                    "no tool calls on retry either; proceeding to verdict as Blocked",
                    test_case.test_id,
                )

    return {
        "messages": [*seed, *replacements, response],
        "pending_tool_calls": response.tool_calls,
        "turn_count": state.get("turn_count", 0) + 1,
        "turn_budget": turn_budget,
        "deadline_at": deadline_at,
        **init_adaptive_fields,  # only non-empty on turn 1; noop thereafter
    }


async def tool_node(state: WorkerState, config: RunnableConfig) -> dict:
    test_id = state["test_case"].test_id
    test_id_var.set(test_id)  # see agent_node's comment
    run_id = run_id_var.get()
    call, remaining = state["pending_tool_calls"][0], state["pending_tool_calls"][1:]
    # Tool NAME only, never call["args"] — same reasoning and same value as
    # auth_tool_node's identical line (nodes/auth/nodes.py): without this, a test case
    # that stops early (a text-only response with unfinished steps, no tool call at all)
    # was completely invisible turn-by-turn — only the final verdict's own retrospective
    # summary showed anything, with no way to see what the agent actually did or when it
    # first went quiet.
    logging.info(
        "tool_node: run %s test=%s turn %d calling %s",
        run_id, test_id, state.get("turn_count", 0), call["name"],
    )

    # ── trigger_rediscovery interception ──────────────────────────────────────
    # Must come BEFORE ask_human and review_if_risky — this tool is never in the MCP
    # tool_map (it's a virtual tool like ask_human_tool), so dispatching it would fail.
    # Sets needs_rediscovery=True and mutation_context so route_after_tool can send
    # execution to rediscovery_node once pending_tool_calls drains to empty.
    if call["name"] == TRIGGER_REDISCOVERY_TOOL_NAME:
        completed_transition = call["args"].get("completed_transition", "a state transition")
        new_observation = call["args"].get("new_observation", "")
        mutation_ctx = f"{completed_transition}. {new_observation}".strip(". ")

        # Guard: refuse the call if the agent has not navigated anywhere yet (browser is
        # still on the default blank page). trigger_rediscovery is only meaningful after
        # the agent has already been working inside the application and completed a real
        # transition (login, onboarding). Calling it on a blank page would send
        # rediscovery_node a useless "blank" snapshot and likely cause the agent to loop
        # without ever navigating, burning the entire turn budget.
        # Heuristic: turn_count == 0 means this is the VERY FIRST tool call of the test
        # case — no navigation could possibly have happened yet.
        if state.get("turn_count", 0) == 0:
            premature_reply = ToolMessage(
                content=(
                    "[trigger_rediscovery called too early — you have not navigated to the "
                    "application yet. This tool is only for use AFTER you have completed a "
                    "login or similar transition WITHIN the app. Please call browser_navigate "
                    "first, complete the earlier numbered steps, and only call "
                    "trigger_rediscovery once you have successfully logged in or passed a "
                    "gate that reveals new application structure.]"
                ),
                tool_call_id=call["id"],
                name=call["name"],
            )
            progress.update(run_id, test_id, phase="running", current_action=None)
            return {"messages": [premature_reply], "pending_tool_calls": remaining}

        # Acknowledge the virtual tool call so the agent gets a ToolMessage back —
        # without this the AIMessage's tool_calls list would have an un-replied call,
        # which many LLM providers reject as a malformed turn.
        reply = ToolMessage(
            content=(
                "[Rediscovery triggered — the system will now observe the current application "
                "state and update your plan. Wait for the updated plan before continuing.]"
            ),
            tool_call_id=call["id"],
            name=call["name"],
        )
        progress.update(
            run_id, test_id,
            phase="rediscovering",
            current_action=f"Re-observing app after: {completed_transition[:60]}",
        )
        return {
            "messages": [reply],
            "pending_tool_calls": remaining,
            "needs_rediscovery": True,
            "mutation_context": mutation_ctx,
        }

    # ── ask_human interception ─────────────────────────────────────────────────
    # Checked BEFORE review_if_risky, not after: agent_loop._is_risky substring-matches
    # RISKY_KEYWORDS against "{name} {args}" — a clarifying question whose text happens
    # to contain one of those words (e.g. mentions "removing" something) would
    # otherwise be misrouted into the risky_action branch — wrong payload shape, and on
    # "approve" would KeyError looking up "ask_human" in tool_map below, where it's
    # never added.
    if call["name"] == ASK_HUMAN_TOOL_NAME:
        question = call["args"].get("question", "")
        progress.update(run_id, test_id, phase="awaiting_input", current_action=f"asking: {question[:80]}")

        async def _tool_map() -> dict:
            _, _, tm = await get_session(session_key(config, test_id))
            return tm

        # core/run_knowledge.py dedupe: if an equivalent question was already asked
        # (and answered) by an earlier test case THIS run, reuse that answer instead of
        # pausing again — worker #2 shouldn't make the human answer the same question
        # twice. Still embeds a fresh browser_snapshot for the same reason
        # ask_human_and_reply's own post-interrupt reply does (agent_loop.py): element
        # refs from before this point may be stale. Uses that same trailing phrase
        # verbatim (not just a similarly-worded one) so stale_snapshot_replacements'
        # _EMBEDDED_SNAPSHOT_MARKER match still finds and compacts this snapshot later,
        # exactly as it does for a real ask_human reply.
        reused = run_knowledge.find_answer(run_id, question)
        if reused is not None:
            try:
                tool_map = await _tool_map()
            except Exception as exc:
                logging.exception("tool_node: failed to open a session for %s — aborting this test case", test_id)
                return {"pending_tool_calls": [], "abort_reason": f"could not open a browser session ({exc})"}
            fresh_snapshot = truncate_tool_result(await invoke_tool_or_error_text(tool_map["browser_snapshot"], {}))
            reply = ToolMessage(
                content=(
                    f"[Already answered earlier in this run by another test case.] {reused['answer']}\n\n"
                    "[Fresh snapshot taken after waiting for this answer — element refs from before this "
                    f"point may be stale, use these instead:]\n{fresh_snapshot}"
                ),
                tool_call_id=call["id"],
                name=call["name"],
            )
            progress.update(run_id, test_id, phase="running", current_action=None)
            progress.bump(run_id, test_id, "asks")
            # Reused answers bypass the interrupt() path entirely — record the event as
            # immediately resolved so the frontend's mutation timeline still shows the
            # question + answer without leaving an unresolved slot.
            progress.add_mutation_event(
                run_id, test_id,
                type="clarification",
                description=question,
                sensitive=bool(reused.get("sensitive", False)),
            )
            progress.resolve_last_mutation(
                run_id, test_id,
                user_decision="[reused from another test case in this run]",
            )
            updates: dict = {"messages": [reply], "pending_tool_calls": remaining}
            if reused["sensitive"]:
                # This worker's OWN sensitive_answers, not the answering worker's — each
                # worker's verdict_node only redacts from its own state (see
                # verdict_node's _redact call), so reuse must re-register the secret here
                # too or it would leak into THIS test case's persisted TestResult.reason.
                updates["sensitive_answers"] = [*state.get("sensitive_answers", []), reused["answer"]]
            return updates

        # Emit the mutation event BEFORE interrupt() — interrupt() pauses execution at
        # the LangGraph level so any write AFTER it only runs post-resume; the frontend
        # needs to see the "waiting for input" event while the pause is active.
        progress.add_mutation_event(
            run_id, test_id,
            type="clarification",
            step=state.get("turn_count", 0),
            description=question,
            sensitive=bool(call["args"].get("sensitive", False)),
        )

        # get_session runs INSIDE ask_human_and_reply, only after interrupt() returns
        # (its own docstring's deliberate ordering) — a failure there would otherwise
        # propagate uncaught out of this node and crash the whole run, same reasoning
        # as agent_node's session-open guard.
        #
        # CRITICAL: GraphInterrupt (raised by interrupt() inside ask_human_and_reply)
        # is a subclass of Exception in LangGraph 1.2.x. Without the explicit re-raise
        # below, `except Exception` would swallow the interrupt and set abort_reason to
        # "could not open a browser session (@interrupt{...})", silently preventing the
        # HITL pause from ever working. GraphInterrupt must bubble up to the LangGraph
        # runtime so it can checkpoint the state and emit a 'paused' SSE event.
        try:
            reply, answer_text, sensitive = await ask_human_and_reply(call, _tool_map, subject_id=test_id)
        except GraphInterrupt:
            raise  # Let LangGraph's interrupt machinery handle this — never catch it
        except Exception as exc:
            logging.exception("tool_node: failed to open a session resuming ask_human for %s — aborting", test_id)
            return {"pending_tool_calls": [], "abort_reason": f"could not open a browser session ({exc})"}
        run_knowledge.record_answer(run_id, question, answer_text, sensitive=sensitive)
        # Resolve the mutation event now that the human has answered — store a masked
        # placeholder for sensitive answers so the frontend never displays the real secret.
        resolved_decision = "[sensitive — not displayed]" if sensitive else answer_text
        progress.resolve_last_mutation(run_id, test_id, user_decision=resolved_decision)
        progress.update(run_id, test_id, phase="running", current_action=None)
        progress.bump(run_id, test_id, "asks")
        # An answered ask_human earns the same budget bonus a handled deviation does
        # (see agent_node) — asking should never be the choice that runs a case out of
        # turns faster than guessing would have. deadline_at is likewise pushed forward
        # a fresh SCENARIO_DEADLINE_SECONDS window (agent_loop.new_deadline) — a real
        # interrupt()/resume just happened, and the wall-clock gap was human response
        # time, not runaway execution.
        new_budget = min((state.get("turn_budget") or MAX_TOOL_TURNS) + TURN_BUDGET_BONUS, MAX_TOOL_TURNS_CEILING)
        updates: dict = {
            "messages": [reply],
            "pending_tool_calls": remaining,
            "turn_budget": new_budget,
            "deadline_at": new_deadline(),
        }
        if sensitive:
            updates["sensitive_answers"] = [*state.get("sensitive_answers", []), answer_text]
        return updates

    progress.update(run_id, test_id, phase="awaiting_input", current_action=f"reviewing: {call['name']}")

    # Emit the mutation event before review_if_risky's interrupt() for the same reason
    # as the ask_human branch above: writes after interrupt() only run post-resume.
    # Only emit if the tool is actually risky — review_if_risky returns None immediately
    # for safe calls, so we'd produce a spurious event.  Use a sentinel flag to track
    # whether we emitted one, so we can resolve it cleanly either way.
    from ..agent_loop import _is_risky as _check_risky  # local import avoids a module-
    # level circular dep on a private name — only needed in this one branch.
    _risky_emitted = False
    if _check_risky(call):
        descriptor = _risky_text_from_args(call.get("args", {}))
        description = f"Risky action requires approval: {call['name']}" + (f" — {descriptor}" if descriptor else "")
        progress.add_mutation_event(
            run_id, test_id,
            type="clarification",
            step=state.get("turn_count", 0),
            description=description,
        )
        _risky_emitted = True

    decision = await review_if_risky(call, subject_id=test_id)
    # A real interrupt()/resume happened whenever decision is not None (review_if_risky
    # returns None with no pause for anything that isn't risky at all) — either way
    # below, deadline_at gets pushed forward a fresh window, same reasoning as
    # ask_human's real-pause branch above.
    deadline_update = {"deadline_at": new_deadline()} if decision is not None else {}
    if decision is not None and not decision.get("approved", False):
        progress.update(run_id, test_id, phase="running", current_action=None)
        if _risky_emitted:
            reason = decision.get("reason") or "not approved"
            progress.resolve_last_mutation(
                run_id, test_id,
                user_decision=f"Blocked: {reason}",
            )
        blocked = ToolMessage(
            content=f"Blocked by human reviewer: {decision.get('reason', 'not approved')}",
            tool_call_id=call["id"],
            name=call["name"],
        )
        return {"messages": [blocked], "pending_tool_calls": remaining, **deadline_update}

    # Approved (or not risky) — resolve any emitted event
    if _risky_emitted and decision is not None:
        progress.resolve_last_mutation(run_id, test_id, user_decision="Approved")

    # NOTE: a resume landing on a different process than the one that paused finds no
    # cached session here and transparently opens a fresh, unnavigated browser instead
    # of failing loudly — see the plan's accepted limitations for this exact scenario.
    key = session_key(config, test_id)
    try:
        _, _, tool_map = await get_session(key)
    except Exception as exc:
        logging.exception("tool_node: failed to open a session for %s — aborting this test case", test_id)
        return {"pending_tool_calls": [], "abort_reason": f"could not open a browser session ({exc})"}

    # core/progress.py: what this test case is doing RIGHT NOW, for the SSE payload —
    # reuses agent_loop._risky_text_from_args' element/name/target extraction, since
    # that's already exactly the "what is this call actually targeting" text this
    # codebase computes elsewhere (there for risk detection, here for display).
    descriptor = _risky_text_from_args(call.get("args", {}))
    action_label = f"{call['name']} — {descriptor}" if descriptor else call["name"]
    progress.update(run_id, test_id, phase="running", current_action=action_label)

    # See ensure_action_overlay's docstring for why this has to be retried here rather
    # than once at session start — cheap no-op once already enabled for this session.
    # Called before dispatch (not just after) so a mutation clip started below already
    # has the overlay baked in the first time it succeeds.
    await ensure_action_overlay(tool_map, key)

    async def _do_action():
        # invoke_tool_or_error_text, not invoke_tool: this IS the tool-dispatch point —
        # a raise here (no enclosing try/except) would crash the whole run, not just
        # fail this one leaf (see mcp.client.invoke_tool's docstring).
        return await invoke_tool_or_error_text(tool_map[call["name"]], call["args"])

    # Only a genuine page mutation gets its own clip — a read (browser_snapshot, etc.)
    # or an LLM turn deciding what to do next never had a camera running in the first
    # place. See capture_mutation_clip's docstring for why a session-length recording
    # was replaced with this. clip_index continues from however many this test case
    # has already produced (video_clips accumulates via WorkerState's operator.add
    # reducer across every prior tool_node call), so filenames stay ordered and unique.
    if call["name"] in MUTATING_TOOL_NAMES:
        result, clip_path = await capture_mutation_clip(
            tool_map, run_dir_for(key), len(state.get("video_clips", [])), _do_action
        )
        new_clips = [clip_path] if clip_path else []
    else:
        result = await _do_action()
        new_clips = []

    result_text = truncate_tool_result(str(result))

    # Deterministic, contextual nudge — not just a system-prompt sentence, which was
    # confirmed directly NOT to reliably stop the model from filling multi-field forms
    # one `browser_type` call per field (found from a live run where it filled 2 of 5
    # signup fields individually and ran out of its turn budget before the rest). Fires
    # exactly when the pattern it's meant to interrupt is actually happening, which
    # tends to land far better than a rule stated once at the top of a long context.
    if call["name"] == "browser_type" and _recent_consecutive_tool_calls(state.get("messages", []), "browser_type") >= 1:
        result_text += (
            "\n\n[This is the 2nd+ field you've typed into one at a time on this page — call browser_fill_form "
            "now instead to fill every remaining empty field on this page in ONE call, like this:\n"
            'browser_fill_form({"fields": ['
            '{"element": "<label>", "target": "<ref from your last snapshot>", "name": "<label>", '
            '"type": "textbox", "value": "<value>"}, ...one entry per remaining field]})]'
        )

    repeats = _recent_repeated_calls(state.get("messages", []), call)
    result_text += _stuck_nudge(state, repeats)

    # Deterministic, self-report-independent budget extension. CONFIRMED live (see
    # nodes.py's turn-budget history) that every self-report mechanism — the PROGRESS
    # `status=deviated` line, `trigger_rediscovery`, even the explicit last-turn
    # ask_human nudge in _stuck_nudge below — can all fail to fire in the SAME run: a
    # gradually-discovered multi-tab wizard where the model just keeps calling real,
    # distinct tools without ever announcing it needs more room. Gating the bonus on an
    # announcement that doesn't reliably come means the earned-extension system doesn't
    # protect the exact case it was built for. Reuses `_stuck_nudge`'s own "not stuck"
    # definition (`repeats + 1 < STUCK_REPEAT_THRESHOLD`) as positive evidence of genuine
    # forward progress instead of inventing a separate threshold — `repeats == 0` was
    # tried first and CONFIRMED live to never fire even once across two full runs:
    # `_recent_repeated_calls` walks `state["messages"]`, whose LAST entry is always the
    # very AIMessage that produced this `call` (added_messages already appended it before
    # tool_node runs), so an ordinary single-call turn matches itself on the very first
    # comparison — `repeats` is >= 1 for essentially every real call, never 0.
    budget = state.get("turn_budget") or MAX_TOOL_TURNS
    turn_count = state.get("turn_count", 0)
    budget_update = {}
    if repeats + 1 < STUCK_REPEAT_THRESHOLD and turn_count >= budget - 1 and budget < MAX_TOOL_TURNS_CEILING:
        new_budget = min(budget + TURN_BUDGET_BONUS, MAX_TOOL_TURNS_CEILING)
        logging.info(
            "tool_node: run %s test=%s auto-extending turn_budget %d -> %d "
            "(genuine progress, no repeated call)",
            run_id, test_id, budget, new_budget,
        )
        budget_update = {"turn_budget": new_budget}

    return {
        "messages": [
            ToolMessage(
                content=result_text,
                tool_call_id=call["id"],
                name=call["name"],
            )
        ],
        "pending_tool_calls": remaining,
        "video_clips": new_clips,
        **deadline_update,
        **budget_update,
    }


def _stuck_nudge(state: WorkerState, repeats: int) -> str:
    """Deterministic, contextual nudges appended to a tool result the moment a bad
    pattern is actually happening — same rationale as the browser_fill_form nudge
    above, which this codebase already confirmed lands far better than a rule stated
    once at the top of a long system prompt. "" when neither pattern fires.

    Deliberately does NOT try to detect a streak of tool-call ERRORS: langchain_mcp_
    adapters' handle_tool_errors=True surfaces the MCP server's own error text
    verbatim, with no stable prefix or shape to match on (confirmed against the
    installed langchain-mcp-adapters — see _handle_mcp_tool_error), so a text-pattern
    check for "was this an error" would be guessing at a format that isn't contractual.
    The two checks below use only turn_count/turn_budget and the exact-repeat count,
    both already tracked deterministically regardless of what any tool result says.
    `repeats` is computed once by the caller (tool_node also uses it to decide whether
    to auto-extend the turn budget) rather than recomputed here.
    """
    parts = []
    if repeats + 1 >= STUCK_REPEAT_THRESHOLD:
        parts.append(
            f"\n\n[You have called this exact action with the exact same arguments {repeats + 1} times in a "
            "row with no new information in between. Either take a materially different approach right now, "
            "or call `ask_human` — do not repeat this again unchanged.]"
        )
    budget = state.get("turn_budget") or MAX_TOOL_TURNS
    if state.get("turn_count", 0) >= budget - 1:
        parts.append(
            "\n\n[You are on your last turn before this test case's budget runs out. "
            "If the goal is not yet achieved: (a) if you are blocked on a login/auth wall "
            "or need information from the reviewer, call `ask_human` — this EXTENDS your "
            "budget by 3 turns so you can finish after the answer arrives; "
            "(b) otherwise finish the remaining steps in this turn with tool calls. "
            "Do NOT produce a text-only response — that ends the test case immediately.]"
        )
    return "".join(parts)


def _redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


async def _abort_verdict(state: WorkerState, config: RunnableConfig, abort_reason: str) -> dict:
    """Deterministic Blocked result for a leaf that agent_node/tool_node cut short
    before ever reaching a real verdict call — a session-open timeout (get_session
    never succeeded, so there is no browser to grade against) or an exceeded
    deadline_at (the browser may itself be the thing that's wedged). Skips the LLM
    verdict call and the fresh-snapshot fetch entirely: asking a possibly-unresponsive
    browser for one more snapshot is exactly the kind of wait this mechanism exists to
    bound, and grading against nothing is meaningless anyway. Still attempts teardown
    (best-effort, tolerant of a session that never opened at all) so the slot and any
    real subprocess this leaf held are actually released.
    """
    test_case = state["test_case"]
    key = session_key(config, test_case.test_id)
    logging.warning("verdict_node: aborting %s without grading — %s", key, abort_reason)

    screenshot_path = ""
    try:
        handle, _, tool_map = await get_session(key, require_existing=True)
    except SessionGoneError:
        handle, tool_map = None, None

    if tool_map is not None:
        try:
            run_dir = run_dir_for(key)
            run_dir.mkdir(parents=True, exist_ok=True)
            screenshot_path = await capture_screenshot(tool_map, run_dir)
            close_tool = tool_map.get("browser_close")
            if close_tool is not None:
                await invoke_tool(close_tool, {})
        except Exception:
            logging.exception("verdict_node: best-effort evidence capture failed while aborting %s", key)
        finally:
            discard_session(key)
            await handle.close()

    progress.update(run_id_var.get(), test_case.test_id, phase="done", current_action=None)
    return {
        "test_results": [
            TestResult(
                test_id=test_case.test_id,
                status="Blocked",
                screenshot_path=screenshot_path,
                trace_path=None,
                video_clips=state.get("video_clips", []),
                reason=f"Execution aborted before grading: {abort_reason}",
                deviations=[],
                amended_steps=[],
                last_step_reached=state.get("turn_count", 0),
            )
        ]
    }


async def verdict_node(state: WorkerState, config: RunnableConfig) -> dict:
    test_case = state["test_case"]
    test_id_var.set(test_case.test_id)  # see agent_node's comment
    run_id = run_id_var.get()

    abort_reason = state.get("abort_reason")
    if abort_reason:
        return await _abort_verdict(state, config, abort_reason)

    progress.update(run_id, test_case.test_id, phase="grading", current_action="grading")
    key = session_key(config, test_case.test_id)
    # require_existing=True: unlike agent_node/tool_node, this must NEVER silently open
    # a fresh browser on a cache miss — see SessionGoneError's docstring
    # (nodes/worker/session.py) for the wrong-verdict failure mode that would cause.
    handle, _, tool_map = await get_session(key, require_existing=True)

    # A fresh, final look at the page, taken right now — not relying on whatever the
    # model remembers from its own turn history, which can be stale (an earlier
    # success message it saw before a page changed again, or an assumption it never
    # actually re-checked). Found directly: a run graded Pass with a reason claiming
    # the agent was created, while the real final dashboard showed "Total Agents: 0" —
    # the remembered conversation didn't match the true current state. This costs one
    # extra MCP call, not an extra LLM call, so it's effectively free.
    #
    # Bounded and tolerant, unlike a bare ainvoke: a hang/timeout here would otherwise
    # leave this leaf's own case ungraded AND (per invoke_tool's docstring) risk
    # crashing every sibling test case's already-good results too. Falling back to a
    # placeholder still produces a well-formed grading call — the verdict prompt below
    # already treats "no evidence to point to" as a Fail, which is the correct outcome
    # when the final page genuinely can't be observed.
    try:
        final_snapshot = truncate_tool_result(await invoke_tool_or_error_text(tool_map["browser_snapshot"], {}))
    except Exception:
        logging.exception("verdict_node: failed to capture the final page snapshot for %s", key)
        final_snapshot = "[final snapshot unavailable — the browser did not respond in time]"

    # Only for a case that named a genuinely visual claim (core/models.py's
    # visual_assertion) — the accessibility tree above is text, and can't show layout,
    # overlap, color, or whether a chart/canvas actually rendered. Best-effort: a failure
    # here degrades to text-only grading (still correct for anything the tree DOES show)
    # rather than losing the verdict entirely.
    visual_evidence_b64 = None
    if test_case.visual_assertion:
        try:
            visual_run_dir = run_dir_for(key)
            visual_run_dir.mkdir(parents=True, exist_ok=True)
            await capture_screenshot(tool_map, visual_run_dir)
            visual_evidence_b64 = base64.b64encode((visual_run_dir / "final.png").read_bytes()).decode("ascii")
        except Exception:
            logging.exception("verdict_node: failed to capture the visual-assertion screenshot for %s", key)

    # Found directly from a live run: a test case that spent most of its turn budget on
    # an unplanned signup detour, then hit the budget with the actual goal (typing
    # and submitting an agent description) never attempted — the final page was
    # visibly untouched — still got graded Pass, with a reason amounting to "reached
    # the right screen." Reaching a page, or believing a flow "should" work, is not
    # evidence the goal was achieved; only call out the budget specifically here (not
    # as a blanket instruction on every verdict) since it's the concrete, confirmed
    # failure mode and this stays a no-op for the common case that finishes on its own.
    #
    # turn_budget may have grown past MAX_TOOL_TURNS (see agent_node/tool_node) if this
    # case handled a real deviation or an ask_human answer along the way — exhausting
    # that EXTENDED budget while still genuinely adapting is graded Blocked, not Fail:
    # the test case never got a real answer about the site's behavior, so it isn't
    # evidence the site is broken. Exhausting the plain, un-extended budget with nothing
    # to show for it is still a Fail — nothing earned the extension in the first place.
    turn_budget = state.get("turn_budget") or MAX_TOOL_TURNS
    ran_out_of_turns = state.get("turn_count", 0) >= turn_budget
    was_extended = turn_budget > MAX_TOOL_TURNS
    budget_note = (
        (
            "\n\nNote: you reached your turn budget limit rather than stopping because the goal was done."
            + (
                " This budget was already extended past the normal limit because this run was handling a "
                "real deviation (a deviation, or an ask_human question, along the way). If you were still "
                "genuinely working through that when the budget ran out, grade this BLOCKED — not Fail — and "
                "name in your reason the specific thing still unresolved."
                if was_extended
                else ""
            )
            + " If you don't have concrete evidence the goal was actually completed, and nothing above "
            "applies, that is a FAIL — do not pass this just because you were heading in the right direction "
            "or reached an intermediate screen."
        )
        if ran_out_of_turns
        else ""
    )

    # When replanning happened, grade against the UPDATED expected result and include plan
    # evolution context so the verdict LLM understands the execution history.
    plan_version = state.get("plan_version") or 0
    plan_history = state.get("plan_history") or []
    current_expected_result = state.get("current_expected_result")
    grading_criterion = current_expected_result or _expected_result(test_case)

    plan_evolution_note = ""
    if plan_version > 0 and plan_history:
        plan_evolution_note = "\n\nNote: this test case was REPLANNED during execution:\n"
        for entry in plan_history:
            if entry.get("replanned"):
                new_steps_preview = "; ".join(entry.get("new_steps", [])[:4])
                if len(entry.get("new_steps", [])) > 4:
                    new_steps_preview += f" (+ {len(entry['new_steps']) - 4} more)"
                plan_evolution_note += (
                    f"  Trigger: {entry.get('trigger', '')}\n"
                    f"  Reason: {entry.get('reason', '')}\n"
                    f"  Updated steps from that point: {new_steps_preview}\n"
                )
        if current_expected_result:
            plan_evolution_note += (
                f"\nThe expected result was also updated: {current_expected_result}\n"
                "Grade against this UPDATED expected result and the updated steps above, "
                "not the original ones."
            )

    verdict_text = (
        f"Here is the ACTUAL page state right now, captured fresh for this verdict — trust this over "
        f"anything you remember from earlier in the conversation if they seem to disagree:\n{final_snapshot}\n\n"
        "You are now the QA reviewer grading this test case. You are grading ONE claim, not the site "
        "in general:\n"
        f"Goal: {test_case.goal}\n"
        f"Category: {test_case.category} — {_category_note(test_case.category)}\n"
        f"Expected result (the ONLY criterion; grade against this, not against your own idea of what "
        f"should have happened): {grading_criterion}\n\n"
        "How to decide:\n"
        "- Pass ONLY if the fresh page state above, or a tool result you can actually point to in this "
        "conversation, shows the expected result. Name that evidence in your reason.\n"
        "- Fail if you cannot point to such evidence, if the run never got far enough to produce it, or "
        "if something else happened instead.\n"
        "- Reaching the right page, an absence of visible errors, or a flow that 'should' work is NOT "
        "evidence of success.\n"
        "- On a negative test case the expected result IS a rejection: the site showing that error is a "
        "PASS, and the site accepting the invalid input is a FAIL.\n"
        "- Grade BLOCKED, not Fail, ONLY when the transcript shows a specific, named external wall "
        "stopped the run before it could reach the expected result — a CAPTCHA, an OTP/email/SMS "
        "verification step, a paywall, or a required credential nobody supplied — or the budget note "
        "below tells you to. Name that exact wall in your reason.\n"
        "- A tool error, a crashed step, or any other unrelated blocker that ISN'T one of those named "
        "walls is a FAIL — say so plainly rather than grading the site's behavior you never got to "
        f"observe.{budget_note}{plan_evolution_note}"
    )
    # List content (text block + image_url block) only when there's actually an image —
    # confirmed against the installed langchain_google_genai's own accepted shape
    # (chat_models.py). Plain string otherwise, unchanged from before this field existed.
    verdict_content = verdict_text
    if visual_evidence_b64:
        verdict_content = [
            {
                "type": "text",
                "text": verdict_text + "\n\nA screenshot of this exact final page state is attached below — this "
                "case was flagged as needing a visual check specifically because its expected result names "
                "something the accessibility tree above can't show (layout, overlap, color, or whether "
                "something actually rendered). Weigh the screenshot for that, not for anything the tree "
                "already told you.",
            },
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{visual_evidence_b64}"}},
        ]

    model = with_fallback(ModelRole.VERDICT, lambda m: m.with_structured_output(Verdict), temperature=0)
    verdict: Verdict = await model.ainvoke(compact_history(state["messages"]) + [HumanMessage(content=verdict_content)])

    # Evidence capture + teardown, wrapped in one try/except/finally so NOTHING in this
    # block can escape verdict_node and trigger a node-level retry. Before this fix, a
    # failure anywhere after discard_session (below) — most plausibly handle.close() —
    # would raise out of the node; LangGraph retries the WHOLE node from the top, which
    # re-entered get_session AFTER the session had already been discarded, silently
    # opening a fresh, unnavigated browser, re-snapshotting a blank page, and re-running
    # the verdict LLM against it (a confirmed failure mode). With require_existing=True
    # above, that re-entry now raises SessionGoneError instead — loud, not silently
    # wrong — and node_retry_on (core/llm.py) refuses to retry that exception anyway.
    # screenshot_path defaults to "" (not None) since TestResult.screenshot_path is a
    # required str; trace_path is genuinely Optional in that schema, so None is its
    # correct "capture failed" value. Video has no equivalent here at all — each clip
    # was already started, padded, and stopped inline by tool_node (via
    # capture_mutation_clip) as its mutation happened, not once at teardown; state's
    # accumulated video_clips is read directly below. Each step's result is only
    # overwritten on success, so a partial failure (e.g. trace capture fails after
    # screenshot already succeeded) still preserves whatever evidence WAS captured.
    screenshot_path = ""
    trace_path = None
    try:
        run_dir = run_dir_for(key)
        run_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = await capture_screenshot(tool_map, run_dir)
        trace_path = await stop_and_capture(tool_map, "browser_stop_tracing", run_dir / "trace.zip")
        close_tool = tool_map.get("browser_close")
        if close_tool is not None:
            await invoke_tool(close_tool, {})
    except Exception:
        logging.exception("verdict_node evidence capture/teardown failed for %s — verdict itself still stands", key)
    finally:
        discard_session(key)  # this test case's browser work is done
        await handle.close()

    # Belt-and-suspenders alongside WORKER_SYSTEM_PROMPT's instruction not to quote a
    # secret verbatim — this is the one place a sensitive ask_human answer could
    # otherwise leak into the SSE stream/persisted report (the raw message history
    # never itself leaves this subgraph; only test_results does). Applied to every new
    # free-text field the verdict LLM can populate, not just `reason` — deviations/
    # amended_steps are exactly as capable of quoting an ask_human answer back verbatim.
    secrets = state.get("sensitive_answers", [])
    reason = _redact(verdict.reason, secrets)
    deviations = [_redact(d, secrets) for d in verdict.deviations]
    amended_steps = [_redact(s, secrets) for s in verdict.amended_steps]

    progress.update(
        run_id,
        test_case.test_id,
        phase="done",
        step_index=verdict.last_step_reached,
        current_action=None,
    )

    return {
        "test_results": [
            TestResult(
                test_id=test_case.test_id,
                status=verdict.status,
                screenshot_path=screenshot_path,
                trace_path=trace_path,
                video_clips=state.get("video_clips", []),
                reason=reason,
                deviations=deviations,
                amended_steps=amended_steps,
                last_step_reached=verdict.last_step_reached,
            )
        ]
    }


async def rediscovery_node(state: WorkerState, config: RunnableConfig) -> dict:
    """Takes a fresh browser snapshot and calls an LLM (WORKER model with structured
    output) to decide whether the remaining steps still make sense from the current
    application state, or need to be regenerated toward the same original objective.

    Called by route_after_tool when needs_rediscovery=True and pending_tool_calls is
    empty — always returns to agent_node via a fixed edge, injecting either an updated
    plan or a "no change needed" acknowledgement as a HumanMessage into the message
    history so the agent picks up from the right place.

    Failure modes are soft: if the snapshot or LLM call fails, execution continues with
    the original remaining steps (logged at ERROR level). The test case never aborts
    here — worst case it runs with a slightly stale plan, which is still better than
    losing the test result entirely.
    """
    test_case = state["test_case"]
    test_id_var.set(test_case.test_id)
    run_id = run_id_var.get()

    progress.update(run_id, test_case.test_id, phase="replanning", current_action="Generating updated plan")

    # Fresh snapshot for the replanning LLM — taken from the already-open browser
    # session (same session agent_node/tool_node use for this test case).
    key = session_key(config, test_case.test_id)
    try:
        _, _, tool_map = await get_session(key)
        fresh_snapshot = truncate_tool_result(
            await invoke_tool_or_error_text(tool_map["browser_snapshot"], {})
        )
    except Exception:
        logging.exception("rediscovery_node: failed to get snapshot for %s — keeping original plan", test_case.test_id)
        fresh_snapshot = "[snapshot unavailable]"

    objective = state.get("objective") or test_case.goal
    mutation_context = state.get("mutation_context") or "a state transition"
    working_steps = state.get("working_steps") or list(test_case.steps)
    plan_version = state.get("plan_version") or 0
    plan_history = list(state.get("plan_history") or [])
    turn_count = state.get("turn_count", 0)

    # Split working_steps into "already executed" and "still remaining" using turn_count
    # as a proxy — not perfect (some turns don't advance a step) but good enough for
    # providing the LLM context about what has and hasn't been done yet.
    completed_steps = working_steps[:turn_count] if turn_count < len(working_steps) else working_steps
    remaining_steps = working_steps[turn_count:] if turn_count < len(working_steps) else []

    completed_block = (
        "\n".join(f"{i + 1}. {s}" for i, s in enumerate(completed_steps)) or "(none yet)"
    )
    remaining_block = (
        "\n".join(f"{i + 1}. {s}" for i, s in enumerate(remaining_steps)) or "(none remaining)"
    )

    rediscovery_prompt = (
        f"ORIGINAL OBJECTIVE: {objective}\n\n"
        f"WHAT JUST HAPPENED: {mutation_context}\n\n"
        f"STEPS EXECUTED BEFORE THIS TRANSITION:\n{completed_block}\n\n"
        f"REMAINING STEPS FROM THE CURRENT PLAN:\n{remaining_block}\n\n"
        f"CURRENT APPLICATION STATE (fresh snapshot, taken right now):\n{fresh_snapshot}\n\n"
        "Decide: do the remaining steps above still make sense from the current position, "
        "given what you can see in the fresh snapshot?\n\n"
        "Set should_replan=False if the remaining steps still describe reachable, visible "
        "elements that lead toward the objective — even if a few labels differ slightly.\n"
        "Set should_replan=True if the remaining steps describe a page or elements that no "
        "longer exist in this new state, or if new required steps are now visible that the "
        "original plan didn't know about.\n\n"
        "If should_replan=True, generate new_steps from the CURRENT position (what you see "
        "in the snapshot) toward the original objective — concrete, actionable steps.\n\n"
        f"Original expected result: {_expected_result(test_case)}\n"
        "If replanning changes what observable success looks like, update updated_expected_result."
    )

    model = with_fallback(
        ModelRole.WORKER,
        lambda m: m.with_structured_output(RediscoveryPlan),
        temperature=0,
    )

    try:
        plan: RediscoveryPlan = await model.ainvoke([HumanMessage(rediscovery_prompt)])
    except Exception:
        logging.exception(
            "rediscovery_node: LLM call failed for %s — keeping original plan", test_case.test_id
        )
        progress.update(run_id, test_case.test_id, phase="running", current_action=None)
        return {"needs_rediscovery": False, "mutation_context": ""}

    new_plan_version = plan_version + 1
    new_plan_history = [
        *plan_history,
        {
            "version": new_plan_version,
            "trigger": mutation_context,
            "original_steps": working_steps,
            "new_steps": plan.new_steps if plan.should_replan else working_steps,
            "reason": plan.reason,
            "replanned": plan.should_replan,
        },
    ]

    if plan.should_replan and plan.new_steps:
        new_working_steps = plan.new_steps
        new_expected = (
            plan.updated_expected_result
            or state.get("current_expected_result")
            or _expected_result(test_case)
        )
        # CONFIRMED live: a genuine replan here used to grant NO extra turns at all,
        # unlike agent_node's other two earned-extension paths (a `deviated` PROGRESS
        # line, an answered ask_human) — a multi-tab wizard the original plan never
        # anticipated (e.g. "fill the form" turning into three separate tabs each with
        # their own fields and a Next button) discovers real, concrete NEW work right
        # here, deterministically, yet the test case kept running on whatever budget it
        # already had and reliably ran out mid-wizard. Unlike the other two bonus paths,
        # this one is NOT a fixed TURN_BUDGET_BONUS: the size of the just-discovered plan
        # is a direct, available signal for how much MORE work is actually left, so the
        # bonus scales with it (still floored at TURN_BUDGET_BONUS so a tiny replan gets
        # at least the same bump the other paths do) — capped at MAX_TOOL_TURNS_CEILING
        # like every other path, so this can never grant unbounded turns.
        new_budget = min(
            (state.get("turn_budget") or MAX_TOOL_TURNS) + max(TURN_BUDGET_BONUS, len(new_working_steps)),
            MAX_TOOL_TURNS_CEILING,
        )
        steps_block = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(new_working_steps))
        plan_message = HumanMessage(
            f"[PLAN UPDATED — {mutation_context}]\n"
            f"Reason: {plan.reason}\n\n"
            f"Your updated steps from the current position toward the original objective:\n"
            f"{steps_block}\n\n"
            f"Updated expected result: {new_expected}\n\n"
            "Continue from step 1 of the updated plan above. Your original objective "
            f"remains: {objective}"
        )
        # Record as a deviation event so the mutation timeline shows the plan change.
        progress.add_mutation_event(
            run_id, test_case.test_id,
            type="deviation",
            step=turn_count,
            description=f"Plan updated after {mutation_context}: {plan.reason}",
        )
        progress.resolve_last_mutation(run_id, test_case.test_id, user_decision="Replanned")
        progress.update(
            run_id, test_case.test_id,
            phase="running",
            current_action=None,
            plan_version=new_plan_version,
            plan_history=new_plan_history,
        )
        return {
            "messages": [plan_message],
            "needs_rediscovery": False,
            "mutation_context": "",
            "working_steps": new_working_steps,
            "plan_version": new_plan_version,
            "plan_history": new_plan_history,
            "current_expected_result": new_expected,
            "turn_budget": new_budget,
            # A real replan just happened via a genuine ask_human-free pause — mirrors
            # why tool_node's ask_human path (agent_node) pushes deadline_at forward on
            # its own extension: the wall-clock this rediscovery LLM call just spent
            # would otherwise count against SCENARIO_DEADLINE_SECONDS the same as a stuck
            # loop would, even though it's exactly the productive work the extra turn
            # budget above is meant to pay for.
            "deadline_at": new_deadline(),
        }
    else:
        # No replanning needed — inject an acknowledgement so the agent's message
        # history stays well-formed and it knows to continue with its current plan.
        no_change_message = HumanMessage(
            f"[REDISCOVERY COMPLETE — no plan change needed]\n"
            f"Reason: {plan.reason}\n\n"
            "Your remaining steps are still valid from the current position. Continue with them."
        )
        progress.update(
            run_id, test_case.test_id,
            phase="running",
            current_action=None,
            plan_version=new_plan_version,
            plan_history=new_plan_history,
        )
        return {
            "messages": [no_change_message],
            "needs_rediscovery": False,
            "mutation_context": "",
            "plan_version": new_plan_version,
            "plan_history": new_plan_history,
        }


def route_after_agent(state: WorkerState) -> str:
    return "tool_node" if state["pending_tool_calls"] else "verdict_node"


def route_after_tool(state: WorkerState) -> str:
    # Checked first: an abort (session-open timeout) leaves pending_tool_calls empty
    # with turn_count possibly still under budget — without this, that would route back
    # to agent_node, which would just hit the same dead session again next turn instead
    # of reaching verdict_node's abort_reason short-circuit.
    if state.get("abort_reason"):
        return "verdict_node"
    # If the agent called trigger_rediscovery, route to rediscovery_node once all
    # pending_tool_calls from that same turn have drained — any remaining calls execute
    # first (in case the agent batched them), then rediscovery_node runs.
    if state.get("needs_rediscovery") and not state["pending_tool_calls"]:
        return "rediscovery_node"
    if state["pending_tool_calls"]:
        return "tool_node"
    budget = state.get("turn_budget") or MAX_TOOL_TURNS
    return "verdict_node" if state["turn_count"] >= budget else "agent_node"


def build_worker_subgraph():
    sub = StateGraph(WorkerState)
    sub.add_node("agent_node", agent_node, retry_policy=LLM_RETRY_POLICY)
    # No retry on tool_node — retrying after a raised exception risks re-invoking a
    # tool that already had a real side effect. mcp/client.py's per-tool
    # handle_tool_error=True already converts tool exceptions into a ToolMessage fed
    # back to agent_node's next turn instead of raising, so this rarely matters.
    sub.add_node("tool_node", tool_node)
    sub.add_node("verdict_node", verdict_node, retry_policy=LLM_RETRY_POLICY)
    # rediscovery_node uses an LLM (structured output) — retry policy mirrors verdict_node.
    sub.add_node("rediscovery_node", rediscovery_node, retry_policy=LLM_RETRY_POLICY)

    sub.add_edge(START, "agent_node")
    sub.add_conditional_edges("agent_node", route_after_agent, ["tool_node", "verdict_node"])
    sub.add_conditional_edges(
        "tool_node", route_after_tool,
        ["tool_node", "agent_node", "verdict_node", "rediscovery_node"],
    )
    # rediscovery_node always returns to agent_node — it injects the updated plan as a
    # HumanMessage and the agent continues from there.
    sub.add_edge("rediscovery_node", "agent_node")
    sub.add_edge("verdict_node", END)

    # No checkpointer passed — inherits the parent graph's, required for interrupt()
    # inside tool_node (added later) to actually persist.
    return sub.compile()
