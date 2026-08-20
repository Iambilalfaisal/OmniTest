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

import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from ...core.llm import LLM_RETRY_POLICY, ModelRole, get_chat_model
from ...core.models import TestResult
from ...core.run_context import test_id_var
from ...core.state import WorkerState
from .evidence import capture_screenshot, ensure_action_overlay, run_dir_for, stop_and_capture
from .session import discard_session, get_session, session_key
from .tools import ASK_HUMAN_TOOL_NAME, ask_human_tool

# Stage 2: dropped from a hardcoded 20 to an env-tunable default of 8. 20 was set
# before two request-reducing fixes landed: multi-action turns (below, WORKER_SYSTEM_PROMPT)
# let one turn drive several tool calls instead of one, and Stage 3's requires_auth +
# shared storage-state injection deletes the ~15-turn signup/login detour that
# originally justified raising this from 15. At 20, one runaway case could still burn a
# large slice of MAX_CONCURRENT_WORKERS x 20 requests/day producing nothing; 8 caps that
# downside while multi-action turns keep a legitimate flow from actually needing more
# turns to fit the same steps. Override via env if a specific site's flows need more.
MAX_TOOL_TURNS = int(os.getenv("MAX_TOOL_TURNS", "8"))

# Backstop cap on any single tool result's size before it enters message history —
# without this, a large tool result (most commonly browser_snapshot's accessibility
# tree on a complex page) gets resent in full on every subsequent turn of this same
# test case's loop, compounding token cost across up to MAX_TOOL_TURNS turns.
MAX_TOOL_RESULT_CHARS = int(os.getenv("MAX_TOOL_RESULT_CHARS", "8000"))

# Never offered to the worker model at all — not a prompt instruction, a hard
# exclusion. Found directly from live runs: the model defaulted to
# `browser_run_code_unsafe` (arbitrary JS in the PLAYWRIGHT SERVER PROCESS, per its own
# tool description — a materially bigger privilege than page-sandboxed JS) for
# completely routine actions it already has dedicated tools for — filling a textbox,
# clicking a button/link, even just logging page content — turning nearly every single
# turn into a risky_action pause. A stronger prompt instruction wasn't tried first here
# because the same class of fix (a soft instruction) already proved unreliable for
# ask_human in this exact codebase. Removing the tool is deterministic, costs nothing
# (no prompt tokens spent trying to talk it out of a tool it can't see), and there's no
# legitimate QA-testing need for server-process-level code execution — `browser_click`,
# `browser_type`, `browser_fill_form`, `browser_select_option`, and `browser_evaluate`
# (page-sandboxed JS, a materially smaller privilege) already cover everything a test
# step should need.
EXCLUDED_TOOL_NAMES = frozenset({"browser_run_code_unsafe"})

WORKER_SYSTEM_PROMPT = """You are a QA test executor. You control a real browser through \
the tools available to you (Playwright, driven by the accessibility tree — call \
`browser_snapshot` whenever you need current element refs before acting).

Your job is to execute the numbered steps you are given, in order, exactly as written, \
and reach the state the test case's "Expected result" describes. That expected result is \
the only thing this test case will be graded on. You are executing someone else's written \
test case, not improvising your own: do not add steps that aren't in the list, do not \
skip a step because you think it's unnecessary, and do not "fix" a test case you think is \
wrong — if a step is genuinely impossible (the element it names does not exist on the \
page, even after a fresh snapshot), stop and report that plainly rather than substituting \
a different element you happened to find.

CRITICAL for negative test cases (Category: negative). The step values are deliberately \
invalid — a wrong password, a malformed email, a blank required field. The site rejecting \
them with a visible error IS the successful outcome you are looking for. When you see that \
error, you are DONE: stop calling tools. Do NOT correct the value and retry, do NOT try a \
different value to "get it to work", and do NOT treat the error as a problem to solve. \
Doing so destroys the test case, because the final page state will no longer show the \
rejection that was being verified.

A blocking cookie banner, consent dialog, or interstitial popup is not a test failure — \
dismiss or accept it and carry on with your steps.

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
than guessing, fabricating a value, or silently giving up. This most commonly means: you \
land on a login or signup screen that isn't part of your steps (even if nothing told you \
to expect it) — stop immediately and ask whether to log in (and with what credentials) or \
sign up, instead of guessing a password or abandoning the goal; or any other required \
field or decision your steps simply don't cover. Never end your turns without either \
having completed the goal or having called `ask_human` — reaching the verdict with the \
goal unresolved and no `ask_human` call means you gave up silently, which is worse than \
asking. Never quote a sensitive value (a password or other secret a human gave you) \
verbatim in your final verdict — refer to it generically instead (e.g. "the provided \
password").

Before you stop calling tools, make sure the browser is actually sitting on the state your \
expected result describes: if your last action submitted a form or navigated, take one \
final `browser_snapshot` so you have SEEN the real outcome instead of assuming it. Then, in \
your final message, state plainly what you actually observed — the real message text, page, \
or element you ended on, and which step you got to. Do not describe what you expected to \
happen or what "should" have happened; only what was on screen. A negative test case whose \
input was correctly rejected counts as completed — that needs no `ask_human` call.
"""

# Genuinely destructive/irreversible actions only. "confirm" and "submit" are
# deliberately NOT here — they matched almost every ordinary form interaction (login,
# signup, "create agent", contact forms), pausing for human review on essentially every
# step instead of only when something real is at stake — found directly from a report
# that risky-action review was firing "after every agent step."
RISKY_KEYWORDS = ("delete", "purchase", "buy", "pay", "remove")

# What "Pass" means for each TestCase.category — without this, the worker/verdict LLMs
# have no way to tell that a `negative` case is SUPPOSED to be rejected by the site.
_CATEGORY_PASS_NOTES = {
    "happy_path": "Pass means the flow completed successfully as expected.",
    "edge_case": "Pass means the site handled this boundary/unusual-but-valid case correctly.",
    "negative": "Pass means the site correctly REJECTED this input/action with a visible "
    "error — if it incorrectly allowed it to succeed, that is a FAIL.",
    "error_handling": "Pass means the site's error state was handled or recovered from gracefully.",
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


def _is_risky(call: dict) -> bool:
    # Any tool whose name itself says "unsafe" (e.g. browser_run_code_unsafe — arbitrary
    # JS execution in the page) is ALWAYS risky, regardless of what its args happen to
    # contain — found by reproducing a real pause where this call only got flagged
    # because the generated code string happened to contain "submit"; without this,
    # the exact same tool call generating different code wouldn't have been caught at
    # all, silently letting the agent run arbitrary code un-reviewed.
    if "unsafe" in call["name"].lower():
        return True
    haystack = f"{call['name']} {call.get('args', {})}".lower()
    return any(word in haystack for word in RISKY_KEYWORDS)


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


def _truncate_tool_result(text: str) -> str:
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    omitted = len(text) - MAX_TOOL_RESULT_CHARS
    return f"{text[:MAX_TOOL_RESULT_CHARS]}\n...[truncated {omitted} more characters — call browser_snapshot again if you need the full current state]"


# Marks a `browser_snapshot` embedded inside an ask_human reply (see tool_node) so
# _compact_history can find and trim it too — that message is named "ask_human", not
# "browser_snapshot", since its tool_call_id must match the ask_human call it's
# replying to.
_EMBEDDED_SNAPSHOT_MARKER = "\n\n[Fresh snapshot taken after waiting for this answer"


def _compact_history(history: list) -> list:
    """Collapses every stale `browser_snapshot` result — whether it's its own tool
    message or embedded inside an ask_human reply — down to a short placeholder for
    THIS call only, keeping just the most recent one in full. The real, untruncated
    history stays in WorkerState/the checkpoint untouched; this only affects what's
    sent to the model.

    An earlier snapshot reflects a page state the agent has almost always already
    acted on or navigated past; resending every one of them in full on every
    subsequent turn is the single biggest driver of this loop's token cost (each
    browser_snapshot result can be thousands of tokens on a complex page, and a plain
    full-history resend compounds that across up to MAX_TOOL_TURNS turns).
    """

    def is_plain_snapshot(m) -> bool:
        return isinstance(m, ToolMessage) and m.name == "browser_snapshot"

    def has_embedded_snapshot(m) -> bool:
        return isinstance(m, ToolMessage) and _EMBEDDED_SNAPSHOT_MARKER in str(m.content)

    snapshot_indices = [i for i, m in enumerate(history) if is_plain_snapshot(m) or has_embedded_snapshot(m)]
    if len(snapshot_indices) <= 1:
        return history

    compacted = list(history)
    placeholder = "[earlier snapshot omitted to save context — call browser_snapshot again if you need current element refs]"
    for i in snapshot_indices[:-1]:
        m = compacted[i]
        if is_plain_snapshot(m):
            compacted[i] = ToolMessage(content=placeholder, tool_call_id=m.tool_call_id, name=m.name)
        else:
            # ask_human reply with an embedded snapshot — keep the human's actual
            # answer (durable context the model still needs), drop just the stale
            # snapshot text appended after it.
            answer_part = str(m.content).split(_EMBEDDED_SNAPSHOT_MARKER, 1)[0]
            compacted[i] = ToolMessage(content=f"{answer_part}\n\n{placeholder}", tool_call_id=m.tool_call_id, name=m.name)
    return compacted


class Verdict(BaseModel):
    status: str = Field(description="Strictly 'Pass' or 'Fail'")
    reason: str = Field(
        description="2-3 sentences citing the specific observed evidence this verdict rests on — the actual "
        "message text, page, or element seen. For a Fail, also say what was observed instead and at which "
        "step the run diverged. No advice, no speculation about causes that weren't observed."
    )


async def agent_node(state: WorkerState, config: RunnableConfig) -> dict:
    test_case = state["test_case"]
    test_id_var.set(test_case.test_id)  # see core/run_context.py — set per-node, not
    # inherited, since LangGraph runs each node as its own separate asyncio task.
    _, tools, tool_map = await get_session(session_key(config, test_case.test_id))
    offered_tools = [t for t in tools if t.name not in EXCLUDED_TOOL_NAMES]

    # `messages` uses the `add_messages` reducer (append-only) — the seed below must be
    # returned alongside the response on turn 1 so it's actually persisted into state.
    # Returning only `[response]` would drop it after this call, leaving turn 2+ with a
    # history that starts on an AIMessage with no preceding user turn — Gemini rejects
    # that ("function call turn [must come] immediately after a user turn or after a
    # function response turn").
    history = state.get("messages")
    seed: list = []
    if not history:
        # Stage 3: inject the shared login (nodes/auth_setup.py) BEFORE the model's first
        # turn, not after — it must land before this test case's own browser_navigate,
        # since a storage-state restore only affects requests made after it, not
        # anything the browser already loaded. Best-effort: a failure here is logged and
        # the test case still proceeds, just unauthenticated — its steps were only
        # written assuming requires_auth in the first place, so this is a real (if
        # unlikely) partial-degradation path, not a silent one.
        auth_state = state.get("auth_storage_state")
        set_storage_state_tool = tool_map.get("browser_set_storage_state")
        if test_case.requires_auth and auth_state and set_storage_state_tool is not None:
            try:
                # auth_state is a file path (see nodes/auth_setup.py) — confirmed against
                # the installed @playwright/mcp's tool schema that browser_set_storage_state
                # restores from a file via `filename`, not an inline blob.
                await set_storage_state_tool.ainvoke({"filename": auth_state})
            except Exception:
                logging.exception(
                    "agent_node: failed to inject shared auth storage state for %s — "
                    "steps will run unauthenticated",
                    test_case.test_id,
                )
        steps_block = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(test_case.steps))
        auth_note = (
            "\nStarting state: you are ALREADY logged in as the shared test account — a valid session was "
            "restored into this browser before your first turn. Do not log in or sign up again."
            if test_case.requires_auth
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

    # 0.3, not 0 — some Gemini variants (confirmed via a runtime warning against the
    # configured WORKER_MODEL) use fixed sampling defaults and ignore temperature=0
    # specifically; a non-zero value is more likely to actually take effect.
    model = get_chat_model(ModelRole.WORKER, temperature=0.3)
    response = await model.bind_tools([*offered_tools, ask_human_tool]).ainvoke(_compact_history(history))

    return {
        "messages": [*seed, response],
        "pending_tool_calls": response.tool_calls,
        "turn_count": state.get("turn_count", 0) + 1,
    }


async def tool_node(state: WorkerState, config: RunnableConfig) -> dict:
    test_id_var.set(state["test_case"].test_id)  # see agent_node's comment
    call, remaining = state["pending_tool_calls"][0], state["pending_tool_calls"][1:]

    # Checked BEFORE _is_risky, not after: _is_risky substring-matches RISKY_KEYWORDS
    # against "{name} {args}" — a clarifying question whose text happens to contain one
    # of those words (e.g. mentions "removing" something) would otherwise be misrouted
    # into the risky_action branch — wrong payload shape, and on "approve" would
    # KeyError looking up "ask_human" in tool_map below, where it's never added.
    if call["name"] == ASK_HUMAN_TOOL_NAME:
        answer = interrupt(
            {
                "type": "clarification",
                "test_id": state["test_case"].test_id,
                "question": call["args"].get("question", ""),
                "context": call["args"].get("context"),
                "sensitive": bool(call["args"].get("sensitive", False)),
            }
        )
        answer_text = answer.get("text", "")

        # A human can take an arbitrary amount of real time to answer — the browser
        # session sits open the whole time, so element refs the model captured before
        # the pause are frequently stale by the time it resumes. Confirmed directly: a
        # resumed worker clicked a pre-pause ref, got "not found," landed on a blank
        # page, and — rather than recovering — restarted its entire test case from step
        # 1, burning through its turn budget on repeated work instead of progress.
        # Embedding a fresh snapshot right here, deterministically, is both CHEAPER and
        # more reliable than a prompt-only fix: it saves an entire extra agent_node ->
        # tool_node round trip the model would otherwise need just to ask for one, and
        # it doesn't depend on the model remembering to.
        _, _, tool_map = await get_session(session_key(config, state["test_case"].test_id))
        fresh_snapshot = _truncate_tool_result(str(await tool_map["browser_snapshot"].ainvoke({})))
        reply = ToolMessage(
            content=(
                f"{answer_text}\n\n[Fresh snapshot taken after waiting for this answer — "
                f"element refs from before this point may be stale, use these instead:]\n{fresh_snapshot}"
            ),
            tool_call_id=call["id"],
            name=call["name"],
        )
        updates: dict = {"messages": [reply], "pending_tool_calls": remaining}
        if call["args"].get("sensitive"):
            updates["sensitive_answers"] = [*state.get("sensitive_answers", []), answer_text]
        return updates

    # Pure check, no side effect yet — so replaying this node on resume is free even
    # though the interrupt() call below pauses execution mid-function.
    if _is_risky(call):
        decision = interrupt(
            {
                "type": "risky_action",
                "test_id": state["test_case"].test_id,
                "tool": call["name"],
                "args": call["args"],
            }
        )
        if not decision.get("approved", False):
            blocked = ToolMessage(
                content=f"Blocked by human reviewer: {decision.get('reason', 'not approved')}",
                tool_call_id=call["id"],
                name=call["name"],
            )
            return {"messages": [blocked], "pending_tool_calls": remaining}

    # NOTE: a resume landing on a different process than the one that paused finds no
    # cached session here and transparently opens a fresh, unnavigated browser instead
    # of failing loudly — see the plan's accepted limitations for this exact scenario.
    key = session_key(config, state["test_case"].test_id)
    _, _, tool_map = await get_session(key)
    result = await tool_map[call["name"]].ainvoke(call["args"])
    result_text = _truncate_tool_result(str(result))

    # See ensure_action_overlay's docstring for why this has to be retried here rather
    # than once at session start — cheap no-op once already enabled for this session.
    await ensure_action_overlay(tool_map, key)

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

    return {
        "messages": [
            ToolMessage(
                content=result_text,
                tool_call_id=call["id"],
                name=call["name"],
            )
        ],
        "pending_tool_calls": remaining,
    }


def _redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


async def verdict_node(state: WorkerState, config: RunnableConfig) -> dict:
    test_case = state["test_case"]
    test_id_var.set(test_case.test_id)  # see agent_node's comment
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
    final_snapshot = _truncate_tool_result(str(await tool_map["browser_snapshot"].ainvoke({})))

    # Found directly from a live run: a test case that spent most of its turn budget on
    # an unplanned signup detour, then hit MAX_TOOL_TURNS with the actual goal (typing
    # and submitting an agent description) never attempted — the final page was
    # visibly untouched — still got graded Pass, with a reason amounting to "reached
    # the right screen." Reaching a page, or believing a flow "should" work, is not
    # evidence the goal was achieved; only call out the budget specifically here (not
    # as a blanket instruction on every verdict) since it's the concrete, confirmed
    # failure mode and this stays a no-op for the common case that finishes on its own.
    ran_out_of_turns = state.get("turn_count", 0) >= MAX_TOOL_TURNS
    budget_note = (
        "\n\nNote: you reached your turn budget limit rather than stopping because the goal was done. If "
        "you don't have concrete evidence the goal was actually completed by this point, that is a FAIL — "
        "do not pass this just because you were heading in the right direction or reached an intermediate "
        "screen."
        if ran_out_of_turns
        else ""
    )

    model = get_chat_model(ModelRole.VERDICT, temperature=0)
    verdict: Verdict = await model.with_structured_output(Verdict).ainvoke(
        _compact_history(state["messages"])
        + [
            HumanMessage(
                f"Here is the ACTUAL page state right now, captured fresh for this verdict — trust this over "
                f"anything you remember from earlier in the conversation if they seem to disagree:\n{final_snapshot}\n\n"
                "You are now the QA reviewer grading this test case. You are grading ONE claim, not the site "
                "in general:\n"
                f"Goal: {test_case.goal}\n"
                f"Category: {test_case.category} — {_category_note(test_case.category)}\n"
                f"Expected result (the ONLY criterion; grade against this, not against your own idea of what "
                f"should have happened): {_expected_result(test_case)}\n\n"
                "How to decide:\n"
                "- Pass ONLY if the fresh page state above, or a tool result you can actually point to in this "
                "conversation, shows the expected result. Name that evidence in your reason.\n"
                "- Fail if you cannot point to such evidence, if the run never got far enough to produce it, or "
                "if something else happened instead.\n"
                "- Reaching the right page, an absence of visible errors, or a flow that 'should' work is NOT "
                "evidence of success.\n"
                "- On a negative test case the expected result IS a rejection: the site showing that error is a "
                "PASS, and the site accepting the invalid input is a FAIL.\n"
                "- A tool error, a crashed step, or an unrelated blocker is a Fail — say so plainly rather than "
                f"grading the site's behavior you never got to observe.{budget_note}"
            )
        ]
    )

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
    # required str; trace_path/video_path are genuinely Optional in that schema, so None
    # is their correct "capture failed" value. Each step's result is only overwritten on
    # success, so a partial failure (e.g. video capture fails after screenshot/trace
    # already succeeded) still preserves whatever evidence WAS captured.
    screenshot_path = ""
    trace_path = video_path = None
    try:
        run_dir = run_dir_for(key)
        run_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = await capture_screenshot(tool_map, run_dir)
        trace_path = await stop_and_capture(tool_map, "browser_stop_tracing", run_dir / "trace.zip")
        video_path = await stop_and_capture(tool_map, "browser_stop_video", run_dir / "video.webm")
        close_tool = tool_map.get("browser_close")
        if close_tool is not None:
            await close_tool.ainvoke({})
    except Exception:
        logging.exception("verdict_node evidence capture/teardown failed for %s — verdict itself still stands", key)
    finally:
        discard_session(key)  # this test case's browser work is done
        await handle.close()

    # Belt-and-suspenders alongside WORKER_SYSTEM_PROMPT's instruction not to quote a
    # secret verbatim — this is the one place a sensitive ask_human answer could
    # otherwise leak into the SSE stream/persisted report (the raw message history
    # never itself leaves this subgraph; only test_results does).
    reason = _redact(verdict.reason, state.get("sensitive_answers", []))

    return {
        "test_results": [
            TestResult(
                test_id=test_case.test_id,
                status=verdict.status,
                screenshot_path=screenshot_path,
                trace_path=trace_path,
                video_path=video_path,
                reason=reason,
            )
        ]
    }


def route_after_agent(state: WorkerState) -> str:
    return "tool_node" if state["pending_tool_calls"] else "verdict_node"


def route_after_tool(state: WorkerState) -> str:
    if state["pending_tool_calls"]:
        return "tool_node"
    return "verdict_node" if state["turn_count"] >= MAX_TOOL_TURNS else "agent_node"


def build_worker_subgraph():
    sub = StateGraph(WorkerState)
    sub.add_node("agent_node", agent_node, retry_policy=LLM_RETRY_POLICY)
    # No retry on tool_node — retrying after a raised exception risks re-invoking a
    # tool that already had a real side effect. mcp/client.py's per-tool
    # handle_tool_error=True already converts tool exceptions into a ToolMessage fed
    # back to agent_node's next turn instead of raising, so this rarely matters.
    sub.add_node("tool_node", tool_node)
    sub.add_node("verdict_node", verdict_node, retry_policy=LLM_RETRY_POLICY)

    sub.add_edge(START, "agent_node")
    sub.add_conditional_edges("agent_node", route_after_agent, ["tool_node", "verdict_node"])
    sub.add_conditional_edges("tool_node", route_after_tool, ["tool_node", "agent_node", "verdict_node"])
    sub.add_edge("verdict_node", END)

    # No checkpointer passed — inherits the parent graph's, required for interrupt()
    # inside tool_node (added later) to actually persist.
    return sub.compile()
