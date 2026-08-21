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
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ...core import progress, run_knowledge
from ...core.llm import LLM_RETRY_POLICY, ModelRole, with_fallback
from ...core.models import TestResult
from ...core.run_context import run_id_var, test_id_var
from ...core.state import WorkerState
from ..agent_loop import (
    ASK_HUMAN_TOOL_NAME,
    DEVIATION_POLICY,
    EXCLUDED_TOOL_NAMES,
    _risky_text_from_args,
    ask_human_and_reply,
    ask_human_tool,
    compact_history,
    review_if_risky,
    stale_snapshot_replacements,
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
from .session import discard_session, get_session, session_key

# Stage 2: dropped from a hardcoded 20 to an env-tunable default of 8. 20 was set
# before two request-reducing fixes landed: multi-action turns (below, WORKER_SYSTEM_PROMPT)
# let one turn drive several tool calls instead of one, and Stage 3's requires_auth +
# shared storage-state injection deletes the ~15-turn signup/login detour that
# originally justified raising this from 15. At 20, one runaway case could still burn a
# large slice of MAX_CONCURRENT_WORKERS x 20 requests/day producing nothing; 8 caps that
# downside while multi-action turns keep a legitimate flow from actually needing more
# turns to fit the same steps. Override via env if a specific site's flows need more.
MAX_TOOL_TURNS = int(os.getenv("MAX_TOOL_TURNS", "8"))

# Earned-extension budget (see agent_node/tool_node below): a worker starts at
# MAX_TOOL_TURNS but gains TURN_BUDGET_BONUS turns, up to MAX_TOOL_TURNS_CEILING, each
# time it actually handles a deviation or an ask_human answer lands — so the cost of
# adapting only rises for the specific case that needed it, rather than raising the flat
# cap (and therefore worst-case quota burn) for every case whether it deviates or not.
MAX_TOOL_TURNS_CEILING = int(os.getenv("MAX_TOOL_TURNS_CEILING", "20"))
TURN_BUDGET_BONUS = int(os.getenv("TURN_BUDGET_BONUS", "3"))

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
error, you are DONE: stop calling tools. Do NOT correct the value and retry, do NOT try a \
different value to "get it to work", and do NOT treat the error as a problem to solve. \
Doing so destroys the test case, because the final page state will no longer show the \
rejection that was being verified. The deviation policy above never overrides this: never \
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


async def agent_node(state: WorkerState, config: RunnableConfig) -> dict:
    test_case = state["test_case"]
    test_id_var.set(test_case.test_id)  # see core/run_context.py — set per-node, not
    # inherited, since LangGraph runs each node as its own separate asyncio task.
    run_id = run_id_var.get()
    # Flips this card from `queued` (route_to_workers' pre-registration) to `running`
    # as soon as this test case's OWN first turn actually starts, rather than only once
    # the (potentially slow) model call below returns.
    progress.update(run_id, test_case.test_id, phase="running", total_steps=len(test_case.steps))
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
        # Stage 3: inject the shared login (nodes/auth/nodes.py) BEFORE the model's first
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
                # auth_state is a file path (see nodes/auth/nodes.py) — confirmed against
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
    model = with_fallback(
        ModelRole.WORKER, lambda m: m.bind_tools([*offered_tools, ask_human_tool]), temperature=0.3
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

    return {
        "messages": [*seed, *replacements, response],
        "pending_tool_calls": response.tool_calls,
        "turn_count": state.get("turn_count", 0) + 1,
        "turn_budget": turn_budget,
    }


async def tool_node(state: WorkerState, config: RunnableConfig) -> dict:
    test_id = state["test_case"].test_id
    test_id_var.set(test_id)  # see agent_node's comment
    run_id = run_id_var.get()
    call, remaining = state["pending_tool_calls"][0], state["pending_tool_calls"][1:]

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
            tool_map = await _tool_map()
            fresh_snapshot = truncate_tool_result(str(await tool_map["browser_snapshot"].ainvoke({})))
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
            updates: dict = {"messages": [reply], "pending_tool_calls": remaining}
            if reused["sensitive"]:
                # This worker's OWN sensitive_answers, not the answering worker's — each
                # worker's verdict_node only redacts from its own state (see
                # verdict_node's _redact call), so reuse must re-register the secret here
                # too or it would leak into THIS test case's persisted TestResult.reason.
                updates["sensitive_answers"] = [*state.get("sensitive_answers", []), reused["answer"]]
            return updates

        reply, answer_text, sensitive = await ask_human_and_reply(call, _tool_map, subject_id=test_id)
        run_knowledge.record_answer(run_id, question, answer_text, sensitive=sensitive)
        progress.update(run_id, test_id, phase="running", current_action=None)
        progress.bump(run_id, test_id, "asks")
        # An answered ask_human earns the same budget bonus a handled deviation does
        # (see agent_node) — asking should never be the choice that runs a case out of
        # turns faster than guessing would have.
        new_budget = min((state.get("turn_budget") or MAX_TOOL_TURNS) + TURN_BUDGET_BONUS, MAX_TOOL_TURNS_CEILING)
        updates: dict = {"messages": [reply], "pending_tool_calls": remaining, "turn_budget": new_budget}
        if sensitive:
            updates["sensitive_answers"] = [*state.get("sensitive_answers", []), answer_text]
        return updates

    progress.update(run_id, test_id, phase="awaiting_input", current_action=f"reviewing: {call['name']}")
    decision = await review_if_risky(call, subject_id=test_id)
    if decision is not None and not decision.get("approved", False):
        progress.update(run_id, test_id, phase="running", current_action=None)
        blocked = ToolMessage(
            content=f"Blocked by human reviewer: {decision.get('reason', 'not approved')}",
            tool_call_id=call["id"],
            name=call["name"],
        )
        return {"messages": [blocked], "pending_tool_calls": remaining}

    # NOTE: a resume landing on a different process than the one that paused finds no
    # cached session here and transparently opens a fresh, unnavigated browser instead
    # of failing loudly — see the plan's accepted limitations for this exact scenario.
    key = session_key(config, test_id)
    _, _, tool_map = await get_session(key)

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
        return await tool_map[call["name"]].ainvoke(call["args"])

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

    result_text += _stuck_nudge(state, call)

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
    }


def _stuck_nudge(state: WorkerState, call: dict) -> str:
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
    """
    parts = []
    repeats = _recent_repeated_calls(state.get("messages", []), call)
    if repeats + 1 >= STUCK_REPEAT_THRESHOLD:
        parts.append(
            f"\n\n[You have called this exact action with the exact same arguments {repeats + 1} times in a "
            "row with no new information in between. Either take a materially different approach right now, "
            "or call `ask_human` — do not repeat this again unchanged.]"
        )
    budget = state.get("turn_budget") or MAX_TOOL_TURNS
    if state.get("turn_count", 0) >= budget - 1:
        parts.append(
            "\n\n[You are on your last turn before this test case's budget runs out. If the goal is not yet "
            "achieved, either finish it in this turn or call `ask_human` now — do not let the budget run out "
            "silently.]"
        )
    return "".join(parts)


def _redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


async def verdict_node(state: WorkerState, config: RunnableConfig) -> dict:
    test_case = state["test_case"]
    test_id_var.set(test_case.test_id)  # see agent_node's comment
    run_id = run_id_var.get()
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
    final_snapshot = truncate_tool_result(str(await tool_map["browser_snapshot"].ainvoke({})))

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

    model = with_fallback(ModelRole.VERDICT, lambda m: m.with_structured_output(Verdict), temperature=0)
    verdict: Verdict = await model.ainvoke(
        compact_history(state["messages"])
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
                "- Grade BLOCKED, not Fail, ONLY when the transcript shows a specific, named external wall "
                "stopped the run before it could reach the expected result — a CAPTCHA, an OTP/email/SMS "
                "verification step, a paywall, or a required credential nobody supplied — or the budget note "
                "below tells you to. Name that exact wall in your reason.\n"
                "- A tool error, a crashed step, or any other unrelated blocker that ISN'T one of those named "
                "walls is a FAIL — say so plainly rather than grading the site's behavior you never got to "
                f"observe.{budget_note}"
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
            await close_tool.ainvoke({})
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


def route_after_agent(state: WorkerState) -> str:
    return "tool_node" if state["pending_tool_calls"] else "verdict_node"


def route_after_tool(state: WorkerState) -> str:
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

    sub.add_edge(START, "agent_node")
    sub.add_conditional_edges("agent_node", route_after_agent, ["tool_node", "verdict_node"])
    sub.add_conditional_edges("tool_node", route_after_tool, ["tool_node", "agent_node", "verdict_node"])
    sub.add_edge("verdict_node", END)

    # No checkpointer passed — inherits the parent graph's, required for interrupt()
    # inside tool_node (added later) to actually persist.
    return sub.compile()
