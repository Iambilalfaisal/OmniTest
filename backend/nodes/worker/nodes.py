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

import os

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, interrupt
from pydantic import BaseModel, Field

from ...core.models import TestResult
from ...core.state import WorkerState
from .evidence import capture_screenshot, run_dir_for, stop_and_capture
from .session import discard_session, get_session, session_key
from .tools import ASK_HUMAN_TOOL_NAME, ask_human_tool

# Raised from 15 after repeated direct evidence (multiple live runs against a real
# signup-gated flow) that 15 wasn't enough turns to get through an unplanned signup
# detour AND the test's actual goal, even after fixing the inefficiencies that were
# wasting turns (stale-ref restarts, one-field-at-a-time form filling). A legitimate
# multi-step flow needs real room; this is evidence-based, not a blanket "give it more".
MAX_TOOL_TURNS = 20

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
`browser_snapshot` whenever you need current element refs before acting). Execute the \
numbered steps below in order to pursue the goal. If a tool call fails or the page looks \
unexpectedly different (e.g. blank), take a fresh `browser_snapshot` and continue from \
your actual progress — never restart the test case from step 1, since that wastes your \
limited turns without giving you any new information. Before typing into a second field \
on the same form, stop and use `browser_fill_form` instead to fill every remaining empty \
field in that ONE call — never fill a multi-field form (e.g. a signup form) one \
`browser_type` call per field. You have a limited number of turns; this is usually the \
difference between finishing a multi-field form in budget or not. Call it exactly like \
this, one entry per field, using each field's own ref from your last snapshot as `target`:
browser_fill_form({"fields": [
  {"element": "Full Name input", "target": "e21", "name": "Full Name", "type": "textbox", "value": "QA Tester"},
  {"element": "Work Email input", "target": "e22", "name": "Work Email", "type": "textbox", "value": "qa+test@example.com"},
  {"element": "Password input", "target": "e23", "name": "Password", "type": "textbox", "value": "TestPass123!"}
]})
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
    reason: str


async def agent_node(state: WorkerState, config: RunnableConfig) -> dict:
    test_case = state["test_case"]
    _, tools, _ = await get_session(session_key(config, test_case.test_id))
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
        steps_block = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(test_case.steps))
        seed = [
            SystemMessage(WORKER_SYSTEM_PROMPT),
            HumanMessage(
                f"Target URL: {state['target_url']}\nGoal: {test_case.goal}\n"
                f"Category: {test_case.category} ({_category_note(test_case.category)})\n\n"
                f"Steps:\n{steps_block}"
            ),
        ]
        history = seed

    # 0.3, not 0 — some Gemini variants (confirmed via a runtime warning against the
    # configured WORKER_MODEL) use fixed sampling defaults and ignore temperature=0
    # specifically; a non-zero value is more likely to actually take effect.
    model = ChatGoogleGenerativeAI(model=os.environ["WORKER_MODEL"], temperature=0.3)
    response = await model.bind_tools([*offered_tools, ask_human_tool]).ainvoke(_compact_history(history))

    return {
        "messages": [*seed, response],
        "pending_tool_calls": response.tool_calls,
        "turn_count": state.get("turn_count", 0) + 1,
    }


async def tool_node(state: WorkerState, config: RunnableConfig) -> dict:
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
    _, _, tool_map = await get_session(session_key(config, state["test_case"].test_id))
    result = await tool_map[call["name"]].ainvoke(call["args"])
    result_text = _truncate_tool_result(str(result))

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
    key = session_key(config, test_case.test_id)
    handle, _, tool_map = await get_session(key)

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

    model = ChatGoogleGenerativeAI(model=os.environ["WORKER_MODEL"], temperature=0)
    verdict: Verdict = await model.with_structured_output(Verdict).ainvoke(
        _compact_history(state["messages"])
        + [
            HumanMessage(
                f"Here is the ACTUAL page state right now, captured fresh for this verdict — trust this over "
                f"anything you remember from earlier in the conversation if they seem to disagree:\n{final_snapshot}\n\n"
                "Give your final verdict on this test case now. Category: "
                f"{test_case.category} ({_category_note(test_case.category)}). Base your verdict strictly on "
                "concrete, observable evidence — a visible confirmation, the expected end state, or a specific "
                "error, depending on the category. Reaching the right page is not itself evidence of success: "
                "if the fresh state above doesn't show explicit confirmation the goal was achieved, that is a "
                f"FAIL, not a Pass.{budget_note}"
            )
        ]
    )

    run_dir = run_dir_for(key)
    run_dir.mkdir(parents=True, exist_ok=True)

    screenshot_path = await capture_screenshot(tool_map, run_dir)
    trace_path = await stop_and_capture(tool_map, "browser_stop_tracing", run_dir / "trace.zip")
    video_path = await stop_and_capture(tool_map, "browser_stop_video", run_dir / "video.webm")

    close_tool = tool_map.get("browser_close")
    if close_tool is not None:
        await close_tool.ainvoke({})
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
    sub.add_node("agent_node", agent_node, retry_policy=RetryPolicy(max_attempts=3))
    # No retry on tool_node — retrying after a raised exception risks re-invoking a
    # tool that already had a real side effect. mcp/client.py's per-tool
    # handle_tool_error=True already converts tool exceptions into a ToolMessage fed
    # back to agent_node's next turn instead of raising, so this rarely matters.
    sub.add_node("tool_node", tool_node)
    sub.add_node("verdict_node", verdict_node, retry_policy=RetryPolicy(max_attempts=3))

    sub.add_edge(START, "agent_node")
    sub.add_conditional_edges("agent_node", route_after_agent, ["tool_node", "verdict_node"])
    sub.add_conditional_edges("tool_node", route_after_tool, ["tool_node", "agent_node", "verdict_node"])
    sub.add_edge("verdict_node", END)

    # No checkpointer passed — inherits the parent graph's, required for interrupt()
    # inside tool_node (added later) to actually persist.
    return sub.compile()
