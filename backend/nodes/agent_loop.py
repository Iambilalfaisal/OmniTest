"""Shared primitives for every LLM-driven tool-calling loop in this codebase: the
worker subgraph (nodes/worker/nodes.py, one instance per TestCase) and the auth
subgraph (nodes/auth/nodes.py, establishing the one shared login). Both are the same
shape — an agent_node making one LLM turn, a tool_node executing one call at a time,
pausing via interrupt() for a human-clarification question or a risky-action review —
so the actual mechanics live here once instead of drifting across two copies.

Extracted from nodes/worker/nodes.py (Stage 0-3's original, and only, tool-calling
loop) when the auth subgraph was built to replace nodes/auth_setup.py's separate
hand-rolled loop — that hand-rolled version had no ask_human escape hatch and no risky-
action review at all, silently degrading every requires_auth case that hit a CAPTCHA or
2FA wall instead of asking, and executing whatever the model returned (short of
browser_run_code_unsafe) with zero human review. Sharing this module is what let the
auth subgraph gain both for free rather than reimplementing them a second time.
"""
from __future__ import annotations

import os
import re
import time
from typing import Awaitable, Callable

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from ..mcp.client import invoke_tool_or_error_text

ASK_HUMAN_TOOL_NAME = "ask_human"

# Coarse per-leaf wall-clock backstop, checked once at the top of every agent_node/
# auth_agent_node turn (not inside tool_node — every individual tool call is already
# bounded by mcp.client.TOOL_CALL_TIMEOUT_SECONDS, so the risk this specifically guards
# against is the SUM across many turns, not any single call: MAX_TOOL_TURNS_CEILING
# already caps turn COUNT, but not turns-times-worst-case-latency, so a case that times
# out on nearly every tool call could still occupy a slot for a very long time without
# this). Deliberately generous relative to a normal test case's real duration — this is
# a backstop for a leaf that is clearly not converging, not a performance target.
SCENARIO_DEADLINE_SECONDS = float(os.getenv("SCENARIO_DEADLINE_SECONDS", "600"))


def new_deadline() -> float:
    """A fresh SCENARIO_DEADLINE_SECONDS window starting now — used both to set a
    leaf's initial deadline on its first turn, and to push it forward after a genuine
    human-in-the-loop pause (ask_human, risky-action review) resumes. A pause's
    wall-clock gap is real human response time, not runaway execution — naively
    continuing a pre-pause countdown would trip the deadline the instant a slow human
    answers. Treating a resume as "you get a fresh window" mirrors the semantics
    MAX_TOOL_TURNS_CEILING's earned extension already uses for turn_budget.
    """
    return time.monotonic() + SCENARIO_DEADLINE_SECONDS


class AskHumanInput(BaseModel):
    question: str = Field(description="The question to ask the human reviewer.")
    context: str | None = Field(default=None, description="Optional extra context to help them answer.")
    sensitive: bool = Field(
        default=False,
        description="Set True if the expected answer is a secret (e.g. a password) so it's masked in the "
        "UI and never echoed back verbatim in the final verdict.",
    )


async def _unreachable(**_kwargs) -> str:
    raise RuntimeError(
        f"{ASK_HUMAN_TOOL_NAME} must be intercepted by a tool_node before this ever runs — "
        "it is deliberately never added to a tool_node's MCP tool_map."
    )


# The callable body is unreachable by design — every tool_node in this codebase always
# intercepts a call to this tool by name, before it ever reaches the generic MCP tool
# dispatch, and this tool is deliberately never added to a tool_node's own MCP tool_map.
ask_human_tool = StructuredTool.from_function(
    coroutine=_unreachable,
    name=ASK_HUMAN_TOOL_NAME,
    description=(
        "Ask the human reviewer a free-text question when you hit real ambiguity you cannot resolve "
        "yourself — e.g. missing login credentials, an unclear requirement, or a decision only a human "
        "can make. Execution pauses until they answer; their answer is given back to you as this tool's "
        "result. Set `sensitive=True` if the answer will be a secret."
    ),
    args_schema=AskHumanInput,
)

# ── trigger_rediscovery virtual tool ─────────────────────────────────────────
# Intercepted by tool_node BEFORE MCP dispatch, exactly like ask_human_tool.
# The agent calls this after completing a significant state transition (e.g. logging
# in) so the system can take a fresh snapshot and regenerate the remaining steps
# against what the app actually looks like now, rather than the pre-transition plan.

TRIGGER_REDISCOVERY_TOOL_NAME = "trigger_rediscovery"


class TriggerRediscoveryInput(BaseModel):
    completed_transition: str = Field(
        description=(
            "Brief description of the significant state change just completed — e.g. "
            "'successfully logged in to the application', 'completed onboarding wizard'."
        )
    )
    new_observation: str = Field(
        description=(
            "What you can now see or access that was inaccessible before the transition — "
            "e.g. 'the main dashboard is visible with a sidebar navigation'."
        )
    )


async def _unreachable_rediscovery(**_kwargs) -> str:
    raise RuntimeError(
        f"{TRIGGER_REDISCOVERY_TOOL_NAME} must be intercepted by tool_node before this ever runs — "
        "it is deliberately never added to a tool_node's MCP tool_map."
    )


# Same interception design as ask_human_tool — tool_node catches this by name, never
# reaches the (unreachable) coroutine body. Not added to any MCP tool_map.
trigger_rediscovery_tool = StructuredTool.from_function(
    coroutine=_unreachable_rediscovery,
    name=TRIGGER_REDISCOVERY_TOOL_NAME,
    description=(
        "Call this tool ONCE, immediately after you complete a significant state transition that "
        "opens up application structure that was completely inaccessible before — most commonly: "
        "successfully logging in (you can now see the authenticated dashboard), or completing an "
        "onboarding wizard (you can now see the main app). After you call this, the system will "
        "take a fresh look at the current page and give you an updated plan from your current "
        "position toward the original objective.\n\n"
        "When to call: you just logged in and the dashboard/home is visible; you just passed an "
        "onboarding gate; any authentication or access-control step that revealed new structure.\n"
        "When NOT to call: ordinary page navigation, filling a form field, clicking a button that "
        "stays on the same screen, or any step that does NOT open new parts of the application. "
        "Also do NOT call it more than once per major transition."
    ),
    args_schema=TriggerRediscoveryInput,
)

# Never offered to any agent_node in this codebase — not a prompt instruction, a hard
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

# Shared between WORKER_SYSTEM_PROMPT (nodes/worker/nodes.py) and AUTH_SETUP_SYSTEM_PROMPT
# (nodes/auth/nodes.py) so both browser-driving loops agree on when to route around a
# real-world surprise silently versus stop and ask. Found directly from a live run: a
# worker told to "stop and report" on ANY step that didn't match reality byte-for-byte
# (an unplanned onboarding screen, an extra required field, a cookie banner) was graded
# Fail on straightforward happy-path cases purely because the written plan and the real
# page weren't identical — never because the site itself misbehaved. Joined into each
# prompt with plain `+` concatenation, not an f-string: WORKER_SYSTEM_PROMPT embeds a
# literal JSON example (`browser_fill_form({"fields": [...`) whose braces an f-string
# would otherwise require escaping.
DEVIATION_POLICY = """## Handling a real page that does not match your steps

Your steps describe the INTENDED path. The real site will sometimes show something they
didn't anticipate. When that happens, decide which of these buckets it's in.

### Handle it yourself, then keep going — and say what you changed in your final report
- A cookie/consent banner, or any interstitial/popup blocking the page.
- An onboarding wizard, product tour, or "what's your role" gate before the real page.
- A newsletter, app-install, or notification-permission modal.
- A control whose visible label differs from the step's wording but is unambiguously the
  one the step means (e.g. the step says "Sign up" and the button reads "Create account").
- One extra required field your steps don't mention, where an obviously-safe value is
  clear from context — invent one rather than leaving it empty.
- An extra "Are you sure?" confirmation on an action your steps already told you to take.
- A tool call that failed once, or a page that looks broken/blank right after navigating —
  take a fresh `browser_snapshot` and try once more before treating it as a real problem.
- Needing to scroll or paginate to bring an element your step already names into view.
- A button or submit control that appears disabled when you arrive at the page — NEVER
  stop because of this. Execute all earlier steps first (fill every field your steps
  describe), THEN attempt to click or interact with it. A button is commonly disabled
  only because a required field is still empty: filling it enables the button. Also, in
  many applications a click on a visually-disabled submit still triggers the
  authentication or validation flow that IS the expected outcome of the test — you must
  attempt the click to find out. Only treat an element as truly unclickable if
  `browser_click` itself returns an error after you have filled all prior fields.

### Stop and call `ask_human` instead — never guess these
- A password, code, or any other secret you were not given.
- An OTP prompt, a CAPTCHA, two-factor auth, an email/SMS verification step, or a paywall.
- A choice with a real business consequence (which plan, which payment method) that your
  steps don't specify.
- Two or more visible controls that could each plausibly be the one a step means.
- An extra required field where guessing wrong would invalidate what this case is
  checking (a phone number, a coupon code, a tax ID) rather than just being cosmetic.

### Never allowed, no matter which bucket you think this is
- Changing a value your steps gave you literally, in quotes.
- Substituting a different control for the one a step names, unless it is unambiguously
  the same control under a different label (see above).
- Skipping the exact step that IS the behavior this test case exists to check.
- Making a deliberately-invalid input valid just so the site accepts it, when the
  rejection itself was the outcome you were checking for.
"""

# Backstop cap on any single tool result's size before it enters message history —
# without this, a large tool result (most commonly browser_snapshot's accessibility
# tree on a complex page) gets resent in full on every subsequent turn of this same
# loop, compounding token cost across however many turns it runs.
MAX_TOOL_RESULT_CHARS = int(os.getenv("MAX_TOOL_RESULT_CHARS", "8000"))


def truncate_tool_result(text: str) -> str:
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    omitted = len(text) - MAX_TOOL_RESULT_CHARS
    return f"{text[:MAX_TOOL_RESULT_CHARS]}\n...[truncated {omitted} more characters — call browser_snapshot again if you need the full current state]"


# Genuinely destructive/irreversible actions only. "confirm" and "submit" are
# deliberately NOT here — they matched almost every ordinary form interaction (login,
# signup, "create agent", contact forms), pausing for human review on essentially every
# step instead of only when something real is at stake — found directly from a report
# that risky-action review was firing "after every agent step."
RISKY_KEYWORDS = ("delete", "purchase", "buy", "pay", "remove")
_RISKY_PATTERN = re.compile(r"\b(?:" + "|".join(RISKY_KEYWORDS) + r")\b")

# Args fields worth scanning for risky INTENT — element/target descriptors the model
# wrote (e.g. "Delete account button"), never a value a human or test data supplied.
# Scanning every arg value (the old behavior) false-positived on ordinary form input —
# a typed email like "buyer@example.com" or a search term containing "removed" — pausing
# runs that had nothing to do with a genuinely destructive action.
_RISKY_ARG_KEYS = ("element", "name", "target", "ref", "selector")


def _risky_text_from_args(args: dict) -> str:
    """Collects descriptor text from a tool call's args, including one level into
    `browser_fill_form`'s `fields` list — the only tool call shape in this codebase
    where the risky descriptors live in a list of dicts rather than the top-level args."""
    parts = []
    for key, value in args.items():
        if key in _RISKY_ARG_KEYS and isinstance(value, str):
            parts.append(value)
        elif key == "fields" and isinstance(value, list):
            for field in value:
                if isinstance(field, dict):
                    parts.extend(str(field[k]) for k in _RISKY_ARG_KEYS if k in field)
    return " ".join(parts)


def _is_risky(call: dict) -> bool:
    # Any tool whose name itself says "unsafe" (e.g. browser_run_code_unsafe — arbitrary
    # JS execution in the page) is ALWAYS risky, regardless of what its args happen to
    # contain — found by reproducing a real pause where this call only got flagged
    # because the generated code string happened to contain "submit"; without this,
    # the exact same tool call generating different code wouldn't have been caught at
    # all, silently letting the agent run arbitrary code un-reviewed.
    if "unsafe" in call["name"].lower():
        return True
    # Word-boundary match (not substring) against tool name + element/target
    # descriptors only (not every arg value) — see _RISKY_PATTERN/_risky_text_from_args
    # above for why: a substring scan over every value flagged ordinary form input
    # (e.g. a typed "buyer@example.com") as a risky action.
    haystack = f"{call['name']} {_risky_text_from_args(call.get('args', {}))}".lower()
    return bool(_RISKY_PATTERN.search(haystack))


async def review_if_risky(call: dict, *, subject_id: str) -> dict | None:
    """If `call` matches `_is_risky`, pauses via `interrupt()` for a human to approve or
    deny it and returns their decision dict (`{"approved": bool, "reason": str, ...}`).
    Returns `None` immediately, with no pause, for anything that isn't risky — a pure
    check with no side effect, so replaying the caller's node on resume is free even
    though the interrupt() call here pauses execution mid-function.

    `subject_id` identifies what this decision is being asked about, for the human
    reviewer/frontend — a TestCase's `test_id` in the worker subgraph, or the fixed
    sentinel `"__auth__"` for the shared-login setup (there's no TestCase there).
    """
    if not _is_risky(call):
        return None
    return interrupt({"type": "risky_action", "test_id": subject_id, "tool": call["name"], "args": call["args"]})


async def ask_human_and_reply(
    call: dict, get_tool_map: Callable[[], Awaitable[dict]], *, subject_id: str
) -> tuple[ToolMessage, str, bool]:
    """Runs the `ask_human` interrupt-and-resume protocol shared by every tool_node in
    this codebase. Pauses via `interrupt()` with the question/context/sensitive flag,
    then on resume embeds a FRESH `browser_snapshot` into the reply — element refs
    captured before a human-length pause are usually stale by the time it resumes.
    Confirmed directly (the original, pre-extraction bug this fixes): a resumed worker
    clicked a pre-pause ref, got "not found," landed on a blank page, and — rather than
    recovering — restarted its entire test case from step 1, burning through its turn
    budget on repeated work instead of progress. Embedding a fresh snapshot right here,
    deterministically, is both cheaper and more reliable than a prompt-only fix: it
    saves an entire extra agent_node -> tool_node round trip the model would otherwise
    need just to ask for one, and it doesn't depend on the model remembering to.

    `get_tool_map` is called only AFTER the interrupt returns, not before — matching
    the original inline implementation's ordering exactly, so the (cheap, cache-hit)
    session lookup never runs on the pre-pause execution that's about to be thrown away
    by interrupt(), only on the resume. Takes a callable rather than a resolved
    tool_map so this module stays agnostic to how each caller manages its own session
    cache (worker/nodes.py's `session_key`-based cache vs. auth/nodes.py's fixed
    `__auth__` key).

    Returns `(reply_message, answer_text, sensitive)` rather than a ready-made state
    update — the caller's own state schema decides what to do with `answer_text` when
    `sensitive` is True (WorkerState appends it to `sensitive_answers` for verdict_node
    to redact; a schema with no such channel, or no answer that could ever reach a
    graded verdict, simply doesn't).
    """
    answer = interrupt(
        {
            "type": "clarification",
            "test_id": subject_id,
            "question": call["args"].get("question", ""),
            "context": call["args"].get("context"),
            "sensitive": bool(call["args"].get("sensitive", False)),
        }
    )
    answer_text = answer.get("text", "")
    tool_map = await get_tool_map()
    # invoke_tool_or_error_text, not invoke_tool: this runs directly inside
    # tool_node/auth_tool_node after a real interrupt()/resume, with no enclosing
    # try/except — a raise here would crash the whole run (see invoke_tool's
    # docstring), not just fail this one leaf's ask_human turn.
    fresh_snapshot = truncate_tool_result(await invoke_tool_or_error_text(tool_map["browser_snapshot"], {}))
    reply = ToolMessage(
        content=(
            f"{answer_text}\n\n[Fresh snapshot taken after waiting for this answer — "
            f"element refs from before this point may be stale, use these instead:]\n{fresh_snapshot}"
        ),
        tool_call_id=call["id"],
        name=call["name"],
    )
    return reply, answer_text, bool(call["args"].get("sensitive", False))


# Marks a `browser_snapshot` embedded inside an ask_human reply (see ask_human_and_reply
# above) so stale_snapshot_replacements can find and trim it too — that message is
# named "ask_human", not "browser_snapshot", since its tool_call_id must match the
# ask_human call it's replying to.
_EMBEDDED_SNAPSHOT_MARKER = "\n\n[Fresh snapshot taken after waiting for this answer"

_STALE_SNAPSHOT_PLACEHOLDER = (
    "[earlier snapshot omitted to save context — call browser_snapshot again if you need current element refs]"
)


def stale_snapshot_replacements(history: list) -> list[ToolMessage]:
    """Placeholder `ToolMessage`s for every stale `browser_snapshot` result in
    `history` — whether it's its own tool message or embedded inside an ask_human
    reply — except the most recent one. Each replacement carries the SAME `id` as the
    message it supersedes: a state's `add_messages` reducer replaces a stored message
    in place when a returned message's `id` matches one already in state (confirmed
    against the installed langgraph 1.2.11 — this is also how every message already in
    state came to have a real `id` in the first place, since `add_messages` auto-
    assigns one to any message that arrives without one). A caller's agent_node returns
    these alongside its response so the compaction lands in the checkpoint, not just in
    what gets sent to the model this turn — see `compact_history` below for the
    outbound-only counterpart, and its docstring for why an earlier snapshot is safe to
    collapse. Returns [] (no persisted change) once nothing is left to collapse,
    including when a message was already replaced on a prior turn — its content
    already equals the placeholder, so re-emitting it would be a pure no-op write.
    """

    def is_plain_snapshot(m) -> bool:
        return isinstance(m, ToolMessage) and m.name == "browser_snapshot"

    def has_embedded_snapshot(m) -> bool:
        return isinstance(m, ToolMessage) and _EMBEDDED_SNAPSHOT_MARKER in str(m.content)

    snapshot_indices = [i for i, m in enumerate(history) if is_plain_snapshot(m) or has_embedded_snapshot(m)]
    if len(snapshot_indices) <= 1:
        return []

    replacements = []
    for i in snapshot_indices[:-1]:
        m = history[i]
        if is_plain_snapshot(m):
            content = _STALE_SNAPSHOT_PLACEHOLDER
        else:
            # ask_human reply with an embedded snapshot — keep the human's actual
            # answer (durable context the model still needs), drop just the stale
            # snapshot text appended after it.
            answer_part = str(m.content).split(_EMBEDDED_SNAPSHOT_MARKER, 1)[0]
            content = f"{answer_part}\n\n{_STALE_SNAPSHOT_PLACEHOLDER}"
        if str(m.content) == content:
            continue
        replacements.append(ToolMessage(content=content, tool_call_id=m.tool_call_id, name=m.name, id=m.id))
    return replacements


def compact_history(history: list) -> list:
    """Applies `stale_snapshot_replacements` for THIS outbound model call only — used
    everywhere a call site needs the compacted view but isn't in a position to persist
    it (verdict_node's own read of `state["messages"]`, and as the base every
    agent_node turn builds on before adding this turn's own new replacements).

    An earlier snapshot reflects a page state the agent has almost always already
    acted on or navigated past; resending every one of them in full on every
    subsequent turn is the single biggest driver of a tool-calling loop's token cost
    (each browser_snapshot result can be thousands of tokens on a complex page, and a
    plain full-history resend compounds that across however many turns it runs).
    """
    replacements = stale_snapshot_replacements(history)
    if not replacements:
        return history
    by_id = {r.id: r for r in replacements}
    return [by_id.get(m.id, m) for m in history]
