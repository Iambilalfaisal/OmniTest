"""Auth subgraph nodes: `auth_agent_node` (one LLM turn) -> `auth_tool_node` (one tool
call) -> `auth_save_node` (persist the shared storage state + teardown), looping
between auth_agent_node and auth_tool_node until the model stops calling tools or
AUTH_SETUP_MAX_TURNS is hit — the exact same shape as nodes/worker/nodes.py's worker
loop, and for the exact same reason: `interrupt()` replay safety.

Replaces the original nodes/auth_setup.py, a hand-rolled `for` loop with no ask_human
escape hatch (every requires_auth case silently degraded to unauthenticated the moment
a CAPTCHA, 2FA code, or rejected credentials came up — there was nowhere to ask) and no
risky-action review at all (the model could act on "Delete account" with zero human
oversight, guarded only by excluding browser_run_code_unsafe). Rebuilding this as a
proper subgraph sharing nodes/agent_loop.py's primitives with the worker subgraph is
what gains both for free instead of reimplementing them a second time.

Runs at most once per run, and only if the approved plan actually has a requires_auth
case — auth_agent_node's first lines short-circuit before opening any browser for the
common no-auth-needed run, exactly like the node this replaces did.

Session lifecycle deliberately goes through nodes/worker/session.py's SHARED
`_SESSIONS` cache (keyed `f"{thread_id}:__auth__"`) rather than a private
AsyncExitStack, the way the original did — that shared cache is what lets this
session survive across auth_agent_node/auth_tool_node/auth_save_node's separate node
invocations, AND across an ask_human/risky-action pause, the same way a worker's
session survives across its own three nodes. It is only safe to rely on that survival
because api.py's `_drive()` no longer sweeps sessions on a mere pause (see that
module's own history for the bug this would otherwise reproduce on the login flow).
"""
from __future__ import annotations

import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from ...core.llm import LLM_RETRY_POLICY, ModelRole, with_fallback
from ..agent_loop import (
    ASK_HUMAN_TOOL_NAME,
    DEVIATION_POLICY,
    EXCLUDED_TOOL_NAMES,
    ask_human_and_reply,
    ask_human_tool,
    compact_history,
    review_if_risky,
    stale_snapshot_replacements,
    truncate_tool_result,
)
from ..worker.evidence import EVIDENCE_DIR, run_dir_for, stop_and_capture
from ..worker.session import discard_session, get_session
from .state import AuthState

# Mechanical, single-purpose loop (sign up or log in, once) — same order-of-magnitude
# budget a real test case's own setup detour used to need, before this subgraph
# existed to eliminate paying that cost once per requires_auth case instead of once
# per run.
AUTH_SETUP_MAX_TURNS = int(os.getenv("AUTH_SETUP_MAX_TURNS", "8"))

# Where the shared login's storage state file is saved, keyed by thread_id (== run_id)
# so concurrent runs in this same process never collide. Reuses EVIDENCE_DIR rather
# than a second top-level directory — this is per-run artifact storage exactly like
# evidence.py's screenshot/trace/video files, just not screenshot/trace/video.
AUTH_STATE_DIR = EVIDENCE_DIR / "auth-state"

# Fixed sentinel subject/session id: this subgraph establishes ONE session per run, not
# one per TestCase, so it has no real test_id to key its session or interrupt payloads
# by. Distinct from any real test_id a planner could generate (those are
# lowercase-hyphenated words per TEST_CASE_AUTHORING_GUIDELINES) so it never collides.
AUTH_SUBJECT_ID = "__auth__"

AUTH_SETUP_SYSTEM_PROMPT = """You are establishing ONE shared logged-in browser session that \
will be reused by several separate test cases afterward. You are NOT running a test case: \
there is nothing to verify, nothing to grade, and no edge case to probe here. Your only job \
is to get this browser into an authenticated state on the target site, as directly as \
possible, and then stop.

How to proceed:
1. Take a `browser_snapshot` of the current page first. Find the way in — usually a 'Log in', \
'Sign in', 'Get started' or 'Sign up' link in the header or nav. Dismiss any cookie or \
consent banner blocking the page; that is not a failure.
2. If existing test credentials are given to you below, LOG IN with those. Use them exactly \
as given — do not modify, guess at, or "correct" them.
3. Only if no credentials were given, sign up a brand new account using the run token you \
were given (e.g. email "qa+<run_token>@example.com", with a strong password like \
"QaTest123!") and complete whatever the sign-up form requires.
4. Fill a multi-field form with ONE `browser_fill_form` call rather than one \
`browser_type` per field — you have very few turns.
5. Stop the moment you can see you are logged in: an account menu or avatar, a dashboard, a \
'Log out' option, or being redirected off the login page onto an app page. Do not explore, do \
not click anything else, do not sign out, do not change any account setting, and do not \
delete anything.

The recipe above is the intended path; the real sign-up/log-in flow may show something it \
didn't anticipate (an onboarding step, an extra field, a renamed button). The same policy a \
test case's own worker follows applies to you too:
""" + DEVIATION_POLICY + """
If you get blocked by something you cannot finish on your own — an email or SMS verification \
code, a CAPTCHA, two-factor auth, or a required payment — call the `ask_human` tool and say \
exactly what blocked you and what you need (e.g. the code that was emailed). Do not retry, do \
not try variations, and do not look for a way around it yourself. The same applies if the \
given credentials are rejected: try once more only if you can see you actually mistyped or \
left a field empty, otherwise call `ask_human` and report that the credentials were rejected \
rather than guessing at a fix. If a human can't resolve it either within your remaining turns, \
stop calling tools and say so plainly — the test cases that wanted this session will still \
run, just unauthenticated, and a clear report of what blocked you is what makes that fixable.
"""


def _session_key(config: RunnableConfig) -> str:
    return f"{config['configurable']['thread_id']}:{AUTH_SUBJECT_ID}"


async def auth_agent_node(state: AuthState, config: RunnableConfig) -> dict:
    # Checked first, before opening a browser, so the common no-auth-needed run pays
    # nothing for this subgraph — exactly like the node this replaces.
    if not any(tc.requires_auth for tc in state["test_cases"]):
        return {"auth_storage_state": None, "pending_tool_calls": []}

    key = _session_key(config)
    try:
        _, tools, tool_map = await get_session(key)
    except Exception:
        logging.exception(
            "auth_agent_node: failed to open a Playwright session — requires_auth "
            "test cases in this run will fall back to running unauthenticated"
        )
        return {"auth_storage_state": None, "pending_tool_calls": []}

    # `messages` uses the `add_messages` reducer (append-only) — the seed below must be
    # returned alongside the response on turn 1 so it's actually persisted into state,
    # same subtlety documented in nodes/worker/nodes.py's agent_node.
    history = state.get("messages")
    seed: list = []
    if not history:
        try:
            await tool_map["browser_navigate"].ainvoke({"url": state["target_url"]})
        except Exception:
            logging.exception(
                "auth_agent_node: failed to navigate to the target URL — requires_auth "
                "test cases in this run will fall back to running unauthenticated"
            )
            discard_session(key)
            return {"auth_storage_state": None, "pending_tool_calls": []}

        known_context = (
            state.get("discovery_context") or "No existing credentials were provided — sign up a new account."
        )
        seed = [
            SystemMessage(AUTH_SETUP_SYSTEM_PROMPT),
            HumanMessage(
                f"Target URL: {state['target_url']}\nRun token: {state.get('run_token', '')}\n"
                f"Known context (existing credentials, if any): {known_context}"
            ),
        ]
        history = seed

    # Computed once, before the outbound call, and returned below alongside this
    # turn's response — see nodes/agent_loop.py's stale_snapshot_replacements for why
    # this persists the compaction into the checkpoint rather than only shrinking what
    # gets sent to the model.
    replacements = stale_snapshot_replacements(history)
    offered_tools = [t for t in tools if t.name not in EXCLUDED_TOOL_NAMES]

    # WORKER, not PLANNER/VERDICT: this is mechanical sign-up/login execution, the
    # exact kind of hot-loop tool-calling Stage 1 keeps on the cheap model.
    model = with_fallback(
        ModelRole.WORKER, lambda m: m.bind_tools([*offered_tools, ask_human_tool]), temperature=0.3
    )
    response = await model.ainvoke(compact_history(history))

    return {
        "messages": [*seed, *replacements, response],
        "pending_tool_calls": response.tool_calls,
        "turn_count": state.get("turn_count", 0) + 1,
    }


async def auth_tool_node(state: AuthState, config: RunnableConfig) -> dict:
    call, remaining = state["pending_tool_calls"][0], state["pending_tool_calls"][1:]
    key = _session_key(config)

    if call["name"] == ASK_HUMAN_TOOL_NAME:

        async def _tool_map() -> dict:
            _, _, tm = await get_session(key)
            return tm

        reply, _answer_text, _sensitive = await ask_human_and_reply(call, _tool_map, subject_id=AUTH_SUBJECT_ID)
        # No sensitive_answers channel here, unlike WorkerState — a shared-login
        # session's transcript never reaches a graded verdict or leaves this subgraph
        # the way a test case's does, so there's nothing downstream for it to leak into.
        return {"messages": [reply], "pending_tool_calls": remaining}

    decision = await review_if_risky(call, subject_id=AUTH_SUBJECT_ID)
    if decision is not None and not decision.get("approved", False):
        blocked = ToolMessage(
            content=f"Blocked by human reviewer: {decision.get('reason', 'not approved')}",
            tool_call_id=call["id"],
            name=call["name"],
        )
        return {"messages": [blocked], "pending_tool_calls": remaining}

    _, _, tool_map = await get_session(key)
    result = await tool_map[call["name"]].ainvoke(call["args"])
    result_text = truncate_tool_result(str(result))

    return {
        "messages": [ToolMessage(content=result_text, tool_call_id=call["id"], name=call["name"])],
        "pending_tool_calls": remaining,
    }


async def auth_save_node(state: AuthState, config: RunnableConfig) -> dict:
    key = _session_key(config)
    # require_existing=True: mirrors verdict_node exactly (nodes/worker/nodes.py) —
    # this node must never silently open a fresh, unnavigated browser only to "save"
    # ITS blank storage state as if it were the real logged-in session.
    handle, _, tool_map = await get_session(key, require_existing=True)

    auth_storage_state = None
    try:
        run_dir = run_dir_for(key)
        run_dir.mkdir(parents=True, exist_ok=True)
        # Finalizes what get_session's cache-miss path auto-started for this session
        # (start_capture, nodes/worker/session.py). No video counterpart: video is no
        # longer session-length for anything in this codebase (nodes/worker/evidence.py's
        # capture_mutation_clip records one short clip per mutating action instead,
        # inline in tool_node as it happens) — auth_tool_node doesn't do that, matching
        # the original nodes/auth_setup.py's behavior of capturing no evidence at all
        # for the shared-login setup, which is never shown as a graded test result.
        await stop_and_capture(tool_map, "browser_stop_tracing", run_dir / "trace.zip")

        storage_state_tool = tool_map.get("browser_storage_state")
        if storage_state_tool is None:
            logging.error(
                "auth_save_node: browser_storage_state tool missing — is the storage "
                "capability enabled (see mcp/client.py's --config)? requires_auth test "
                "cases in this run will fall back to running unauthenticated"
            )
        else:
            # browser_storage_state saves cookies/local storage to a FILE (confirmed
            # against the installed @playwright/mcp's tool schema — its one param is
            # `filename`, not an inline blob), matching evidence.py's
            # browser_start_video pattern. This same path is carried through
            # QAState.auth_storage_state to agent_node (nodes/worker/nodes.py), which
            # passes it straight back as browser_set_storage_state's `filename`.
            AUTH_STATE_DIR.mkdir(parents=True, exist_ok=True)
            storage_state_path = AUTH_STATE_DIR / f"{config['configurable']['thread_id']}.json"
            await storage_state_tool.ainvoke({"filename": str(storage_state_path)})
            auth_storage_state = str(storage_state_path)
    except Exception:
        logging.exception(
            "auth_save_node: failed to capture the shared login's storage state — "
            "requires_auth test cases in this run will fall back to running unauthenticated"
        )
    finally:
        close_tool = tool_map.get("browser_close")
        if close_tool is not None:
            try:
                await close_tool.ainvoke({})
            except Exception:
                logging.exception("auth_save_node: failed to close the shared-login browser")
        discard_session(key)  # the shared login's browser work is done
        await handle.close()

    return {"auth_storage_state": auth_storage_state}


def route_after_auth_agent(state: AuthState) -> str:
    if not any(tc.requires_auth for tc in state["test_cases"]):
        return END  # auth_agent_node short-circuited above; no session was ever opened
    return "auth_tool_node" if state["pending_tool_calls"] else "auth_save_node"


def route_after_auth_tool(state: AuthState) -> str:
    if state["pending_tool_calls"]:
        return "auth_tool_node"
    return "auth_save_node" if state["turn_count"] >= AUTH_SETUP_MAX_TURNS else "auth_agent_node"


def build_auth_subgraph():
    sub = StateGraph(AuthState)
    sub.add_node("auth_agent_node", auth_agent_node, retry_policy=LLM_RETRY_POLICY)
    # No retry on auth_tool_node — same reasoning as nodes/worker/nodes.py's tool_node:
    # retrying after a raised exception risks re-invoking a tool that already had a
    # real side effect.
    sub.add_node("auth_tool_node", auth_tool_node)
    sub.add_node("auth_save_node", auth_save_node, retry_policy=LLM_RETRY_POLICY)

    sub.add_edge(START, "auth_agent_node")
    sub.add_conditional_edges("auth_agent_node", route_after_auth_agent, ["auth_tool_node", "auth_save_node", END])
    sub.add_conditional_edges(
        "auth_tool_node", route_after_auth_tool, ["auth_tool_node", "auth_agent_node", "auth_save_node"]
    )
    sub.add_edge("auth_save_node", END)

    # No checkpointer passed — inherits the parent graph's, required for interrupt()
    # inside auth_tool_node to actually persist, same as build_worker_subgraph().
    return sub.compile()
