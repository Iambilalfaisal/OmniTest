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
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from ...core.llm import LLM_RETRY_POLICY, ModelRole, with_fallback
from ...mcp.client import extract_eval_value, invoke_tool, invoke_tool_or_error_text
from ..agent_loop import (
    ASK_HUMAN_TOOL_NAME,
    DEVIATION_POLICY,
    EXCLUDED_TOOL_NAMES,
    SCENARIO_DEADLINE_SECONDS,
    ask_human_and_reply,
    ask_human_tool,
    compact_history,
    new_deadline,
    review_if_risky,
    stale_snapshot_replacements,
    truncate_tool_result,
)
from ..worker.evidence import EVIDENCE_DIR, run_dir_for, stop_and_capture
from ..worker.session import SessionGoneError, discard_session, get_session
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


# A password-type textbox still visible on screen is the strongest, lowest-false-positive
# signal available that the shared-login attempt did NOT actually reach an authenticated
# state — a real app's authenticated pages essentially never render a live password input,
# regardless of the site's own markup/URL conventions (unlike a URL-path check, which
# would need per-site guessing). Matches Playwright's ARIA snapshot format, e.g.
# `- textbox "Password" [ref=e6]`; role and accessible name can appear in either order
# depending on the control's markup, so both orderings are checked.
_PASSWORD_FIELD_RE = re.compile(
    r"textbox[^\n]{0,80}password|password[^\n]{0,80}textbox", re.IGNORECASE
)


async def _looks_unauthenticated(tool_map: dict) -> bool:
    """Best-effort check used by auth_save_node right before it would otherwise trust
    whatever session state exists as "the shared login." Returns False (i.e. "go ahead
    and save it") on any error taking the snapshot — this check exists to catch a KNOWN
    failure mode (saving a still-on-the-login-page session as if it were authenticated),
    not to become a new reason a session that might genuinely be fine gets discarded.
    """
    try:
        snapshot = await invoke_tool_or_error_text(tool_map["browser_snapshot"], {})
    except Exception:
        return False
    return bool(_PASSWORD_FIELD_RE.search(snapshot))


async def auth_agent_node(state: AuthState, config: RunnableConfig) -> dict:
    # Checked first, before opening a browser, so the common no-auth-needed run pays
    # nothing for this subgraph — exactly like the node this replaces.
    if not any(tc.requires_auth for tc in state["test_cases"]):
        return {"auth_storage_state": None, "authenticated_landing_url": None, "pending_tool_calls": []}

    # Wall-clock backstop (agent_loop.SCENARIO_DEADLINE_SECONDS), mirroring
    # nodes/worker/nodes.py's agent_node exactly — see that copy's comment for why this
    # bounds the SUM across turns rather than duplicating TOOL_CALL_TIMEOUT_SECONDS'
    # per-call bound. auth_save_node turns abort_reason into a clean, unauthenticated
    # fallback instead of leaving this subgraph stuck.
    now = time.monotonic()
    deadline_at = state.get("deadline_at")
    if deadline_at is None:
        deadline_at = now + SCENARIO_DEADLINE_SECONDS
    elif now > deadline_at:
        return {"pending_tool_calls": [], "abort_reason": f"exceeded its {SCENARIO_DEADLINE_SECONDS:.0f}s execution deadline"}

    key = _session_key(config)
    try:
        _, tools, tool_map = await get_session(key)
    except Exception:
        logging.exception(
            "auth_agent_node: failed to open a Playwright session — requires_auth "
            "test cases in this run will fall back to running unauthenticated"
        )
        return {"auth_storage_state": None, "authenticated_landing_url": None, "pending_tool_calls": []}

    # `messages` uses the `add_messages` reducer (append-only) — the seed below must be
    # returned alongside the response on turn 1 so it's actually persisted into state,
    # same subtlety documented in nodes/worker/nodes.py's agent_node.
    history = state.get("messages")
    seed: list = []
    if not history:
        try:
            await invoke_tool(tool_map["browser_navigate"], {"url": state["target_url"]})
        except Exception:
            logging.exception(
                "auth_agent_node: failed to navigate to the target URL — requires_auth "
                "test cases in this run will fall back to running unauthenticated"
            )
            discard_session(key)
            return {"auth_storage_state": None, "authenticated_landing_url": None, "pending_tool_calls": []}

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
        "deadline_at": deadline_at,
    }


async def auth_tool_node(state: AuthState, config: RunnableConfig) -> dict:
    call, remaining = state["pending_tool_calls"][0], state["pending_tool_calls"][1:]
    key = _session_key(config)

    if call["name"] == ASK_HUMAN_TOOL_NAME:

        async def _tool_map() -> dict:
            _, _, tm = await get_session(key)
            return tm

        # get_session runs INSIDE ask_human_and_reply, only after interrupt() returns
        # — a failure there would otherwise propagate uncaught out of this node and
        # crash the whole run (see nodes/worker/nodes.py's tool_node, same reasoning).
        try:
            reply, _answer_text, _sensitive = await ask_human_and_reply(call, _tool_map, subject_id=AUTH_SUBJECT_ID)
        except Exception as exc:
            logging.exception("auth_tool_node: failed to open a session resuming ask_human — aborting auth setup")
            return {"pending_tool_calls": [], "abort_reason": f"could not open a browser session ({exc})"}
        # No sensitive_answers channel here, unlike WorkerState — a shared-login
        # session's transcript never reaches a graded verdict or leaves this subgraph
        # the way a test case's does, so there's nothing downstream for it to leak into.
        # deadline_at is pushed forward a fresh window (agent_loop.new_deadline) — a
        # real interrupt()/resume just happened, so the wall-clock gap was human
        # response time, not runaway execution (see agent_node's identical reasoning).
        return {"messages": [reply], "pending_tool_calls": remaining, "deadline_at": new_deadline()}

    decision = await review_if_risky(call, subject_id=AUTH_SUBJECT_ID)
    deadline_update = {"deadline_at": new_deadline()} if decision is not None else {}
    if decision is not None and not decision.get("approved", False):
        blocked = ToolMessage(
            content=f"Blocked by human reviewer: {decision.get('reason', 'not approved')}",
            tool_call_id=call["id"],
            name=call["name"],
        )
        return {"messages": [blocked], "pending_tool_calls": remaining, **deadline_update}

    try:
        _, _, tool_map = await get_session(key)
    except Exception as exc:
        logging.exception("auth_tool_node: failed to open a session — aborting auth setup")
        return {"pending_tool_calls": [], "abort_reason": f"could not open a browser session ({exc})"}
    # Tool NAME only, never call["args"] — args are exactly where the credentials being
    # typed into the login form live. Diagnostic-only: without this, the shared-login
    # attempt's own turn-by-turn trajectory was completely invisible in the console log
    # (its `messages` never reach any report — see this module's own docstring), so a
    # run that burned its full AUTH_SETUP_MAX_TURNS budget gave no way to tell whether
    # it was genuinely stuck, looping, or just slow, short of replaying a video/trace.
    logging.info(
        "auth_tool_node: run %s turn %d calling %s",
        config["configurable"]["thread_id"],
        state.get("turn_count", 0),
        call["name"],
    )
    # invoke_tool_or_error_text, not invoke_tool: this IS the tool-dispatch point, with
    # no enclosing try/except — a raise here would crash the whole run, not just fail
    # this subgraph (see mcp.client.invoke_tool's docstring).
    result_text = truncate_tool_result(await invoke_tool_or_error_text(tool_map[call["name"]], call["args"]))

    return {
        "messages": [ToolMessage(content=result_text, tool_call_id=call["id"], name=call["name"])],
        "pending_tool_calls": remaining,
        **deadline_update,
    }


async def auth_save_node(state: AuthState, config: RunnableConfig) -> dict:
    key = _session_key(config)
    # require_existing=True: mirrors verdict_node exactly (nodes/worker/nodes.py) —
    # this node must never silently open a fresh, unnavigated browser only to "save"
    # ITS blank storage state as if it were the real logged-in session.
    #
    # SessionGoneError is caught here, not left to propagate: this node carries
    # LLM_RETRY_POLICY, and node_retry_on (core/llm.py) explicitly refuses to retry
    # SessionGoneError, so an uncaught raise here crashes the WHOLE run immediately
    # (confirmed against the installed langgraph 1.2.11 — one Send-spawned branch's
    # uncaught exception aborts every concurrently-running branch, not just this
    # subgraph). This is a genuine pre-existing gap, not just a new abort_reason path:
    # auth_agent_node's OWN existing get_session failure (a few lines up, unrelated to
    # anything new here) already returns early with no session ever cached — which
    # used to reach this exact line and crash the run despite that comment's promise
    # of "falls back to running unauthenticated". Catching it here is what actually
    # makes that promise true, for that pre-existing path as well as the new
    # deadline/session-open-timeout abort_reason paths below.
    try:
        handle, _, tool_map = await get_session(key, require_existing=True)
    except SessionGoneError:
        return {"auth_storage_state": None, "authenticated_landing_url": None}

    abort_reason = state.get("abort_reason")
    if abort_reason:
        logging.warning(
            "auth_save_node: aborting shared-login setup for run %s — %s",
            config["configurable"]["thread_id"],
            abort_reason,
        )
        close_tool = tool_map.get("browser_close")
        if close_tool is not None:
            try:
                await invoke_tool(close_tool, {})
            except Exception:
                logging.exception("auth_save_node: failed to close the shared-login browser while aborting")
        discard_session(key)
        await handle.close()
        return {"auth_storage_state": None, "authenticated_landing_url": None}

    auth_storage_state = None
    authenticated_landing_url = None
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

        # Logged unconditionally regardless of outcome below — closes the observability
        # gap that made two separate live runs indistinguishable from the console alone:
        # did the shared login actually reach an authenticated page, and if so, which
        # URL? extract_eval_value (mcp/client.py), not invoke_tool_or_error_text's plain
        # str() — that stringifies the raw content-block list through repr(), which
        # mangles every real newline in the wrapped "### Result" text into a literal
        # backslash-n, confirmed live to make the value unparseable downstream. Best-
        # effort: any failure here must never block the real save/no-save decision below.
        try:
            final_url = extract_eval_value(
                await invoke_tool(tool_map["browser_evaluate"], {"function": "() => window.location.href"})
            )
        except Exception:
            final_url = "(could not read)"
        logging.info(
            "auth_save_node: run %s shared-login attempt ended at %s",
            config["configurable"]["thread_id"],
            final_url,
        )

        storage_state_tool = tool_map.get("browser_storage_state")
        if storage_state_tool is None:
            logging.error(
                "auth_save_node: browser_storage_state tool missing — is the storage "
                "capability enabled (see mcp/client.py's --config)? requires_auth test "
                "cases in this run will fall back to running unauthenticated"
            )
        elif await _looks_unauthenticated(tool_map):
            # CONFIRMED live: this node used to save+return whatever session state existed
            # unconditionally, with no check that the login/signup attempt actually
            # succeeded — a login that failed (wrong/missing credentials, a CAPTCHA, a
            # site quirk the auth agent couldn't get past within AUTH_SETUP_MAX_TURNS)
            # still produced a non-null auth_storage_state pointing at an unauthenticated
            # session. Every requires_auth worker that restored it then got told "you are
            # ALREADY logged in" (nodes/worker/nodes.py's agent_node) while actually
            # looking at a login page it had no steps of its own to handle, with only
            # ask_human left as a legitimate way out. Refusing to save here instead routes
            # those cases through the ALREADY-correct unauthenticated degrade path.
            logging.warning(
                "auth_save_node: the shared-login attempt for run %s ended on what still "
                "looks like a login/signup screen (a password field is still visible) — "
                "not saving this as a valid session; requires_auth test cases in this run "
                "will fall back to running unauthenticated",
                config["configurable"]["thread_id"],
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
            await invoke_tool(storage_state_tool, {"filename": str(storage_state_path)})
            auth_storage_state = str(storage_state_path)
            # Only set alongside a genuine save, in this same branch — restoring valid
            # cookies into a requires_auth worker's browser does not guarantee its own
            # steps' target_url shows authenticated content (confirmed live: a site whose
            # root path is a public/marketing page regardless of session, only
            # recognizing auth under a deeper path like "/dashboard"). agent_node
            # (nodes/worker/nodes.py) uses this to redirect that worker's own first
            # navigate step here instead, when it looks like a plain navigate to target_url.
            if final_url and final_url != "(could not read)":
                authenticated_landing_url = final_url
    except Exception:
        logging.exception(
            "auth_save_node: failed to capture the shared login's storage state — "
            "requires_auth test cases in this run will fall back to running unauthenticated"
        )
    finally:
        close_tool = tool_map.get("browser_close")
        if close_tool is not None:
            try:
                await invoke_tool(close_tool, {})
            except Exception:
                logging.exception("auth_save_node: failed to close the shared-login browser")
        discard_session(key)  # the shared login's browser work is done
        await handle.close()

    return {"auth_storage_state": auth_storage_state, "authenticated_landing_url": authenticated_landing_url}


def route_after_auth_agent(state: AuthState) -> str:
    if not any(tc.requires_auth for tc in state["test_cases"]):
        return END  # auth_agent_node short-circuited above; no session was ever opened
    return "auth_tool_node" if state["pending_tool_calls"] else "auth_save_node"


def route_after_auth_tool(state: AuthState) -> str:
    # Checked first: an abort (session-open timeout) leaves pending_tool_calls empty
    # with turn_count possibly still under AUTH_SETUP_MAX_TURNS — without this, that
    # would route back to auth_agent_node instead of auth_save_node's abort handling,
    # same reasoning as nodes/worker/nodes.py's route_after_tool.
    if state.get("abort_reason"):
        return "auth_save_node"
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
