"""Stage 3 — establishes ONE shared logged-in session before the worker fan-out, so
every TestCase with `requires_auth=True` can restore it into its own isolated browser
instead of repeating its own sign-up/login — the ~15-turn tax the plan's decisive-number
table is built around. Runs at most once per run, and only if the approved plan actually
has a requires_auth case (checked first, before opening a browser, so the common
no-auth-needed run pays nothing for this node). A failure here is non-fatal: those cases
just fall back to running unauthenticated (see their own docstring note below for what
that means) rather than failing the whole run over one setup step.
"""
from __future__ import annotations

import logging
import os
from contextlib import AsyncExitStack

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from ..core.llm import ModelRole, get_chat_model
from ..core.state import QAState
from ..mcp.client import open_playwright_session
from .worker.evidence import EVIDENCE_DIR

# Where auth_setup_node saves the shared login's storage state file, keyed by thread_id
# (== run_id) so concurrent runs in this same process never collide. Reusing EVIDENCE_DIR
# rather than a second top-level directory — this is per-run artifact storage exactly like
# evidence.py's screenshot/trace/video files, just not screenshot/trace/video.
AUTH_STATE_DIR = EVIDENCE_DIR / "auth-state"

# Mechanical, single-purpose loop (sign up or log in, once) — same order-of-magnitude
# budget a real test case's own setup detour used to need, before this node existed to
# eliminate paying that cost once per requires_auth case instead of once per run.
AUTH_SETUP_MAX_TURNS = int(os.getenv("AUTH_SETUP_MAX_TURNS", "8"))

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

If you get blocked by something you cannot finish on your own — an email or SMS verification \
code, a CAPTCHA, two-factor auth, or a required payment — STOP CALLING TOOLS IMMEDIATELY and \
say in your final message exactly what blocked you. Do not retry, do not try variations, and \
do not look for a way around it. The same applies if the given credentials are rejected: try \
once more only if you can see you actually mistyped or left a field empty, otherwise stop and \
report that the credentials were rejected. Burning every turn on a wall you cannot get past \
is worse than stopping — the test cases that wanted this session will still run, just \
unauthenticated, and a clear report of what blocked you is what makes that fixable.
"""


async def auth_setup_node(state: QAState, config: RunnableConfig) -> dict:
    if not any(tc.requires_auth for tc in state["test_cases"]):
        return {"auth_storage_state": None}

    known_context = (
        state.get("discovery_context") or "No existing credentials were provided — sign up a new account."
    )

    async with AsyncExitStack() as stack:
        try:
            tools = await open_playwright_session(stack)
        except Exception:
            logging.exception(
                "auth_setup_node: failed to open a Playwright session — requires_auth "
                "test cases in this run will fall back to running unauthenticated"
            )
            return {"auth_storage_state": None}

        tool_map = {t.name: t for t in tools}
        navigate_tool = tool_map.get("browser_navigate")
        storage_state_tool = tool_map.get("browser_storage_state")
        if navigate_tool is None or storage_state_tool is None:
            logging.error(
                "auth_setup_node: browser_navigate or browser_storage_state tool missing — is "
                "--caps=storage set on PLAYWRIGHT_MCP_ARGS? requires_auth test cases in this "
                "run will fall back to running unauthenticated"
            )
            return {"auth_storage_state": None}

        try:
            await navigate_tool.ainvoke({"url": state["target_url"]})

            # WORKER, not PLANNER/VERDICT: this is mechanical sign-up/login execution,
            # the exact kind of hot-loop tool-calling Stage 1 keeps on the cheap model.
            model = get_chat_model(ModelRole.WORKER, temperature=0.3)
            offered_tools = [t for t in tools if t.name != "browser_run_code_unsafe"]
            history: list = [
                SystemMessage(AUTH_SETUP_SYSTEM_PROMPT),
                HumanMessage(
                    f"Target URL: {state['target_url']}\nRun token: {state.get('run_token', '')}\n"
                    f"Known context (existing credentials, if any): {known_context}"
                ),
            ]

            for _ in range(AUTH_SETUP_MAX_TURNS):
                response = await model.bind_tools(offered_tools).ainvoke(history)
                history.append(response)
                if not response.tool_calls:
                    break
                for call in response.tool_calls:
                    tool = tool_map.get(call["name"])
                    result_text = (
                        f"Unknown tool {call['name']!r}"
                        if tool is None
                        else str(await tool.ainvoke(call["args"]))
                    )
                    history.append(ToolMessage(content=result_text, tool_call_id=call["id"], name=call["name"]))

            # browser_storage_state saves cookies/local storage to a FILE (confirmed
            # against the installed @playwright/mcp's tool schema — its one param is
            # `filename`, not an inline blob), matching evidence.py's browser_start_video
            # pattern: give it an absolute path up front, since that's the only place the
            # destination can be set. auth_storage_state below carries that same path
            # (still a plain, JSON-checkpointable string) through to agent_node
            # (nodes/worker/nodes.py), which passes it straight back as
            # browser_set_storage_state's `filename` to restore from it.
            AUTH_STATE_DIR.mkdir(parents=True, exist_ok=True)
            storage_state_path = AUTH_STATE_DIR / f"{config['configurable']['thread_id']}.json"
            await storage_state_tool.ainvoke({"filename": str(storage_state_path)})
            auth_storage_state = str(storage_state_path)
        except Exception:
            logging.exception(
                "auth_setup_node: shared login attempt failed — requires_auth test cases "
                "in this run will fall back to running unauthenticated"
            )
            return {"auth_storage_state": None}

    return {"auth_storage_state": auth_storage_state}
