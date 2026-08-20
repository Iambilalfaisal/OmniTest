"""Playwright MCP session lifecycle: one persistent session per (thread, test case),
opened lazily by `get_session` and cached process-locally in `_SESSIONS` so
agent_node/tool_node/verdict_node — each a separate LangGraph node invocation — read
and write through the SAME browser across however many turns one test case's
tool-calling loop takes.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack

from langchain_core.runnables import RunnableConfig

from ...mcp.client import open_playwright_session
from .evidence import discard_action_overlay, start_capture


class SessionGoneError(Exception):
    """Raised by `get_session(key, require_existing=True)` when the session isn't
    cached. Used only by verdict_node, which must NEVER transparently open a fresh
    browser on a retry — doing so silently re-snapshots a blank page and re-runs the
    verdict LLM against it, producing a plausible-looking WRONG verdict (a confirmed
    failure mode — see verdict_node's own comments for the two real incidents this
    class of bug caused). Every other caller (agent_node, tool_node) keeps the existing
    lenient default (`require_existing=False`), since a fresh session there is a
    legitimate, if degraded, recovery path for a resume landing on a different process
    than the one that paused (see get_session's docstring below).
    """


class SessionHandle:
    """Owns one persistent Playwright MCP session's open/close lifecycle on a single
    dedicated asyncio task.

    Required because anyio's task groups (used internally by the MCP stdio transport)
    enforce that a cancel scope's __aenter__ and __aexit__ run on the SAME task — but
    LangGraph runs agent_node/tool_node/verdict_node as separate tasks (each is its own
    `asyncio.create_task(...)` internally), so a plain AsyncExitStack opened in one node
    and closed in another raises "Attempted to exit cancel scope in a different task
    than it was entered in". Reading/writing through an already-open session isn't
    affected by this — only the open/close pair is task-locked — so agent_node/
    tool_node/verdict_node still call `.ainvoke()` on `tools` directly from their own
    separate tasks; only opening and closing happens inside `_run`, both on the one
    dedicated task that owns this handle.
    """

    def __init__(self) -> None:
        self.tools: list = []
        self.error: BaseException | None = None
        self._ready = asyncio.Event()
        self._close_requested = asyncio.Event()
        self._closed = asyncio.Event()
        self._task = asyncio.create_task(self._run())  # reference kept for GC safety

    async def _run(self) -> None:
        try:
            async with AsyncExitStack() as stack:
                self.tools = await open_playwright_session(stack)
                self._ready.set()
                await self._close_requested.wait()
        except Exception as exc:  # noqa: BLE001 - re-raised to the waiting caller below
            self.error = exc
            self._ready.set()
        finally:
            self._closed.set()

    async def wait_ready(self) -> list:
        await self._ready.wait()
        if self.error is not None:
            raise self.error
        return self.tools

    async def close(self) -> None:
        self._close_requested.set()
        await self._closed.wait()


# "{thread_id}:{test_id}" -> (handle, tools, tool_map). A live Playwright/MCP subprocess
# can't be checkpointed (not serializable), so it's cached process-locally and shared by
# agent_node/tool_node/verdict_node across however many separate node invocations one
# test case's tool-calling loop takes. A cache miss transparently opens a fresh browser
# rather than erroring for agent_node/tool_node's default (lenient) get_session call —
# see the plan's "accepted limitations" for what that means for a risky-action resume
# landing on a different process than the one that paused. verdict_node instead calls
# get_session(key, require_existing=True) and gets a loud SessionGoneError on a miss —
# see that class's docstring for why a fresh/blank browser is never an acceptable
# substitute there.
_SESSIONS: dict[str, tuple] = {}


def session_key(config: RunnableConfig, test_id: str) -> str:
    return f"{config['configurable']['thread_id']}:{test_id}"


async def get_session(key: str, *, require_existing: bool = False) -> tuple:
    if key not in _SESSIONS:
        if require_existing:
            raise SessionGoneError(key)
        handle = SessionHandle()
        tools = await handle.wait_ready()
        tool_map = {t.name: t for t in tools}
        await start_capture(tool_map, key)
        _SESSIONS[key] = (handle, tools, tool_map)
    return _SESSIONS[key]


def discard_session(key: str) -> None:
    """Drop the cache entry for a session verdict_node is about to close itself —
    separate from close_sessions_for_thread's pop+close since verdict_node already
    holds the handle and closes it directly.
    """
    _SESSIONS.pop(key, None)
    discard_action_overlay(key)


async def close_sessions_for_thread(thread_id: str) -> None:
    """Best-effort cleanup for a run that ended — successfully or via a crash — without
    every branch reaching verdict_node's own cleanup above. Without this, a crash
    anywhere in agent_node/tool_node (e.g. the retries-exhausted case) would leak that
    test case's browser subprocess forever, since nothing else ever closes it.
    Called from api.py's `_drive()` after every run, success or failure; a no-op on the
    happy path since verdict_node already popped and closed its own session by then.
    """
    prefix = f"{thread_id}:"
    for key in [k for k in _SESSIONS if k.startswith(prefix)]:
        handle, _, _ = _SESSIONS.pop(key)
        discard_action_overlay(key)
        try:
            await handle.close()
        except Exception:
            logging.exception("failed to close leaked Playwright session for %s", key)
