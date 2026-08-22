"""Playwright MCP session lifecycle: one persistent session per (thread, test case),
opened lazily by `get_session` and cached process-locally in `_SESSIONS` so
agent_node/tool_node/verdict_node — each a separate LangGraph node invocation — read
and write through the SAME browser across however many turns one test case's
tool-calling loop takes.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Collection
from contextlib import AsyncExitStack

from langchain_core.runnables import RunnableConfig

from ...mcp.client import open_playwright_session
from .evidence import discard_action_overlay, start_capture

# Idle TTL for a cached session. Exists because api.py's `_drive()` deliberately does
# NOT sweep sessions on a pause (see the `if finished:` guard there) — a paused run's
# browser must stay alive for the human to answer against. Without a backstop, a pause
# that is never resumed (the human closes the tab, abandons the run) would pin its
# `npx @playwright/mcp` subprocess and Chromium for the life of the process. 30 minutes
# is far past any realistic time-to-answer and far short of "forever".
SESSION_IDLE_TTL_SECONDS = float(os.getenv("SESSION_IDLE_TTL_SECONDS", "1800"))
SESSION_REAP_INTERVAL_SECONDS = float(os.getenv("SESSION_REAP_INTERVAL_SECONDS", "300"))

# Wall-clock bound on OPENING a session (spawning `npx @playwright/mcp`, launching
# Chromium, completing the MCP handshake) — separate from TOOL_CALL_TIMEOUT_SECONDS
# (agent_loop.py), which bounds a call made THROUGH an already-open session. A slow
# first-time `npx` package fetch or a wedged subprocess previously had no bound at all:
# `SessionHandle.wait_ready()` just awaited an `asyncio.Event` that only a successful
# (or failed) `_run()` ever sets.
SESSION_OPEN_TIMEOUT_SECONDS = float(os.getenv("SESSION_OPEN_TIMEOUT_SECONDS", "60"))


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

    async def wait_ready(self, *, timeout: float | None = None) -> list:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # `close()` can't abandon a stuck open: it only sets `_close_requested`,
            # which `_run()` doesn't check until AFTER `open_playwright_session`
            # returns — so a genuinely wedged open would never see it. Cancelling
            # `_task` directly interrupts whatever `_run()` is awaiting; its
            # `AsyncExitStack` still unwinds normally on cancellation.
            await self.cancel()
            raise
        if self.error is not None:
            raise self.error
        return self.tools

    async def cancel(self) -> None:
        """Abandons an in-flight open that's taking too long (see `wait_ready`'s
        timeout above). Idempotent — safe to call again from `close()`/`get_session`'s
        cleanup even after this already ran."""
        self._task.cancel()
        try:
            await self._task
        except BaseException:  # noqa: BLE001 - task cancellation/any startup failure, already abandoning it
            pass
        self._closed.set()

    async def close(self) -> None:
        self._close_requested.set()
        await self._closed.wait()


# "{thread_id}:{test_id}" -> (handle, tools, tool_map). A live Playwright/MCP subprocess
# can't be checkpointed (not serializable), so it's cached process-locally and shared by
# agent_node/tool_node/verdict_node across however many separate node invocations one
# test case's tool-calling loop takes. A cache miss transparently opens a fresh browser
# rather than erroring for agent_node/tool_node's default (lenient) get_session call —
# see the plan's "accepted limitations" for what that means for a risky-action resume
# landing on a different process than the one that paused, OR a resume after
# SESSION_IDLE_TTL_SECONDS elapsed (below). verdict_node instead calls
# get_session(key, require_existing=True) and gets a loud SessionGoneError on a miss —
# see that class's docstring for why a fresh/blank browser is never an acceptable
# substitute there.
_SESSIONS: dict[str, tuple] = {}

# key -> time.monotonic() of the last get_session() hit. Kept OUT of the _SESSIONS
# tuple deliberately — that tuple is unpacked positionally at four call sites in
# nodes.py, and widening it here would touch all of them for a bookkeeping field none
# of them read (same reasoning evidence.py's _action_overlay_enabled set already uses).
# monotonic(), not time(), so a system clock change can't make a live session look stale.
_LAST_USED: dict[str, float] = {}


def session_key(config: RunnableConfig, test_id: str) -> str:
    return f"{config['configurable']['thread_id']}:{test_id}"


async def get_session(key: str, *, require_existing: bool = False) -> tuple:
    if key not in _SESSIONS:
        if require_existing:
            raise SessionGoneError(key)
        # Detector for the class of bug this module used to have (a session silently
        # destroyed and replaced mid-test-case, e.g. by api.py sweeping on a pause
        # instead of only on a finished run): two of these lines for the SAME key
        # within one run's log means exactly that happened.
        logging.info("opening Playwright session for %s", key)
        handle = SessionHandle()
        try:
            tools = await handle.wait_ready(timeout=SESSION_OPEN_TIMEOUT_SECONDS)
            tool_map = {t.name: t for t in tools}
            await start_capture(tool_map, key)
        except Exception:
            # Pre-existing leak this fix closes as a side effect: on any failure here
            # (not just the new open-timeout — start_capture failing too), `handle`
            # was never stored in `_SESSIONS` and therefore never closed by anything —
            # an orphaned `npx @playwright/mcp` subprocess/Chromium for the life of the
            # process. `close()` is safe even if `wait_ready` already cancelled it
            # (idempotent, see `SessionHandle.cancel`).
            await handle.close()
            raise
        _SESSIONS[key] = (handle, tools, tool_map)
    _LAST_USED[key] = time.monotonic()
    return _SESSIONS[key]


def discard_session(key: str) -> None:
    """Drop the cache entry for a session verdict_node is about to close itself —
    separate from close_sessions_for_thread's pop+close since verdict_node already
    holds the handle and closes it directly.
    """
    _SESSIONS.pop(key, None)
    _LAST_USED.pop(key, None)
    discard_action_overlay(key)


async def _close_key(key: str) -> None:
    handle, _, _ = _SESSIONS.pop(key)
    _LAST_USED.pop(key, None)
    discard_action_overlay(key)
    try:
        await handle.close()
    except Exception:
        logging.exception("failed to close leaked Playwright session for %s", key)


async def close_sessions_for_thread(thread_id: str) -> None:
    """Best-effort cleanup for a run that ENDED — successfully or via a crash — without
    every branch reaching verdict_node's own cleanup above. Without this, a crash
    anywhere in agent_node/tool_node (e.g. the retries-exhausted case) would leak that
    test case's browser subprocess forever, since nothing else ever closes it.

    Called from api.py's `_drive()` only when the run is actually over — NOT when it
    merely paused for a human-in-the-loop interrupt. A paused worker resumes into this
    SAME cached session (tool_node's interrupt() calls sit before their get_session, so
    the pause never repopulates the cache); sweeping it here on a pause used to close
    the very browser the resumed node needed, silently replacing it with a blank one
    and clobbering the in-progress video recording — see this module's SessionGoneError
    docstring and api.py's `_drive()` for the guard that now prevents that. A no-op on
    the happy path since verdict_node already popped and closed its own session by then.
    """
    prefix = f"{thread_id}:"
    for key in [k for k in _SESSIONS if k.startswith(prefix)]:
        await _close_key(key)


async def close_idle_sessions(
    *, ttl_seconds: float = SESSION_IDLE_TTL_SECONDS, exempt_threads: Collection[str] = ()
) -> int:
    """Bounded backstop for the one leak `close_sessions_for_thread` no longer covers:
    a run that paused for a human and was then never resumed. `exempt_threads` should
    hold the thread_ids of every run with a live `_drive()` task — a long Gemini backoff
    inside a single node can legitimately go quiet for minutes, and reaping a session
    out from under an executing (not paused) node would recreate the exact
    blank-browser failure this whole mechanism exists to prevent. thread_id is a bare
    uuid4 (api.py) so it never contains ':' — split(":", 1)[0] is exact.
    """
    cutoff = time.monotonic() - ttl_seconds
    stale = [
        k
        for k in _SESSIONS
        if _LAST_USED.get(k, 0.0) < cutoff and k.split(":", 1)[0] not in exempt_threads
    ]
    for key in stale:
        logging.warning(
            "reaping Playwright session %s — idle > %.0fs (a paused run that was never "
            "resumed in time; resuming it now would have landed on a fresh, "
            "unnavigated browser anyway)",
            key,
            ttl_seconds,
        )
        await _close_key(key)
    return len(stale)


async def close_all_sessions() -> None:
    """Process-shutdown sweep: everything still cached, including paused runs — nothing
    here can survive the process. An unclosed handle means an orphaned
    `npx @playwright/mcp` subprocess (and Chromium) outliving the server.
    """
    for key in list(_SESSIONS):
        await _close_key(key)
