"""Per-caller MultiServerMCPClient factory for the Playwright MCP server.

The planner and every parallel `worker_node` branch each call `create_playwright_client()`
to get their own client — and therefore their own `npx @playwright/mcp` subprocess /
browser instance — instead of sharing one singleton, so parallel branches never race on
the same tab.
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import AsyncExitStack
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# Wall-clock bound on a SINGLE MCP tool call (browser_click, browser_navigate, ...).
# Before this, nothing anywhere in this codebase bounded how long a stalled browser/page
# could hold a node — and therefore its MAX_CONCURRENT_WORKERS slot — since
# `tool.ainvoke()` was called bare everywhere. `asyncio.TimeoutError` IS `TimeoutError`
# on the installed Python (3.12) and is already an `Exception` subclass, so every
# existing `except Exception` around a tool call (or the deliberate lack of one, where a
# raise is expected to propagate) keeps behaving exactly as it does for any other tool
# failure — this only bounds how long that failure can take to surface, converting an
# unbounded hang into a bounded one.
TOOL_CALL_TIMEOUT_SECONDS = float(os.getenv("TOOL_CALL_TIMEOUT_SECONDS", "90"))


async def invoke_tool(tool: BaseTool, args: dict, *, timeout: float = TOOL_CALL_TIMEOUT_SECONDS):
    """The ONE way any MCP tool is called in this codebase — every bare `tool.ainvoke()`
    call site was replaced with this. Raises `TimeoutError` (== `asyncio.TimeoutError`)
    after `timeout` seconds instead of hanging forever; callers that already wrap their
    call in `except Exception` (or a caller like planner_node that's meant to fail the
    whole node/run on any tool error) get identical behavior to today, just time-bounded.

    NOT safe to use at a genuine tool-dispatch point inside `nodes/worker/nodes.py`'s
    `tool_node` or `nodes/auth/nodes.py`'s `auth_tool_node` — a raise there is uncaught
    and propagates out of the node; LangGraph's executor re-raises the first exception
    across ALL concurrently-running Send-spawned branches on exit (confirmed against the
    installed langgraph 1.2.11's `PregelExecutor`), so one leaf's tool timeout would
    crash the WHOLE run, including every sibling test case's already-good results. Use
    `invoke_tool_or_error_text` there instead.
    """
    return await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)


async def invoke_tool_or_error_text(tool: BaseTool, args: dict, *, timeout: float = TOOL_CALL_TIMEOUT_SECONDS) -> str:
    """Same as `invoke_tool`, but NEVER raises — a timeout becomes descriptive text
    instead, exactly like `langchain_mcp_adapters`' `handle_tool_errors=True` (below,
    `open_playwright_session`) already converts a genuine MCP-side tool error into text
    fed back to the model rather than a raised exception. Used at the actual
    tool-dispatch point inside tool_node/auth_tool_node (and agent_loop.py's
    ask_human_and_reply post-resume snapshot fetch) — the places a raise would crash
    the whole run per `invoke_tool`'s docstring — so a stuck browser reads to the model
    as "that action failed," letting the existing turn budget / stuck-nudge machinery
    handle it as any other tool error, instead of taking down every sibling test case.
    """
    try:
        return str(await invoke_tool(tool, args, timeout=timeout))
    except asyncio.TimeoutError:
        return f"[tool call '{tool.name}' timed out after {timeout:.0f}s — the page or browser may be unresponsive]"


PLAYWRIGHT_MCP_COMMAND = os.getenv("PLAYWRIGHT_MCP_COMMAND", "npx")
# --isolated is required, not optional, given our design: @playwright/mcp defaults to a
# PERSISTENT shared browser profile, but we spin up one subprocess per parallel worker —
# without --isolated, concurrent workers would fight over the same profile directory
# instead of getting independent browsers.
# --headless is required, not optional, for our product: @playwright/mcp launches a real,
# visible browser window on the host desktop by default ("headed") — every automated
# session (planner crawl, each worker, each discovery-chat dive) would otherwise pop its
# own Chromium window in front of whatever the person running this is doing. Evidence
# capture (screenshot/trace/video, evidence.py) works identically headless — nothing is
# lost by hiding it.
# See https://playwright.dev/docs/getting-started-mcp and https://github.com/microsoft/playwright-mcp.
PLAYWRIGHT_MCP_ARGS = os.getenv("PLAYWRIGHT_MCP_ARGS", "-y @playwright/mcp@latest --isolated --headless").split()

# devtools opts into browser_start_tracing/browser_start_video (+ stop_ variants) for
# nodes/worker.py's evidence capture, plus a few extra low-risk tools (browser_annotate,
# browser_highlight, video-chapter tools) as a side effect, since we bind everything
# indiscriminately. storage opts into browser_storage_state/browser_set_storage_state,
# used by nodes/auth/nodes.py (Stage 3) to capture one shared login once and restore it
# into each requires_auth TestCase's own isolated browser, instead of every such case
# paying its own ~15-turn signup/login tax.
#
# CONFIRMED live against the installed @playwright/mcp (0.0.79): passing MULTIPLE
# capabilities via `--caps=devtools,storage` (or two repeated `--caps` flags) silently
# yields NEITHER — comma-joined values aren't split, and repeated flags overwrite rather
# than accumulate. Verified directly, tool-by-tool: `--caps=storage` alone correctly
# exposes all 17 storage/cookie tools; `--caps=devtools` alone correctly exposes all 7
# tracing/video tools; `--caps=devtools,storage` together exposes ZERO of either, falling
# silently back to the 24-tool default — this is exactly what made every requires_auth
# TestCase run unauthenticated and every trace/video capture silently no-op. The `--config
# <path>` JSON form does NOT share this bug (confirmed the same way): its `capabilities`
# array combines correctly, so that's what's used here instead of `--caps=...`.
_CONFIG_PATH = Path(__file__).resolve().parent / "playwright-mcp-config.json"
PLAYWRIGHT_MCP_ARGS = [*PLAYWRIGHT_MCP_ARGS, f"--config={_CONFIG_PATH}"]


def create_playwright_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "playwright": {
                "command": PLAYWRIGHT_MCP_COMMAND,
                "args": PLAYWRIGHT_MCP_ARGS,
                "transport": "stdio",
            }
        }
    )


async def open_playwright_session(stack: AsyncExitStack) -> list:
    """Open ONE persistent MCP session for the caller's whole browser interaction and
    return its tools, registering the session on `stack` for cleanup.

    `MultiServerMCPClient.get_tools()` — the simple, session-less API — opens a
    brand-new session (and, for our stdio transport, a brand-new subprocess/browser)
    for every single tool call; that's its documented behavior ("A new session will be
    created for each tool call"). That's wrong for us: navigating in one call and then
    clicking/snapshotting in the next needs to see the SAME browser, and a start/stop
    pair like tracing or video recording needs to run against one continuous session —
    otherwise "stop" just operates on an unrelated, blank browser that never started
    anything. This uses `client.session(...)` + `load_mcp_tools(session)` instead,
    matching `MultiServerMCPClient`'s own "explicitly starting a session" pattern, so
    every tool returned here shares the one session until `stack` is closed.
    """
    client = create_playwright_client()
    session = await stack.enter_async_context(client.session("playwright"))
    return await load_mcp_tools(session, handle_tool_errors=True)


async def get_accessibility_snapshot(tools: list, url: str) -> dict:
    """Navigate to `url` and return the page's accessibility tree snapshot."""
    tool_map = {tool.name: tool for tool in tools}
    await invoke_tool(tool_map["browser_navigate"], {"url": url})
    return await invoke_tool(tool_map["browser_snapshot"], {})


def _coerce_text(result) -> str:
    """MCP tool results surface through langchain_mcp_adapters, whose exact wire shape
    isn't nailed down anywhere else in this codebase either — get_accessibility_snapshot
    above is already annotated `-> dict` while being fed straight into a prompt's
    `.format()` call. Treat this the same way core/memory.py's own TODO(verify) treats an
    analogous uncertainty: this assumes `.ainvoke()` returns (or stringifies cleanly to)
    plain text; confirm against the installed langchain-mcp-adapters version if a crawl
    digest reads oddly in practice.
    """
    return result if isinstance(result, str) else str(result)


def _coerce_json_array(result) -> list[dict]:
    try:
        parsed = json.loads(_coerce_text(result))
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


async def navigate(tools: list, url: str) -> None:
    """Navigate without snapshotting — reused by planner_explore.crawl_site's page visits."""
    tool_map = {tool.name: tool for tool in tools}
    await invoke_tool(tool_map["browser_navigate"], {"url": url})


async def shallow_snapshot(tools: list, *, depth: int) -> str:
    """Depth-limited accessibility snapshot of the CURRENT page — used for a crawl's
    per-page digest instead of a full tree, which would blow up the planning prompt
    across several pages. `browser_snapshot`'s `depth` param limits the snapshot tree's
    depth (confirmed against the installed @playwright/mcp); planner_explore.py
    additionally hard-truncates the result by character count regardless, as a backstop.
    """
    tool_map = {tool.name: tool for tool in tools}
    result = await invoke_tool(tool_map["browser_snapshot"], {"depth": depth})
    return _coerce_text(result)


async def get_page_title(tools: list) -> str:
    tool_map = {tool.name: tool for tool in tools}
    result = await invoke_tool(tool_map["browser_evaluate"], {"function": "() => document.title"})
    return _coerce_text(result).strip()


async def list_page_links(tools: list) -> list[dict]:
    """Read-only DOM query enumerating <a href> elements on the CURRENT page as
    [{"text": ..., "href": <absolute URL>}, ...]. Deliberately NOT derived from the
    accessibility-tree snapshot, which omits link hrefs entirely (only role/name/ref) —
    link discovery needs the actual URL to follow.

    Uses browser_evaluate with a FIXED, hardcoded, side-effect-free DOM read (never
    LLM-authored) — this is the only script ever passed to browser_evaluate anywhere in
    this codebase, so despite that tool's generic "can run arbitrary JS" classification,
    this specific call has no mutation risk.
    """
    tool_map = {tool.name: tool for tool in tools}
    result = await invoke_tool(
        tool_map["browser_evaluate"],
        {
            "function": (
                "() => Array.from(document.querySelectorAll('a[href]'))"
                ".map(a => ({text: (a.innerText || a.textContent || '').trim().slice(0, 80), href: a.href}))"
            )
        },
    )
    return _coerce_json_array(result)
