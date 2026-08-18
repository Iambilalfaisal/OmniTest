"""Per-caller MultiServerMCPClient factory for the Playwright MCP server.

The planner and every parallel `worker_node` branch each call `create_playwright_client()`
to get their own client — and therefore their own `npx @playwright/mcp` subprocess /
browser instance — instead of sharing one singleton, so parallel branches never race on
the same tab.
"""
from __future__ import annotations

import os
from contextlib import AsyncExitStack

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

PLAYWRIGHT_MCP_COMMAND = os.getenv("PLAYWRIGHT_MCP_COMMAND", "npx")
# --isolated is required, not optional, given our design: @playwright/mcp defaults to a
# PERSISTENT shared browser profile, but we spin up one subprocess per parallel worker —
# without --isolated, concurrent workers would fight over the same profile directory
# instead of getting independent browsers.
# --caps=devtools opts into browser_start_tracing/browser_start_video (+ stop_ variants)
# for nodes/worker.py's evidence capture; it also exposes a few extra low-risk tools
# (browser_annotate, browser_highlight, browser_resume, video-chapter tools) to the
# worker LLM's tool-calling set as a side effect, since we bind everything indiscriminately.
# See https://playwright.dev/docs/getting-started-mcp and https://github.com/microsoft/playwright-mcp.
PLAYWRIGHT_MCP_ARGS = os.getenv(
    "PLAYWRIGHT_MCP_ARGS", "-y @playwright/mcp@latest --isolated --caps=devtools"
).split()


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
    await tool_map["browser_navigate"].ainvoke({"url": url})
    return await tool_map["browser_snapshot"].ainvoke({})
