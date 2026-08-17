"""Per-caller MultiServerMCPClient factory for the Playwright MCP server.

The planner and every parallel `worker_node` branch each call `create_playwright_client()`
to get their own client — and therefore their own `npx @playwright/mcp` subprocess /
browser instance — instead of sharing one singleton, so parallel branches never race on
the same tab.
"""
from __future__ import annotations

import os

from langchain_mcp_adapters.client import MultiServerMCPClient

PLAYWRIGHT_MCP_COMMAND = os.getenv("PLAYWRIGHT_MCP_COMMAND", "npx")
PLAYWRIGHT_MCP_ARGS = os.getenv("PLAYWRIGHT_MCP_ARGS", "-y @playwright/mcp").split()


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


async def get_playwright_tools(client: MultiServerMCPClient) -> list:
    """Load this client's tools and make them self-healing.

    NOTE: `handle_tool_errors` is not a `MultiServerMCPClient`/connection-config kwarg
    in langchain-mcp-adapters as of this writing — there's no such constructor option.
    The actual hook lives per-tool: `BaseTool.handle_tool_error` (singular), which
    catches a raised `ToolException` and feeds its message back to the model as the
    tool's output instead of raising, so the agent sees the failure and can retry with
    corrected arguments. We set it on every tool returned here. Verify this still
    matches your installed langchain-mcp-adapters / langchain-core versions.
    """
    tools = await client.get_tools(server_name="playwright")
    for tool in tools:
        tool.handle_tool_error = True
    return tools


async def get_accessibility_snapshot(tools: list, url: str) -> dict:
    """Navigate to `url` and return the page's accessibility tree snapshot."""
    tool_map = {tool.name: tool for tool in tools}
    await tool_map["browser_navigate"].ainvoke({"url": url})
    return await tool_map["browser_snapshot"].ainvoke({})
