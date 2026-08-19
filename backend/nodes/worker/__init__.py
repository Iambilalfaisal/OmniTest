"""Worker subgraph: `agent_node` (one LLM turn) -> `tool_node` (one tool call) ->
`verdict_node` (final Pass/Fail + evidence capture), looping between `agent_node` and
`tool_node` until the model stops calling tools or `MAX_TOOL_TURNS` is hit. See
nodes.py's module docstring for why the loop is split into one node per unit of work.

Split by concern: session.py (Playwright/MCP session lifecycle), evidence.py
(screenshot/trace/video capture), nodes.py (the LangGraph nodes + subgraph assembly).
Only `build_worker_subgraph` and `close_sessions_for_thread` are used outside this
package (graph/builder.py and api.py, respectively) — everything else here is
package-internal.
"""
from .nodes import build_worker_subgraph
from .session import close_sessions_for_thread

__all__ = ["build_worker_subgraph", "close_sessions_for_thread"]
