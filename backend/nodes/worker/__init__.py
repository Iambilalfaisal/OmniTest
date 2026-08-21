"""Worker subgraph: `agent_node` (one LLM turn) -> `tool_node` (one tool call) ->
`verdict_node` (final Pass/Fail + evidence capture), looping between `agent_node` and
`tool_node` until the model stops calling tools or `MAX_TOOL_TURNS` is hit. See
nodes.py's module docstring for why the loop is split into one node per unit of work.

Split by concern: session.py (Playwright/MCP session lifecycle), evidence.py
(screenshot/trace/video capture), nodes.py (the LangGraph nodes + subgraph assembly).
`build_worker_subgraph` (graph/builder.py), `close_sessions_for_thread` (api.py's
`_drive()`, run-ended cleanup), and `close_idle_sessions`/`close_all_sessions` (api.py's
lifespan — the paused-and-abandoned-run backstop and shutdown sweep) are the public
surface for callers outside `nodes/` entirely.

`nodes/auth/nodes.py` is a partial exception: it imports `session.py`'s
`get_session`/`discard_session` and `evidence.py`'s `EVIDENCE_DIR`/`run_dir_for`/
`stop_and_capture` directly, deliberately reaching past this package's own boundary.
That's not an oversight — the auth subgraph's shared-login session MUST live in this
SAME process-global `_SESSIONS` cache (not a duplicate one) for `close_sessions_for_thread`
and the idle-session reaper to sweep it exactly like a worker's session, and its evidence
capture (started automatically by `get_session`, since it goes through the same
cache-miss path a worker's session does) needs the same finalize-then-close pairing
`verdict_node` uses. Everything else here stays package-internal.
"""
from .nodes import build_worker_subgraph
from .session import (
    SESSION_REAP_INTERVAL_SECONDS,
    close_all_sessions,
    close_idle_sessions,
    close_sessions_for_thread,
)

__all__ = [
    "build_worker_subgraph",
    "close_sessions_for_thread",
    "close_idle_sessions",
    "close_all_sessions",
    "SESSION_REAP_INTERVAL_SECONDS",
]
