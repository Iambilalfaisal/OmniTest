"""ContextVars carrying run_id/test_id through the current asyncio task tree, so
logging_config.py's log filter can attach them without threading extra parameters
through every function signature.

`asyncio.create_task()` copies the creator's `contextvars.Context` at creation time —
this is the SAME isolation mechanism LangSmith's tracing already relies on for per-call
context, and it's why setting `run_id_var` once, before `graph.ainvoke()` starts, is
enough to reach every LangGraph-internal task it spawns afterward (including parallel
`Send`-spawned worker branches): each new task's context is a copy taken at that later
point, so it already carries the value. The reverse is NOT true — a `.set()` inside one
already-created task (e.g. one worker node's own task) does not propagate to a sibling
or successor task's independently-created context, which is why `test_id_var` has to be
set at the top of each of agent_node/tool_node/verdict_node individually rather than
once anywhere upstream (see nodes/worker/nodes.py and the module docstring for the
"separate task per node" fact this depends on).
"""
from __future__ import annotations

from contextvars import ContextVar

run_id_var: ContextVar[str] = ContextVar("run_id", default="-")
test_id_var: ContextVar[str] = ContextVar("test_id", default="-")
