"""Worker subgraph: `agent_node` (one LLM turn) -> `tool_node` (one tool call) ->
`verdict_node` (final Pass/Fail + evidence capture), looping between `agent_node` and
`tool_node` until the model stops calling tools or `MAX_TOOL_TURNS` is hit.

Restructured from a single-function loop into one node per unit of work so each is
independently checkpointed — required for `tool_node`'s risky-action `interrupt()` to
pause/resume without redoing prior real browser actions. LangGraph re-executes a node
function from the top on resume; splicing an interrupt into a single multi-turn loop
would replay every earlier turn's tool calls.
"""
from __future__ import annotations

import os
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, interrupt
from pydantic import BaseModel, Field

from ..core.models import TestResult
from ..core.state import WorkerState
from ..mcp.client import create_playwright_client, get_playwright_tools

EVIDENCE_DIR = Path(__file__).resolve().parent.parent / "evidence"
MAX_TOOL_TURNS = 15

WORKER_SYSTEM_PROMPT = """You are a QA test executor. You control a real browser through \
the tools available to you (Playwright, driven by the accessibility tree — call \
`browser_snapshot` whenever you need current element refs before acting). Execute the \
numbered steps below in order to pursue the goal. If a tool call fails, read the error \
and adjust your next call rather than repeating the same failing approach — you have a \
limited number of turns. Once every step has been attempted, stop calling tools and wait \
for the verdict question.
"""

# "{thread_id}:{test_id}" -> (client, tools, tool_map). A live Playwright/MCP subprocess
# can't be checkpointed (not serializable), so it's cached process-locally and shared by
# agent_node/tool_node/verdict_node across however many separate node invocations one
# test case's tool-calling loop takes. A cache miss transparently opens a fresh browser
# rather than erroring — see the plan's "accepted limitations" for what that means for
# a risky-action resume landing on a different process than the one that paused.
_SESSIONS: dict[str, tuple] = {}


def _session_key(config: RunnableConfig, test_id: str) -> str:
    return f"{config['configurable']['thread_id']}:{test_id}"


async def _get_session(session_key: str) -> tuple:
    if session_key not in _SESSIONS:
        client = create_playwright_client()
        tools = await get_playwright_tools(client)
        _SESSIONS[session_key] = (client, tools, {t.name: t for t in tools})
    return _SESSIONS[session_key]


class Verdict(BaseModel):
    status: str = Field(description="Strictly 'Pass' or 'Fail'")
    reason: str


async def agent_node(state: WorkerState, config: RunnableConfig) -> dict:
    test_case = state["test_case"]
    _, tools, _ = await _get_session(_session_key(config, test_case.test_id))

    messages = state.get("messages")
    if not messages:
        steps_block = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(test_case.steps))
        messages = [
            SystemMessage(WORKER_SYSTEM_PROMPT),
            HumanMessage(
                f"Target URL: {state['target_url']}\nGoal: {test_case.goal}\n\nSteps:\n{steps_block}"
            ),
        ]

    model = ChatOpenAI(model=os.environ["WORKER_MODEL"], temperature=0)
    response = await model.bind_tools(tools).ainvoke(messages)

    return {
        "messages": [response],
        "pending_tool_calls": response.tool_calls,
        "turn_count": state.get("turn_count", 0) + 1,
    }


RISKY_KEYWORDS = ("delete", "purchase", "buy", "pay", "confirm", "submit", "remove")


def _is_risky(call: dict) -> bool:
    haystack = f"{call['name']} {call.get('args', {})}".lower()
    return any(word in haystack for word in RISKY_KEYWORDS)


async def tool_node(state: WorkerState, config: RunnableConfig) -> dict:
    call, remaining = state["pending_tool_calls"][0], state["pending_tool_calls"][1:]

    # Pure check, no side effect yet — so replaying this node on resume is free even
    # though the interrupt() call below pauses execution mid-function.
    if _is_risky(call):
        decision = interrupt(
            {
                "type": "risky_action",
                "test_id": state["test_case"].test_id,
                "tool": call["name"],
                "args": call["args"],
            }
        )
        if not decision.get("approved", False):
            blocked = ToolMessage(
                content=f"Blocked by human reviewer: {decision.get('reason', 'not approved')}",
                tool_call_id=call["id"],
            )
            return {"messages": [blocked], "pending_tool_calls": remaining}

    # NOTE: a resume landing on a different process than the one that paused finds no
    # cached session here and transparently opens a fresh, unnavigated browser instead
    # of failing loudly — see the plan's accepted limitations for this exact scenario.
    _, _, tool_map = await _get_session(_session_key(config, state["test_case"].test_id))
    result = await tool_map[call["name"]].ainvoke(call["args"])
    return {
        "messages": [ToolMessage(content=str(result), tool_call_id=call["id"])],
        "pending_tool_calls": remaining,
    }


async def verdict_node(state: WorkerState, config: RunnableConfig) -> dict:
    test_case = state["test_case"]
    session_key = _session_key(config, test_case.test_id)
    _, _, tool_map = await _get_session(session_key)

    model = ChatOpenAI(model=os.environ["WORKER_MODEL"], temperature=0)
    verdict: Verdict = await model.with_structured_output(Verdict).ainvoke(
        state["messages"] + [HumanMessage("Give your final verdict on this test case now.")]
    )

    screenshot_path = await _capture_screenshot(tool_map, session_key)
    close_tool = tool_map.get("browser_close")
    if close_tool is not None:
        await close_tool.ainvoke({})
    _SESSIONS.pop(session_key, None)  # this test case's browser work is done

    return {
        "test_results": [
            TestResult(
                test_id=test_case.test_id,
                status=verdict.status,
                screenshot_path=screenshot_path,
                trace_path=None,  # TODO(verify): see mcp/client.py — trace/video capture unconfirmed
                video_path=None,
                reason=verdict.reason,
            )
        ]
    }


async def _capture_screenshot(tool_map: dict, session_key: str) -> str:
    # session_key contains ":" (invalid in a Windows path component) — sanitize for the dir name.
    run_dir = EVIDENCE_DIR / session_key.replace(":", "_")
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "final.png"
    await tool_map["browser_take_screenshot"].ainvoke({"filename": str(path)})
    return str(path.relative_to(EVIDENCE_DIR.parent))


def route_after_agent(state: WorkerState) -> str:
    return "tool_node" if state["pending_tool_calls"] else "verdict_node"


def route_after_tool(state: WorkerState) -> str:
    if state["pending_tool_calls"]:
        return "tool_node"
    return "verdict_node" if state["turn_count"] >= MAX_TOOL_TURNS else "agent_node"


def build_worker_subgraph():
    sub = StateGraph(WorkerState)
    sub.add_node("agent_node", agent_node, retry_policy=RetryPolicy(max_attempts=3))
    # No retry on tool_node — retrying after a raised exception risks re-invoking a
    # tool that already had a real side effect. mcp/client.py's per-tool
    # handle_tool_error=True already converts tool exceptions into a ToolMessage fed
    # back to agent_node's next turn instead of raising, so this rarely matters.
    sub.add_node("tool_node", tool_node)
    sub.add_node("verdict_node", verdict_node, retry_policy=RetryPolicy(max_attempts=3))

    sub.add_edge(START, "agent_node")
    sub.add_conditional_edges("agent_node", route_after_agent, ["tool_node", "verdict_node"])
    sub.add_conditional_edges("tool_node", route_after_tool, ["tool_node", "agent_node", "verdict_node"])
    sub.add_edge("verdict_node", END)

    # No checkpointer passed — inherits the parent graph's, required for interrupt()
    # inside tool_node (added later) to actually persist.
    return sub.compile()
