"""Worker node: a tool-calling agent that executes one TestCase's steps against a
fresh, isolated Playwright browser context, then captures evidence immediately
before closing. One `worker_node` branch runs per TestCase (spawned by
`route_to_workers`'s Send-based fan-out); each returns a single-element
`test_results` list that the `operator.add` reducer on QAState.test_results merges
together.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
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


class Verdict(BaseModel):
    status: str = Field(description="Strictly 'Pass' or 'Fail'")
    reason: str


async def worker_node(state: WorkerState) -> dict:
    test_case = state["test_case"]
    client = create_playwright_client()
    tools = await get_playwright_tools(client)
    tool_map = {tool.name: tool for tool in tools}

    # Suffixed so two runs that both name a case e.g. "test_1" don't collide on disk —
    # test_id alone isn't guaranteed globally unique since QAState carries no run_id.
    run_dir = EVIDENCE_DIR / f"{test_case.test_id}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = ChatOpenAI(model=os.getenv("WORKER_MODEL", "gpt-4o"), temperature=0)
    model_with_tools = model.bind_tools(tools)

    steps_block = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(test_case.steps))
    messages: list = [
        SystemMessage(WORKER_SYSTEM_PROMPT),
        HumanMessage(
            f"Target URL: {state['target_url']}\nGoal: {test_case.goal}\n\nSteps:\n{steps_block}"
        ),
    ]

    for _ in range(MAX_TOOL_TURNS):
        response: AIMessage = await model_with_tools.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for call in response.tool_calls:
            result = await tool_map[call["name"]].ainvoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    verdict: Verdict = await model.with_structured_output(Verdict).ainvoke(
        messages + [HumanMessage("Give your final verdict on this test case now.")]
    )

    screenshot_path = await _capture_screenshot(tool_map, run_dir)
    # TODO(verify): @playwright/mcp's trace/video capture is either a launch-time
    # CLI flag or a dedicated tool depending on version — not confirmed here, so
    # these are left unpopulated until that's checked against your installed version.
    trace_path = None
    video_path = None

    close_tool = tool_map.get("browser_close")
    if close_tool is not None:
        await close_tool.ainvoke({})

    return {
        "test_results": [
            TestResult(
                test_id=test_case.test_id,
                status=verdict.status,
                screenshot_path=screenshot_path,
                trace_path=trace_path,
                video_path=video_path,
                reason=verdict.reason,
            )
        ]
    }


async def _capture_screenshot(tool_map: dict, run_dir: Path) -> str:
    path = run_dir / "final.png"
    await tool_map["browser_take_screenshot"].ainvoke({"filename": str(path)})
    return str(path.relative_to(EVIDENCE_DIR.parent))
