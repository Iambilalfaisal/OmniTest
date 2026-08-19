"""Generic dynamic human-in-the-loop tool: lets a worker ask a free-text question mid-
execution whenever it hits real ambiguity (missing credentials, an unclear requirement),
instead of only pausing at the two static checkpoints this codebase had before
(plan_review_node's whole-plan approval, tool_node's risky-action keyword match).

The callable body below is unreachable by design — nodes.py's tool_node always
intercepts a call to this tool by name, before it ever reaches the generic MCP tool
dispatch, and this tool is deliberately never added to tool_node's MCP tool_map.
"""
from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

ASK_HUMAN_TOOL_NAME = "ask_human"


class AskHumanInput(BaseModel):
    question: str = Field(description="The question to ask the human reviewer.")
    context: str | None = Field(default=None, description="Optional extra context to help them answer.")
    sensitive: bool = Field(
        default=False,
        description="Set True if the expected answer is a secret (e.g. a password) so it's masked in the "
        "UI and never echoed back verbatim in the final verdict.",
    )


async def _unreachable(**_kwargs) -> str:
    raise RuntimeError(
        f"{ASK_HUMAN_TOOL_NAME} must be intercepted by tool_node before this ever runs — "
        "it is deliberately never added to tool_node's MCP tool_map."
    )


ask_human_tool = StructuredTool.from_function(
    coroutine=_unreachable,
    name=ASK_HUMAN_TOOL_NAME,
    description=(
        "Ask the human reviewer a free-text question when you hit real ambiguity you cannot resolve "
        "yourself — e.g. missing login credentials, an unclear requirement, or a decision only a human "
        "can make. Execution pauses until they answer; their answer is given back to you as this tool's "
        "result. Set `sensitive=True` if the answer will be a secret."
    ),
    args_schema=AskHumanInput,
)
