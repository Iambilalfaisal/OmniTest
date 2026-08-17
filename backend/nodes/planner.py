"""Planner node: reads the target page's accessibility tree and turns the user's
plain-language instruction into a small set of TestCases — each a goal plus an
ordered list of plain-language steps for a worker to execute.
"""
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from ..core.memory import MEMORY_NAMESPACE, domain_of, format_memories_for_prompt
from ..core.models import TestPlan
from ..core.state import QAState
from ..mcp.client import create_playwright_client, get_accessibility_snapshot, get_playwright_tools

PLANNER_PROMPT = """You are a QA engineer. Given the accessibility tree of a web page \
and a testing instruction, write a small, high-value set of test cases that exercise \
the instruction. Each test case is a goal plus an ordered list of plain-language steps \
(e.g. "Click the 'Sign in' button", "Type 'demo@site.com' into the email field") — the \
worker executing these steps only has accessibility-tree-based Playwright tools, not CSS \
selectors, so phrase steps in terms of visible roles/labels/text.

Instruction: {instruction}
URL: {url}

Prior learnings about this site:
{memory_context}
Accessibility tree:
{tree}
"""


def _planner_llm():
    # Constructed lazily so importing this module (e.g. at API startup) doesn't
    # require OPENAI_API_KEY until a run actually reaches the planner node.
    # PLANNER_MODEL is required with no fallback — set it explicitly in .env.
    return ChatOpenAI(model=os.environ["PLANNER_MODEL"], temperature=0).with_structured_output(TestPlan)


async def _retrieve_memory_context(target_url: str, instruction: str, store) -> str:
    if store is None:
        return "No prior learnings recorded for this site yet.\n"
    domain = domain_of(target_url)
    items = await store.asearch((MEMORY_NAMESPACE, domain), query=instruction, limit=5)
    return format_memories_for_prompt(items)


async def planner_node(state: QAState, *, store=None) -> dict:
    client = create_playwright_client()
    tools = await get_playwright_tools(client)
    tree = await get_accessibility_snapshot(tools, state["target_url"])
    memory_context = await _retrieve_memory_context(state["target_url"], state["instruction"], store)

    plan: TestPlan = await _planner_llm().ainvoke(
        PLANNER_PROMPT.format(
            instruction=state["instruction"],
            url=state["target_url"],
            memory_context=memory_context,
            tree=tree,
        )
    )

    return {"test_cases": plan.test_cases}
