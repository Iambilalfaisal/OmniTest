"""Planner node: reads the target page's accessibility tree plus a shallow site map,
and turns the user's instruction (optionally enriched by a discovery-chat conversation)
into a comprehensive set of TestCases — happy path AND edge/negative/error-handling
cases — each fully self-contained (setup steps like sign-up/login inlined) since every
test case runs in its own isolated browser session in parallel with no shared state.
"""
from __future__ import annotations

import os
from contextlib import AsyncExitStack

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.store.base import BaseStore

from ..core.memory import get_cached_site_map, retrieve_memory_context, save_site_map
from ..core.models import TEST_CASE_AUTHORING_GUIDELINES, TestPlan
from ..core.run_planning import ensure_unique_test_ids, generate_run_token
from ..core.state import QAState
from ..mcp.client import get_accessibility_snapshot, open_playwright_session
from .planner_explore import crawl_site, format_site_map_for_prompt

PLANNER_CRAWL_MAX_PAGES = int(os.getenv("PLANNER_CRAWL_MAX_PAGES", "8"))
PLANNER_CRAWL_MAX_DEPTH = int(os.getenv("PLANNER_CRAWL_MAX_DEPTH", "2"))

PLANNER_PROMPT = f"""You are a QA engineer. Given a testing instruction, a summary of the \
target site, and any context already gathered from the user, write a comprehensive, \
high-value set of test cases that exercise the instruction.

{TEST_CASE_AUTHORING_GUIDELINES}

Instruction: {{instruction}}
Target URL: {{url}}

Context from the user (credentials, preferences, or clarifications gathered before planning):
{{known_context}}
Prior learnings about this site (from previous runs):
{{memory_context}}
Site map (from a shallow, read-only crawl of pages linked from the target page — use this to
understand what other pages/flows exist, e.g. a separate login or account page, even though
only the target page's full accessibility tree is given below):
{{site_map}}
Accessibility tree of the target page ({{url}}):
{{tree}}
"""


def _planner_llm():
    # Constructed lazily so importing this module (e.g. at API startup) doesn't
    # require GOOGLE_API_KEY until a run actually reaches the planner node.
    # PLANNER_MODEL is required with no fallback — set it explicitly in .env.
    return ChatGoogleGenerativeAI(model=os.environ["PLANNER_MODEL"], temperature=0).with_structured_output(
        TestPlan
    )


async def planner_node(state: QAState, *, store: BaseStore | None = None) -> dict:
    run_token = generate_run_token()

    async with AsyncExitStack() as stack:
        tools = await open_playwright_session(stack)
        tree = await get_accessibility_snapshot(tools, state["target_url"])

        site_map = await get_cached_site_map(store, state["target_url"])
        if site_map is None:
            site_map = await crawl_site(
                tools,
                state["target_url"],
                max_pages=PLANNER_CRAWL_MAX_PAGES,
                max_depth=PLANNER_CRAWL_MAX_DEPTH,
                start_already_loaded=True,  # tools already navigated to target_url above
            )
            await save_site_map(store, state["target_url"], site_map)

    memory_context = await retrieve_memory_context(state["target_url"], state["instruction"], store)
    known_context = state.get("discovery_context") or "No additional context from the user was provided before planning.\n"

    plan: TestPlan = await _planner_llm().ainvoke(
        PLANNER_PROMPT.format(
            instruction=state["instruction"],
            url=state["target_url"],
            run_token=run_token,
            known_context=known_context,
            memory_context=memory_context,
            site_map=format_site_map_for_prompt(site_map, exclude_url=state["target_url"]),
            tree=tree,
        )
    )

    return {"test_cases": ensure_unique_test_ids(plan.test_cases), "run_token": run_token}
