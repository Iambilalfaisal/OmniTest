"""Planner node: reads the target page's accessibility tree plus a shallow site map,
and turns the user's instruction (optionally enriched by a discovery-chat conversation)
into a comprehensive set of TestCases — happy path AND edge/negative/error-handling
cases — each fully self-contained (setup steps like sign-up/login inlined) since every
test case runs in its own isolated browser session in parallel with no shared state.
"""
from __future__ import annotations

import os
from contextlib import AsyncExitStack

from langgraph.store.base import BaseStore

from ..core.llm import ModelRole, with_fallback
from ..core.memory import get_cached_site_map, retrieve_memory_context, save_site_map
from ..core.models import TEST_CASE_AUTHORING_GUIDELINES, TestPlan
from ..core.run_planning import ensure_expected_result, ensure_unique_test_ids, generate_run_token
from ..core.state import QAState
from ..mcp.client import get_accessibility_snapshot, open_playwright_session
from .planner_explore import crawl_site, format_site_map_for_prompt

PLANNER_CRAWL_MAX_PAGES = int(os.getenv("PLANNER_CRAWL_MAX_PAGES", "8"))
PLANNER_CRAWL_MAX_DEPTH = int(os.getenv("PLANNER_CRAWL_MAX_DEPTH", "2"))

# Layout is deliberate, and matters more here than in a normal prompt because this call
# is one-shot (no chance to correct itself) and runs on whatever cheap model PLANNER_MODEL
# names: ROLE first, then the static authoring rules (stable prefix — same text every run,
# so it's the cacheable part), then this run's inputs, then the bulky evidence blocks
# (site map, accessibility tree), and finally a SHORT restatement of the task. The closing
# restatement is not redundant: a weak model's adherence degrades across a long context,
# and the last thing it reads before answering should be what to do, not several thousand
# tokens of accessibility tree.
PLANNER_PROMPT = f"""You are a senior QA engineer writing the end-to-end test plan for a
website you have just been given access to. You have no knowledge of this site beyond the
evidence below: the user's instruction, whatever they told you up front, what previous
runs of this tool learned, a shallow site map, and the target page's accessibility tree.

Your plan is not a document for a human to interpret — another agent executes your steps
literally, one at a time, in a real browser, and then grades each case Pass/Fail strictly
against the expected_result you wrote. Write it for someone who cannot ask you anything.

{TEST_CASE_AUTHORING_GUIDELINES}

# This run

Testing instruction from the user: {{instruction}}
Target URL: {{url}}

Context the user gave before planning (credentials, preferences, clarifications) — prefer
any real credentials here over inventing an account:
{{known_context}}
Prior learnings about this site, from previous runs — treat these as already verified
about this specific site, and do not write test cases that contradict them:
{{memory_context}}
Site map, from a shallow read-only crawl of pages linked from the target page — use it to
know which other pages and flows exist (e.g. a separate login or account page), even
though only the target page's full accessibility tree is given below:
{{site_map}}
Accessibility tree of the target page ({{url}}) — the ground truth for element labels on
this page; use these exact labels in your steps:
{{tree}}

# Now write the plan

Cover the instruction above, choosing the flows from the coverage checklist that this site
actually has and that the instruction is actually about. Give every test case a concrete,
observable expected_result, and run the section 7 self-check over each one before you
answer.
"""


def _planner_llm():
    # get_chat_model (core/llm.py) is itself lazy — importing this module (e.g. at API
    # startup) still doesn't require GOOGLE_API_KEY until a run actually reaches the
    # planner node. PLANNER_MODEL is required — set it explicitly in .env; an optional
    # PLANNER_FALLBACK_MODEL (OpenRouter) takes over on a Gemini rate-limit/server error.
    return with_fallback(ModelRole.PLANNER, lambda m: m.with_structured_output(TestPlan), temperature=0)


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

    # ensure_expected_result guards against a live-confirmed gap: langchain-google-genai's
    # structured output does not actually enforce TestCase.expected_result as required, so
    # a case missing it must be backfilled here — before it reaches plan_review's
    # model_dump(), the worker, or verdict_node — rather than trusted as always present.
    test_cases = ensure_expected_result(ensure_unique_test_ids(plan.test_cases))
    return {"test_cases": test_cases, "run_token": run_token}
