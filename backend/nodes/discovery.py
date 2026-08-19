"""Discovery node: the chat-first planning conversation that runs BEFORE any test
executes — explores the site, proposes/revises a candidate test plan, and asks the user
proactively about known ambiguities (credentials, destructive actions) — see
graph/discovery_graph.py for why the interrupt() that pauses each turn lives in a
SEPARATE node from this one (replay safety: this node does all the real work — crawling
and LLM calls — and must never itself pause, or a resume would redo it).
"""
from __future__ import annotations

import os
from contextlib import AsyncExitStack

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.store.base import BaseStore

from ..core.discovery_state import DiscoveryState
from ..core.memory import get_cached_site_map, retrieve_memory_context, save_site_map
from ..core.models import TEST_CASE_AUTHORING_GUIDELINES, DiscoveryTurn, SiteMap
from ..core.run_planning import generate_run_token
from ..mcp.client import open_playwright_session
from .planner import PLANNER_CRAWL_MAX_DEPTH, PLANNER_CRAWL_MAX_PAGES
from .planner_explore import crawl_site, format_site_map_for_prompt

# Total additional on-demand crawls allowed across one whole discovery conversation,
# beyond the automatic upfront shallow crawl — keeps a chatty conversation from turning
# into an unbounded number of extra browser round trips.
MAX_EXTRA_DIVES = int(os.getenv("MAX_EXTRA_DIVES", "5"))

DISCOVERY_SYSTEM_PROMPT = f"""You are a QA engineer having a conversation with a user to scope out an \
end-to-end test plan for their website, BEFORE any test actually runs. You already have a shallow, \
read-only site map (crawled automatically) as context — use it to understand what pages/flows exist.

Your job each turn:
1. Propose or revise a complete candidate test plan (grouped conceptually by feature/area) covering \
the happy path and meaningful edge/negative/error-handling cases relevant to what the user has asked \
for, or to the site in general if they haven't specified anything narrow yet.
2. Proactively surface anything you need to know before the plan can actually run — most commonly: if \
the site map shows a login/signup wall, ask whether the user has existing test credentials you should \
use, or whether you should test via the sign-up flow instead; also ask about any destructive-looking \
action (payments, account deletion, etc.) the user would rather you skip or handle carefully.
3. If you'd answer better with a deeper look at a specific page/area not yet in the site map, set \
"explore_more" to that URL and a short reason — you'll be given a fresh look at it before your next reply.
4. Once the user has clearly approved out loud (e.g. "looks good", "go ahead") OR your plan already \
covers what they asked for well, you may set "ready_to_run" to true as a hint for the UI — but this is \
only ever a hint: the run does NOT start from this field, only from the user's explicit Approve button.
5. Keep "assistant_message" natural and conversational — a short summary of what changed, plus, when \
relevant, your question. Don't re-paste the whole plan in prose; the plan is shown to the user separately.

{TEST_CASE_AUTHORING_GUIDELINES}
"""


def _discovery_llm():
    # Lazy for the same reason as planner._planner_llm — reuses PLANNER_MODEL rather than
    # a separate required env var, since the chat conversation IS the planner for a
    # chat-approved run; can be split into its own model later if cost/latency tuning
    # turns out to need it.
    return ChatGoogleGenerativeAI(model=os.environ["PLANNER_MODEL"], temperature=0).with_structured_output(
        DiscoveryTurn
    )


def _format_candidate_plan(plan) -> str:
    if plan is None or not plan.test_cases:
        return "(no plan proposed yet)"
    lines = []
    for tc in plan.test_cases:
        precond = f" [setup: {'; '.join(tc.preconditions)}]" if tc.preconditions else ""
        steps = "\n".join(f"    {i + 1}. {step}" for i, step in enumerate(tc.steps))
        lines.append(f"- ({tc.category}/{tc.priority}) {tc.goal}{precond}\n{steps}")
    return "\n".join(lines)


def _context_message(site_context: SiteMap, candidate_plan) -> HumanMessage:
    """Fresh, transient addendum appended to the call each turn (never persisted into
    `messages`) so the model always sees the latest site map/plan rather than having to
    reconstruct them from its own prior chat replies."""
    return HumanMessage(
        "[Current context, not something the user said]\n"
        f"Site map so far:\n{format_site_map_for_prompt(site_context)}\n"
        f"Current candidate plan:\n{_format_candidate_plan(candidate_plan)}\n"
    )


async def _crawl(target_url: str, store: BaseStore | None, *, start_url: str | None = None, already_visited=None) -> SiteMap:
    """Upfront crawl (start_url=None): cache-first, cached on success. On-demand dive
    (start_url set): never cached under the domain's main site-map key — it's a narrower,
    conversation-specific look, not the general site map future runs should reuse.
    """
    if start_url is None:
        cached = await get_cached_site_map(store, target_url)
        if cached is not None:
            return cached

    async with AsyncExitStack() as stack:
        tools = await open_playwright_session(stack)
        site_map = await crawl_site(
            tools,
            start_url or target_url,
            max_pages=PLANNER_CRAWL_MAX_PAGES,
            max_depth=PLANNER_CRAWL_MAX_DEPTH,
            already_visited=already_visited,
        )

    if start_url is None:
        await save_site_map(store, target_url, site_map)
    return site_map


def _merge_site_maps(base: SiteMap, extra: SiteMap) -> SiteMap:
    seen = {p.url for p in base.pages}
    merged_pages = list(base.pages) + [p for p in extra.pages if p.url not in seen]
    return SiteMap(pages=merged_pages, truncated=base.truncated or extra.truncated)


async def discovery_agent_node(state: DiscoveryState, *, store: BaseStore | None = None) -> dict:
    # `messages` uses the `add_messages` reducer (append-only) — the seed below must be
    # returned alongside the turn's reply on turn 1 so it's actually persisted, same
    # subtlety documented in nodes/worker/nodes.py's agent_node.
    history = state.get("messages")
    seed: list = []
    site_context = state.get("site_context")
    extra_dives_used = state.get("extra_dives_used", 0)
    run_token = state.get("run_token") or ""

    if not history:
        run_token = generate_run_token()
        site_context = await _crawl(state["target_url"], store)
        memory_context = await retrieve_memory_context(
            state["target_url"], state["starting_idea"] or "general exploration", store
        )
        seed = [
            SystemMessage(DISCOVERY_SYSTEM_PROMPT.format(run_token=run_token)),
            HumanMessage(
                f"{state['starting_idea'] or 'Explore this site and propose a test plan.'}\n\n"
                f"Prior learnings about this site (from previous runs):\n{memory_context}"
            ),
        ]
        history = seed

    turn: DiscoveryTurn = await _discovery_llm().ainvoke(
        history + [_context_message(site_context, state.get("candidate_plan"))]
    )

    if turn.explore_more is not None and extra_dives_used < MAX_EXTRA_DIVES:
        already_visited = {p.url for p in site_context.pages}
        extra = await _crawl(state["target_url"], store, start_url=turn.explore_more.url, already_visited=already_visited)
        site_context = _merge_site_maps(site_context, extra)
        extra_dives_used += 1
        turn = await _discovery_llm().ainvoke(
            history + [_context_message(site_context, state.get("candidate_plan"))]
        )

    return {
        "messages": [*seed, AIMessage(turn.assistant_message)],
        "site_context": site_context,
        "extra_dives_used": extra_dives_used,
        "candidate_plan": turn.candidate_plan,
        "run_token": run_token,
        "turn_count": state.get("turn_count", 0) + 1,
    }
