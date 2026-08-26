"""Discovery node: the chat-first planning conversation that runs BEFORE any test
executes — explores the site, proposes/revises a candidate test plan, and asks the user
proactively about known ambiguities (credentials, destructive actions) — see
graph/discovery_graph.py for why the interrupt() that pauses each turn lives in a
SEPARATE node from this one (replay safety: this node does all the real work — crawling
and LLM calls — and must never itself pause, or a resume would redo it).

`explore_more`'s dive is likewise deliberately deferred rather than run inline: an
earlier version crawled the requested page and re-invoked the LLM a SECOND time in the
same turn, so the model could see it — costing a second, largely redundant call every
time the model asked for a dive (up to MAX_EXTRA_DIVES per conversation), since the
first call's turn was simply thrown away. `DiscoveryState.pending_dive` instead carries
the request across the turn boundary: the crawl happens at the TOP of the NEXT
invocation, before that turn's one LLM call, and reads more like a real conversation
("let me look at /checkout" -> the next reply has it) than a same-turn interruption would.
"""
from __future__ import annotations

import os
from contextlib import AsyncExitStack

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.store.base import BaseStore

from ..core.discovery_state import DiscoveryState
from ..core.llm import ModelRole, with_fallback
from ..core.memory import drop_semantic_duplicates, get_cached_site_map, retrieve_memory_context, save_site_map
from ..core.models import TEST_CASE_AUTHORING_GUIDELINES, DiscoveryTurn, SiteMap
from ..core.run_planning import drop_duplicate_scenarios, ensure_expected_result, ensure_features, generate_run_token
from ..mcp.client import open_playwright_session
from .planner import PLANNER_CRAWL_MAX_DEPTH, PLANNER_CRAWL_MAX_PAGES
from .planner_explore import crawl_site, format_site_map_for_prompt

# Total additional on-demand crawls allowed across one whole discovery conversation,
# beyond the automatic upfront shallow crawl — keeps a chatty conversation from turning
# into an unbounded number of extra browser round trips.
MAX_EXTRA_DIVES = int(os.getenv("MAX_EXTRA_DIVES", "5"))

DISCOVERY_SYSTEM_PROMPT = f"""You are a senior QA engineer scoping an end-to-end test plan \
together with the person who owns the website, in a chat, BEFORE anything runs. You already \
have a shallow, read-only site map (crawled automatically) — use it to know what pages and \
flows exist. The plan you end this conversation with is executed literally, step by step, by \
another agent in a real browser, and graded strictly against each case's expected_result.

Every turn, do all of the following:

1. Return a COMPLETE candidate plan — every test case you still believe in, revised for \
whatever the user just said. It is not a diff: re-include the unchanged cases too. Never \
silently drop a case you proposed earlier unless the user asked you to, or it turned out not \
to apply (and if so, say which one and why in your message). Losing coverage between turns is \
the worst thing you can do here.

2. Write every case to final quality right now — real values, real element labels from the \
site map, one concrete observable expected_result each (see the authoring rules below). Never \
propose a placeholder case you intend to fill in later, and never a case whose expected_result \
is "it works". Group them mentally by feature/area so the plan reads coherently, high-priority \
flows first.

3. Proactively ask about whatever would block this plan from running well. In priority order:
   - Credentials: if the site map shows a login or signup wall, ask whether they have an \
existing test account you should use (and to paste the email and password), or whether you \
should exercise the sign-up flow and create your own. Say which you will assume if they don't \
answer.
   - Destructive actions: name the specific actions in your plan that would delete data, spend \
money, email real people, or change account settings, and ask whether to include them, skip \
them, or substitute something safe.
   - Scope and priority: if their idea is broad ("test my site"), state which areas you intend \
to cover and in what order and ask them to confirm or reorder — do not ask an open-ended \
"what would you like me to test?".
   Ask at most two questions per turn, and only ones you genuinely cannot answer from the site \
map yourself. If you have no real question, don't invent one — say what you'd run next instead.

4. If answering well needs a closer look at a specific page or area not yet in the site map, \
set "explore_more" to that URL with a short reason — you get a fresh look at it before your \
next reply. Don't use it for a page already in the site map.

5. Set "ready_to_run" to true only once your plan covers what the user asked for and no \
blocking question is outstanding (or the user told you to go ahead). It is only ever a UI hint: \
the run starts from their explicit Approve button, never from this field.

6. Keep "assistant_message" short and conversational: what changed since your last turn, and \
your question if you have one. Do not re-paste the plan as prose — the user sees the plan \
itself in a panel beside the chat.

{TEST_CASE_AUTHORING_GUIDELINES}
"""

# Appended to DISCOVERY_SYSTEM_PROMPT only for mode="quick" (DiscoveryState.mode) — the
# base prompt above is written for a multi-turn conversation; this is what makes "Quick
# Start" actually behave like a single-turn proposal-then-approve flow instead of just
# changing the opening message. No `{`/`}` in here — appended before the one `.format
# (run_token=...)` call in discovery_agent_node, so it must stay free of stray braces
# (see core/models.py's TEST_CASE_AUTHORING_GUIDELINES docstring for the same constraint).
DISCOVERY_QUICK_ADDENDUM = """

QUICK MODE: the user wants a single-turn result, not a back-and-forth conversation. In \
this one turn, propose a complete plan covering every major flow the site map shows \
evidence of, state the assumption you're making for anything you'd normally ask about \
(which credentials to use, whether to include a destructive action) rather than asking, \
and set ready_to_run to true as soon as the plan meets the quality bar above — only ask a \
question if running as-is would be genuinely unsafe (e.g. no way to sign in and no site-map \
evidence of a sign-up flow either)."""


def _discovery_llm():
    # Lazy for the same reason as planner._planner_llm — reuses PLANNER_MODEL rather than
    # a separate required env var, since the chat conversation IS the planner for a
    # chat-approved run; can be split into its own model later if cost/latency tuning
    # turns out to need it. Same PLANNER_FALLBACK_MODEL (OpenRouter) as the planner node
    # takes over on a Gemini rate-limit/server error — see core/llm.py's with_fallback.
    return with_fallback(ModelRole.PLANNER, lambda m: m.with_structured_output(DiscoveryTurn), temperature=0)


def _format_candidate_plan(plan) -> str:
    if plan is None or not plan.test_cases:
        return "(no plan proposed yet)"
    lines = []
    for tc in plan.test_cases:
        precond = f" [setup: {'; '.join(tc.preconditions)}]" if tc.preconditions else ""
        steps = "\n".join(f"    {i + 1}. {step}" for i, step in enumerate(tc.steps))
        # expected_result is echoed back deliberately: it's the field the run is graded
        # on, and a turn that can't see the oracle it wrote last turn tends to quietly
        # regress it to something vaguer while "revising" an unrelated case. getattr, not
        # direct access: a candidate_plan resumed from a checkpoint written before
        # ensure_expected_result existed (core/run_planning.py) may still lack it.
        expected_value = getattr(tc, "expected_result", None)
        expected = f"\n    Expected: {expected_value}" if expected_value else ""
        lines.append(f"- ({tc.category}/{tc.priority}) {tc.goal}{precond}\n{steps}{expected}")
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
        system_prompt = DISCOVERY_SYSTEM_PROMPT + (
            DISCOVERY_QUICK_ADDENDUM if state.get("mode") == "quick" else ""
        )
        seed = [
            SystemMessage(system_prompt.format(run_token=run_token)),
            HumanMessage(
                f"{state['starting_idea'] or 'Explore this site and propose a test plan.'}\n\n"
                f"Prior learnings about this site (from previous runs):\n{memory_context}"
            ),
        ]
        history = seed
    elif state.get("pending_dive") is not None:
        # A prior turn's explore_more is consumed HERE, at the top of the turn that
        # follows it, instead of inline in the same turn that requested it — see this
        # module's docstring for why doing it inline cost a second, largely redundant
        # LLM call (up to MAX_EXTRA_DIVES times per conversation) to re-answer a turn
        # the model had already just answered, without the new page. Deferring it one
        # turn costs nothing: the user's next reply is the natural point to fold in what
        # the dive found, and it reads more like a real conversation ("let me look at
        # /checkout" -> next turn has it) than a mid-turn interruption would.
        already_visited = {p.url for p in site_context.pages}
        extra = await _crawl(
            state["target_url"], store, start_url=state["pending_dive"], already_visited=already_visited
        )
        site_context = _merge_site_maps(site_context, extra)

    turn: DiscoveryTurn = await _discovery_llm().ainvoke(
        history + [_context_message(site_context, state.get("candidate_plan"))]
    )

    next_pending_dive = None
    if turn.explore_more is not None and extra_dives_used < MAX_EXTRA_DIVES:
        next_pending_dive = turn.explore_more.url
        extra_dives_used += 1

    # See core/run_planning.py's ensure_expected_result docstring: the model can and does
    # omit this required field, and this candidate_plan is read again next turn (by
    # _format_candidate_plan/_context_message above) as well as by the frontend — so it
    # must be backfilled every turn, not only once at final approval in api.py.
    # ensure_features is the same defense for Feature/feature_id. drop_duplicate_scenarios
    # is the same defense for a confirmed-live failure mode: a turn's structured output
    # emitting the SAME candidate plan twice in one call — applied here too so the review
    # panel the user approves from already shows the deduped set, not just the final
    # approved run (api.py's own call is what actually matters for execution; this one is
    # for an honest review UI). drop_semantic_duplicates runs second (only over whatever
    # exact-match didn't already remove, since it's the one that costs an embedding call)
    # to catch the SAME failure worded differently rather than repeated verbatim.
    fixed_test_cases = ensure_expected_result(drop_duplicate_scenarios(turn.candidate_plan.test_cases))
    fixed_test_cases = await drop_semantic_duplicates(fixed_test_cases)
    fixed_features, fixed_test_cases = ensure_features(turn.candidate_plan.features, fixed_test_cases)
    candidate_plan = turn.candidate_plan.model_copy(
        update={"test_cases": fixed_test_cases, "features": fixed_features}
    )

    return {
        "messages": [*seed, AIMessage(turn.assistant_message)],
        "site_context": site_context,
        "extra_dives_used": extra_dives_used,
        "candidate_plan": candidate_plan,
        "run_token": run_token,
        "turn_count": state.get("turn_count", 0) + 1,
        "pending_dive": next_pending_dive,
    }
