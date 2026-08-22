"""Recon subgraph nodes: `recon_agent_node` (one LLM turn) -> `recon_tool_node` (one
tool call) -> `recon_plan_node` (final synthesis + teardown) — the same shape as
nodes/worker/nodes.py's worker loop and nodes/auth/nodes.py's auth loop, and for the
same reason: `interrupt()` replay safety.

Runs once per Feature the planner/discovery chat identified (Send-spawned, one instance
per Feature — see graph/builder.py), INTERACTIVELY exploring that feature's real flows:
clicking into a sign-up form to see whether it also offers Google OAuth, phone/OTP, an
invite code, a multi-step wizard, and reading the site's OWN real validation messages.
This is the evidence-acquisition step the rest of the architecture is built around — a
strictly read-only crawl (nodes/planner_explore.py's crawl_site) can never see any of
this, since every one of those flows sits behind a click, and TEST_CASE_AUTHORING_GUIDELINES
correctly forbids the planner from inventing a flow it has no evidence for.

`recon_plan_node` is this subgraph's `verdict_node`: ONE final structured-output LLM
call over the whole exploration transcript, turning grounded observations into ranked
ScenarioProposals — never the exploring agent itself, which would mean asking a
hot-loop, cheap-model turn to also reason well about "is this scenario worth proposing
and how does it rank," the same worker/verdict split this codebase already uses.

Session lifecycle goes through nodes/worker/session.py's SHARED `_SESSIONS` cache
(keyed `f"{thread_id}:recon:{feature_id}"`), exactly like nodes/auth/nodes.py's shared
login — already swept by `close_sessions_for_thread`'s prefix scan and the idle-session
reaper with no changes needed there, and safe to pause via interrupt()/ask_human the
same way a worker's or auth's session is.
"""
from __future__ import annotations

import logging
import os
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ...core import progress
from ...core.llm import LLM_RETRY_POLICY, ModelRole, with_fallback
from ...core.models import TEST_CASE_AUTHORING_GUIDELINES, FlowReport
from ...core.run_context import run_id_var
from ...mcp.client import invoke_tool, invoke_tool_or_error_text
from ..agent_loop import (
    ASK_HUMAN_TOOL_NAME,
    EXCLUDED_TOOL_NAMES,
    SCENARIO_DEADLINE_SECONDS,
    ask_human_and_reply,
    ask_human_tool,
    compact_history,
    new_deadline,
    review_if_risky,
    stale_snapshot_replacements,
    truncate_tool_result,
)
from ..worker.session import SessionGoneError, discard_session, get_session
from .state import ReconState

# Exploring ONE feature's several possible flows genuinely needs more turns than
# executing one scripted test case's steps (nodes/worker/nodes.py's MAX_TOOL_TURNS=8) —
# but this is still a single mechanical exploration pass, not an open-ended budget.
RECON_MAX_TURNS = int(os.getenv("RECON_MAX_TURNS", "12"))

# Per-Feature cap on how many of recon_plan_node's own ranked ScenarioProposals survive
# into real TestCases — applied per FlowReport (by its own `rank`, lower = more
# important) before this subgraph returns. A SEPARATE, global SCENARIOS_PER_RUN_MAX cap
# (graph/builder.py's recon join node, which sees every Feature's FlowReports at once)
# truncates again across the whole run — this one bounds a single chatty Feature from
# dominating that shared budget before the run-wide cap even gets a chance to balance it.
SCENARIOS_PER_FEATURE_MAX = int(os.getenv("SCENARIOS_PER_FEATURE_MAX", "6"))

RECON_SUBJECT_PREFIX = "__recon__"


def _subject_id(feature_id: str) -> str:
    """Distinct from any real test_id (lowercase-hyphenated words per
    TEST_CASE_AUTHORING_GUIDELINES) or AUTH_SUBJECT_ID (nodes/auth/nodes.py's
    "__auth__") — identifies this Feature's exploration to interrupt()/ask_human
    payloads and the frontend's human-review panel.
    """
    return f"{RECON_SUBJECT_PREFIX}{feature_id}"


def _session_key(config: RunnableConfig, feature_id: str) -> str:
    return f"{config['configurable']['thread_id']}:recon:{feature_id}"


RECON_SYSTEM_PROMPT = """You are a QA reconnaissance agent. Your job is NOT to test or grade anything — you \
are exploring one real, specific area of a real website to find out what actually exists there, so a later \
step can write well-grounded test cases from what you observed instead of guessing at what a generic site \
might have.

You have been assigned ONE feature to explore: {feature_name} — {feature_description}

How to explore:
1. Take a `browser_snapshot` first, then navigate toward this feature from the target page — click whatever \
visibly leads there (e.g. a 'Sign up' or 'Log in' link), or navigate directly if you already know the URL.
2. Once there, look for EVERY distinct way to accomplish this feature, not just the first one you see: the \
primary form, but also any alternate button (Google/OAuth, phone number), an invite-code field, a \
multi-step flow, or any other visible variation. Click far enough into each one to see its real fields and \
any real validation/error text — you do NOT need to complete any flow end to end.
3. Deliberately trigger at least one validation error where practical (submit a required field empty, or \
type an obviously invalid value) so you can report the SITE'S OWN real error wording, not a guess at it.
4. Note every distinct field label, button, and validation message you actually saw, and note plainly \
anything that stopped you going further (a CAPTCHA, an OTP/email verification step, a paywall, a required \
invite code you don't have) — never guess what is behind something you couldn't get past.
5. You do not need to finish signing up, logging in, or submitting anything for real — reaching far enough \
to SEE what a flow needs is the goal, not completing it. Never delete, purchase, or pay for anything.

If you hit something you cannot safely finish exploring on your own — a secret you were not given, a real \
payment, a destructive action — call `ask_human` and say exactly what you found and what you would need to \
go further. Otherwise, once you believe you have seen what this feature actually offers (or your turns are \
running out), stop calling tools and summarize what you found in plain text — a partial report of what you \
DID see is far more useful than continuing to explore with nothing left to act on.
"""

# Built with `+`, not an f-string, for the same reason WORKER_SYSTEM_PROMPT/
# AUTH_SETUP_SYSTEM_PROMPT already do (nodes/worker/nodes.py, nodes/auth/nodes.py):
# TEST_CASE_AUTHORING_GUIDELINES (core/models.py) contains one deliberately-unescaped
# `{run_token}` placeholder, meant to survive into a LATER `.format()` call — an
# f-string embedding it here would substitute its text immediately, leaving the
# literal, uninterpolated characters "{run_token}" in the final prompt instead of the
# real value. `{feature_name}` is filled by that same later `.format()` call.
RECON_PLAN_PROMPT = (
    "Exploration is over. Based ONLY on what you actually observed above (never invent a field, button, or "
    'message nothing above mentions), report every distinct flow you found for this feature ("{feature_name}") '
    "and propose ranked test scenarios for each. Use this run token for any generated test data a scenario "
    "would need (e.g. a fresh signup email): {run_token}\n\n"
) + TEST_CASE_AUTHORING_GUIDELINES


class ScenarioProposalOut(BaseModel):
    goal: str = Field(description="Same contract as TestCase.goal — one sentence, '<action> should <outcome>'.")
    category: str = Field(description="One of: happy_path, edge_case, negative, error_handling.")
    priority: str = Field(default="medium", description="One of: high, medium, low.")
    rationale: str = Field(
        description="Why this scenario matters for THIS application specifically, grounded in what was "
        "actually observed during exploration — never a generic justification."
    )
    rank: int = Field(description="Priority ordering within this flow — lower is more important.")
    expected_result: str = Field(
        description="Same contract as TestCase.expected_result — the ONE observable outcome that decides "
        "Pass/Fail, naming something actually seen (or clearly implied) during exploration. Never vague."
    )
    steps: list[str] = Field(
        description="Same contract as TestCase.steps — ordered, one action per step, every typed value "
        "literal and in quotes, using the EXACT visible label you saw during exploration. Include getting "
        "to this flow's entry point as the first steps, since this scenario will run in a fresh, isolated "
        "browser that starts from the target URL, not from wherever you happened to explore from."
    )


class FlowReportOut(BaseModel):
    flow_name: str = Field(description="Human-readable name for this specific flow, e.g. 'Google OAuth'.")
    observed_fields: list[str] = Field(default_factory=list)
    observed_validation: list[str] = Field(default_factory=list)
    blocked_by: str | None = Field(default=None)
    scenarios: list[ScenarioProposalOut] = Field(default_factory=list)


class ReconOutput(BaseModel):
    """Structured-output contract recon_plan_node's synthesis LLM call is constrained
    to. Kept local to this module (not core/models.py) — same convention as
    nodes/worker/nodes.py's Verdict — since this shape only exists to be immediately
    converted into FlowReport/ScenarioProposal (core/models.py) via _to_flow_reports
    below, never persisted or passed around on its own.
    """

    flows: list[FlowReportOut] = Field(
        description="One entry per DISTINCT flow actually observed for this feature — e.g. separate entries "
        "for 'Email + Password' and 'Google OAuth' if both exist. Empty list if nothing could be explored."
    )


def _recon_plan_llm():
    return with_fallback(ModelRole.PLANNER, lambda m: m.with_structured_output(ReconOutput), temperature=0)


def _to_flow_reports(feature_id: str, output: ReconOutput) -> list[FlowReport]:
    reports = []
    for i, flow in enumerate(output.flows):
        flow_id = f"{feature_id}-flow-{i + 1}"
        # Truncate by rank (ascending — lower is more important), not arbitrarily —
        # see SCENARIOS_PER_FEATURE_MAX's own comment for why this per-flow cap is
        # separate from the run-wide one applied later in graph/builder.py.
        ranked = sorted(flow.scenarios, key=lambda s: s.rank)[:SCENARIOS_PER_FEATURE_MAX]
        reports.append(
            FlowReport(
                feature_id=feature_id,
                flow_id=flow_id,
                flow_name=flow.flow_name,
                observed_fields=flow.observed_fields,
                observed_validation=flow.observed_validation,
                blocked_by=flow.blocked_by,
                scenarios=[
                    {
                        "goal": s.goal,
                        "category": s.category,
                        "priority": s.priority,
                        "rationale": s.rationale,
                        "rank": s.rank,
                        "expected_result": s.expected_result,
                        "steps": s.steps,
                    }
                    for s in ranked
                ],
            )
        )
    return reports


async def recon_agent_node(state: ReconState, config: RunnableConfig) -> dict:
    feature = state["feature"]

    # Wall-clock backstop (agent_loop.SCENARIO_DEADLINE_SECONDS) — mirrors
    # nodes/worker/nodes.py's agent_node exactly. recon_plan_node turns abort_reason
    # into a partial/empty result for just this Feature instead of crashing the run.
    now = time.monotonic()
    deadline_at = state.get("deadline_at")
    if deadline_at is None:
        deadline_at = now + SCENARIO_DEADLINE_SECONDS
    elif now > deadline_at:
        return {"pending_tool_calls": [], "abort_reason": f"exceeded its {SCENARIO_DEADLINE_SECONDS:.0f}s execution deadline"}

    key = _session_key(config, feature.feature_id)
    try:
        _, tools, tool_map = await get_session(key)
    except Exception as exc:
        logging.exception("recon_agent_node: failed to open a session for %s — aborting this feature's recon", feature.feature_id)
        return {"pending_tool_calls": [], "abort_reason": f"could not open a browser session ({exc})"}
    offered_tools = [t for t in tools if t.name not in EXCLUDED_TOOL_NAMES]

    history = state.get("messages")
    seed: list = []
    if not history:
        try:
            await invoke_tool(tool_map["browser_navigate"], {"url": state["target_url"]})
        except Exception:
            logging.exception("recon_agent_node: failed to navigate to the target URL for %s", feature.feature_id)
            discard_session(key)
            return {"pending_tool_calls": [], "abort_reason": "failed to navigate to the target URL"}

        seed = [
            SystemMessage(
                RECON_SYSTEM_PROMPT.format(feature_name=feature.name, feature_description=feature.description)
            ),
            HumanMessage(
                f"Target URL: {state['target_url']}\n"
                f"Known context (existing credentials/preferences, if any): "
                f"{state.get('discovery_context') or 'None provided.'}"
            ),
        ]
        history = seed

    replacements = stale_snapshot_replacements(history)

    # WORKER, not PLANNER: this is hot-loop, mechanical tool-calling exploration — the
    # exact same reasoning nodes/auth/nodes.py's auth_agent_node already uses. The one
    # LLM call that needs to reason WELL about what was found (recon_plan_node, below)
    # uses PLANNER instead, mirroring the worker/verdict model split.
    model = with_fallback(
        ModelRole.WORKER, lambda m: m.bind_tools([*offered_tools, ask_human_tool]), temperature=0.3
    )
    response = await model.ainvoke(compact_history(history))

    return {
        "messages": [*seed, *replacements, response],
        "pending_tool_calls": response.tool_calls,
        "turn_count": state.get("turn_count", 0) + 1,
        "deadline_at": deadline_at,
    }


async def recon_tool_node(state: ReconState, config: RunnableConfig) -> dict:
    feature = state["feature"]
    subject_id = _subject_id(feature.feature_id)
    key = _session_key(config, feature.feature_id)
    call, remaining = state["pending_tool_calls"][0], state["pending_tool_calls"][1:]

    if call["name"] == ASK_HUMAN_TOOL_NAME:

        async def _tool_map() -> dict:
            _, _, tm = await get_session(key)
            return tm

        try:
            reply, _answer_text, _sensitive = await ask_human_and_reply(call, _tool_map, subject_id=subject_id)
        except Exception as exc:
            logging.exception("recon_tool_node: failed to open a session resuming ask_human for %s", feature.feature_id)
            return {"pending_tool_calls": [], "abort_reason": f"could not open a browser session ({exc})"}
        # No sensitive_answers channel here, unlike WorkerState — an exploration
        # transcript never reaches a graded verdict or leaves this subgraph the way a
        # test case's does, so there's nothing downstream for a secret to leak into.
        return {"messages": [reply], "pending_tool_calls": remaining, "deadline_at": new_deadline()}

    decision = await review_if_risky(call, subject_id=subject_id)
    deadline_update = {"deadline_at": new_deadline()} if decision is not None else {}
    if decision is not None and not decision.get("approved", False):
        blocked = ToolMessage(
            content=f"Blocked by human reviewer: {decision.get('reason', 'not approved')}",
            tool_call_id=call["id"],
            name=call["name"],
        )
        return {"messages": [blocked], "pending_tool_calls": remaining, **deadline_update}

    try:
        _, _, tool_map = await get_session(key)
    except Exception as exc:
        logging.exception("recon_tool_node: failed to open a session for %s", feature.feature_id)
        return {"pending_tool_calls": [], "abort_reason": f"could not open a browser session ({exc})"}
    result_text = truncate_tool_result(await invoke_tool_or_error_text(tool_map[call["name"]], call["args"]))

    return {
        "messages": [ToolMessage(content=result_text, tool_call_id=call["id"], name=call["name"])],
        "pending_tool_calls": remaining,
        **deadline_update,
    }


async def recon_plan_node(state: ReconState, config: RunnableConfig) -> dict:
    feature = state["feature"]
    key = _session_key(config, feature.feature_id)
    run_id = run_id_var.get()

    abort_reason = state.get("abort_reason")
    if abort_reason:
        logging.warning("recon_plan_node: %s aborted before synthesis — %s", feature.feature_id, abort_reason)
        progress.update_feature(run_id, feature.feature_id, phase="done", scenario_count=0)
        # Best-effort teardown, tolerant of a session that never opened at all (a
        # session-open timeout means there's nothing to close) — mirrors
        # nodes/worker/nodes.py's _abort_verdict / nodes/auth/nodes.py's auth_save_node.
        try:
            handle, _, tool_map = await get_session(key, require_existing=True)
        except SessionGoneError:
            return {"flow_reports": []}
        close_tool = tool_map.get("browser_close")
        if close_tool is not None:
            try:
                await invoke_tool(close_tool, {})
            except Exception:
                logging.exception("recon_plan_node: failed to close browser while aborting %s", feature.feature_id)
        discard_session(key)
        await handle.close()
        return {"flow_reports": []}

    # require_existing=True: same reasoning as verdict_node/auth_save_node — this node
    # must never silently open a fresh, unnavigated browser only to "synthesize" a
    # report about a page it never actually explored.
    handle, _, tool_map = await get_session(key, require_existing=True)

    output = await _recon_plan_llm().ainvoke(
        compact_history(state["messages"])
        + [
            HumanMessage(
                RECON_PLAN_PROMPT.format(feature_name=feature.name, run_token=state.get("run_token", ""))
            )
        ]
    )
    flow_reports = _to_flow_reports(feature.feature_id, output)
    progress.update_feature(
        run_id, feature.feature_id, phase="done", scenario_count=sum(len(fr.scenarios) for fr in flow_reports)
    )

    try:
        close_tool = tool_map.get("browser_close")
        if close_tool is not None:
            await invoke_tool(close_tool, {})
    except Exception:
        logging.exception("recon_plan_node: failed to close browser for %s", feature.feature_id)
    finally:
        discard_session(key)
        await handle.close()

    return {"flow_reports": flow_reports}


def route_after_recon_agent(state: ReconState) -> str:
    return "recon_tool_node" if state["pending_tool_calls"] else "recon_plan_node"


def route_after_recon_tool(state: ReconState) -> str:
    # Checked first: an abort (session-open timeout) leaves pending_tool_calls empty
    # with turn_count possibly still under RECON_MAX_TURNS — without this, that would
    # route back to recon_agent_node instead of recon_plan_node's abort handling, same
    # reasoning as nodes/worker/nodes.py's route_after_tool.
    if state.get("abort_reason"):
        return "recon_plan_node"
    if state["pending_tool_calls"]:
        return "recon_tool_node"
    return "recon_plan_node" if state["turn_count"] >= RECON_MAX_TURNS else "recon_agent_node"


def build_recon_subgraph():
    sub = StateGraph(ReconState)
    sub.add_node("recon_agent_node", recon_agent_node, retry_policy=LLM_RETRY_POLICY)
    # No retry on recon_tool_node — same reasoning as nodes/worker/nodes.py's
    # tool_node: retrying after a raised exception risks re-invoking a tool that
    # already had a real side effect.
    sub.add_node("recon_tool_node", recon_tool_node)
    sub.add_node("recon_plan_node", recon_plan_node, retry_policy=LLM_RETRY_POLICY)

    sub.add_edge(START, "recon_agent_node")
    sub.add_conditional_edges("recon_agent_node", route_after_recon_agent, ["recon_tool_node", "recon_plan_node"])
    sub.add_conditional_edges(
        "recon_tool_node", route_after_recon_tool, ["recon_tool_node", "recon_agent_node", "recon_plan_node"]
    )
    sub.add_edge("recon_plan_node", END)

    # No checkpointer passed — inherits the parent graph's, required for interrupt()
    # inside recon_tool_node to actually persist, same as build_worker_subgraph()/
    # build_auth_subgraph().
    return sub.compile()
