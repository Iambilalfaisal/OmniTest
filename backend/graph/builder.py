"""DAG assembly: planner_node -> plan_review_node (human-in-the-loop) ->
auth_setup_node -> [recon_node fan-out -> recon_join_node barrier] ->
Send-based fan-out over worker_node -> reporter_node -> memory_node."""
from __future__ import annotations

import logging
import os

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from ..core import progress
from ..core.llm import LLM_RETRY_POLICY
from ..core.memory import drop_semantic_duplicates
from ..core.models import FlowReport, ScenarioProposal, TestCase
from ..core.run_planning import ensure_expected_result
from ..core.state import QAState
from ..nodes.auth import build_auth_subgraph
from ..nodes.memory import memory_node
from ..nodes.planner import planner_node
from ..nodes.recon import build_recon_subgraph
from ..nodes.reporter import reporter_node
from ..nodes.worker import build_worker_subgraph

# Kill switch (plan decision D3): unset/false makes the run behave EXACTLY as it did
# before recon existed — route_to_recon skips straight to recon_join_node, which is
# then a harmless no-op (flow_reports stays empty, so it contributes no new test
# cases). Deliberately NOT required — recon is new, unverified-by-a-live-run behavior
# (see the plan's own step 8), so it must be an explicit opt-in, not a silent default.
RECON_ENABLED = os.getenv("RECON_ENABLED", "false").strip().lower() in ("1", "true", "yes")

# Global cap across ALL Features' recon-discovered scenarios combined — separate from
# nodes/recon/nodes.py's own SCENARIOS_PER_FEATURE_MAX, which only bounds a single
# chatty Feature. This is the one that actually bounds total run cost/wall-clock when
# several Features each recon'd close to their own per-feature cap. Recon proposes
# RANKED scenarios; this is where the budget truncates — see recon_join_node.
SCENARIOS_PER_RUN_MAX = int(os.getenv("SCENARIOS_PER_RUN_MAX", "18"))

_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


async def plan_review_node(state: QAState) -> dict:
    """Pauses for a human to approve or edit the generated test plan before any
    worker touches the browser. Portable across instances/restarts — unlike the
    risky-action interrupt in nodes/worker.py, no live external resource is
    involved here, so this checkpoint can be resumed from anywhere.
    """
    decision = interrupt(
        {"type": "plan_review", "test_cases": [tc.model_dump() for tc in state["test_cases"]]}
    )
    test_cases = (
        [TestCase(**tc) for tc in decision["test_cases"]]
        if decision.get("test_cases")
        else state["test_cases"]
    )
    return {"test_cases": test_cases, "plan_approved": bool(decision.get("approved", False))}


def route_after_plan_review(state: QAState) -> str:
    return "auth_setup_node" if state["plan_approved"] else "reporter_node"


def route_to_recon(state: QAState, config: RunnableConfig):
    """auth_setup_node's outgoing edge. When RECON_ENABLED, the plan has at least one
    Feature, and this run's mode is "explore" (QAState.discovery_mode — recon's whole
    point is grounding scenarios in a deep per-Feature exploration, confirmed at ~2.5
    minutes on a real run, and is opt-IN per mode rather than opt-out: "quick" wants a
    single fast pass, and "My Own Plan" (discovery_mode=None — never went through
    discovery at all) means the user already told us exactly what to test, so recon
    silently adding MORE scenarios beyond that would defeat the entire point of that
    mode, not just cost it extra tokens/time), Sends one recon_node instance per Feature
    — each interactively explores that Feature's real flows (nodes/recon/) BEFORE any
    worker touches the browser for graded execution, since a strictly read-only crawl
    (nodes/planner_explore.py) can never see a flow that sits behind a click. Otherwise
    routes straight to recon_join_node, which is then a harmless no-op pass-through —
    this is what makes RECON_ENABLED=false (or a non-explore-mode run) degrade to
    exactly today's behavior, not a parallel code path.

    Also pre-registers every Feature in core/progress.py as `exploring`, before any
    recon_node branch has actually started — same rationale as route_to_workers' own
    pre-registration below: without this, the SSE `progress` payload (api.py) would show
    nothing at all for however long recon takes, indistinguishable from a stalled run.
    """
    if not RECON_ENABLED or not state.get("features") or state.get("discovery_mode") != "explore":
        return "recon_join_node"
    run_id = config["configurable"]["thread_id"]
    for feature in state["features"]:
        progress.register_feature(run_id, feature.feature_id, name=feature.name)

    # Pre-register the BASELINE (planner-authored) test cases as `queued` RIGHT NOW,
    # before recon starts — same rationale as route_to_workers' own pre-registration
    # (see its docstring) and register_feature above: without this, test cards only
    # appear once recon_join_node finishes (after all features are explored), making
    # the UI show nothing but "Exploring application features…" for the full recon
    # duration (confirmed at ~2.5 minutes when RECON_MAX_TURNS=12). Pre-registering
    # here lets the cards appear as QUEUED immediately after the plan is approved,
    # making the run's progress visible during the entire recon phase.
    # Recon-discovered test cases can't be pre-registered (they don't exist yet) —
    # they appear via route_to_workers' loop once recon finishes, which is correct.
    for tc in state.get("test_cases", []):
        progress.register(run_id, tc.test_id, total_steps=len(tc.steps))

    baseline_cases = state.get("test_cases", [])
    auth_state = state.get("auth_storage_state")
    return [
        Send(
            "recon_node",
            {
                "target_url": state["target_url"],
                "run_token": state["run_token"],
                "discovery_context": state.get("discovery_context", ""),
                "feature": feature,
                # Filtered to THIS Feature — see nodes/recon/state.py's ReconState.existing_test_cases
                # for why recon's synthesis step needs this to avoid restating baseline coverage.
                "existing_test_cases": [tc for tc in baseline_cases if tc.feature_id == feature.feature_id],
                # Unconditional (unlike route_to_workers' per-test-case requires_auth gate) —
                # recon has no per-Feature requires_auth concept, and exploring already
                # logged in never hurts a flow that doesn't need auth. See ReconState's
                # own comment for the login-wall failure mode this avoids.
                "auth_storage_state": auth_state,
            },
        )
        for feature in state["features"]
    ]


def _test_case_from_scenario(flow: FlowReport, scenario: ScenarioProposal, index: int) -> TestCase:
    return TestCase(
        test_id=f"{flow.flow_id}-{index + 1}",
        goal=scenario.goal,
        category=scenario.category,
        priority=scenario.priority,
        requires_auth=False,  # recon writes each scenario's OWN entry steps inline (see ScenarioProposal.steps)
        preconditions=[f"Discovered by recon: {flow.flow_name}"],
        expected_result=scenario.expected_result,
        steps=scenario.steps,
        feature_id=flow.feature_id,
        flow_id=flow.flow_id,
        origin="recon",
        discovery_rationale=scenario.rationale,
    )


async def recon_join_node(state: QAState) -> dict:
    """Barrier node reached via a PLAIN edge from recon_node (build_graph below), not a
    conditional one. CONFIRMED 2026-08-22, empirically, against installed langgraph
    1.2.11: this is required for correctness, not a style choice — a conditional edge
    issuing a LATER Send into a node a normal edge already settled past double-fires the
    downstream node (reporter_node) with corrupted, compounding state (reproduced
    directly in a minimal harness). A plain edge forces LangGraph to wait for EVERY
    currently-active recon_node branch (one per Feature) to finish before this runs even
    once, so route_to_workers (next) always sees the FULL, final set of
    recon-discovered scenarios — never a partial one. When route_to_recon skipped recon
    entirely, `flow_reports` is simply empty and this is a harmless no-op.

    Applies SCENARIOS_PER_RUN_MAX globally across ALL Features' FlowReports combined,
    sorted by (priority, rank) — nodes/recon/nodes.py's own SCENARIOS_PER_FEATURE_MAX
    already capped each Feature individually; this only trims further if MULTIPLE
    Features each proposed close to their own per-feature cap. Truncation is logged, not
    silent — a silently-capped run would read as complete coverage when it isn't.

    async (unlike most of this module's plain functions): needs to await
    drop_semantic_duplicates below, which is not something a LangGraph reducer
    (core/state.py's _merge_test_cases, the actual consumer of this node's returned
    test_cases) could ever do — reducers run synchronously inline during a channel
    update, with no support for awaiting I/O. This node is where the async embedding
    call has to happen instead, before the reducer's synchronous exact-match pass ever
    sees the result.
    """
    all_scenarios = [(flow, s) for flow in state.get("flow_reports", []) for s in flow.scenarios]
    all_scenarios.sort(key=lambda item: (_PRIORITY_ORDER.get(item[1].priority, 1), item[1].rank))
    kept = all_scenarios[:SCENARIOS_PER_RUN_MAX]
    dropped = len(all_scenarios) - len(kept)
    if dropped > 0:
        logging.warning(
            "recon_join_node: SCENARIOS_PER_RUN_MAX=%d — dropped %d of %d recon-discovered scenarios",
            SCENARIOS_PER_RUN_MAX,
            dropped,
            len(all_scenarios),
        )

    by_flow_index: dict[str, int] = {}
    accepted = []
    for flow, scenario in kept:
        i = by_flow_index.get(flow.flow_id, 0)
        accepted.append(_test_case_from_scenario(flow, scenario, i))
        by_flow_index[flow.flow_id] = i + 1

    # Same backstop api.py/planner.py already apply to a baseline plan's TestCases
    # (core/run_planning.py's own docstring: Gemini's structured output can silently
    # omit a required field, confirmed live) — recon-discovered cases sit behind the
    # SAME kind of structured-output call (nodes/recon/nodes.py's ScenarioProposalOut)
    # and had never gotten this applied before now.
    accepted = ensure_expected_result(accepted)
    # Semantic backstop for D3's own acknowledged risk: recon-discovered scenarios
    # restating the planner's baseline in different words — recon_agent_node's
    # existing_test_cases context (nodes/recon/state.py) only ever told the model not
    # to; this is the deterministic check for when it does anyway. Compared against
    # state["test_cases"] as it stands ENTERING this node — the baseline plan, untouched
    # since planner_node/plan_review_node, since nothing between there and here writes
    # to it.
    accepted = await drop_semantic_duplicates(accepted, reference=state.get("test_cases", []))
    return {"test_cases": accepted}


def route_to_workers(state: QAState, config: RunnableConfig):
    """recon_join_node's outgoing edge (formerly auth_setup_node's directly — recon_node
    /recon_join_node now sit between the two; see build_graph below). By construction
    (see route_entry and route_after_plan_review) this is only ever reached with
    plan_approved already True, so unlike the pre-Stage-3 version of this function
    there's no unapproved-plan branch to check here anymore. Stage 3: auth_storage_state
    (set once by auth_setup_node) is only threaded into a given worker's payload when
    that test case actually declared requires_auth — a case that didn't ask for a shared
    login shouldn't have one silently injected into its browser.

    Reads state["test_cases"] AFTER recon_join_node's merge (core/state.py's
    _merge_test_cases reducer) — this function itself needs no recon-awareness at all,
    since a recon-discovered scenario is a real TestCase by the time it gets here,
    indistinguishable from a planner-authored one to everything downstream.

    Also pre-registers every test case in core/progress.py as `queued`, before any
    worker_node branch has actually started — this is the one point in the whole graph
    where every TestCase in the approved (and now recon-augmented) plan is visited
    together with the run's thread_id, so it's the only place that can seed the FULL set
    up front. Without this, the SSE `progress` payload (api.py) would only ever show a
    test case once its own agent_node's first turn happens to land, so a card that
    hasn't been scheduled onto a free MAX_CONCURRENT_WORKERS slot yet would be
    indistinguishable from one that doesn't exist.
    """
    run_id = config["configurable"]["thread_id"]
    auth_state = state.get("auth_storage_state")
    sends = []
    for test in state["test_cases"]:
        progress.register(run_id, test.test_id, total_steps=len(test.steps))
        sends.append(
            Send(
                "worker_node",
                {
                    "target_url": state["target_url"],
                    "test_case": test,
                    "auth_storage_state": auth_state if test.requires_auth else None,
                },
            )
        )
    return sends


def route_entry(state: QAState):
    """Chat-approved runs (backend/api.py's POST /discover/{id}/message approve handler)
    arrive with test_cases/plan_approved already set by the discovery conversation —
    skip planner_node's LLM call and plan_review_node's interrupt (already approved
    conversationally) straight to auth_setup_node. Runs started the old way (POST /runs,
    or any future programmatic caller) arrive with plan_approved=False and take the
    original planner_node -> plan_review_node path, unchanged.
    """
    if state.get("plan_approved") and state.get("test_cases"):
        return "auth_setup_node"
    return "planner_node"


def build_graph(checkpointer, store=None):
    graph = StateGraph(QAState)

    graph.add_node("planner_node", planner_node, retry_policy=LLM_RETRY_POLICY)
    graph.add_node("plan_review_node", plan_review_node)
    # Subgraph-as-node, like worker_node below — no retry_policy at this level, since
    # each of its own nodes (nodes/auth/nodes.py, nodes/recon/nodes.py) already carries
    # its own.
    graph.add_node("auth_setup_node", build_auth_subgraph())
    graph.add_node("recon_node", build_recon_subgraph())
    graph.add_node("recon_join_node", recon_join_node)
    graph.add_node("worker_node", build_worker_subgraph())
    graph.add_node("reporter_node", reporter_node)
    graph.add_node("memory_node", memory_node)

    graph.add_conditional_edges(START, route_entry, ["planner_node", "auth_setup_node"])
    graph.add_edge("planner_node", "plan_review_node")
    graph.add_conditional_edges("plan_review_node", route_after_plan_review, ["auth_setup_node", "reporter_node"])
    graph.add_conditional_edges("auth_setup_node", route_to_recon, ["recon_node", "recon_join_node"])
    # PLAIN edge, not conditional — see recon_join_node's own docstring for why this is
    # load-bearing, not stylistic: it's what forces LangGraph to wait for every
    # Send-spawned recon_node branch (one per Feature) before recon_join_node runs.
    graph.add_edge("recon_node", "recon_join_node")
    graph.add_conditional_edges("recon_join_node", route_to_workers, ["worker_node"])
    graph.add_edge("worker_node", "reporter_node")
    graph.add_edge("reporter_node", "memory_node")
    graph.add_edge("memory_node", END)

    return graph.compile(checkpointer=checkpointer, store=store)
