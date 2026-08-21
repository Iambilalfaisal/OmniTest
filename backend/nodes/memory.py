"""memory_node: after a run's results are final, distills a few reusable facts about
the target site and writes them to the long-term store. Best-effort by design — a
failure here must never fail the run's visible report, so exceptions are logged and
swallowed rather than propagated.
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage
from langgraph.store.base import BaseStore

from ..core.memory import MEMORY_NAMESPACE, domain_of, get_memory_manager
from ..core.state import QAState


async def memory_node(state: QAState, *, store: BaseStore | None = None) -> dict:
    if store is None:
        return {}

    try:
        domain = domain_of(state["target_url"])
        results_by_id = {r.test_id: r for r in state["test_results"]}

        transcript_lines = [f"Instruction: {state['instruction']}", f"URL: {state['target_url']}", ""]
        for test_case in state["test_cases"]:
            result = results_by_id.get(test_case.test_id)
            if result is None:
                continue
            # expected_result is included so the extractor can tell an actual site
            # constraint ("expected a rejection, got one, with this wording") from a test
            # case that simply expected the wrong thing — without it, a Fail reason alone
            # reads the same either way and gets persisted as a bogus "site quirk". getattr,
            # not direct access: this best-effort node must never raise on a test_case from
            # a checkpoint predating ensure_expected_result (core/run_planning.py).
            #
            # deviations/amended_steps (TestResult, populated from the adaptive worker's
            # Verdict — nodes/worker/nodes.py) are what let a recorded site_quirk feed
            # the NEXT run's planner_node (core/memory.py's retrieve_memory_context is
            # already injected into PLANNER_PROMPT) with something concrete enough to
            # write straight into that flow's steps — "this signup form also asks for a
            # Company name" — instead of the next run's worker hitting, and adapting to,
            # the exact same surprise from scratch. getattr here too, same reason as
            # expected_result above: a TestResult from a checkpoint predating these
            # fields has no value for them.
            deviations = getattr(result, "deviations", None) or []
            amended_steps = getattr(result, "amended_steps", None) or []
            transcript_lines.append(
                f"Goal: {test_case.goal}\nCategory: {test_case.category}\n"
                f"Expected: {getattr(test_case, 'expected_result', None) or '(not specified)'}\n"
                f"Status: {result.status}\nReason: {result.reason}\n"
                f"Deviations from the written steps: {'; '.join(deviations) if deviations else '(none)'}\n"
                f"Steps as actually executed: {'; '.join(amended_steps) if amended_steps else '(matched the written steps)'}\n"
            )
        transcript = "\n".join(transcript_lines)

        # create_memory_manager(...).ainvoke(...) returns list[ExtractedMemory], a
        # NamedTuple of (id, content) — content is already the SiteMemory instance
        # langmem constructed (confirmed against the installed langmem 0.0.30).
        extracted = await get_memory_manager().ainvoke({"messages": [HumanMessage(transcript)]})
        for item in extracted:
            await store.aput((MEMORY_NAMESPACE, domain), item.id, item.content.model_dump())
    except Exception:
        logging.exception("memory_node failed for %s — run's report is unaffected", state.get("target_url"))

    return {}
