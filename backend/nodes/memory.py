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
            transcript_lines.append(
                f"Goal: {test_case.goal}\nCategory: {test_case.category}\nStatus: {result.status}\nReason: {result.reason}\n"
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
