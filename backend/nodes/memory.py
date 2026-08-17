"""memory_node: after a run's results are final, distills a few reusable facts about
the target site and writes them to the long-term store. Best-effort by design — a
failure here must never fail the run's visible report, so exceptions are logged and
swallowed rather than propagated.
"""
from __future__ import annotations

import logging
import uuid

from ..core.memory import MEMORY_NAMESPACE, domain_of, get_memory_manager
from ..core.models import SiteMemory
from ..core.state import QAState


async def memory_node(state: QAState, *, store=None) -> dict:
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
            transcript_lines.append(f"Goal: {test_case.goal}\nStatus: {result.status}\nReason: {result.reason}\n")
        transcript = "\n".join(transcript_lines)

        # TODO(verify): return shape of create_memory_manager(...).ainvoke(...) — assumed
        # to be a list of SiteMemory-like items or plain dicts matching that schema.
        extracted = await get_memory_manager().ainvoke({"messages": [{"role": "user", "content": transcript}]})
        for item in extracted:
            memory = item if isinstance(item, SiteMemory) else SiteMemory(**item)
            await store.aput((MEMORY_NAMESPACE, domain), str(uuid.uuid4()), memory.model_dump())
    except Exception:
        logging.exception("memory_node failed for %s — run's report is unaffected", state.get("target_url"))

    return {}
