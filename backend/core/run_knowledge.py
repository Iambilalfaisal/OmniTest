"""Per-run, in-process knowledge shared across a run's PARALLEL worker branches — so
one test case's answered `ask_human` question, or a deviation it worked out on its own,
can save a sibling test case from hitting the exact same wall a second time. Same
design precedent as core/llm_metrics.py: a module-level dict keyed by run_id (==
LangGraph's `thread_id`), single-process/single-event-loop (confirmed there — see that
module's own docstring), so no new QAState channel/reducer is needed and nothing here
has to reason about concurrent writes across processes.

Two independent kinds of entry, deliberately not merged into one shape:

- `facts` — short, non-secret text a worker noticed while adapting to a real page (a
  deviation, from WORKER_SYSTEM_PROMPT's `PROGRESS: ... note=...` line). Safe to inject
  into ANY sibling's prompt context as-is (see `context_block`).
- `answers` — an `ask_human` question and its answer. NEVER injected into general
  prompt context, even when not sensitive: only `find_answer` (an exact, narrow lookup
  keyed by the question itself) can retrieve one, and it hands the real value back to
  whichever tool_node explicitly asked for it — that caller is responsible for treating
  a sensitive one exactly as if its OWN interrupt() had just returned it (i.e. adding it
  to its own WorkerState.sensitive_answers so verdict_node's redaction still catches
  it). Keeping this split is what makes a sensitive answer's blast radius stay bounded
  to workers that actually hit an equivalent question, not every worker in the run.

Honest limits: process-local and best-effort, matching session.py/llm_metrics.py — a
resume landing on a different process than the one that recorded a fact or answer loses
it, same as those. Question matching is plain normalized-string equality, not semantic
similarity — deliberate: a false REUSE would feed a worker a wrong answer, which is
worse than one extra `ask_human` interrupt for a reworded-but-equivalent question.
"""
from __future__ import annotations

import re
from typing import Any

_FACTS: dict[str, list[str]] = {}
_ANSWERS: dict[str, list[dict[str, Any]]] = {}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def record_fact(run_id: str, note: str) -> None:
    """Called from agent_node when a turn's parsed PROGRESS line reports `deviated`
    with a non-empty note — see nodes/worker/nodes.py's _parse_progress_line. Best-
    effort by construction (the note is free text a model chose to write), so no
    validation beyond "non-empty" is attempted here.
    """
    note = note.strip()
    if not note:
        return
    _FACTS.setdefault(run_id, []).append(note)


def context_block(run_id: str, *, max_facts: int) -> str:
    """The most recent `max_facts` facts, formatted for direct inclusion in a fresh
    worker's outbound prompt. "" when there is nothing yet — the common case for the
    first test case(s) to reach agent_node in a run, and for every run where nothing
    deviated at all.
    """
    facts = _FACTS.get(run_id, [])
    if not facts:
        return ""
    lines = "\n".join(f"- {fact}" for fact in facts[-max_facts:])
    return (
        "Learned during this run by other test cases against this same site — treat as "
        f"true, not merely possible:\n{lines}"
    )


def find_answer(run_id: str, question: str) -> dict[str, Any] | None:
    """An already-recorded answer to a normalized-equal `question` asked earlier in
    THIS run, or None. Returns the full entry (`{"answer": str, "sensitive": bool}`),
    not just the answer text, since a caller reusing a sensitive answer must know to
    track it as sensitive itself (see this module's docstring).
    """
    norm = _normalize(question)
    for entry in _ANSWERS.get(run_id, []):
        if entry["norm_question"] == norm:
            return {"answer": entry["answer"], "sensitive": entry["sensitive"]}
    return None


def record_answer(run_id: str, question: str, answer: str, *, sensitive: bool) -> None:
    """Called from tool_node's ask_human branch AFTER interrupt() returns — never
    before, for the same reason ask_human_and_reply's own get_tool_map() call is
    deferred until after (see agent_loop.py's docstring): execution before an
    interrupt() call is discarded on resume, so writing here first would record an
    answer that node invocation never actually finished handling.
    """
    _ANSWERS.setdefault(run_id, []).append(
        {"question": question, "norm_question": _normalize(question), "answer": answer, "sensitive": sensitive}
    )


def discard(run_id: str) -> None:
    _FACTS.pop(run_id, None)
    _ANSWERS.pop(run_id, None)
