"""Per-run LLM request accounting: how many logical LLM calls a run made, split by
model, plus how many ultimately failed and how many of those were rate-limited (and by
which class — see core/llm.rate_limit_class). Surfaces into reporter_node's `summary`
(nodes/reporter.py) so "requests per run" becomes a visible metric — the metric that
actually matters when the binding constraint is API quota rather than wall clock (see
the optimization plan this module is Stage 0 of). LangSmith already traces every call in
detail; what it doesn't give us is a same-process, always-on counter foldable straight
into a run's own summary/history row without a separate query.

Single-process/single-event-loop (confirmed elsewhere in this codebase — see
nodes/worker/session.py's own module docstring), so a module-level dict keyed by run_id
(== LangGraph's `thread_id`) is enough — same precedent as session.py's `_SESSIONS`. No
new QAState channel/reducer needed, which matters: QAState.test_cases has no reducer
today, and parallel worker branches writing one un-reduced channel raises
`InvalidUpdateError` (see core/state.py's `_keep_latest` docstring for a reproduced
case) — a state channel would need one too, and everything this needs is already
process-local exactly like session state is.

Honest limits: this counts LOGICAL calls (one per `.ainvoke()` on a chat model), not
raw HTTP round trips — the SDK's own retries (core/llm.GEMINI_MAX_RETRIES) happen below
this layer and aren't separately countable here; that's what the `google_genai._api_client`
log line (core/logging_config.py) is for instead. Embedding calls emit nothing —
`GoogleGenerativeAIEmbeddings` isn't callback-aware (confirmed: it doesn't inherit
`_BaseGoogleGenerativeAI`) and is out of scope regardless (it already defaults to zero
retries with no way to configure any). Only the QA run path is wired up (api.py's
`_run_config`) — the discovery chat's own LLM calls use a separate, uninstrumented
config, since Stage 0 scopes this to the run path that actually fans out into
concurrent, rate-limited workers.
"""
from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import AsyncCallbackHandler

from .llm import find_gemini_api_error, rate_limit_class

_METRICS: dict[str, dict[str, Any]] = {}


def _blank() -> dict[str, Any]:
    return {"requests": 0, "by_model": {}, "errors": 0, "rate_limited": {"rpm": 0, "rpd": 0}}


class LlmUsageCallback(AsyncCallbackHandler):
    """Attached once per run via api.py's `_run_config()`. LangChain propagates
    callbacks through a ContextVar into every nested `.ainvoke()` call in the graph —
    including ones that never explicitly forward `config=` (confirmed against the
    installed langchain-core: `ensure_config()` reads `var_child_runnable_config` before
    falling back to an empty config) — the same mechanism that already makes LangSmith
    tracing work transparently in this codebase without every call site passing config.
    """

    def __init__(self, run_key: str) -> None:
        self._run_key = run_key

    def _bucket(self) -> dict[str, Any]:
        return _METRICS.setdefault(self._run_key, _blank())

    async def on_chat_model_start(
        self,
        serialized: dict,
        messages: list,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ) -> None:
        bucket = self._bucket()
        bucket["requests"] += 1
        # ls_model_name is the standard LangSmith metadata key every chat model
        # populates via _get_ls_params() (confirmed directly against the installed
        # ChatGoogleGenerativeAI) — reusing it avoids re-deriving the model name from
        # `serialized`'s less stable shape.
        model = (metadata or {}).get("ls_model_name", "unknown")
        bucket["by_model"][model] = bucket["by_model"].get(model, 0) + 1

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        bucket = self._bucket()
        bucket["errors"] += 1
        api_error = find_gemini_api_error(error)
        if api_error is None:
            return
        cls = rate_limit_class(api_error)
        if cls is not None:
            bucket["rate_limited"][cls] += 1


def snapshot(run_key: str) -> dict[str, Any]:
    return _METRICS.get(run_key, _blank())


def discard(run_key: str) -> None:
    _METRICS.pop(run_key, None)
