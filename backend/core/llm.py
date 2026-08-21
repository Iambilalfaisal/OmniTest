"""Central Gemini chat-model factory + the one RetryPolicy every LLM-calling graph node
uses — exists to fix a verified amplification bug.

`ChatGoogleGenerativeAI`'s own client already retries a 429/5xx up to `max_retries`
times — a cheap, SDK-level HTTP retry (google-genai's own backoff: 1.0s initial, base 2,
max 60s, jitter). Before this module, EVERY LLM-calling node ALSO carried
`RetryPolicy(max_attempts=3)`, which re-runs the WHOLE node body (re-crawling a page,
re-snapshotting a browser, reconstructing the model) on top of that SDK retry — worst
case 6 SDK attempts (the old, never-overridden default) x 3 node attempts = 18 round
trips for one logical call. `LLM_RETRY_POLICY` below stops the node layer from retrying
anything the SDK layer already tried (and is either still trying or has given up on),
while still letting node-level retries catch genuine non-API failures (e.g. a Playwright
tool-call exception, a connection drop before a request is even sent).
"""
from __future__ import annotations

import logging
import os
from enum import Enum

# google-genai is the underlying Gemini SDK — a transitive dependency of
# langchain-google-genai (not listed separately in requirements.txt), always present
# alongside it since langchain_google_genai imports it directly itself.
from google.genai.errors import APIError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.types import RetryPolicy, default_retry_on

# Total HTTP attempts INCLUDING the original request (confirmed against the installed
# langchain-google-genai: it builds `HttpRetryOptions(attempts=max_retries)` verbatim) —
# not "retries in addition to the original". 3 = original + 2 retries.
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))


class ModelRole(str, Enum):
    PLANNER = "planner"
    WORKER = "worker"
    VERDICT = "verdict"
    MEMORY = "memory"


# First env var present (and truthy) wins; the LAST name in each tuple is required (no
# fallback), so a genuinely unconfigured role still fails loudly at first use exactly
# like the old `os.environ["..._MODEL"]` call sites did. VERDICT falls back to
# WORKER_MODEL, so introducing VERDICT_MODEL is a config-only, backward-compatible
# change: unset, verdict_node resolves to the same model it always has.
_ROLE_ENV: dict[ModelRole, tuple[str, ...]] = {
    ModelRole.PLANNER: ("PLANNER_MODEL",),
    ModelRole.WORKER: ("WORKER_MODEL",),
    ModelRole.VERDICT: ("VERDICT_MODEL", "WORKER_MODEL"),
    ModelRole.MEMORY: ("MEMORY_EXTRACTION_MODEL",),
}


def _resolve_model_name(role: ModelRole) -> str:
    names = _ROLE_ENV[role]
    for name in names[:-1]:
        value = os.environ.get(name)
        if value:
            return value
    return os.environ[names[-1]]  # required — raises KeyError if truly unset


def get_chat_model(role: ModelRole, *, temperature: float = 0.0, **kwargs) -> ChatGoogleGenerativeAI:
    """Lazy — called at use time, not import time, so importing a module that reaches
    this (e.g. at API startup, before any run) doesn't require GOOGLE_API_KEY until a
    run actually needs it. Matches the convention this codebase already used per-call-site
    before this module existed (planner._planner_llm, discovery._discovery_llm,
    core.memory.get_memory_manager) — this just centralizes it so GEMINI_MAX_RETRIES and
    future per-role tuning live in one place instead of drifting across five sites.
    """
    return ChatGoogleGenerativeAI(
        model=_resolve_model_name(role),
        temperature=temperature,
        max_retries=GEMINI_MAX_RETRIES,
        **kwargs,
    )


def find_gemini_api_error(exc: BaseException) -> APIError | None:
    """Walk `__cause__` looking for the real `google.genai.errors.APIError`.

    A 429/5xx from Gemini does NOT surface as `APIError` directly through
    `ChatGoogleGenerativeAI` for client errors: confirmed directly against the installed
    langchain-google-genai, `chat_models._handle_client_error` catches the real
    `google.genai.errors.ClientError` and re-raises `ChatGoogleGenerativeAIError(msg)
    from e` — a bare-`Exception` subclass with the original only reachable via
    `__cause__`. `isinstance(wrapped, APIError)` is `False`, so a predicate written as a
    plain `isinstance` check would silently miss every rate-limit error while looking
    correct. 5xx (`ServerError`) is NOT wrapped by that path and arrives as a raw
    `APIError` — handled here too, since walking `__cause__` finds it on the first
    iteration (an `APIError` is trivially its own cause-chain match).
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, APIError):
            return current
        seen.add(id(current))
        current = current.__cause__
    return None


def rate_limit_class(err: APIError) -> str | None:
    """"rpd" | "rpm" | None for a 429. RPM/TPM clear in seconds — the SDK's own backoff
    (GEMINI_MAX_RETRIES above) is the right response. RPD does NOT clear until the
    provider's daily quota reset, so a run hitting it should fail fast rather than back
    off into a wall it can't get past before the reset. Classified from the violation
    type Gemini puts in the 429 body; an unrecognized 429 is treated as the more common,
    actually-recoverable RPM case rather than silently dropped.
    """
    if err.code != 429:
        return None
    haystack = f"{err.details} {err}".lower().replace(" ", "")
    if "perday" in haystack:
        return "rpd"
    return "rpm"


# Per-role OpenRouter model id, e.g. "openai/gpt-oss-20b:free" — optional. Unset (the
# default) means that role has no fallback and behaves exactly as before this existed.
# MEMORY has no entry: get_memory_manager() (core/memory.py) hands its model straight to
# langmem's create_memory_manager(), which does its own internal bind_tools/
# with_structured_output wiring — a fallback swapped in here wouldn't have those
# BaseChatModel methods for langmem to call, so MEMORY_MODEL has no fallback for now.
_FALLBACK_ROLE_ENV: dict[ModelRole, str] = {
    ModelRole.PLANNER: "PLANNER_FALLBACK_MODEL",
    ModelRole.WORKER: "WORKER_FALLBACK_MODEL",
    ModelRole.VERDICT: "VERDICT_FALLBACK_MODEL",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _build_fallback_model(role: ModelRole, *, temperature: float, **kwargs) -> ChatOpenAI | None:
    model_name = os.environ.get(_FALLBACK_ROLE_ENV.get(role, ""))
    if not model_name:
        return None
    # OPENROUTER_API_KEY is only required once a fallback model is actually configured
    # AND the primary actually fails — checked here (not at import time) so a role with
    # no fallback configured never needs this var at all.
    return ChatOpenAI(
        model=model_name,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=temperature,
        **kwargs,
    )


def _is_fallback_worthy(exc: BaseException) -> bool:
    """429 (rate/quota — RPM or RPD) or 5xx: the classes an SDK-level retry has already
    had its shot at and a NODE-level retry (same provider, same request) can't fix
    either — RPD doesn't clear same-day, a persistent 5xx isn't transient. Deliberately
    narrower than "any Gemini error": a genuine 400 (a real bug in our own request
    shape) should surface loudly instead of being masked behind an unrelated OpenRouter
    failure — see node_retry_on's docstring for the same reasoning applied to node
    retries instead of cross-provider fallback.
    """
    api_error = find_gemini_api_error(exc)
    if api_error is None:
        return False
    return rate_limit_class(api_error) is not None or api_error.code >= 500


class _ChatModelWithFallback:
    """Wraps an already-fully-composed primary chain (e.g.
    `get_chat_model(role).with_structured_output(Schema)`) with a same-shape fallback
    chain built from an OpenRouter model. Deliberately NOT built by calling
    `Runnable.with_fallbacks()` on the bare chat model before `.with_structured_output()`/
    `.bind_tools()`: `with_fallbacks()` returns a `RunnableWithFallbacks`, which is not a
    `BaseChatModel` and so loses exactly those methods every call site in this codebase
    chains on afterward. Composing chain_fn onto each model first (see `with_fallback()`
    below) and wrapping the RESULT instead sidesteps that — this class only ever needs
    `.ainvoke()`, which is all any call site in this codebase actually calls.

    The fallback chain itself is built lazily, on first actual use inside `ainvoke()`,
    NOT in `__init__`/`with_fallback()` — `_build_fallback_model()` requires
    OPENROUTER_API_KEY, and building it eagerly would mean a blank/invalid key breaks
    EVERY call the moment a `*_FALLBACK_MODEL` is configured, even on the (normal) path
    where the primary Gemini call succeeds and the fallback is never needed. Confirmed
    directly: this was the shape of the bug before the fix (`ChatOpenAI.__init__`
    itself raises on a blank key, before any Gemini call ever happens).
    """

    def __init__(self, primary, role: ModelRole, chain_fn, temperature: float, kwargs: dict):
        self._primary = primary
        self._role = role
        self._chain_fn = chain_fn
        self._temperature = temperature
        self._kwargs = kwargs
        self._fallback = None

    async def ainvoke(self, *args, **kwargs):
        try:
            return await self._primary.ainvoke(*args, **kwargs)
        except Exception as exc:
            if not _is_fallback_worthy(exc):
                raise
            logging.warning(
                "%s role's primary model failed (%s) — falling back to OpenRouter", self._role.value, exc
            )
            if self._fallback is None:
                fallback_model = _build_fallback_model(self._role, temperature=self._temperature, **self._kwargs)
                self._fallback = self._chain_fn(fallback_model)
            return await self._fallback.ainvoke(*args, **kwargs)


def with_fallback(role: ModelRole, chain_fn, *, temperature: float = 0.0, **kwargs):
    """`chain_fn` is whatever a call site would otherwise chain directly onto
    `get_chat_model(role, ...)` — e.g. `lambda m: m.with_structured_output(Schema)` or
    `lambda m: m.bind_tools(tools)`. Applies it to the primary Gemini model now, and — if
    `role` has a fallback model configured (see `_FALLBACK_ROLE_ENV`) — arranges to apply
    it to an OpenRouter model LATER, only if the primary actually fails with a
    rate-limit/server error (see `_is_fallback_worthy`, `_ChatModelWithFallback`).
    Returns just `chain_fn(primary)` — identical to today's direct-chaining call sites —
    when no fallback is configured for this role.
    """
    primary = chain_fn(get_chat_model(role, temperature=temperature, **kwargs))
    if not os.environ.get(_FALLBACK_ROLE_ENV.get(role, "")):
        return primary
    return _ChatModelWithFallback(primary, role, chain_fn, temperature, kwargs)


def node_retry_on(exc: Exception) -> bool:
    """`RetryPolicy.retry_on` for every LLM-calling graph node (see LLM_RETRY_POLICY).

    False for anything traceable to a real Gemini API error — the SDK's own HTTP-level
    retry has already retried 408/429/5xx before this exception ever reaches here, and a
    NODE retry re-runs the whole node body on top of that (see this module's docstring
    for the 18-round-trip amplification this replaces). Also False for
    `SessionGoneError` (nodes/worker/session.py) — retrying that would just raise
    identically every time, since the session is gone for good, not transiently
    unavailable. Everything else falls through to LangGraph's own default
    classification (genuine connection errors retry; programming-error classes like
    ValueError/TypeError don't).
    """
    # Local import: nodes/worker/session.py doesn't import anything from core/, and
    # nothing here needs SessionGoneError until a node actually raises one — importing
    # it at call time rather than module load time sides steps having to reason about
    # import order between core/ and nodes/ at all.
    from ..nodes.worker.session import SessionGoneError

    if find_gemini_api_error(exc) is not None:
        return False
    if isinstance(exc, SessionGoneError):
        return False
    return default_retry_on(exc)


LLM_RETRY_POLICY = RetryPolicy(max_attempts=3, retry_on=node_retry_on)
