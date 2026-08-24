"""Long-term semantic memory: per-domain facts learned from past runs, persisted in
Postgres via pgvector and retrieved by planner_node before drafting a new plan.

Shares the same Postgres database as the checkpointer (backend/graph/checkpointer.py)
— same DATABASE_URL — via its own connection pool rather than a literally shared pool
object, since AsyncPostgresStore's exact pool-sharing API wasn't confirmed available.
Called far less often than the checkpointer (~2x per run vs. once per superstep), so its
pool is sized smaller. pgvector is confirmed installed and enabled on this database
(built from source on Windows via VS Build Tools + nmake, since no prebuilt binaries or
Stack Builder option existed for this Postgres install).
"""
from __future__ import annotations

import logging
import math
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator
from urllib.parse import urlparse

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.store.base import BaseStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from langmem import create_memory_manager

from .llm import ModelRole, get_chat_model
from .models import SiteMemory, SiteMap, TestCase

MEMORY_NAMESPACE = "site_memory"
DATABASE_URL = os.environ["DATABASE_URL"]  # required — fail fast at import time
MEMORY_POOL_MAX_SIZE = int(os.getenv("MEMORY_POOL_MAX_SIZE", "5"))

# Direct-key (not embedding-search) slot in the same per-domain namespace SiteMemory
# facts use — a crawled site map is structured data, not a prose fact worth embedding.
SITE_MAP_KEY = "site_map"
SITE_MAP_TTL_HOURS = int(os.getenv("SITE_MAP_TTL_HOURS", "24"))


def domain_of(target_url: str) -> str:
    # langgraph.store.base's namespace validation rejects any label containing a
    # period — real domains (example.com, etc.) almost always have one, so sanitize.
    return urlparse(target_url).netloc.replace(".", "_")


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    # Lazy, matching this codebase's existing convention (planner._planner_llm, etc.)
    # so importing this module doesn't require the key until a store is actually built.
    return GoogleGenerativeAIEmbeddings(model=os.environ["MEMORY_EMBEDDING_MODEL"])


@asynccontextmanager
async def make_store() -> AsyncIterator[AsyncPostgresStore]:
    dims = int(os.environ["MEMORY_EMBEDDING_DIMS"])
    # pool_config, not a bare connection string — confirmed against the installed
    # langgraph-checkpoint-postgres 3.1.2 that from_conn_string builds and owns an
    # AsyncConnectionPool internally when this is set (falling back to a single
    # AsyncConnection otherwise, which is what this used to run on). A single
    # connection here was a smaller risk than the checkpointer's — this store is only
    # touched ~twice per run (planner_node's read, memory_node's write) — but still
    # serializes concurrent runs' memory reads/writes against each other for no reason.
    async with AsyncPostgresStore.from_conn_string(
        DATABASE_URL,
        pool_config={"min_size": 1, "max_size": MEMORY_POOL_MAX_SIZE},
        index={"dims": dims, "embed": get_embeddings(), "fields": ["summary"]},
    ) as store:
        await store.setup()  # idempotent DDL — creates the pgvector-backed store tables
        yield store


def get_memory_manager():
    # get_chat_model (core/llm.py) is itself lazy, for the same reason as
    # get_embeddings() above.
    # TODO(verify): SiteMemory.related_goal is Optional (str | None) — some
    # langchain-google-genai versions have had quirks with Optional/nullable fields
    # in structured-output/function-calling schemas. Confirm langmem's extraction
    # against this schema actually works with Gemini before relying on it.
    model = get_chat_model(ModelRole.MEMORY, temperature=0)
    return create_memory_manager(
        model,
        schemas=[SiteMemory],
        instructions=(
            "Extract at most a few distilled, reusable facts about this site from the QA run "
            "transcript below — only failure patterns and structural/behavioral quirks worth "
            "remembering the next time someone plans tests for this same site. Each fact must "
            "be specific enough to change a future test case: name the flow, the field, and "
            "the exact message or rule text the run actually observed. Quote real text where "
            "the transcript has it.\n"
            "Pay special attention to constraints on account creation and login — password "
            "rules ('passwords must be at least 8 characters'), verification requirements "
            "('requires email confirmation before first login'), rate limiting or lockout, "
            "and the exact wording of rejection messages. These are precisely what a future "
            "planning run needs before it writes its own inline signup/login steps or decides "
            "what a negative auth case should expect to see.\n"
            "Do NOT record: anything you only inferred rather than observed in the transcript; "
            "generic testing advice; restatements of a test case's own goal; or one-off "
            "environment noise (a timeout, a flaky load). If nothing meets that bar, extract "
            "nothing — an empty result is better than a vague fact that misleads the next run."
        ),
        enable_inserts=True,
    )


async def get_cached_site_map(store: BaseStore | None, target_url: str) -> SiteMap | None:
    """Direct-key lookup of a previously crawled SiteMap for this domain, or None if
    absent/stale. Not the semantic/embedding search used for SiteMemory facts — a crawled
    site map is structured data, not a prose fact worth embedding.
    """
    if store is None:
        return None
    domain = domain_of(target_url)
    item = await store.aget((MEMORY_NAMESPACE, domain), SITE_MAP_KEY)
    if item is None:
        return None
    site_map = SiteMap(**item.value)
    if site_map.crawled_at is None:
        return None
    age_hours = (datetime.now(timezone.utc) - datetime.fromisoformat(site_map.crawled_at)).total_seconds() / 3600
    return site_map if age_hours < SITE_MAP_TTL_HOURS else None


async def save_site_map(store: BaseStore | None, target_url: str, site_map: SiteMap) -> None:
    if store is None:
        return
    domain = domain_of(target_url)
    stamped = site_map.model_copy(update={"crawled_at": datetime.now(timezone.utc).isoformat()})
    # TODO(verify): make_store()'s embedding index is configured with fields=["summary"]
    # (SiteMemory's field) — SiteMap.model_dump() has no "summary" key. Confirm
    # AsyncPostgresStore.aput no-ops the indexer gracefully for a value lacking the
    # configured field rather than erroring.
    await store.aput((MEMORY_NAMESPACE, domain), SITE_MAP_KEY, stamped.model_dump())


async def retrieve_memory_context(target_url: str, query: str, store: BaseStore | None) -> str:
    """Shared by the one-shot planner and the discovery chat — both want prior
    cross-run learnings about the target site before drafting/revising a plan."""
    if store is None:
        return "No prior learnings recorded for this site yet.\n"
    domain = domain_of(target_url)
    items = await store.asearch((MEMORY_NAMESPACE, domain), query=query, limit=5)
    return format_memories_for_prompt(items)


def format_memories_for_prompt(items) -> str:
    if not items:
        return "No prior learnings recorded for this site yet.\n"
    lines = []
    for item in items:
        value = item.value if hasattr(item, "value") else item
        goal = f' (goal: "{value.get("related_goal")}")' if value.get("related_goal") else ""
        lines.append(f"- [{value.get('kind')}]{goal} {value.get('summary')}")
    return "\n".join(lines) + "\n"


# Threshold for "same behavior under test, worded differently" vs. "genuinely different
# case that happens to read similarly" — deliberately high. core/run_planning.py's
# drop_duplicate_scenarios already catches an EXACT (goal, steps) repeat for free, with
# zero embedding cost and zero false-positive risk; this only needs to catch the gap
# above that — near-duplicate wording — so it can afford to be conservative. Too low a
# threshold silently drops real coverage (e.g. "invalid email format" vs "empty email
# field" are both email-related negative cases that could embed close together despite
# testing different things) — a false-positive here is worse than an occasional
# near-duplicate slipping through, since dropped coverage is invisible, not reported.
SCENARIO_SIMILARITY_THRESHOLD = float(os.getenv("SCENARIO_SIMILARITY_THRESHOLD", "0.93"))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


async def drop_semantic_duplicates(candidates: list[TestCase], reference: list[TestCase] | None = None) -> list[TestCase]:
    """Drops a candidate TestCase whose `goal` embeds too close (cosine similarity >=
    SCENARIO_SIMILARITY_THRESHOLD) to one already accepted — either an earlier entry in
    `candidates` itself, or one in `reference` (an already-committed list candidates
    should never restate, e.g. graph/builder.py's recon_join_node comparing recon's own
    proposals against the planner's baseline for the SAME Feature).

    Backstop layered ON TOP of drop_duplicate_scenarios (core/run_planning.py), not a
    replacement for it: that one is free (no I/O, no false-positive risk) and catches
    exact repeats; this one costs one batched embedding call and exists specifically for
    wording drift a plain string match can't see — e.g. recon proposing "Creating a new
    account should reach the dashboard" when the baseline already has "Registering a
    brand-new unique account should succeed and land on the dashboard". Embeds only
    `goal` (TEST_CASE_AUTHORING_GUIDELINES' own "one sentence, what is being tested"
    field), not `steps` — step text is largely shared boilerplate ("Navigate to X, click
    Y") across unrelated cases and would dilute the similarity signal that actually
    distinguishes one behavior under test from another.

    Best-effort: an embedding-call failure returns `candidates` unfiltered rather than
    raising — this is a quality improvement on top of an already-correct plan, not
    something that should block a run or crash a node over a transient API error.
    """
    if not candidates:
        return candidates
    reference = reference or []
    texts = [tc.goal for tc in [*reference, *candidates]]
    try:
        vectors = await get_embeddings().aembed_documents(texts, task_type="SEMANTIC_SIMILARITY")
    except Exception:
        logging.exception("drop_semantic_duplicates: embedding call failed — skipping semantic dedup for this batch")
        return candidates

    # Walked in order so "first occurrence wins" matches drop_duplicate_scenarios'
    # own semantics — kept_vectors seeds from `reference` (never itself filtered; those
    # are already-committed cases) and grows as each candidate survives.
    kept_vectors = vectors[: len(reference)]
    result = []
    for tc, vec in zip(candidates, vectors[len(reference):]):
        if any(_cosine_similarity(vec, kv) >= SCENARIO_SIMILARITY_THRESHOLD for kv in kept_vectors):
            continue
        kept_vectors.append(vec)
        result.append(tc)
    return result
