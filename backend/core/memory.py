"""Long-term semantic memory: per-domain facts learned from past runs, persisted in
Postgres via pgvector and retrieved by planner_node before drafting a new plan.

Shares the same Postgres database as the checkpointer (backend/graph/checkpointer.py)
— same DATABASE_URL — via its own connection pool rather than a literally shared pool
object, since AsyncPostgresStore's exact pool-sharing API wasn't confirmed available.
pgvector is confirmed installed and enabled on this database (built from source on
Windows via VS Build Tools + nmake, since no prebuilt binaries or Stack Builder option
existed for this Postgres install).
"""
from __future__ import annotations

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
from .models import SiteMemory, SiteMap

MEMORY_NAMESPACE = "site_memory"
DATABASE_URL = os.environ["DATABASE_URL"]  # required — fail fast at import time

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
    async with AsyncPostgresStore.from_conn_string(
        DATABASE_URL,
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
