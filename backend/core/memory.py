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
from typing import AsyncIterator
from urllib.parse import urlparse

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.store.postgres.aio import AsyncPostgresStore
from langmem import create_memory_manager

from .models import SiteMemory

MEMORY_NAMESPACE = "site_memory"
DATABASE_URL = os.environ["DATABASE_URL"]  # required — fail fast at import time


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
    # Lazy for the same reason as get_embeddings() above.
    # TODO(verify): SiteMemory.related_goal is Optional (str | None) — some
    # langchain-google-genai versions have had quirks with Optional/nullable fields
    # in structured-output/function-calling schemas. Confirm langmem's extraction
    # against this schema actually works with Gemini before relying on it.
    model = ChatGoogleGenerativeAI(model=os.environ["MEMORY_EXTRACTION_MODEL"], temperature=0)
    return create_memory_manager(
        model,
        schemas=[SiteMemory],
        instructions=(
            "Extract at most a few distilled, reusable facts about this site from the QA "
            "run transcript below — only failure patterns and structural/behavioral quirks "
            "worth remembering for future test planning. Skip anything not worth persisting."
        ),
        enable_inserts=True,
    )


def format_memories_for_prompt(items) -> str:
    if not items:
        return "No prior learnings recorded for this site yet.\n"
    lines = []
    for item in items:
        value = item.value if hasattr(item, "value") else item
        goal = f' (goal: "{value.get("related_goal")}")' if value.get("related_goal") else ""
        lines.append(f"- [{value.get('kind')}]{goal} {value.get('summary')}")
    return "\n".join(lines) + "\n"
