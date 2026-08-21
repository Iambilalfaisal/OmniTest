"""AsyncPostgresSaver lifecycle: a pooled set of connections shared by every graph run
in this process. Acquired once at API startup via the FastAPI lifespan and passed into
build_graph(checkpointer=...); never constructed per-request.

Backed by an AsyncConnectionPool, not a single AsyncConnection — the worker subgraph
writes one checkpoint per superstep (agent_node -> tool_node -> tool_node -> ...), across
MAX_CONCURRENT_WORKERS parallel branches, plus one full-checkpoint READ per second per
SSE-connected client (backend/api.py's run_events). A single connection serializes all
of that onto one round trip at a time and was confirmed (backend/api.py's own comment on
_background_tasks) to raise `psycopg.OperationalError("another command is already in
progress")` under exactly this load — that error is the symptom, the single connection
is the cause.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

DATABASE_URL = os.environ["DATABASE_URL"]  # required — fail fast at import time

# Sized at roughly MAX_CONCURRENT_WORKERS + a handful of readers (SSE pollers, history
# writes) rather than mirroring HISTORY_POOL_MAX_SIZE (core/history.py) — this pool
# carries a materially heavier and bursty write load (a checkpoint per tool call, not
# per status transition), so it gets a bigger floor.
CHECKPOINTER_POOL_MAX_SIZE = int(os.getenv("CHECKPOINTER_POOL_MAX_SIZE", "10"))


@asynccontextmanager
async def make_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    # Connection kwargs match AsyncPostgresSaver.from_conn_string's own single-connection
    # setup exactly (confirmed against the installed langgraph-checkpoint-postgres
    # 3.1.2's source) — autocommit (checkpoint writes manage their own transactions),
    # prepare_threshold=0 (psycopg's server-side prepared statements don't survive a
    # pool handing a connection to a different caller), dict_row (the saver indexes
    # query results by column name). AsyncPostgresSaver itself has no from_conn_string
    # pool-config option (unlike AsyncPostgresStore below), so the pool is built here
    # and passed straight to the constructor, which accepts a pool or a bare connection.
    pool = AsyncConnectionPool(
        DATABASE_URL,
        min_size=1,
        max_size=CHECKPOINTER_POOL_MAX_SIZE,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open()
    try:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()  # idempotent: creates checkpoint tables/indices if missing
        yield saver
    finally:
        await pool.close()
