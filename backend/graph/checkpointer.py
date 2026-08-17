"""AsyncPostgresSaver lifecycle: one pooled connection shared by every graph run in
this process. Acquired once at API startup via the FastAPI lifespan and passed into
build_graph(checkpointer=...); never constructed per-request.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

DATABASE_URL = os.environ["DATABASE_URL"]  # required — fail fast at import time


@asynccontextmanager
async def make_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as saver:
        await saver.setup()  # idempotent: creates checkpoint tables/indices if missing
        yield saver
