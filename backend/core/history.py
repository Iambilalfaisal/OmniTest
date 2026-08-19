"""Lightweight session-history metadata store: records when a run/discovery session
starts and how its status changes over time, powering the /history and /history/stats
endpoints in backend/api.py.

Deliberately separate from both the checkpointer (backend/graph/checkpointer.py) and the
long-term memory store (backend/core/memory.py) — this is plain relational bookkeeping,
not LangGraph-internal state, and gets its own dedicated connection pool rather than
either of those modules' single-connection-plus-lock pattern: a history write happens on
every status transition across every in-flight run and must never contend with or
serialize behind the checkpointer's existing single-connection bottleneck.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator, Literal

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

DATABASE_URL = os.environ["DATABASE_URL"]  # required — fail fast at import time
HISTORY_POOL_MAX_SIZE = int(os.getenv("HISTORY_POOL_MAX_SIZE", "5"))

SessionKind = Literal["run", "discovery"]

# No FK on parent_id (deliberately decoupled, auxiliary metadata) and no trigram index for
# URL search (ILIKE '%...%' sequential scan is fine at expected volume — hundreds/thousands
# of session rows, not events — and avoids requiring another Postgres extension after
# core/memory.py's own account of how painful getting pgvector built on this Windows
# install already was).
_DDL = """
CREATE TABLE IF NOT EXISTS history_sessions (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL CHECK (kind IN ('run', 'discovery')),
    target_url  TEXT NOT NULL,
    label       TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL CHECK (status IN (
                    'running','paused','done','error',
                    'in_progress','approved','cancelled'
                )),
    summary     JSONB,
    parent_id   TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_history_sessions_kind       ON history_sessions (kind);
CREATE INDEX IF NOT EXISTS idx_history_sessions_status     ON history_sessions (status);
CREATE INDEX IF NOT EXISTS idx_history_sessions_created_at ON history_sessions (created_at DESC);
"""


class HistoryStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def create_session(
        self,
        *,
        id: str,
        kind: SessionKind,
        target_url: str,
        label: str,
        status: str,
        parent_id: str | None = None,
    ) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO history_sessions (id, kind, target_url, label, status, parent_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (id, kind, target_url, label, status, parent_id),
            )

    async def update_status(self, id: str, status: str, *, summary: dict | None = None) -> None:
        # COALESCE(%s, summary): pause/error transitions pass summary=None and leave any
        # already-stored summary value untouched.
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                UPDATE history_sessions
                SET status = %s, updated_at = now(), summary = COALESCE(%s, summary)
                WHERE id = %s
                """,
                (status, Jsonb(summary) if summary is not None else None, id),
            )

    async def list_sessions(
        self,
        *,
        kind: str | None = None,
        status: str | None = None,
        url_contains: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        where: list[str] = []
        params: list = []
        if kind:
            where.append("kind = %s")
            params.append(kind)
        if status:
            where.append("status = %s")
            params.append(status)
        if url_contains:
            where.append("target_url ILIKE %s")
            params.append(f"%{url_contains}%")
        if since:
            where.append("created_at >= %s")
            params.append(since)
        if until:
            where.append("created_at <= %s")
            params.append(until)
        clause = f"WHERE {' AND '.join(where)}" if where else ""

        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(f"SELECT COUNT(*) AS n FROM history_sessions {clause}", params)
                total = (await cur.fetchone())["n"]

                await cur.execute(
                    f"""
                    SELECT id, kind, target_url, label, status, summary, parent_id, created_at, updated_at
                    FROM history_sessions {clause}
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    [*params, limit, offset],
                )
                items = await cur.fetchall()
        return items, total

    async def get_stats(self, *, since: datetime | None = None, until: datetime | None = None) -> dict:
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("SELECT COUNT(*) AS n FROM history_sessions")
                total_sessions = (await cur.fetchone())["n"]

                await cur.execute("SELECT kind, COUNT(*) AS n FROM history_sessions GROUP BY kind")
                by_kind = {row["kind"]: row["n"] for row in await cur.fetchall()}

                await cur.execute("SELECT status, COUNT(*) AS n FROM history_sessions GROUP BY status")
                by_status = {row["status"]: row["n"] for row in await cur.fetchall()}

                trend_where = ["kind = 'run'", "status = 'done'"]
                trend_params: list = []
                if since:
                    trend_where.append("updated_at >= %s")
                    trend_params.append(since)
                if until:
                    trend_where.append("updated_at <= %s")
                    trend_params.append(until)

                await cur.execute(
                    f"""
                    SELECT
                        date_trunc('day', updated_at) AS day,
                        COALESCE(SUM((summary->>'passed')::int), 0) AS passed,
                        COALESCE(SUM((summary->>'failed')::int), 0) AS failed,
                        COUNT(*) AS runs_completed
                    FROM history_sessions
                    WHERE {' AND '.join(trend_where)}
                    GROUP BY 1
                    ORDER BY 1
                    """,
                    trend_params,
                )
                trend_rows = await cur.fetchall()

        trend = [
            {
                "date": row["day"].date().isoformat(),
                "passed": row["passed"],
                "failed": row["failed"],
                "runs_completed": row["runs_completed"],
            }
            for row in trend_rows
        ]
        total_passed = sum(r["passed"] for r in trend)
        total_failed = sum(r["failed"] for r in trend)
        overall_pass_rate = (
            total_passed / (total_passed + total_failed) if (total_passed + total_failed) > 0 else None
        )

        return {
            "total_sessions": total_sessions,
            "by_kind": by_kind,
            "by_status": by_status,
            "overall_pass_rate": overall_pass_rate,
            "trend": trend,
        }


@asynccontextmanager
async def make_history_store() -> AsyncIterator[HistoryStore]:
    pool = AsyncConnectionPool(
        DATABASE_URL,
        min_size=1,
        max_size=HISTORY_POOL_MAX_SIZE,
        open=False,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    await pool.open()
    try:
        async with pool.connection() as conn:
            await conn.execute(_DDL)  # idempotent — safe to run on every startup
        yield HistoryStore(pool)
    finally:
        await pool.close()
