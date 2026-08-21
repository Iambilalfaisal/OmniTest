"""FastAPI server: drives graph runs in the background and lets clients observe
progress (including human-in-the-loop pauses) by polling checkpointed state.
"""
from __future__ import annotations

import asyncio
import sys

# Must happen before any event loop is created (i.e. before uvicorn/anything else
# touches asyncio) — psycopg's async mode raises on Windows' default ProactorEventLoop
# ("Psycopg cannot use the 'ProactorEventLoop' to run in async mode"), which would
# otherwise break both the checkpointer and core/memory.py's PostgresMemoryStore at
# startup. Confirmed by hitting this directly while testing on Windows.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Loaded before the graph/node imports below, since those transitively construct
# LangChain/LangSmith clients that read OPENAI_API_KEY / LANGCHAIN_* from os.environ.
#
# Path is explicit (not load_dotenv()'s default search) because that search is
# frame-based: it walks up from the caller's __file__ unless it decides it's running
# "interactively", in which case it falls back to os.getcwd() instead. uvicorn
# --reload on Windows launches the real server via multiprocessing's spawn method,
# whose bootstrap process has no __main__.__file__ — dotenv's interactive-detection
# trips on that and searches from cwd, which misses backend/.env entirely whenever
# uvicorn is started from the repo root instead of from backend/.
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# core.logging_config/core.run_context are deliberately imported and applied here,
# before the heavier graph/node imports below — those are the ones that transitively
# construct LangChain/LangSmith/Gemini clients and could plausibly log something at
# import time. Root had NO logging configuration at all before this (confirmed: no
# basicConfig/dictConfig anywhere in backend/) — logging.info calls were silently
# dropped and logging.exception calls printed bare, with no timestamp or run context.
from .core.logging_config import configure_logging  # noqa: E402

configure_logging()

import contextlib
import json
import logging
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime
from typing import AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .core import llm_metrics
from .core.discovery_state import DiscoveryState
from .core.history import HistoryStore, make_history_store
from .core.memory import make_store
from .core.models import SiteMap, TestPlan
from .core.run_context import run_id_var
from .core.run_planning import ensure_expected_result, ensure_unique_test_ids
from .core.state import QAState
from .graph.builder import build_graph
from .graph.checkpointer import make_checkpointer
from .graph.discovery_graph import build_discovery_graph
from .nodes.worker import (
    SESSION_REAP_INTERVAL_SECONDS,
    close_all_sessions,
    close_idle_sessions,
    close_sessions_for_thread,
)

EVIDENCE_DIR = Path(__file__).resolve().parent / "evidence"
# Only ever created lazily by a run's own evidence capture (nodes/worker/evidence.py) —
# a fresh clone with no run yet has no such directory, and StaticFiles(directory=...)
# raises at import time (before the app can even start) if the path doesn't exist.
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
# Stage 1: floor(RPM_limit / requests_per_worker_per_minute) for WORKER_MODEL — see the
# worked example (RPM_limit=15, requests_per_worker_per_minute=3.4 => 4) in backend/.env's
# own comment above this same var. The "4" here is only the fallback for a completely
# unconfigured environment; the real, reasoned value lives in .env.
MAX_CONCURRENT_WORKERS = int(os.getenv("MAX_CONCURRENT_WORKERS", "4"))
SSE_POLL_INTERVAL_SECONDS = float(os.getenv("SSE_POLL_INTERVAL_SECONDS", "1.0"))
MAX_DISCOVERY_TURNS = int(os.getenv("MAX_DISCOVERY_TURNS", "20"))

# Every _drive() fired via asyncio.create_task (start_run / resume_run / a discovery
# approval) is tracked here so a graceful shutdown (uvicorn --reload picking up a file
# change mid-run, in particular) can wait for in-flight graph work to reach its next
# checkpoint before the AsyncExitStack below tears down the checkpointer's connection.
# Without this, a shutdown racing a live checkpoint write on that single connection
# raises psycopg.OperationalError("another command is already in progress") — confirmed
# by hitting this directly when --reload restarted mid-run during development.
_background_tasks: set[asyncio.Task] = set()

# thread_ids with a live _drive() task right now — so close_idle_sessions' periodic
# sweep (below) never reaps a session out from under a run that is actively executing a
# node (as opposed to genuinely paused). A long Gemini backoff inside one node can go
# quiet for minutes; that must not look like an abandoned pause. Deliberately a
# separate set from _background_tasks, not derived from it, since membership here needs
# to start/end exactly at _drive()'s own start/finally, not at task-object lifetime.
_active_threads: set[str] = set()


def _track(task: asyncio.Task) -> asyncio.Task:
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def _reap_idle_sessions_forever() -> None:
    """Periodic backstop for the one leak api.py's own pause-safety guard (see
    `_drive()`'s `finally` below) deliberately accepts: a run that paused for a human
    and was then never resumed. Started in `lifespan()`, held in a plain local variable
    there rather than passed through `_track()` — `_background_tasks` is gathered on
    shutdown, and gathering an infinite loop would hang shutdown forever.
    """
    while True:
        await asyncio.sleep(SESSION_REAP_INTERVAL_SECONDS)
        try:
            await close_idle_sessions(exempt_threads=_active_threads)
        except Exception:
            logging.exception("idle Playwright session reaper tick failed — retrying next tick")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Graph construction can't happen at bare module-import time — building the
    # async Postgres saver/store needs a running event loop.
    async with AsyncExitStack() as stack:
        checkpointer = await stack.enter_async_context(make_checkpointer())
        # Not wrapped in try/except like the memory store below — history has no
        # extension dependency (no pgvector) so it should be at least as reliable as the
        # checkpointer itself, which also isn't degrade-on-failure.
        app.state.history = await stack.enter_async_context(make_history_store())

        store = None
        try:
            store = await stack.enter_async_context(make_store())
        except Exception:
            # pgvector is confirmed installed/enabled, so this should rarely trip —
            # kept as a safety net (e.g. a transient connection failure) rather than
            # removed. Degrade gracefully rather than block startup — planner_node/
            # memory_node already handle store=None (no memory context / no-op write).
            logging.exception(
                "Long-term memory store unavailable — continuing without it; "
                "runs will work, just without cross-run memory."
            )

        app.state.run_graph = build_graph(checkpointer, store=store)
        app.state.discovery_graph = build_discovery_graph(checkpointer, store=store)

        reaper = asyncio.create_task(_reap_idle_sessions_forever())
        try:
            yield
        finally:
            reaper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reaper

            if _background_tasks:
                logging.info(
                    "Shutting down — waiting for %d in-flight run(s) to reach their next "
                    "checkpoint before closing database connections...",
                    len(_background_tasks),
                )
                await asyncio.gather(*_background_tasks, return_exceptions=True)

            # After the gather above, so a run that reaches verdict_node on its way to
            # its next checkpoint still gets to close its own session (and capture its
            # evidence) first — this only mops up what's left, including runs that are
            # currently paused for a human who never got to respond before shutdown.
            await close_all_sessions()


app = FastAPI(title="OmniTest Engine", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/evidence", StaticFiles(directory=str(EVIDENCE_DIR)), name="evidence")


class RunRequest(BaseModel):
    target_url: str
    instruction: str
    discovery_context: str = ""


class RunHandle(BaseModel):
    run_id: str


class ResumeRequest(BaseModel):
    resume: dict[str, dict]


class DiscoveryStartRequest(BaseModel):
    target_url: str
    starting_idea: str = ""


class DiscoveryMessageRequest(BaseModel):
    action: Literal["reply", "approve", "cancel"]
    text: str | None = None


class HistorySession(BaseModel):
    id: str
    kind: Literal["run", "discovery"]
    target_url: str
    label: str
    status: str
    summary: dict | None = None
    parent_id: str | None = None
    created_at: datetime
    updated_at: datetime


class HistoryListResponse(BaseModel):
    items: list[HistorySession]
    total: int
    limit: int
    offset: int


class HistoryStatsResponse(BaseModel):
    total_sessions: int
    by_kind: dict[str, int]
    by_status: dict[str, int]
    overall_pass_rate: float | None
    trend: list[dict]


def _run_config(run_id: str) -> dict:
    # callbacks: attached once here, not per-node — LangChain propagates it through a
    # ContextVar into every nested `.ainvoke()` the run makes, including calls that
    # never explicitly forward `config=` (see llm_metrics.LlmUsageCallback's docstring).
    # Only the QA run path is instrumented this way; the discovery chat's own config
    # (_discovery_config below) is deliberately left out of Stage 0's scope.
    return {
        "configurable": {"thread_id": run_id},
        "max_concurrency": MAX_CONCURRENT_WORKERS,
        "callbacks": [llm_metrics.LlmUsageCallback(run_id)],
    }


async def _update_history_status(history: HistoryStore, session_id: str, status: str, *, summary: dict | None = None) -> None:
    # A transient history-DB hiccup must never mask a run's real outcome or skip
    # close_sessions_for_thread's cleanup in _drive() below — log and move on.
    try:
        await history.update_status(session_id, status, summary=summary)
    except Exception:
        logging.exception("failed to update history status for %s -> %s", session_id, status)


async def _drive(graph, input_, config: dict, history: HistoryStore) -> None:
    """Runs the graph to completion or its next pause. Fire-and-forget from a route
    handler — a crash here (retries exhausted) leaves the checkpoint stalled with
    nothing automatically re-driving it; see the plan's accepted limitations.
    """
    thread_id = config["configurable"]["thread_id"]
    # asyncio.create_task (start_run/resume_run) copies the CURRENT contextvars.Context
    # at task-creation time, so setting this before graph.ainvoke() reaches every
    # LangGraph-internal task it spawns afterward, including parallel Send-spawned
    # worker branches — see core/run_context.py's docstring.
    run_id_var.set(thread_id)
    _active_threads.add(thread_id)
    # Default covers the crash path below, where an exception can happen before
    # `snapshot` is ever fetched — see the finally block's comment for why a crash
    # counts as "finished" for llm_metrics AND session-cleanup purposes.
    finished = True
    try:
        await graph.ainvoke(input_, config=config)
        snapshot = await graph.aget_state(config)
        finished = not snapshot.next
        if finished:
            await _update_history_status(history, thread_id, "done", summary=snapshot.values.get("summary", {}))
        else:
            await _update_history_status(history, thread_id, "paused")
    except Exception:
        logging.exception("run %s crashed", thread_id)
        await _update_history_status(history, thread_id, "error")
    finally:
        _active_threads.discard(thread_id)
        # Guarded by `finished` for the SAME reason llm_metrics.discard below it already
        # was — this guard was previously MISSING here, which was a real, confirmed bug.
        # On a PAUSE, graph.ainvoke() above RETURNS NORMALLY with `__interrupt__` in its
        # output rather than raising (confirmed against the installed langgraph 1.2.11:
        # pregel/_loop.py's __aexit__ suppresses GraphInterrupt for a non-nested graph),
        # so this `finally` used to run at every human-in-the-loop pause and close the
        # very Playwright session the paused worker is about to resume against.
        # tool_node calls interrupt() BEFORE its get_session() (nodes/worker/nodes.py),
        # so the resumed node cache-missed and transparently opened a fresh, unnavigated
        # browser: an ask_human resume's "fresh snapshot" snapshotted about:blank, an
        # approved risky action ran against a blank page, and re-opening the session
        # silently overwrote the pre-pause video recording. A pause is not a terminal
        # state — nothing here should be torn down.
        #
        # Every genuine leak is still reaped: `finished` defaults to True above, so the
        # crash path (ainvoke or aget_state raising) still sweeps here, and
        # _update_history_status swallows its own exceptions so it can never strand
        # this at False. The one case this deliberately gives up on — a run that pauses
        # and is NEVER resumed — is covered by close_idle_sessions' TTL reaper
        # (_reap_idle_sessions_forever, started in lifespan()) and by close_all_sessions()
        # on shutdown.
        if finished:
            await close_sessions_for_thread(thread_id)
            # Only discard the accumulated LLM-request counter once this run is truly
            # over — "paused" means a human-in-the-loop interrupt is pending and
            # _drive() runs again on resume (resume_run), reusing this SAME thread_id,
            # and should keep accumulating onto the same total rather than restarting
            # from zero. A crash counts as terminal too: per this function's own
            # docstring, nothing automatically re-drives a crashed run, so the entry
            # would otherwise leak in llm_metrics._METRICS forever.
            llm_metrics.discard(thread_id)


def _model_dump(value):
    return value.model_dump() if hasattr(value, "model_dump") else value


def _pending_interrupts(snapshot) -> list[dict]:
    # Confirmed against the installed langgraph: StateSnapshot.interrupts is a
    # top-level tuple[Interrupt, ...] (aggregated across all tasks), and Interrupt
    # has exactly .id / .value fields.
    return [{"id": intr.id, "type": intr.value.get("type"), "payload": intr.value} for intr in snapshot.interrupts]


async def _run_exists(history: HistoryStore, run_id: str, snapshot) -> bool:
    """A checkpoint's `snapshot.values` is empty in two indistinguishable-by-itself
    cases: the run_id was never created, OR `start_run`/`send_discovery_message`'s
    `history.create_session()` has already returned but the graph's first checkpoint
    write hasn't landed yet (a real window — `_drive()` is fired via
    `asyncio.create_task` and is not awaited before the route handler returns). Treating
    the second case as 404 was a genuine race: a client that calls GET /runs/{id}/events
    immediately after POST /runs (exactly what a normal client does) could see a 404 for
    a run_id that is, in fact, valid and about to make progress. The history row is
    written first and synchronously, so checking it too closes the window.
    """
    if snapshot.values:
        return True
    return await history.get_session(run_id) is not None


def _discovery_config(discovery_id: str) -> dict:
    # Attaches the same LlmUsageCallback _run_config() uses for an actual run.
    # discovery_agent_node calls PLANNER_MODEL every turn (nodes/discovery.py) and is
    # the entry point most users actually start from — leaving it uninstrumented left
    # the majority of planning-phase quota usage invisible to llm_metrics, despite
    # quota being the documented binding constraint on this system.
    return {
        "configurable": {"thread_id": discovery_id},
        "callbacks": [llm_metrics.LlmUsageCallback(discovery_id)],
    }


def _discovery_turn_payload(snapshot) -> dict:
    """The one (there's never a fan-out in this graph) pending discovery_turn
    interrupt's payload, in the shape both POST /discover and POST /discover/{id}/message
    return for an in-progress conversation."""
    pending = _pending_interrupts(snapshot)
    if not pending:
        return {}
    payload = pending[0]["payload"]
    return {
        "assistant_message": payload.get("assistant_message", ""),
        "candidate_plan": payload.get("candidate_plan", []),
        "turn_count": payload.get("turn_count", 0),
        "max_turns": MAX_DISCOVERY_TURNS,
    }


def _discovery_transcript(messages) -> list[dict]:
    return [
        {"role": "assistant" if isinstance(m, AIMessage) else "user", "text": m.content}
        for m in messages or []
        if not isinstance(m, SystemMessage)
    ]


@app.post("/runs", response_model=RunHandle)
async def start_run(req: RunRequest, request: Request) -> RunHandle:
    run_id = str(uuid.uuid4())
    initial_state: QAState = {
        "target_url": req.target_url,
        "instruction": req.instruction,
        "discovery_context": req.discovery_context,
        "run_token": "",
        "test_cases": [],
        "test_results": [],
        "summary": {},
        "plan_approved": False,
        "auth_storage_state": None,
    }
    # Insert BEFORE firing _drive() so a client hitting GET /history immediately after
    # this call always sees the row.
    await request.app.state.history.create_session(
        id=run_id, kind="run", target_url=req.target_url, label=req.instruction, status="running"
    )
    _track(
        asyncio.create_task(
            _drive(request.app.state.run_graph, initial_state, _run_config(run_id), request.app.state.history)
        )
    )
    return RunHandle(run_id=run_id)


@app.post("/runs/{run_id}/resume")
async def resume_run(run_id: str, req: ResumeRequest, request: Request) -> dict:
    graph = request.app.state.run_graph
    config = {"configurable": {"thread_id": run_id}}
    snapshot = await graph.aget_state(config)
    if not await _run_exists(request.app.state.history, run_id, snapshot):
        raise HTTPException(status_code=404, detail="unknown run_id")

    pending_ids = {i["id"] for i in _pending_interrupts(snapshot)}
    resume_ids = set(req.resume.keys())
    if pending_ids != resume_ids:
        # Exact-match, not just coverage: a checkpoint can have multiple simultaneous
        # pending interrupts (e.g. two parallel workers each hitting a risky action in
        # the same superstep) and Command(resume=...) must resolve all of them at once.
        raise HTTPException(
            status_code=400,
            detail={
                "message": "resume payload must cover exactly the currently-pending interrupts",
                "missing": sorted(pending_ids - resume_ids),
                "unknown": sorted(resume_ids - pending_ids),
            },
        )

    _track(asyncio.create_task(_drive(graph, Command(resume=req.resume), _run_config(run_id), request.app.state.history)))
    return {"status": "resumed", "run_id": run_id}


@app.get("/runs/{run_id}/events")
async def run_events(run_id: str, request: Request) -> EventSourceResponse:
    graph = request.app.state.run_graph
    history = request.app.state.history
    config = {"configurable": {"thread_id": run_id}}

    snapshot = await graph.aget_state(config)
    if not await _run_exists(history, run_id, snapshot):
        raise HTTPException(status_code=404, detail="unknown run_id")

    async def event_stream() -> AsyncIterator[dict]:
        emitted_interrupt_ids: frozenset = frozenset()
        while True:
            snapshot = await graph.aget_state(config)
            pending = _pending_interrupts(snapshot)
            pending_ids = frozenset(i["id"] for i in pending)

            # Checked BEFORE the `not snapshot.next` done-branch below: a crashed run
            # (retries exhausted inside a node, `_drive()`'s `except Exception` caught
            # it) leaves `snapshot.next` truthy — LangGraph doesn't clear the pending
            # task just because it failed — and there are no pending interrupts either,
            # so without this check every tick fell into the `progress` branch below and
            # this stream ran `while True` forever, once a second, with the client never
            # told the run was dead. history_sessions is the only place "crashed" is
            # actually recorded (_drive's `_update_history_status(..., "error")`); the
            # checkpoint itself has no error status of its own.
            row = await history.get_session(run_id)
            if row and row["status"] == "error":
                yield {"event": "error", "data": json.dumps({"message": "run crashed", "run_id": run_id})}
                return

            if pending_ids and pending_ids != emitted_interrupt_ids:
                emitted_interrupt_ids = pending_ids
                yield {"event": "paused", "data": json.dumps({"interrupts": pending})}
            elif not snapshot.next:
                yield {
                    "event": "done",
                    "data": json.dumps(
                        {
                            "summary": snapshot.values.get("summary", {}),
                            "test_cases": [_model_dump(tc) for tc in snapshot.values.get("test_cases", [])],
                            "test_results": [_model_dump(r) for r in snapshot.values.get("test_results", [])],
                            "plan_approved": snapshot.values.get("plan_approved", False),
                        }
                    ),
                }
                return
            elif not pending_ids:
                yield {
                    "event": "progress",
                    "data": json.dumps(
                        {
                            "test_cases": [_model_dump(tc) for tc in snapshot.values.get("test_cases", [])],
                            "test_results": [_model_dump(r) for r in snapshot.values.get("test_results", [])],
                        }
                    ),
                }

            await asyncio.sleep(SSE_POLL_INTERVAL_SECONDS)

    return EventSourceResponse(event_stream())


@app.post("/runs/{run_id}/retry")
async def retry_run(run_id: str, request: Request) -> dict:
    """Re-drives a crashed run from its last good checkpoint. `_drive()`'s own
    docstring has always admitted a crash "leaves the checkpoint stalled with nothing
    automatically re-driving it" — this is that missing re-drive, made explicit and
    human-triggered rather than automatic (an automatic retry loop on a genuinely
    broken plan/site would just burn quota re-failing the same way).
    """
    graph = request.app.state.run_graph
    history = request.app.state.history
    config = {"configurable": {"thread_id": run_id}}

    row = await history.get_session(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="unknown run_id")
    if row["status"] != "error":
        raise HTTPException(status_code=409, detail=f"run is not in an error state (status: {row['status']})")

    snapshot = await graph.aget_state(config)
    if not snapshot.next:
        # The checkpoint says the run actually finished (or a resume already fixed it)
        # even though history still says "error" — nothing left to re-drive. This can
        # only happen if a previous retry succeeded but its history update raced this
        # request; safer to say so than to silently re-run a finished graph.
        raise HTTPException(status_code=409, detail="run has no pending work to resume")

    await _update_history_status(history, run_id, "running")
    # input_=None: NOT Command(resume=...) — there is no pending interrupt to answer,
    # just a superstep that never finished writing its checkpoint because the process
    # died mid-node. Invoking again with no new input re-plans from the last
    # successfully checkpointed state, which re-executes exactly that unfinished
    # superstep — the same fault-tolerance mechanism LangGraph documents for resuming a
    # thread after a crash, distinct from the interrupt-resume path resume_run uses.
    _track(asyncio.create_task(_drive(graph, None, _run_config(run_id), history)))
    return {"status": "retrying", "run_id": run_id}


@app.get("/runs/{run_id}/report")
async def get_report(run_id: str, request: Request) -> dict:
    graph = request.app.state.run_graph
    snapshot = await graph.aget_state({"configurable": {"thread_id": run_id}})
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="unknown run_id")
    if snapshot.next:
        raise HTTPException(status_code=404, detail="run not finished yet")

    return {
        "summary": snapshot.values.get("summary", {}),
        "test_cases": [_model_dump(tc) for tc in snapshot.values.get("test_cases", [])],
        "test_results": [_model_dump(r) for r in snapshot.values.get("test_results", [])],
        "plan_approved": snapshot.values.get("plan_approved", False),
    }


@app.post("/discover")
async def start_discovery(req: DiscoveryStartRequest, request: Request) -> dict:
    """Starts a chat-first discovery conversation: runs discovery to its first pause and
    returns synchronously (unlike /runs, a discovery turn is always bounded to exactly
    one interrupt with no fan-out, so awaiting it directly here is safe and simpler than
    the SSE-poll pattern used for actual runs).
    """
    discovery_id = f"disc-{uuid.uuid4()}"
    initial_state: DiscoveryState = {
        "target_url": req.target_url,
        "starting_idea": req.starting_idea,
        "messages": [],
        "site_context": SiteMap(pages=[]),
        "extra_dives_used": 0,
        "candidate_plan": None,
        "run_token": "",
        "turn_count": 0,
        "status": "in_progress",
        "pending_dive": None,
    }
    await request.app.state.history.create_session(
        id=discovery_id,
        kind="discovery",
        target_url=req.target_url,
        label=req.starting_idea or "(no starting idea — exploratory)",
        status="in_progress",
    )
    config = _discovery_config(discovery_id)
    await request.app.state.discovery_graph.ainvoke(initial_state, config=config)
    snapshot = await request.app.state.discovery_graph.aget_state(config)
    return {"discovery_id": discovery_id, **_discovery_turn_payload(snapshot)}


@app.get("/discover/{discovery_id}")
async def get_discovery(discovery_id: str, request: Request) -> dict:
    """Rehydrates a discovery conversation — there's no SSE stream backing the chat, so
    a reload/reconnect needs to re-fetch the transcript/candidate plan/status directly.
    """
    config = _discovery_config(discovery_id)
    snapshot = await request.app.state.discovery_graph.aget_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="unknown discovery_id")

    site_context: SiteMap = snapshot.values.get("site_context") or SiteMap(pages=[])
    candidate_plan: TestPlan | None = snapshot.values.get("candidate_plan")
    return {
        "status": snapshot.values.get("status", "in_progress"),
        "target_url": snapshot.values.get("target_url"),
        "turn_count": snapshot.values.get("turn_count", 0),
        "max_turns": MAX_DISCOVERY_TURNS,
        "transcript": _discovery_transcript(snapshot.values.get("messages")),
        "candidate_plan": [tc.model_dump() for tc in candidate_plan.test_cases] if candidate_plan else [],
        "site_pages_explored": [{"url": p.url, "title": p.title} for p in site_context.pages],
    }


@app.post("/discover/{discovery_id}/message")
async def send_discovery_message(discovery_id: str, req: DiscoveryMessageRequest, request: Request) -> dict:
    graph = request.app.state.discovery_graph
    config = _discovery_config(discovery_id)
    snapshot = await graph.aget_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="unknown discovery_id")

    pending = _pending_interrupts(snapshot)
    if not pending:
        raise HTTPException(status_code=409, detail="no pending discovery turn to respond to")
    if req.action == "reply":
        if snapshot.values.get("turn_count", 0) >= MAX_DISCOVERY_TURNS:
            raise HTTPException(status_code=400, detail="turn limit reached — approve or cancel")
        if not req.text:
            raise HTTPException(status_code=400, detail="text is required for a reply")

    resume = {pending[0]["id"]: {"action": req.action, "text": req.text}}
    await graph.ainvoke(Command(resume=resume), config=config)
    snapshot = await graph.aget_state(config)
    status = snapshot.values.get("status")

    if status == "cancelled":
        await _update_history_status(
            request.app.state.history, discovery_id, "cancelled", summary={"llm": llm_metrics.snapshot(discovery_id)}
        )
        llm_metrics.discard(discovery_id)
        return {"status": "cancelled"}

    if status == "approved":
        await _update_history_status(
            request.app.state.history, discovery_id, "approved", summary={"llm": llm_metrics.snapshot(discovery_id)}
        )
        llm_metrics.discard(discovery_id)

        candidate_plan: TestPlan = snapshot.values["candidate_plan"]
        # ensure_expected_result: discovery_agent_node already backfills this every turn,
        # this is defense in depth for the handoff into an actual run.
        test_cases = ensure_expected_result(ensure_unique_test_ids(candidate_plan.test_cases))
        instruction = snapshot.values.get("starting_idea") or "(test plan authored via discovery chat)"

        # CONFIRMED live: this used to be hardcoded to "" here, so any credentials the
        # user typed into the discovery chat (the whole reason discovery_agent_node
        # proactively asks for them — see DISCOVERY_SYSTEM_PROMPT) never reached
        # auth_setup_node, which reads exactly this field to decide "log in with the
        # given account" vs "sign up a fresh one." Every requires_auth case ran against
        # a throwaway signup (or, if that signup didn't finish in auth_setup_node's turn
        # budget, unauthenticated) instead of the real account the user provided — traced
        # directly to a test case that should have redirected off an authenticated
        # /login and instead just saw the plain sign-in form. Reconstructed here from the
        # human's own chat turns (excludes the synthetic "[Current context...]" addendum
        # and the assistant's replies), since that transcript is exactly where those
        # credentials/preferences live.
        discovery_context = "\n".join(
            m["text"] for m in _discovery_transcript(snapshot.values.get("messages")) if m["role"] == "user" and m["text"]
        )
        run_id = str(uuid.uuid4())
        initial_run_state: QAState = {
            "target_url": snapshot.values["target_url"],
            "instruction": instruction,
            "discovery_context": discovery_context,
            "run_token": snapshot.values.get("run_token", ""),
            "test_cases": test_cases,
            "test_results": [],
            "summary": {},
            "plan_approved": True,
            "auth_storage_state": None,
        }
        try:
            await request.app.state.history.create_session(
                id=run_id,
                kind="run",
                target_url=snapshot.values["target_url"],
                label=instruction,
                status="running",
                parent_id=discovery_id,
            )
        except Exception:
            # A missing history row only means this run won't show up in History, not
            # that the run fails — the client still needs run_id to navigate onward.
            logging.exception("failed to record history session for run %s", run_id)
        _track(
            asyncio.create_task(
                _drive(request.app.state.run_graph, initial_run_state, _run_config(run_id), request.app.state.history)
            )
        )
        return {"status": "approved", "run_id": run_id}

    return {"status": "in_progress", **_discovery_turn_payload(snapshot)}


@app.get("/history", response_model=HistoryListResponse)
async def list_history(
    request: Request,
    kind: Literal["run", "discovery"] | None = None,
    status: str | None = None,
    url: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 20,
    offset: int = 0,
) -> HistoryListResponse:
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    items, total = await request.app.state.history.list_sessions(
        kind=kind, status=status, url_contains=url, since=since, until=until, limit=limit, offset=offset
    )
    return HistoryListResponse(items=items, total=total, limit=limit, offset=offset)


@app.get("/history/stats", response_model=HistoryStatsResponse)
async def history_stats(request: Request, since: datetime | None = None, until: datetime | None = None) -> HistoryStatsResponse:
    stats = await request.app.state.history.get_stats(since=since, until=until)
    return HistoryStatsResponse(**stats)


if __name__ == "__main__":
    # `python -m backend.api` from the repo root — honors HOST/PORT from .env.
    # (Running via the `uvicorn backend.api:app` CLI instead binds to whatever
    # --host/--port you pass it, since .env is loaded after uvicorn already
    # opened its socket.)
    import uvicorn

    uvicorn.run(
        "backend.api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
