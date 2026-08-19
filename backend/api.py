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

from .core.discovery_state import DiscoveryState
from .core.history import HistoryStore, make_history_store
from .core.memory import make_store
from .core.models import SiteMap, TestPlan
from .core.run_planning import ensure_unique_test_ids
from .core.state import QAState
from .graph.builder import build_graph
from .graph.checkpointer import make_checkpointer
from .graph.discovery_graph import build_discovery_graph
from .nodes.worker import close_sessions_for_thread

EVIDENCE_DIR = Path(__file__).resolve().parent / "evidence"
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


def _track(task: asyncio.Task) -> asyncio.Task:
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


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
        yield

        if _background_tasks:
            logging.info(
                "Shutting down — waiting for %d in-flight run(s) to reach their next "
                "checkpoint before closing database connections...",
                len(_background_tasks),
            )
            await asyncio.gather(*_background_tasks, return_exceptions=True)


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
    return {"configurable": {"thread_id": run_id}, "max_concurrency": MAX_CONCURRENT_WORKERS}


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
    try:
        await graph.ainvoke(input_, config=config)
        snapshot = await graph.aget_state(config)
        if snapshot.next:
            await _update_history_status(history, thread_id, "paused")
        else:
            await _update_history_status(history, thread_id, "done", summary=snapshot.values.get("summary", {}))
    except Exception:
        logging.exception("run %s crashed", thread_id)
        await _update_history_status(history, thread_id, "error")
    finally:
        # No-op on the happy path (verdict_node already closed its own session) — this
        # only matters when a crash left a worker's Playwright session cached but never
        # closed, which would otherwise leak that browser subprocess forever.
        await close_sessions_for_thread(thread_id)


def _model_dump(value):
    return value.model_dump() if hasattr(value, "model_dump") else value


def _pending_interrupts(snapshot) -> list[dict]:
    # Confirmed against the installed langgraph: StateSnapshot.interrupts is a
    # top-level tuple[Interrupt, ...] (aggregated across all tasks), and Interrupt
    # has exactly .id / .value fields.
    return [{"id": intr.id, "type": intr.value.get("type"), "payload": intr.value} for intr in snapshot.interrupts]


def _discovery_config(discovery_id: str) -> dict:
    return {"configurable": {"thread_id": discovery_id}}


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
    if not snapshot.values:
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
    config = {"configurable": {"thread_id": run_id}}

    snapshot = await graph.aget_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="unknown run_id")

    async def event_stream() -> AsyncIterator[dict]:
        emitted_interrupt_ids: frozenset = frozenset()
        while True:
            snapshot = await graph.aget_state(config)
            pending = _pending_interrupts(snapshot)
            pending_ids = frozenset(i["id"] for i in pending)

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
        await _update_history_status(request.app.state.history, discovery_id, "cancelled")
        return {"status": "cancelled"}

    if status == "approved":
        await _update_history_status(request.app.state.history, discovery_id, "approved")

        candidate_plan: TestPlan = snapshot.values["candidate_plan"]
        test_cases = ensure_unique_test_ids(candidate_plan.test_cases)
        instruction = snapshot.values.get("starting_idea") or "(test plan authored via discovery chat)"
        run_id = str(uuid.uuid4())
        initial_run_state: QAState = {
            "target_url": snapshot.values["target_url"],
            "instruction": instruction,
            "discovery_context": "",
            "run_token": snapshot.values.get("run_token", ""),
            "test_cases": test_cases,
            "test_results": [],
            "summary": {},
            "plan_approved": True,
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
