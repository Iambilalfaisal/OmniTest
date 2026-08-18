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
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .core.memory import make_store
from .core.state import QAState
from .graph.builder import build_graph
from .graph.checkpointer import make_checkpointer
from .nodes.worker import close_sessions_for_thread

EVIDENCE_DIR = Path(__file__).resolve().parent / "evidence"
MAX_CONCURRENT_WORKERS = int(os.getenv("MAX_CONCURRENT_WORKERS", "4"))
SSE_POLL_INTERVAL_SECONDS = float(os.getenv("SSE_POLL_INTERVAL_SECONDS", "1.0"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Graph construction can't happen at bare module-import time — building the
    # async Postgres saver/store needs a running event loop.
    async with AsyncExitStack() as stack:
        checkpointer = await stack.enter_async_context(make_checkpointer())

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

        app.state.graph = build_graph(checkpointer, store=store)
        yield


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


class RunHandle(BaseModel):
    run_id: str


class ResumeRequest(BaseModel):
    resume: dict[str, dict]


def _run_config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}, "max_concurrency": MAX_CONCURRENT_WORKERS}


async def _drive(graph, input_, config: dict) -> None:
    """Runs the graph to completion or its next pause. Fire-and-forget from a route
    handler — a crash here (retries exhausted) leaves the checkpoint stalled with
    nothing automatically re-driving it; see the plan's accepted limitations.
    """
    thread_id = config["configurable"]["thread_id"]
    try:
        await graph.ainvoke(input_, config=config)
    except Exception:
        logging.exception("run %s crashed", thread_id)
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


@app.post("/runs", response_model=RunHandle)
async def start_run(req: RunRequest, request: Request) -> RunHandle:
    run_id = str(uuid.uuid4())
    initial_state: QAState = {
        "target_url": req.target_url,
        "instruction": req.instruction,
        "test_cases": [],
        "test_results": [],
        "summary": {},
        "plan_approved": False,
    }
    asyncio.create_task(_drive(request.app.state.graph, initial_state, _run_config(run_id)))
    return RunHandle(run_id=run_id)


@app.post("/runs/{run_id}/resume")
async def resume_run(run_id: str, req: ResumeRequest, request: Request) -> dict:
    graph = request.app.state.graph
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

    asyncio.create_task(_drive(graph, Command(resume=req.resume), _run_config(run_id)))
    return {"status": "resumed", "run_id": run_id}


@app.get("/runs/{run_id}/events")
async def run_events(run_id: str, request: Request) -> EventSourceResponse:
    graph = request.app.state.graph
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
    graph = request.app.state.graph
    snapshot = await graph.aget_state({"configurable": {"thread_id": run_id}})
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="unknown run_id")
    if snapshot.next:
        raise HTTPException(status_code=404, detail="run not finished yet")

    return {
        "summary": snapshot.values.get("summary", {}),
        "test_results": [_model_dump(r) for r in snapshot.values.get("test_results", [])],
        "plan_approved": snapshot.values.get("plan_approved", False),
    }


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
