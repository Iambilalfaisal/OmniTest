"""FastAPI server: kicks off graph runs and streams node-by-node progress over SSE."""
from __future__ import annotations

# Loaded before the graph/node imports below, since those transitively construct
# LangChain/LangSmith clients that read OPENAI_API_KEY / LANGCHAIN_* from os.environ.
from dotenv import load_dotenv

load_dotenv()

import json
import os
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .core.state import QAState
from .graph.builder import build_graph

EVIDENCE_DIR = Path(__file__).resolve().parent / "evidence"

app = FastAPI(title="OmniTest Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/evidence", StaticFiles(directory=str(EVIDENCE_DIR)), name="evidence")

graph = build_graph()
_runs: dict[str, dict] = {}  # run_id -> {"target_url": ..., "instruction": ...}
_reports: dict[str, dict] = {}  # run_id -> {"summary": ..., "test_results": [...]}, once finished


class RunRequest(BaseModel):
    target_url: str
    instruction: str


class RunHandle(BaseModel):
    run_id: str


@app.post("/runs", response_model=RunHandle)
async def start_run(req: RunRequest) -> RunHandle:
    run_id = str(uuid.uuid4())
    _runs[run_id] = {"target_url": req.target_url, "instruction": req.instruction}
    return RunHandle(run_id=run_id)


@app.get("/runs/{run_id}/events")
async def run_events(run_id: str) -> EventSourceResponse:
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown run_id")

    async def event_stream() -> AsyncIterator[dict]:
        # MVP: the graph executes once per SSE connection; a client reconnect
        # would start a fresh run rather than resuming the in-flight one.
        initial_state: QAState = {
            "target_url": run["target_url"],
            "instruction": run["instruction"],
            "test_cases": [],
            "test_results": [],
            "summary": {},
        }
        final_results = []
        async for event in graph.astream(initial_state, stream_mode="updates"):
            node_name, payload = next(iter(event.items()))
            yield {"event": node_name, "data": _to_json(payload)}
            if node_name == "worker_node":
                final_results.extend(payload["test_results"])
            if node_name == "reporter_node":
                _reports[run_id] = {
                    "summary": payload["summary"],
                    "test_results": [r.model_dump() for r in final_results],
                }

    return EventSourceResponse(event_stream())


@app.get("/runs/{run_id}/report")
async def get_report(run_id: str) -> dict:
    report = _reports.get(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail="run not finished yet")
    return report


def _to_json(payload: dict) -> str:
    def default(value):
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return str(value)

    return json.dumps(payload, default=default)


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
