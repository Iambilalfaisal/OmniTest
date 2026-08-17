# OmniTest: Autonomous AI Quality Assurance 🧪

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Map%2FReduce-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Playwright MCP](https://img.shields.io/badge/Playwright-MCP-green.svg)](https://github.com/microsoft/playwright-mcp)

**OmniTest** is a full-stack, autonomous AI testing platform. It uses the Model Context Protocol (MCP) to read the semantic accessibility trees of web applications, generates isolated test cases, and executes them concurrently using LangGraph's dynamic routing.

Unlike traditional headless testing scripts, OmniTest features a **live Next.js execution canvas** that streams the AI's reasoning, map-reduce fan-out, and artifact collection (screenshots, `.webm` videos, and `.zip` traces) in real time via Server-Sent Events (SSE).

## ✨ System Architecture

OmniTest operates on a decoupled frontend/backend architecture:

1. **The AI Engine (FastAPI + LangGraph):** 
   - Uses `langchain-mcp-adapters` to connect a `MultiServerMCPClient` to the official Playwright server. 
   - Analyzes user intent, translates it into strict `TestCase` Pydantic models, and uses LangGraph's `Send` API to spin up parallel, isolated browser contexts.
2. **The Dashboard (Next.js):** 
   - Connects to the FastAPI backend to trigger test runs.
   - Consumes SSE streams to render live "Worker Cards" as parallel tests execute.
   - Embeds the official Playwright Trace Viewer to inspect the DOM state of failed tests directly in the browser.

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Node.js 18+ 
* An OpenAI or Anthropic API Key

### 1. Backend Setup (FastAPI Engine)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install AI orchestration dependencies
pip install -r requirements.txt
cp .env.example .env       # add your ANTHROPIC_API_KEY

# Install Playwright browser binaries for the MCP server
npx playwright install --with-deps chromium

# Start the API server from the repo root, so `backend` resolves as a package
# (backend/mcp/ is a submodule of `backend`, not a top-level `mcp` — running it
# from inside backend/ would put that dir on sys.path and shadow the real `mcp`
# PyPI package that langchain-mcp-adapters depends on)
cd ..
uvicorn backend.api:app --reload --port 8000
```

### 2. Frontend Setup (Next.js Dashboard)

```bash
cd frontend
npm install
cp .env.example .env.local

npm run dev
```

Open http://localhost:3000, start a run, and watch the worker cards fill in live as
the backend streams `planner` / `worker` / `reporter` events over SSE.

## 📁 Project Layout

```
omnitest/
├── backend/                  # FastAPI & LangGraph Engine (Python)
│   ├── core/
│   │   ├── state.py          # QAState TypedDict & operator.add reducer
│   │   └── models.py         # Pydantic schemas for the LLM
│   ├── mcp/
│   │   └── client.py         # langchain-mcp-adapters MultiServerMCPClient setup
│   ├── nodes/
│   │   ├── planner.py        # Accessibility tree analysis & test generation
│   │   ├── worker.py         # Playwright execution & artifact capture
│   │   └── reporter.py       # Metrics aggregation
│   ├── graph/
│   │   └── builder.py        # DAG routing & Send API logic
│   ├── evidence/              # Local storage for traces, videos, and screenshots
│   ├── api.py                 # FastAPI server & SSE streaming endpoints
│   └── requirements.txt
│
├── frontend/                  # Next.js UI Dashboard (Node.js)
│   ├── src/
│   │   ├── app/
│   │   │   ├── run/           # Live execution canvas (consumes SSE)
│   │   │   ├── reports/       # Evidence gallery & metrics
│   │   │   └── layout.tsx
│   │   └── components/
│   │       ├── WorkerCard.tsx   # Live status card for parallel nodes
│   │       └── TraceViewer.tsx  # Iframe embed for Playwright traces
│   ├── package.json
│   └── tailwind.config.ts
│
├── .gitignore
└── README.md
```