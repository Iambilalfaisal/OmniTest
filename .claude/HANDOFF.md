# HANDOFF — OmniTest — 11 tasks

## GOAL
Implement dynamic, application-grounded scenario discovery for OmniTest (a LangGraph AI
QA agent) — a "recon" stage that interactively probes each feature to ground scenario
generation in real observed UI, replacing/supplementing the static per-feature
checklist — and harden the surrounding pipeline as live runs surface real bugs.

## STACK / ENV
Python backend (LangGraph 1.2.11, FastAPI `api.py`, Postgres+pgvector via
`langgraph-checkpoint-postgres`/`langgraph-store-postgres`), Next.js/TypeScript
frontend. LLM: `gemini-3.5-flash-lite` for planner/worker/verdict/recon roles, shared
RPM=15 bucket + a token-bucket rate limiter (`backend/.env`). `MAX_CONCURRENT_WORKERS=5`.
Windows dev machine; backend venv at `backend/.venv`.

## ARCHIVE (compressed earlier tasks)
- Verified `dynamic-scenario-recon.md`'s build-order steps 4-7 (recon subgraph, graph
  wiring, reporter rollup, frontend grouping) were code-complete but never live-run
  verified; updated the plan doc accordingly.
- Root-caused and fixed 3 real bugs found via live runs:
  1. **Duplicate test cases** — Gemini structured output regenerating the same
     scenario(s) twice in one call. Fixed with BOTH a prompt-level self-check bullet
     (root cause, `TEST_CASE_AUTHORING_GUIDELINES`) AND a reinstated exact-match code
     backstop `drop_duplicate_scenarios` (`core/run_planning.py`, wired into `api.py`,
     `planner.py`, `discovery.py`, and `core/state.py`'s `_merge_test_cases` reducer).
     Confirmed via direct Postgres checkpoint inspection that a discovery-chat turn
     doubled a 5-case plan to 10 (`origin: planner`, not recon) — proof prompt-only
     enforcement is NOT reliable enough alone on this model.
  2. **Recon scenarios skipped the `expected_result` backstop** — same Gemini
     unenforced-required-field risk `TestCase.expected_result` already had a fix for,
     never extended to recon's `ScenarioProposalOut`. Fixed: defensive `getattr` reads
     in `_to_flow_reports` (`nodes/recon/nodes.py`) + `ensure_expected_result` applied
     in `recon_join_node` (`graph/builder.py`).
  3. **Recon never inherited the shared login** — any auth-gated feature got explored
     logged out, hit a wall, discovered nothing. Fixed by threading
     `auth_storage_state` through `ReconState` → `route_to_recon`'s Send payload →
     `recon_agent_node` (injected before its first `browser_navigate`, mirroring
     `agent_node`'s existing ordering).
- Added **semantic (embedding-based) duplicate detection** as a backstop layered on
  top of the exact-match one: `drop_semantic_duplicates` (`core/memory.py`, cosine
  similarity ≥0.93 on `goal` text, `GoogleGenerativeAIEmbeddings.aembed_documents
  (task_type="SEMANTIC_SIMILARITY")`). Wired into `recon_join_node` (now `async def`,
  compares recon's proposals against baseline `state["test_cases"]`),
  `discovery_agent_node`, `planner_node`, `api.py`'s approval handler. Key constraint:
  `_merge_test_cases` is a LangGraph reducer and MUST stay synchronous — the async
  embedding call had to move into the producing node (`recon_join_node`), not the
  reducer.
- Enabled `RECON_ENABLED=true` in `backend/.env` (was unset/false); documented
  `RECON_ENABLED`/`RECON_MAX_TURNS`/`SCENARIOS_PER_FEATURE_MAX`/
  `SCENARIOS_PER_RUN_MAX`/`SCENARIO_SIMILARITY_THRESHOLD` in `.env` + `.env.example`
  (neither had them before).
- Diagnosed a live "Connection lost — reconnecting" incident by querying the Postgres
  checkpoint directly: found uncommitted `pending_writes` mid-superstep, consistent
  with `_drive()`'s in-process driving task dying mid-run. Most likely cause: `uvicorn
  --reload` restarting the backend on every file edit made during this session while a
  run was in flight — a documented, accepted limitation (`_drive()`'s own docstring:
  no auto-resume-on-restart), not a code bug. **User has not yet confirmed via their
  backend terminal.**
- Rejected an external research proposal to reorder the graph (recon before planning)
  — grounded in the plan's own D3 decision (planner keeps a baseline, recon tops up,
  specifically so `RECON_ENABLED=false` degrades gracefully and plan review doesn't
  wait on full recon). Full current architecture (2 graphs — `discovery_graph.py` +
  the main run graph in `graph/builder.py`; 3 subgraphs — auth/recon/worker, all the
  same 3-node agent-loop shape) was walked in detail earlier; re-read those files
  directly if needed rather than re-deriving from this summary.

## RECENT TASKS (full detail)
- T9 Analyzed two pieces of external architecture research against the actual graph;
  rejected a proposed reordering (recon-before-planning) citing the plan's own D3
  decision; accepted and implemented 2 narrower ideas: (a) HTML validation-attribute
  scraping guidance added to `RECON_SYSTEM_PROMPT` (`nodes/recon/nodes.py`, new step 4,
  via `browser_evaluate` — `required`/`minlength`/`pattern`/`type`/`disabled`), (b) two
  new `TestCategory` values, `security` and `state_interaction`, threaded through
  `core/models.py` (`TestCategory` Literal + `TEST_CASE_AUTHORING_GUIDELINES` +
  `TestCase.category` field description), `nodes/worker/nodes.py`
  (`_CATEGORY_PASS_NOTES`), `nodes/recon/nodes.py` (`ScenarioProposalOut.category`
  description), and `frontend/src/components/WorkerCard.tsx`
  (`CATEGORY_STYLES`/`CATEGORY_LABELS`, teal/indigo). Verified: `py_compile` +
  real-package-context import checks on every touched backend file; confirmed the
  fragile `TEST_CASE_AUTHORING_GUIDELINES` `.format()` contract still round-trips on
  both `RECON_PLAN_PROMPT` and `PLANNER_PROMPT` with real kwargs (not just import-time).
- T10 Built a new GLOBAL Claude Code skill, `handoff`
  (`C:\Users\PC\.claude\skills\handoff\SKILL.md`), moving the full handoff procedure
  (task ledger, rolling `ARCHIVE` compression, exact template) out of the global
  `CLAUDE.md` §9 so it only loads into context when actually needed. Verified skill
  file location/frontmatter format and `SessionStart` hook capabilities via one
  `claude-code-guide` subagent call (permission asked and given first, per CLAUDE.md
  §8) — confirmed `~/.claude/skills/<name>/SKILL.md` with plain `name`/`description`
  frontmatter is correct; confirmed there is NO true auto-context-injection via hooks
  (a `SessionStart` hook can only cat a file to stdout, which the model still has to
  parse from the transcript) — the one thing that DOES auto-load a file into context
  is a project's own root `CLAUDE.md` using `@path` import syntax, documented in the
  skill as an opt-in the user must confirm per-project, not something wired in
  automatically. Trimmed global `CLAUDE.md` §9 from ~100 lines to ~20 (kept the
  trigger list visible, moved ledger/compression/template detail into the skill).
- T11 Answered a factual question (no code change): confirmed `~/.claude/` (global
  `CLAUDE.md`, the new `handoff` skill, settings, memory) is stored under the Windows
  user profile, not scoped to the authenticated Anthropic account — switching Claude
  accounts on the same machine reads the identical local setup with no extra work.
  Immediately after, the user correctly pointed out this session was already well past
  every handoff threshold and I hadn't invoked the new skill — this handoff is that
  correction, invoked via `Skill(skill="handoff")`.

## CURRENT STATE
- Verified working: every backend file touched this session imports/compiles cleanly
  (`py_compile` + real package-context `import backend.X` checks, run repeatedly,
  most recently after the category-taxonomy change). The `TEST_CASE_AUTHORING_
  GUIDELINES` `.format()` contract confirmed intact with real kwargs, not just at
  import time.
- Unverified: NOTHING from this session has been exercised by an actual live run yet.
  Recon's auth-injection, the `expected_result` backstop, semantic dedup, the two new
  categories, and the HTML-validation-attribute recon guidance are import-checked
  only.
- Broken / unresolved: "Connection lost — reconnecting" on runs `399ccd7e-06aa-4a96-
  811f-87ff133c9992` and `cd302666-c652-4632-9945-08e847352cf7` — root cause diagnosed
  (see ARCHIVE) but not confirmed by the user checking their backend terminal, and not
  something fixable in code (architectural, accepted limitation of `_drive()`).
- Uncommitted changes in: `backend/core/models.py`, `backend/core/run_planning.py`,
  `backend/core/state.py`, `backend/core/memory.py`, `backend/api.py`,
  `backend/graph/builder.py`, `backend/nodes/planner.py`, `backend/nodes/discovery.py`,
  `backend/nodes/recon/nodes.py`, `backend/nodes/recon/state.py`,
  `backend/nodes/worker/nodes.py`, `backend/.env`, `backend/.env.example`,
  `frontend/src/components/WorkerCard.tsx`, plus
  `C:\Users\PC\.claude\plans\dynamic-scenario-recon.md`. Nothing committed to git this
  entire session.

## KEY FILES
- `backend/graph/builder.py` — main graph DAG; `recon_join_node` is now `async def`
  (semantic dedup + `expected_result` backstop + auth threading all land here).
- `backend/nodes/recon/nodes.py` — recon subgraph; `RECON_SYSTEM_PROMPT` now has
  HTML-attribute-scraping guidance (step 4); auth injection before first navigate.
- `backend/core/run_planning.py` — `drop_duplicate_scenarios` (exact-match backstop).
- `backend/core/memory.py` — `drop_semantic_duplicates` (new, embedding-based
  backstop) alongside the existing long-term memory/RAG functions.
- `backend/core/models.py` — `TestCategory` now has 6 values; `TEST_CASE_AUTHORING_
  GUIDELINES` is shared prompt text embedded via `.format()` in 3 places — any edit
  MUST preserve the "only `{run_token}` unescaped brace" rule.
- `C:\Users\PC\.claude\plans\dynamic-scenario-recon.md` — design doc; §8 now says
  steps 1-7 code-complete, step 8 (live run) still not done.
- `C:\Users\PC\.claude\skills\handoff\SKILL.md` — this session's own meta-work.

## DECISIONS & CONSTRAINTS
- Do NOT reorder the graph to put recon before planning — explicitly considered and
  rejected (see ARCHIVE).
- Prompt-only enforcement (self-check bullets) is NOT sufficient alone for Gemini
  structured-output reliability on this project's cheap model — always pair with a
  deterministic code-level backstop. Confirmed twice this session.
- LangGraph reducers (`Annotated[..., reducer_fn]`) must be synchronous — async work
  needed while merging state has to happen in the producing node before the reducer
  ever sees it.
- `~/.claude/skills/` and the global `CLAUDE.md` are machine/OS-user-scoped, not
  Anthropic-account-scoped.

## NEXT STEP
1. User checks their backend terminal for a crash/restart around the "Connection
   lost" incidents, to confirm the `--reload`-during-edits hypothesis, THEN does one
   live run (RECON_ENABLED=true, already set) without concurrent file edits — nothing
   from this entire session has been live-verified yet.

## COMMANDS FOR THE USER TO RUN
- Check the backend terminal/process log around the timestamps of runs `399ccd7e-...`
  and `cd302666-...` for a crash or reload-restart line.
- `cd backend && python -c "import graph.builder, nodes.recon.nodes, nodes.reporter, nodes.worker.nodes, core.models, core.memory"`
  — final combined import sanity check from a normal shell (already run piecemeal via
  the venv directly this session; this is for the user's own confirmation).
- A live run against a real target with `RECON_ENABLED=true`, started only after file
  edits stop landing, checking specifically: recon actually authenticates, grounded
  (not generic) scenarios appear, no duplicate scenarios reappear, and the new
  `security`/`state_interaction` categories show up where relevant.

## GOTCHAS
- If running via `uvicorn --reload`, any file edit — including future ones from a
  continued session — restarts the backend mid-run and orphans any in-flight
  execution (the checkpoint survives; nothing drives it forward). Avoid editing
  backend files while a run that matters is executing.
- `TEST_CASE_AUTHORING_GUIDELINES` tolerates exactly one unescaped brace pair,
  `{run_token}` — a future edit introducing a stray `{`/`}` will only raise at the
  LATER `.format()` call site (`recon_plan_node`, `planner_node`), not at import time.
  Retest with the actual `.format(...)` call after touching this string, not just an
  import check.
- `SCENARIO_SIMILARITY_THRESHOLD` (0.93) and the two new categories are untuned — the
  first live run may show the threshold is too strict/loose, or that the model rarely
  reaches for `security`/`state_interaction` without more explicit prompting.
