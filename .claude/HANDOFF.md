# HANDOFF — OmniTest — reconstructed from repo state (no prior in-session tasks)

## GOAL
Implement dynamic, application-grounded scenario discovery: a new **recon** stage that
interactively probes each feature (clicks past the landing page, into forms/wizards/OAuth)
to ground scenario generation in real observed UI, replacing the static per-feature
checklist in `TEST_CASE_AUTHORING_GUIDELINES`. Full design in
`C:\Users\PC\.claude\plans\dynamic-scenario-recon.md`.

## STACK / ENV
Python backend (LangGraph 1.2.11, FastAPI `api.py`), Next.js/TypeScript frontend.
LLM: `gemini-3.5-flash-lite` for planner/worker/verdict, shared RPM=15 bucket
(`backend/.env`). `MAX_CONCURRENT_WORKERS=5`, `MAX_TOOL_TURNS=8` (ceiling 20).

## ARCHIVE (compressed earlier tasks)
n/a — this is the first turn of this conversation. No task ledger exists yet.

## CURRENT STATE (as observed directly from git + files, not from conversation memory)

**Plan doc's own build-order checklist says only steps 1–3 are done** (timeouts + rate
limiter; models/state/reducer; late-Send fan-in safety check). **But `git diff --stat`
shows far more is already written in the working tree**, matching steps 4–7 too:

- `backend/nodes/recon/` (new, untracked: `__init__.py`, `state.py`, `nodes.py`) — the
  recon subgraph (step 4)
- `backend/graph/builder.py` — 157 lines changed, includes `RECON_ENABLED` env flag
  (`builder.py:29`) and a route function (`builder.py:63-76`, doc'd as auth_setup's
  outgoing edge, degrades to today's behavior when `RECON_ENABLED=false` or no features)
  — this is the barrier-variant wiring from §7.1 (step 5)
- `backend/nodes/reporter.py` — 72 lines changed — likely `by_feature`/`by_flow` rollup
  (step 6)
- `frontend/src/components/RunPageClient.tsx` (+119), `WorkerCard.tsx` (+53),
  `frontend/src/app/reports/page.tsx` (+26) — likely two-level feature/flow grouping
  (step 7)
- Also touched beyond the plan's file list: `backend/mcp/client.py` (+64/-,
  `nodes/worker/session.py` (+48), `nodes/worker/evidence.py`, `nodes/auth/nodes.py` (+96)

No `TODO`/`FIXME`/`NotImplementedError` markers found in any of these files — code reads
as complete, not stubbed.

- **Verified:** nothing in this conversation. The plan doc records earlier
  verification for steps 1–3 only (import/compile checks, reducer collision test,
  timeout/rate-limiter kwarg construction) — dated 2026-08-22, presumably from a prior
  session.
- **Unverified:** whether the step 4–7 code (recon subgraph, graph wiring, reporter
  rollup, frontend grouping) actually works — no live run, no compile check performed
  this turn. Per the plan's own Risk 7: "Recon quality is unverifiable without a live
  run" — code-reading is explicitly called out as insufficient for this class of change.
- **Broken:** unknown — not run this turn.
- **Uncommitted changes in:** all 21 files listed in `git status` (see below), plus
  untracked `backend/nodes/recon/` and `.claude/`.

```
M backend/.env.example                      M backend/nodes/agent_loop.py
M backend/api.py                            M backend/nodes/auth/nodes.py
M backend/core/llm.py                       M backend/nodes/auth/state.py
M backend/core/models.py                    M backend/nodes/discovery.py
M backend/core/progress.py                  M backend/nodes/planner.py
M backend/core/run_planning.py              M backend/nodes/reporter.py
M backend/core/state.py                     M backend/nodes/worker/evidence.py
M backend/graph/builder.py                  M backend/nodes/worker/nodes.py
M backend/mcp/client.py                     M backend/nodes/worker/session.py
?? backend/nodes/recon/                     M frontend/src/app/reports/page.tsx
                                             M frontend/src/components/RunPageClient.tsx
                                             M frontend/src/components/WorkerCard.tsx
```

## KEY FILES
- `C:\Users\PC\.claude\plans\dynamic-scenario-recon.md` — the full design doc; open in
  IDE at session start. Decisions in §7, build order in §8.
- `backend/graph/builder.py:29,63-76` — `RECON_ENABLED` flag + the barrier-variant
  routing function that gates the whole recon path.
- `backend/nodes/recon/{state.py,nodes.py}` — new recon subgraph, mirrors
  `backend/nodes/auth/` shape (agent/tool/save-style nodes).
- `backend/core/state.py` — should contain the custom `test_cases` merge reducer
  (§7.5 of the plan) — **not yet confirmed present**, only inferred from the plan and
  the 60-line diff.
- `backend/nodes/reporter.py` — should contain `by_feature`/`by_flow` rollup — not
  yet read this turn.

## DECISIONS & CONSTRAINTS (from the plan doc, confirmed 2026-08-22)
- Threads-in-worker, deepening the read-only crawl, worker-does-discovery, and a
  separate task queue are all rejected (§3) — do not revisit these.
- Recon fans out at the **feature** level, not the leaf test case (§7.2).
- Overlap variant (recon running concurrently with baseline workers, both feeding
  `reporter` directly) is **confirmed unsafe** — double-fires `reporter_node` with
  corrupted state. **Barrier variant only**: recon → plain-edge join node → join fans
  out to `worker_node` in one wave → `reporter` (§7.1).
- `test_cases` needs a **custom** reducer (concat + re-run `ensure_unique_test_ids`),
  not `operator.add` — plain concatenation can't dedupe collided ids (§7.5).
- Three timeout layers required and marked DONE: per-tool-call, per-session-open,
  per-leaf deadline (§7.3) — deadline must reset on human-in-the-loop resume, not be
  naive wall-clock.
- Global token-bucket rate limiter (`InMemoryRateLimiter`) added ahead of 429s,
  marked DONE (§7.4).
- Recon may interact and submit non-destructive forms; risky actions (delete/purchase/
  pay/remove) still hit the existing `review_if_risky` interrupt — no new gating (D1).
- Budgets are env-tunable, defaults near today's run size: `SCENARIOS_PER_FEATURE_MAX≈6`,
  `SCENARIOS_PER_RUN_MAX≈15–18` (D2). Truncation must be reported, never silent.
- FlowReport cache: 24h TTL, same pattern as existing `site_map` cache (D4).

## NEXT STEP
1. Read `backend/core/state.py` and `backend/nodes/reporter.py` in full to confirm
   whether the `test_cases` custom reducer (§7.5) and `by_feature` rollup are actually
   implemented as the diff size suggests, or partially stubbed.
2. Update the plan doc's §8 build-order checklist to match reality (it currently
   understates progress — only steps 1–3 are marked done).
3. Run a backend import/compile check (ask user to run — see below) before attempting
   a live run, since none has happened yet this session.

## COMMANDS FOR ME TO RUN
- `cd backend && python -c "import graph.builder, nodes.recon.nodes, nodes.reporter"` —
  confirms the new recon subgraph and modified reporter still import cleanly with
  `RECON_ENABLED=false` (default).
- A live run against a real target app (per plan Risk 7, this is the only way to
  confirm recon quality) — do not run unprompted, it drives a real browser against a
  real site and may submit forms.

## GOTCHAS
- `RECON_ENABLED` defaults to `false` — a plain compile/import check will not exercise
  the new recon path at all; that requires explicitly setting the env var.
- Any in-flight/checkpointed runs from before this change will fail validation against
  the new `TestCase` fields (plan explicitly accepts this — not a bug to fix).
- `.env` (not `.env.example`) needs the new vars manually if not already present:
  `RECON_ENABLED`, `RECON_MAX_TURNS`, `SCENARIOS_PER_FEATURE_MAX`,
  `SCENARIOS_PER_RUN_MAX`, `FLOW_REPORT_TTL_HOURS`, `TOOL_CALL_TIMEOUT_SECONDS`,
  `SESSION_OPEN_TIMEOUT_SECONDS`, `SCENARIO_DEADLINE_SECONDS`,
  `LLM_REQUESTS_PER_SECOND`, `LLM_MAX_BURST` — `.env.example` was diffed (+28 lines) so
  likely documents these, but the real `.env` should be checked separately.
- Per [[omnitest_optimization_plan]] and [[feedback_no_external_deps_for_evidence]]
  memory notes: this project's changes have historically only been confirmed correct by
  a live run, never by reading code alone — hold that bar here too.
