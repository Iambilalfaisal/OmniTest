# HANDOFF — OmniTest — 19 tasks

## GOAL
Fix the shared-authentication pipeline and turn-budget exhaustion in OmniTest (AI-driven
QA test runner: LangGraph + Playwright MCP + Gemini) so a login is established once and
reused correctly across parallel test cases, and so legitimate multi-step flows (e.g. a
multi-tab project-creation wizard) get enough turns/guidance to actually finish instead of
failing incomplete.

## STACK / ENV
Python/FastAPI + LangGraph backend, Postgres checkpointer (localhost:5432), Playwright MCP
(`npx @playwright/mcp`), Gemini `gemini-3.5-flash-lite` via langchain-google-genai
(OpenRouter free-tier fallback configured for 429/5xx via `*_FALLBACK_MODEL` env vars).
Next.js frontend. Windows machine, venv at `backend/.venv`. Target app under test:
`https://nucleusone-dev.aiimone.com/`, test creds `fatima.shahzad@acme-one.com` /
`Test123!@#`.

## ARCHIVE (compressed earlier tasks)
- Fixed the shared-auth pipeline end-to-end: `auth-wall reconsideration gate` narrowed to
  genuine give-up phrases, `auth_note` gated on `auth_restored`, `ensure_requires_auth()`
  backstop added (core/run_planning.py), `TEST_CASE_AUTHORING_GUIDELINES` strengthened,
  `discovery_context` fallback to `raw_plan_text` fixed (api.py), `auth_save_node` now
  verifies the browser actually left the login screen before saving state, compound
  "login then X" plans now decompose into two test cases (custom_plan.py), credential
  redaction added at the API output boundary only, `extract_eval_value` fixed to parse
  `browser_evaluate`'s raw MCP result before any `str()`/`repr()`, `authenticated_landing_url`
  threaded through so a `requires_auth` case navigates straight past the public landing page.
- **Key discovery**: `Verdict.deviations`/`amended_steps` (the UI's "Plan Amendments"
  panel) is the final verdict LLM call's own ONE-SHOT RETROSPECTIVE narrative — never
  proof a live turn-budget mechanism (PROGRESS `deviated` line, `trigger_rediscovery`)
  actually fired. Counting `httpx: ... generateContent` lines per `test=<id>` tag in the
  console log is the only reliable way to know real turn usage.
- Raised `MAX_TOOL_TURNS_CEILING` 20→30, `MAX_TOOL_TURNS` (base) 10→16, gave
  `rediscovery_node`'s replan path a scaled bonus it previously granted none of. Added
  per-turn diagnostic logging (`tool_node`/`agent_node` no-tool-call content) — this is
  what made T16 below possible to actually diagnose instead of guess at.

## RECENT TASKS (full detail)
- T16 Live-diagnosed a `create-project` run against the T13-15 fixes: turn_budget stayed
  flat at 16 the ENTIRE run — zero bonus ever applied. Confirmed via console log that
  THREE separate self-report mechanisms all failed to fire in this one run: no `deviated`
  PROGRESS line, no `trigger_rediscovery` call, and the model ignored even the explicit
  `_stuck_nudge` "this is your last turn, call ask_human" injected message at turn 15.
- T17 Added a deterministic, self-report-independent turn-budget auto-extension in
  `tool_node` (nodes/worker/nodes.py) — grants `TURN_BUDGET_BONUS` more turns (capped at
  `MAX_TOOL_TURNS_CEILING`) whenever the agent is near its budget AND its most recent
  call isn't part of a stuck-repeat streak, with NO dependence on the model announcing
  anything. First attempt gated on `repeats == 0` and silently never fired in two full
  live runs — root cause: `_recent_repeated_calls` walks `state["messages"]`, whose last
  entry is always the very AIMessage that produced the current call (already appended by
  `add_messages` before `tool_node` runs), so an ordinary single-call turn always matches
  itself and `repeats` is never actually 0. Fixed to reuse `_stuck_nudge`'s own existing
  "not stuck" definition — `repeats + 1 < STUCK_REPEAT_THRESHOLD` — instead of inventing a
  separate threshold. **Confirmed live** after the fix: `turn_budget` auto-extended
  16→21→26→30 across a single `create-project` run, reaching the full ceiling instead of
  dying at the base 16.
- T18 Generalized the "auth-wall reconsideration gate" (agent_node) to catch ANY
  no-tool-call response, not just auth-signal phrases — needed because T16 showed
  `create-project` stopping early on a text-only "PROGRESS: step=3 status=on_track
  note=Filling in project details form fields" turn with 10 turns still in budget. First
  version fired on every no-tool-call-while-budget-remains, which **regressed** a
  genuinely-finished `user-login` test (confirmed live): it had already reached the
  dashboard and reported success at turn 7, but got rejected and re-prompted every turn
  through turn 12, wasting 5 turns restating the same completed outcome. Fixed by gating
  the new generic branch on `parsed_step is not None and parsed_step < len(test_case.steps)`
  — only retry when the model's OWN step count says it isn't on the plan's last step yet,
  leaving a genuine final report (step count matches, or no PROGRESS line at all)
  untouched. **Confirmed live** after the fix: `user-login` passed cleanly with no
  wasted retries in the next two runs.
- T19 Added a new WORKER_SYSTEM_PROMPT paragraph ("Multi-step wizards with an unspecified
  required field") after live evidence (and explicit user direction) showed
  `create-project`'s real remaining blocker isn't turns — it's the "Add Members"/"Roles
  and Tasks" wizard steps requiring a SPECIFIC value (which team member, which role) that
  the test case never specifies. Told the agent explicitly to call `ask_human` the moment
  it hits such a field instead of guessing, repeatedly searching, or clicking through
  hoping it's optional — mirrors the existing login-wall `ask_human` example's structure
  and vividness rather than relying on the one buried generic line that already existed.
  **NOT YET CONFIRMED** — the very next test run crashed on `verdict_node` (see Broken,
  below) before reaching a verdict, so whether the agent actually calls `ask_human` here
  is still unknown.

## CURRENT STATE
- Verified working live (T17, T18): turn-budget auto-extension fires correctly
  (16→21→26→30 observed); the no-tool-call reconsideration gate no longer regresses an
  already-finished test case.
- Verified working (earlier, still holds): `login-happy-path`/`user-login` passes cleanly
  across every run this session (5 consecutive runs, no failures).
- Unverified: T19's `ask_human` prompt addition — untested due to the crash below.
  `create-project` has NOT yet passed end-to-end in any run this session; best result so
  far is `Blocked` (reached "Add Members" step, ran out of the full 30-turn ceiling) —
  real progress, not a bug, but still incomplete.
- Broken (external, NOT a code bug): the run submitted to test T19 crashed —
  `verdict_node`'s LLM call fell back to OpenRouter (`VERDICT_FALLBACK_MODEL`) and that
  fallback's configured free-tier slug (`openai/gpt-oss-20b`) now 404s — OpenRouter says
  "This model is unavailable for free." Almost certainly triggered by Gemini's free-tier
  quota being exhausted after 5 consecutive full live runs in one session. Needs either a
  quota reset (wait) or the user updating `VERDICT_FALLBACK_MODEL`/`WORKER_FALLBACK_MODEL`
  in `.env` to a currently-live free OpenRouter slug — not something to guess at blindly.
- Uncommitted changes in (never committed this session): `backend/nodes/worker/nodes.py`
  (this session's T17/T18/T19 edits, on top of the prior session's uncommitted changes to
  the same file plus `backend/nodes/auth/nodes.py`, `backend/nodes/auth/state.py`,
  `backend/core/state.py`, `backend/core/run_planning.py`, `backend/core/models.py`,
  `backend/nodes/planner.py`, `backend/nodes/discovery.py`, `backend/nodes/custom_plan.py`,
  `backend/graph/builder.py`, `backend/api.py`, `backend/mcp/client.py`).

## KEY FILES
- `backend/nodes/worker/nodes.py` — worker subgraph (`agent_node`/`tool_node`/
  `rediscovery_node`/`verdict_node`), `WORKER_SYSTEM_PROMPT`, turn-budget constants
  (`MAX_TOOL_TURNS=16`, `MAX_TOOL_TURNS_CEILING=30`, `TURN_BUDGET_BONUS=5`,
  `STUCK_REPEAT_THRESHOLD=3`). This session's new deterministic auto-extension lives in
  `tool_node`'s normal-dispatch return block; the generalized no-tool-call gate and the
  new wizard-field `ask_human` prompt paragraph live in `agent_node`/`WORKER_SYSTEM_PROMPT`.
- `backend/nodes/auth/nodes.py` — shared-login subgraph, untouched this session.
- `backend/core/llm.py` — `*_FALLBACK_MODEL` role→env-var mapping (`ModelRole.VERDICT` →
  `VERDICT_FALLBACK_MODEL`) — this is what's currently misconfigured/dead on OpenRouter.
- `backend/api.py` — `POST /runs` (accepts `raw_plan_text` for "My Own Plan" mode),
  `GET /runs/{id}/report` (404 until finished), `GET /runs/{id}/events` (SSE; emits
  `paused` with `{"interrupts":[...]}` on an `ask_human`/risky-action pause, `done` with
  the final summary, `error` on a crash), `POST /runs/{id}/resume` (body
  `{"resume": {<interrupt_id>: <answer>}}` — `{"text": "..."}` for a `clarification` type,
  `{"approved": true, "reason": "..."}` for a `risky_action` type).

## DECISIONS & CONSTRAINTS
- Model (`gemini-3.5-flash-lite`) is repeatedly confirmed unreliable at every self-report
  mechanism tried so far (PROGRESS `deviated` line, `trigger_rediscovery`, an explicit
  injected "last turn" nudge) — always prefer a deterministic, code-level signal
  (`repeats`/`turn_count`/`parsed_step`, all already tracked) over trusting the model to
  announce anything, but verify the deterministic signal's ACTUAL behavior live before
  trusting it (T17's `repeats==0` bug: correct-looking code that never fired because of
  how `_recent_repeated_calls` self-matches the current turn).
- A broadened "retry on no tool call" gate MUST distinguish a genuinely-finished text-only
  report from a prematurely-stopped one (T18) — gate on the model's own `parsed_step` vs.
  `len(test_case.steps)`, never on "no tool call + budget remains" alone, or it forces
  already-finished tests into pointless retry loops.
- Server must be FULLY restarted after backend edits, not just left to `--reload` — this
  session again confirmed `--reload` cannot be trusted to have picked up an edit; a full
  stop/restart was done before every one of this session's 5 test runs.
- Don't guess new OpenRouter model slugs — check what's actually live on the account first
  (see Broken, above); this is the user's account/config to change, not something to
  silently patch.

## NEXT STEP
1. Resolve the OpenRouter fallback crash first (wait for Gemini quota reset, or user
   updates `VERDICT_FALLBACK_MODEL`/`WORKER_FALLBACK_MODEL` in `.env` to a live free slug).
2. Re-run the same repro ("Test the login using these creds fatima.shahzad@acme-one.com
   and Test123!@# and create a Project" against `https://nucleusone-dev.aiimone.com/`) and
   watch specifically for T19: does the agent now call `ask_human` when it hits the "Add
   Members"/"Roles and Tasks" wizard steps instead of guessing or grinding turns? If it
   pauses, a human (or the automated SSE-watch pattern below) needs to answer and resume.
   If `create-project` still doesn't finish, get the fresh console log before changing
   anything else — don't guess at another ceiling bump without new evidence.

## COMMANDS FOR THE USER TO RUN
- `cd C:\Users\PC\Desktop\Repo-Cleaning\OmniTest; backend\.venv\Scripts\Activate.ps1;
  uvicorn backend.api:app --reload --port 8000` — starts the server. Always confirm a
  FULL restart happened (stop the old process, don't rely on `--reload` alone).
- To reproduce headlessly via the API instead of the UI (what this session used):
  `curl -s -X POST http://127.0.0.1:8000/runs -H "Content-Type: application/json" -d
  '{"target_url":"https://nucleusone-dev.aiimone.com/","instruction":"...","raw_plan_text":
  "Test the login using these creds fatima.shahzad@acme-one.com and Test123!@# and create
  a Project"}'` then poll `GET /runs/{run_id}/report` (404 until finished) or stream
  `GET /runs/{run_id}/events` to catch a `paused` event and resume it — see KEY FILES
  above for the exact resume payload shape.

## GOTCHAS
- `uvicorn --reload` cannot be trusted to pick up an edit before a test run — always fully
  stop and restart.
- Launching uvicorn WITHOUT `--reload` fails to start at all in this environment (a
  Postgres connection-pool timeout tied to Windows' `ProactorEventLoop` vs. psycopg's
  async mode) — always include `--reload` even for a one-off run.
- `browser_evaluate`'s raw MCP tool result must go through `extract_eval_value`
  (mcp/client.py) — never `str()`/`repr()` it directly, that mangles embedded newlines.
- Counting `httpx: ... generateContent` lines per `test=<id>` tag in the console log is
  the only reliable way to know real per-test turn usage — the UI's live progress bar and
  the verdict's narrated "amendments" can both be misleading (see the `Verdict.deviations`
  discovery in ARCHIVE).
- `_recent_repeated_calls` (nodes/worker/nodes.py) always counts the CURRENT pending call
  against itself first — `repeats` is a "not stuck" measure (compare against
  `STUCK_REPEAT_THRESHOLD`), never a "did anything happen yet" measure; don't reach for
  `repeats == 0` again, it's structurally unreachable for a normal single-call turn.
- A live run against the real target app costs real Gemini free-tier quota — this session
  burned 5 full runs (~35-52 LLM calls each) and very likely exhausted the daily quota,
  which is what triggered the OpenRouter-fallback crash. Space out live-run verification.
