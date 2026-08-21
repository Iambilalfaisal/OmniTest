"use client";

import { useState } from "react";

export type TestCategory = "happy_path" | "edge_case" | "negative" | "error_handling";
export type TestPriority = "high" | "medium" | "low";

export type TestCase = {
  test_id: string;
  goal: string;
  category: TestCategory;
  priority: TestPriority;
  preconditions: string[];
  // The oracle the run is graded against (backend/core/models.py). Optional here only so
  // a plan persisted before the field existed still renders instead of crashing.
  expected_result?: string;
  steps: string[];
};

export type TestResult = {
  test_id: string;
  // "Pass" | "Fail" | "Blocked" (backend/nodes/worker/nodes.py's Verdict) — kept as a
  // plain string here rather than a union so a result from before "Blocked" existed
  // still renders instead of a type-narrowing crash.
  status: string;
  screenshot_path: string;
  trace_path?: string | null;
  // One short clip per mutating action, not one video for the whole test case — see
  // backend/nodes/worker/evidence.py's capture_mutation_clip. Play with
  // ClipSequencePlayer below, which advances through them automatically.
  video_clips?: string[];
  reason: string;
  // What the adaptive worker had to work around this run, and the steps as actually
  // executed when they diverged from the written plan (backend/core/models.py's
  // TestResult) — optional so a result from before these fields existed still renders.
  deviations?: string[];
  amended_steps?: string[];
  last_step_reached?: number;
};

export type WorkerPhase = "queued" | "running" | "awaiting_input" | "grading" | "done";

// Live, step-level detail for a test case that hasn't reached its final TestResult yet
// (backend/core/progress.py, streamed in via the SSE `progress` event's
// `worker_progress` map) — one entry per test_id, absent once no longer needed (the
// backend clears its whole run on completion; a card with a `result` simply stops
// reading this).
export type WorkerProgress = {
  phase: WorkerPhase;
  step_index: number;
  total_steps: number;
  current_action: string | null;
  turn: number;
  budget: number | null;
  deviations: number;
  asks: number;
  updated_at: number;
};

export const WORKER_PHASE_LABELS: Record<WorkerPhase, string> = {
  queued: "Queued",
  running: "Running",
  awaiting_input: "Needs input",
  grading: "Grading",
  done: "Done",
};

// Plays a list of short per-action clips back to back so they read as one continuous
// video, with no server-side stitching needed: WebM containers can't be safely
// concatenated by joining bytes, and each clip is already a valid, independent file
// (confirmed live against the installed @playwright/mcp — browser_start_video/
// browser_stop_video can be called repeatedly within one session). `key={src}` forces
// the <video> element to remount on each clip change, which is what makes the browser
// reliably load and play the new source instead of sometimes sticking on the old one.
export function ClipSequencePlayer({
  clips,
  apiBase,
  className,
}: {
  clips: string[];
  apiBase: string;
  className?: string;
}) {
  const [index, setIndex] = useState(0);
  if (clips.length === 0) return null;
  const clampedIndex = Math.min(index, clips.length - 1);
  const src = `${apiBase}/${clips[clampedIndex]}`;

  return (
    <div className={className}>
      <video
        key={src}
        src={src}
        controls
        muted
        // Clip 1 waits for the reviewer to press play, same as the old single-video
        // behavior; every clip after that autoplays so playback doesn't stall waiting
        // for a click each time one clip ends.
        autoPlay={clampedIndex > 0}
        preload="metadata"
        onEnded={() => setIndex((i) => Math.min(i + 1, clips.length - 1))}
        className="w-full rounded-xl border border-white/10 bg-black"
      />
      {clips.length > 1 && (
        <div className="mt-1.5 flex items-center justify-between text-[10px] uppercase tracking-[0.2em] text-slate-500">
          <span>
            Clip {clampedIndex + 1} of {clips.length}
          </span>
          <div className="flex gap-3">
            <button
              type="button"
              onClick={() => setIndex((i) => Math.max(i - 1, 0))}
              disabled={clampedIndex === 0}
              className="disabled:opacity-30"
            >
              ← Prev
            </button>
            <button
              type="button"
              onClick={() => setIndex((i) => Math.min(i + 1, clips.length - 1))}
              disabled={clampedIndex === clips.length - 1}
              className="disabled:opacity-30"
            >
              Next →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

const STATUS_STYLES: Record<string, string> = {
  queued: "border-slate-500/30 bg-slate-500/10 text-slate-300",
  running: "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
  grading: "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
  // review/awaiting_input/Blocked share the same "needs attention" amber — all three
  // mean the same thing to a reviewer skimming cards: something here needs a look.
  review: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  awaiting_input: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  Blocked: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  Pass: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  Fail: "border-rose-500/30 bg-rose-500/10 text-rose-300",
  default: "border-slate-500/30 bg-slate-500/10 text-slate-300",
};

export const CATEGORY_STYLES: Record<TestCategory, string> = {
  happy_path: "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
  edge_case: "border-violet-500/30 bg-violet-500/10 text-violet-300",
  negative: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  error_handling: "border-orange-500/30 bg-orange-500/10 text-orange-300",
};

export const CATEGORY_LABELS: Record<TestCategory, string> = {
  happy_path: "Happy path",
  edge_case: "Edge case",
  negative: "Negative",
  error_handling: "Error handling",
};

// History session status/kind pills (frontend/src/components/HistoryDashboard.tsx).
// Colocated here since this file is already the de facto shared pill-style module.
// Colors chosen by MEANING, not per literal status string, reusing this app's existing
// pass/fail/running/review convention above rather than inventing a new hue per status:
// running/in_progress share the "in progress" cyan already used for a live TestResult;
// done/approved share the "good, terminal" emerald already used for Pass; paused reuses
// the "needs attention" amber already used for review; error reuses the "critical" rose
// already used for Fail; cancelled gets neutral slate (a deliberate non-outcome, not a
// failure). `kind` (run vs discovery) gets two fresh hues — blue/fuchsia — distinct from
// every category/status hue above so it never impersonates either dimension.
export type SessionStatus = "running" | "paused" | "done" | "error" | "in_progress" | "approved" | "cancelled";
export type SessionKind = "run" | "discovery";

export const SESSION_STATUS_STYLES: Record<SessionStatus, string> = {
  running: "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
  in_progress: "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
  paused: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  done: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  approved: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  error: "border-rose-500/30 bg-rose-500/10 text-rose-300",
  cancelled: "border-slate-500/30 bg-slate-500/10 text-slate-300",
};

export const SESSION_STATUS_LABELS: Record<SessionStatus, string> = {
  running: "Running",
  in_progress: "In progress",
  paused: "Paused",
  done: "Done",
  approved: "Approved",
  error: "Error",
  cancelled: "Cancelled",
};

export const KIND_STYLES: Record<SessionKind, string> = {
  run: "border-blue-500/30 bg-blue-500/10 text-blue-300",
  discovery: "border-fuchsia-500/30 bg-fuchsia-500/10 text-fuchsia-300",
};

export const KIND_LABELS: Record<SessionKind, string> = {
  run: "Run",
  discovery: "Discovery",
};

export default function WorkerCard({
  testCase,
  result,
  progress,
  awaitingApproval,
  apiBase,
  onViewTrace,
}: {
  testCase: TestCase;
  result?: TestResult;
  // Live detail while this test case has no `result` yet — see WorkerProgress above.
  // Ignored once `result` exists (the card has its final story to tell at that point).
  progress?: WorkerProgress;
  awaitingApproval?: boolean;
  apiBase?: string;
  onViewTrace?: (url: string) => void;
}) {
  const status = result?.status ?? (awaitingApproval ? "review" : progress?.phase ?? "running");
  const statusLabel =
    status === "review" ? "Needs approval" : (WORKER_PHASE_LABELS as Record<string, string>)[status] ?? status;

  // Which step ticks to draw as done/current/pending. Once a result exists,
  // last_step_reached (Verdict, backend/nodes/worker/nodes.py) is authoritative; before
  // that, progress.step_index is a live, best-effort high-water mark (see
  // core/progress.py's monotonic clamp) — ambiguous between "just finished this step"
  // and "currently on it", so the step AT that index is drawn as `current`, not `done`,
  // rather than claiming more certainty than the underlying signal actually has.
  const reachedStep = result ? result.last_step_reached ?? testCase.steps.length : progress?.step_index ?? 0;

  return (
    <article className="glass-panel rounded-3xl p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.28em] text-slate-400">
            <span>Test case</span>
            {testCase.category && (
              <span className={`rounded-full border px-2 py-0.5 font-semibold ${CATEGORY_STYLES[testCase.category] ?? ""}`}>
                {CATEGORY_LABELS[testCase.category] ?? testCase.category}
              </span>
            )}
          </div>
          <h3 className="mt-2 text-xl font-semibold text-white">{testCase.goal}</h3>
          {testCase.preconditions?.length > 0 && (
            <p className="mt-1 text-xs italic text-slate-400">Setup: {testCase.preconditions.join("; ")}</p>
          )}
          {testCase.expected_result && (
            <p className="mt-1 text-xs text-slate-400">
              <span className="font-semibold text-slate-300">Expected: </span>
              {testCase.expected_result}
            </p>
          )}
        </div>
        <span className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.2em] ${STATUS_STYLES[status] ?? STATUS_STYLES.default}`}>
          {statusLabel}
        </span>
      </div>

      {!result && progress && (
        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-slate-400">
          <span className="relative flex h-2 w-2 shrink-0">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-cyan-400 opacity-75" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-cyan-400" />
          </span>
          <span>
            Turn {progress.turn}
            {progress.budget ? `/${progress.budget}` : ""}
            {progress.current_action ? ` · ${progress.current_action}` : ""}
          </span>
          {progress.deviations > 0 && (
            <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium text-amber-300">
              {progress.deviations} adapted
            </span>
          )}
        </div>
      )}

      <div className="mt-5 space-y-3">
        {testCase.steps.map((step, index) => {
          const stepNum = index + 1;
          const isCurrent = !result && stepNum === reachedStep;
          const isDone = stepNum < reachedStep || (!!result && stepNum <= reachedStep);
          return (
            <div
              key={`${testCase.test_id}-step-${index}`}
              className="flex items-start gap-3 rounded-2xl border border-white/5 bg-slate-950/35 px-3 py-2"
            >
              <div
                className={`mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold ${
                  isDone
                    ? "bg-emerald-500/20 text-emerald-300"
                    : isCurrent
                      ? "animate-pulse bg-cyan-500/20 text-cyan-300"
                      : "bg-slate-800 text-slate-200"
                }`}
              >
                {isDone ? "✓" : stepNum}
              </div>
              <p className="text-sm leading-6 text-slate-300">{step}</p>
            </div>
          );
        })}
      </div>

      {result && (result.deviations?.length || result.amended_steps?.length) ? (
        <div className="mt-3 rounded-2xl border border-amber-500/20 bg-amber-500/5 p-3">
          <div className="text-[10px] uppercase tracking-[0.25em] text-amber-300/80">Plan amendments</div>
          {result.deviations && result.deviations.length > 0 && (
            <ul className="mt-2 space-y-1 text-xs leading-5 text-slate-300">
              {result.deviations.map((deviation, i) => (
                <li key={i}>• {deviation}</li>
              ))}
            </ul>
          )}
          {result.amended_steps && result.amended_steps.length > 0 && (
            <p className="mt-2 text-xs leading-5 text-slate-400">
              <span className="font-semibold text-slate-300">Executed as: </span>
              {result.amended_steps.join(" → ")}
            </p>
          )}
        </div>
      ) : null}

      {result && (
        <div className="mt-5 rounded-2xl border border-white/10 bg-slate-950/40 p-3">
          <div className="text-[10px] uppercase tracking-[0.25em] text-slate-400">Verdict</div>
          <p className="mt-2 text-sm leading-6 text-slate-200">{result.reason}</p>

          {apiBase && result.screenshot_path && (
            <a
              href={`${apiBase}/${result.screenshot_path}`}
              target="_blank"
              rel="noreferrer"
              className="mt-3 block overflow-hidden rounded-xl border border-white/10 bg-slate-950/60"
            >
              <img
                src={`${apiBase}/${result.screenshot_path}`}
                alt={`Final screenshot for ${testCase.test_id}`}
                className="max-h-56 w-full object-contain"
                loading="lazy"
              />
            </a>
          )}

          {apiBase && result.video_clips && result.video_clips.length > 0 && (
            <ClipSequencePlayer clips={result.video_clips} apiBase={apiBase} className="mt-3" />
          )}

          {apiBase && result.trace_path && onViewTrace && (
            <button
              type="button"
              onClick={() => onViewTrace(`${apiBase}/${result.trace_path}`)}
              className="mt-3 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3 py-1 text-[11px] font-medium text-cyan-200 transition hover:border-cyan-400/50"
            >
              View trace
            </button>
          )}
        </div>
      )}
    </article>
  );
}
