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
  status: string;
  screenshot_path: string;
  trace_path?: string | null;
  video_path?: string | null;
  reason: string;
};

const STATUS_STYLES: Record<string, string> = {
  running: "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
  review: "border-amber-500/30 bg-amber-500/10 text-amber-300",
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
  awaitingApproval,
  apiBase,
  onViewTrace,
}: {
  testCase: TestCase;
  result?: TestResult;
  awaitingApproval?: boolean;
  apiBase?: string;
  onViewTrace?: (url: string) => void;
}) {
  const status = result?.status ?? (awaitingApproval ? "review" : "running");

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
          {status === "review" ? "Needs approval" : status}
        </span>
      </div>

      <div className="mt-5 space-y-3">
        {testCase.steps.map((step, index) => (
          <div key={`${testCase.test_id}-step-${index}`} className="flex items-start gap-3 rounded-2xl border border-white/5 bg-slate-950/35 px-3 py-2">
            <div className="mt-0.5 flex h-6 w-6 items-center justify-center rounded-full bg-slate-800 text-[10px] font-semibold text-slate-200">
              {index + 1}
            </div>
            <p className="text-sm leading-6 text-slate-300">{step}</p>
          </div>
        ))}
      </div>

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

          {apiBase && result.video_path && (
            <video
              src={`${apiBase}/${result.video_path}`}
              controls
              preload="metadata"
              className="mt-3 w-full rounded-xl border border-white/10 bg-black"
            />
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
