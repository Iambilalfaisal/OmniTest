"use client";

import { useEffect, useState } from "react";

// ─── shared model types ──────────────────────────────────────────────────────

export type TestCategory =
  | "happy_path"
  | "edge_case"
  | "negative"
  | "error_handling"
  | "security"
  | "state_interaction";
export type TestPriority = "high" | "medium" | "low";

export type TestCase = {
  test_id: string;
  goal: string;
  category: TestCategory;
  priority: TestPriority;
  preconditions: string[];
  expected_result?: string;
  steps: string[];
  feature_id?: string | null;
  flow_id?: string | null;
  origin?: "planner" | "recon";
  discovery_rationale?: string | null;
};

export type Feature = {
  feature_id: string;
  name: string;
  description: string;
};

export type FeaturePhase = "exploring" | "done";

export type FeatureProgress = {
  name: string;
  phase: FeaturePhase;
  scenario_count: number;
  updated_at: number;
};

export const FEATURE_PHASE_LABELS: Record<FeaturePhase, string> = {
  exploring: "Analyzing application…",
  done:      "Discovery complete",
};

export type TestResult = {
  test_id: string;
  status: string;
  screenshot_path: string;
  trace_path?: string | null;
  video_clips?: string[];
  reason: string;
  deviations?: string[];
  amended_steps?: string[];
  last_step_reached?: number;
};

export type WorkerPhase = "queued" | "running" | "awaiting_input" | "grading" | "done" | "rediscovering" | "replanning";

// Mirrors backend/core/progress.py's MutationEvent TypedDict — a single recorded
// adaptation event streamed live via the SSE `worker_progress` payload so the
// WorkerCard can render a mutation timeline while the test is still running.
export type MutationEventType = "deviation" | "clarification" | "risky_blocked";

export type MutationEvent = {
  type: MutationEventType;
  step: number;
  description: string;
  user_decision: string | null;
  sensitive: boolean;
  timestamp: number;
  resolved: boolean;
};

export type PlanHistoryEntry = {
  version: number;
  trigger: string;
  original_steps: string[];
  new_steps: string[];
  reason: string;
  replanned: boolean;
};

export type WorkerProgress = {
  phase: WorkerPhase;
  step_index: number;
  total_steps: number;
  current_action: string | null;
  turn: number;
  budget: number | null;
  deviations: number;
  asks: number;
  mutation_events?: MutationEvent[];
  // Adaptive replanning — mirrors backend/core/progress.py plan_version/plan_history
  plan_version?: number;
  plan_history?: PlanHistoryEntry[];
  updated_at: number;
};

export const WORKER_PHASE_LABELS: Record<WorkerPhase, string> = {
  queued:         "Queued",
  running:        "Running",
  awaiting_input: "Needs input",
  grading:        "Grading",
  done:           "Done",
  rediscovering:  "Re-observing",
  replanning:     "Replanning",
};

// ─── in-card interrupt types ──────────────────────────────────────────────────

export type CardInterruptType = "risky_action" | "clarification";

export type CardInterrupt = {
  id: string;
  type: CardInterruptType;
  payload: {
    type: CardInterruptType;
    test_id: string;
    // risky_action fields
    tool?: string;
    args?: Record<string, unknown>;
    // clarification fields
    question?: string;
    context?: string | null;
    sensitive?: boolean;
  };
};

export type CardDecision =
  | { approved: boolean; reason?: string }  // risky_action
  | { text: string };                        // clarification

// ─── style tables ─────────────────────────────────────────────────────────────

const STATUS_STYLES: Record<string, string> = {
  queued:         "border-slate-500/30 bg-slate-500/10 text-slate-300",
  running:        "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
  grading:        "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
  rediscovering:  "border-violet-500/30 bg-violet-500/10 text-violet-300",
  replanning:     "border-violet-500/30 bg-violet-500/10 text-violet-300",
  review:         "border-amber-500/30 bg-amber-500/10 text-amber-300",
  awaiting_input: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  Blocked:        "border-amber-500/30 bg-amber-500/10 text-amber-300",
  Pass:           "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  Fail:           "border-rose-500/30 bg-rose-500/10 text-rose-300",
  default:        "border-slate-500/30 bg-slate-500/10 text-slate-300",
};

export const CATEGORY_STYLES: Record<TestCategory, string> = {
  happy_path:       "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
  edge_case:        "border-violet-500/30 bg-violet-500/10 text-violet-300",
  negative:         "border-amber-500/30 bg-amber-500/10 text-amber-300",
  error_handling:   "border-orange-500/30 bg-orange-500/10 text-orange-300",
  security:         "border-teal-500/30 bg-teal-500/10 text-teal-300",
  state_interaction:"border-indigo-500/30 bg-indigo-500/10 text-indigo-300",
};

export const CATEGORY_LABELS: Record<TestCategory, string> = {
  happy_path:        "Happy path",
  edge_case:         "Edge case",
  negative:          "Negative",
  error_handling:    "Error handling",
  security:          "Security",
  state_interaction: "State interaction",
};

// History session style tables (used by HistoryDashboard)
export type SessionStatus = "running" | "paused" | "done" | "error" | "in_progress" | "approved" | "cancelled";
export type SessionKind   = "run" | "discovery";

export const SESSION_STATUS_STYLES: Record<SessionStatus, string> = {
  running:     "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
  in_progress: "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
  paused:      "border-amber-500/30 bg-amber-500/10 text-amber-300",
  done:        "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  approved:    "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  error:       "border-rose-500/30 bg-rose-500/10 text-rose-300",
  cancelled:   "border-slate-500/30 bg-slate-500/10 text-slate-300",
};

export const SESSION_STATUS_LABELS: Record<SessionStatus, string> = {
  running:     "Running",
  in_progress: "In progress",
  paused:      "Paused",
  done:        "Done",
  approved:    "Approved",
  error:       "Error",
  cancelled:   "Cancelled",
};

export const KIND_STYLES: Record<SessionKind, string> = {
  run:       "border-blue-500/30 bg-blue-500/10 text-blue-300",
  discovery: "border-fuchsia-500/30 bg-fuchsia-500/10 text-fuchsia-300",
};

export const KIND_LABELS: Record<SessionKind, string> = {
  run:       "Run",
  discovery: "Discovery",
};

// ─── ClipSequencePlayer ───────────────────────────────────────────────────────

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
        autoPlay={clampedIndex > 0}
        preload="metadata"
        onEnded={() => setIndex((i) => Math.min(i + 1, clips.length - 1))}
        className="w-full rounded-xl border border-white/10 bg-black"
      />
      {clips.length > 1 && (
        <div className="mt-1.5 flex items-center justify-between text-[10px] uppercase tracking-[0.2em] text-slate-500">
          <span>Clip {clampedIndex + 1} of {clips.length}</span>
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

// ─── MutationTimeline ─────────────────────────────────────────────────────────

const MUTATION_TYPE_STYLES: Record<MutationEventType, { border: string; bg: string; dot: string; label: string }> = {
  deviation:     { border: "border-amber-500/25",  bg: "bg-amber-500/5",   dot: "bg-amber-400",   label: "Plan adaptation"    },
  clarification: { border: "border-violet-500/25", bg: "bg-violet-500/5",  dot: "bg-violet-400",  label: "Input requested"    },
  risky_blocked: { border: "border-rose-500/25",   bg: "bg-rose-500/5",    dot: "bg-rose-400",    label: "Action blocked"     },
};

function MutationTimeline({
  events,
  isLive,
}: {
  events: MutationEvent[];
  isLive: boolean;
}) {
  const [open, setOpen] = useState(true);
  if (events.length === 0) return null;

  return (
    <div className="mt-3 rounded-2xl border border-white/8 bg-slate-950/30">
      {/* Header row */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-3.5 py-2.5 text-left"
      >
        <span className="text-[10px] uppercase tracking-[0.25em] text-slate-400">
          Mutation history · {events.length} event{events.length !== 1 ? "s" : ""}
        </span>
        <span className={`text-[10px] text-slate-600 transition-transform duration-200 ${open ? "rotate-180" : ""}`}>▾</span>
      </button>

      {open && (
        <div className="flex flex-col gap-2 px-3.5 pb-3.5">
          {events.map((evt, i) => {
            const style = MUTATION_TYPE_STYLES[evt.type] ?? MUTATION_TYPE_STYLES.deviation;
            const pending = !evt.resolved && isLive;

            return (
              <div
                key={i}
                className={`rounded-xl border px-3 py-2.5 ${style.border} ${style.bg}`}
              >
                {/* Row header */}
                <div className="flex items-center gap-2">
                  <span className={`relative flex h-2 w-2 shrink-0`}>
                    {pending && (
                      <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${style.dot}`} />
                    )}
                    <span className={`relative inline-flex h-2 w-2 rounded-full ${style.dot} ${pending ? "opacity-80" : "opacity-60"}`} />
                  </span>
                  <span className="text-[10px] uppercase tracking-[0.22em] text-slate-400">
                    {style.label}
                    {evt.step > 0 && <span className="ml-1.5 text-slate-500"> · step {evt.step}</span>}
                  </span>
                  {pending && (
                    <span className="ml-auto text-[10px] text-amber-300/70 animate-pulse">waiting…</span>
                  )}
                  {evt.resolved && (
                    <span className="ml-auto text-[10px] text-emerald-400/70">✓ resolved</span>
                  )}
                </div>

                {/* Description */}
                <p className="mt-1.5 text-xs leading-5 text-slate-200">{evt.description}</p>

                {/* User decision (post-resume) */}
                {evt.resolved && evt.user_decision && (
                  <div className="mt-2 flex items-start gap-2 rounded-lg border border-white/8 bg-slate-950/40 px-2.5 py-2">
                    <span className="mt-0.5 text-[10px] text-slate-500">↩</span>
                    <p className="text-xs text-slate-300">
                      {evt.sensitive && evt.user_decision !== "[sensitive — not displayed]"
                        ? "[sensitive — hidden]"
                        : evt.user_decision}
                    </p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ─── PlanEvolutionTimeline ────────────────────────────────────────────────────

function PlanEvolutionTimeline({
  testCase,
  progress,
}: {
  testCase: TestCase;
  progress?: WorkerProgress;
}) {
  const [open, setOpen] = useState(false);

  const planVersion = progress?.plan_version ?? 0;
  const planHistory = progress?.plan_history ?? [];
  // Only render if at least one replanning round happened
  const replanRounds = planHistory.filter((e) => e.replanned);
  if (planVersion === 0 || replanRounds.length === 0) return null;

  return (
    <div className="mt-3 rounded-2xl border border-violet-500/20 bg-violet-500/5">
      {/* Header */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-3.5 py-2.5 text-left"
      >
        <div className="flex items-center gap-2">
          <span className="flex h-1.5 w-1.5 rounded-full bg-violet-400" />
          <span className="text-[10px] uppercase tracking-[0.25em] text-violet-300/80">
            Plan updated · {replanRounds.length} revision{replanRounds.length !== 1 ? "s" : ""}
          </span>
        </div>
        <span className={`text-[10px] text-slate-600 transition-transform duration-200 ${open ? "rotate-180" : ""}`}>
          ▾
        </span>
      </button>

      {open && (
        <div className="flex flex-col gap-3 px-3.5 pb-3.5">
          {/* Original plan */}
          <div className="rounded-xl border border-white/6 bg-slate-950/30 px-3 py-2.5">
            <div className="mb-1.5 text-[10px] uppercase tracking-[0.2em] text-slate-500">
              Original plan
            </div>
            <ol className="space-y-0.5">
              {testCase.steps.map((step, i) => (
                <li key={i} className="flex items-start gap-2 text-xs text-slate-400">
                  <span className="mt-0.5 shrink-0 text-[10px] text-slate-600">{i + 1}.</span>
                  <span className="leading-5">{step}</span>
                </li>
              ))}
            </ol>
          </div>

          {/* Each replan round */}
          {replanRounds.map((entry) => (
            <div key={entry.version} className="rounded-xl border border-violet-500/20 bg-violet-500/5 px-3 py-2.5">
              <div className="mb-1 flex items-center gap-2">
                <span className="rounded-full border border-violet-400/40 bg-violet-500/15 px-2 py-0.5 text-[10px] font-semibold text-violet-300">
                  Revision {entry.version}
                </span>
              </div>
              <p className="mb-1 text-[11px] italic text-violet-200/70">
                After: {entry.trigger}
              </p>
              <p className="mb-2 text-xs text-slate-400">{entry.reason}</p>
              <div className="text-[10px] uppercase tracking-[0.18em] text-slate-500 mb-1">
                Updated steps
              </div>
              <ol className="space-y-0.5">
                {entry.new_steps.map((step, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-slate-200">
                    <span className="mt-0.5 shrink-0 text-[10px] text-violet-400/70">{i + 1}.</span>
                    <span className="leading-5">{step}</span>
                  </li>
                ))}
              </ol>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── In-card interrupt helpers ─────────────────────────────────────────────────

const RISKY_ACTION_VERBS: Record<string, string> = {
  browser_click:         "click",
  browser_type:          "type into",
  browser_fill_form:     "fill out",
  browser_select_option: "choose an option in",
  browser_press_key:     "press a key on",
  browser_drag:          "drag",
  browser_hover:         "hover over",
  browser_file_upload:   "upload a file to",
  browser_navigate:      "go to",
};

function describeAction(tool: string, args: Record<string, unknown>): string {
  if (tool === "browser_run_code_unsafe" && typeof args.code === "string") {
    const snippet = args.code.trim().slice(0, 100);
    return `Run custom code: "${snippet}${args.code.trim().length > 100 ? "…" : ""}"`;
  }
  const verb   = RISKY_ACTION_VERBS[tool] ?? tool.replace(/^browser_/, "").replace(/_/g, " ");
  const target =
    typeof args.element === "string" && args.element ? `"${args.element}"`
    : typeof args.url   === "string" && args.url     ? args.url
    : typeof args.text  === "string" && args.text    ? `"${args.text}"`
    : typeof args.value === "string" && args.value   ? `"${args.value}"`
    : "this element";
  return `${verb.charAt(0).toUpperCase() + verb.slice(1)} ${target}`;
}

// ─── InCardInterruptPanel ─────────────────────────────────────────────────────

function InCardInterruptPanel({
  interrupt,
  onAnswer,
  answered,
}: {
  interrupt: CardInterrupt;
  onAnswer: (id: string, decision: CardDecision) => void;
  answered: boolean;
}) {
  const [riskyDecision, setRiskyDecision] = useState<boolean | null>(null);
  const [answerText, setAnswerText]       = useState("");

  if (answered) {
    return (
      <div className="mt-3 flex items-center gap-2 rounded-2xl border border-emerald-500/20 bg-emerald-500/5 px-4 py-2.5 text-xs text-emerald-300/80">
        <span>✓</span>
        <span>Decision recorded — waiting for other tests to answer…</span>
      </div>
    );
  }

  if (interrupt.type === "risky_action") {
    const { tool, args } = interrupt.payload as { tool: string; args: Record<string, unknown> };
    const description = describeAction(tool, args);

    return (
      <div className="mt-3 rounded-2xl border border-amber-400/30 bg-amber-500/5 p-4">
        <div className="flex items-center gap-2">
          <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-amber-400/50 bg-amber-500/15 text-[10px] text-amber-300">
            !
          </span>
          <span className="text-[10px] uppercase tracking-[0.25em] text-amber-300/80">Agent needs approval</span>
        </div>
        <p className="mt-2 text-sm font-medium text-white">{description}</p>
        <details className="mt-2">
          <summary className="cursor-pointer text-[10px] text-slate-500 hover:text-slate-300">
            Technical details
          </summary>
          <pre className="mt-1.5 overflow-x-auto rounded-xl bg-slate-950/60 p-2.5 text-[10px] text-slate-300">
            {tool}{"\n"}{JSON.stringify(args, null, 2)}
          </pre>
        </details>
        <div className="mt-3 flex gap-2">
          <button
            type="button"
            onClick={() => setRiskyDecision(true)}
            className={`rounded-full border px-3 py-1.5 text-xs font-medium transition ${
              riskyDecision === true
                ? "border-emerald-400/60 bg-emerald-500/20 text-emerald-200"
                : "border-white/10 text-slate-300 hover:border-emerald-400/40"
            }`}
          >
            Yes, go ahead
          </button>
          <button
            type="button"
            onClick={() => setRiskyDecision(false)}
            className={`rounded-full border px-3 py-1.5 text-xs font-medium transition ${
              riskyDecision === false
                ? "border-rose-400/60 bg-rose-500/20 text-rose-200"
                : "border-white/10 text-slate-300 hover:border-rose-400/40"
            }`}
          >
            No, skip
          </button>
          {riskyDecision !== null && (
            <button
              type="button"
              onClick={() =>
                onAnswer(interrupt.id, riskyDecision
                  ? { approved: true }
                  : { approved: false, reason: "Blocked by reviewer" })
              }
              className="ml-auto rounded-full border border-cyan-400/40 bg-cyan-500/10 px-3 py-1.5 text-xs font-semibold text-cyan-200 transition hover:bg-cyan-500/20"
            >
              Confirm
            </button>
          )}
        </div>
      </div>
    );
  }

  // clarification
  const { question, context, sensitive } = interrupt.payload as {
    question: string;
    context?: string | null;
    sensitive?: boolean;
  };

  return (
    <div className="mt-3 rounded-2xl border border-amber-400/30 bg-amber-500/5 p-4">
      <div className="flex items-center gap-2">
        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-amber-400/50 bg-amber-500/15 text-[10px] text-amber-300">
          ?
        </span>
        <span className="text-[10px] uppercase tracking-[0.25em] text-amber-300/80">Agent is asking</span>
      </div>
      <p className="mt-2 text-sm font-semibold text-white">{question}</p>
      {context && <p className="mt-1 text-xs text-slate-400">{context}</p>}
      <div className="mt-3 flex gap-2">
        <input
          type={sensitive ? "password" : "text"}
          autoComplete="off"
          value={answerText}
          onChange={(e) => setAnswerText(e.target.value)}
          placeholder={sensitive ? "Your answer (hidden)" : "Type your answer…"}
          className="flex-1 rounded-xl border border-white/10 bg-slate-950/60 px-3 py-2 text-sm text-white outline-none transition focus:border-cyan-400/70 focus:ring-2 focus:ring-cyan-500/30"
        />
        <button
          type="button"
          disabled={!answerText.trim()}
          onClick={() => onAnswer(interrupt.id, { text: answerText.trim() })}
          className="rounded-xl border border-cyan-400/40 bg-cyan-500/10 px-3 py-2 text-xs font-semibold text-cyan-200 transition disabled:opacity-40 hover:bg-cyan-500/20"
        >
          Send
        </button>
      </div>
    </div>
  );
}

// ─── WorkerCard ───────────────────────────────────────────────────────────────

export default function WorkerCard({
  testCase,
  result,
  progress,
  interrupt,
  onAnswer,
  answered,
  apiBase,
  onViewTrace,
}: {
  testCase: TestCase;
  result?: TestResult;
  progress?: WorkerProgress;
  interrupt?: CardInterrupt;
  onAnswer?: (id: string, decision: CardDecision) => void;
  answered?: boolean;
  apiBase?: string;
  onViewTrace?: (url: string) => void;
}) {
  const hasInterrupt = !!interrupt && !result;

  const status =
    result?.status ??
    (hasInterrupt ? "review" : progress?.phase ?? "queued");

  const statusLabel =
    status === "review"
      ? "Needs input"
      : (WORKER_PHASE_LABELS as Record<string, string>)[status] ?? status;

  // Auto-expand cards that are live or need attention
  const isActivePhase = (s: string) =>
    s === "running" || s === "awaiting_input" || s === "review" ||
    s === "grading" || s === "rediscovering" || s === "replanning";

  const [expanded, setExpanded] = useState(isActivePhase(status));

  useEffect(() => {
    if (isActivePhase(status)) {
      setExpanded(true);
    }
  }, [status]);

  // Step progress tracking
  const reachedStep = result
    ? (result.last_step_reached ?? testCase.steps.length)
    : (progress?.step_index ?? 0);

  const isLive = !result && (
    status === "running" || status === "grading" ||
    status === "rediscovering" || status === "replanning"
  );
  const isTerminal = !!result;

  return (
    <article
      className={`glass-panel flex flex-col rounded-3xl transition-all duration-300 ${
        hasInterrupt
          ? "border-amber-400/30 shadow-[0_0_24px_rgba(251,191,36,0.07)]"
          : isLive
            ? "border-cyan-400/20 shadow-[0_0_24px_rgba(57,231,211,0.06)]"
            : ""
      }`}
    >
      {/* ── card header ── */}
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-start justify-between gap-4 p-5 text-left"
        aria-expanded={expanded}
      >
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2 text-[10px] uppercase tracking-[0.28em] text-slate-400">
            <span>Test case</span>
            {testCase.category && (
              <span className={`rounded-full border px-2 py-0.5 font-semibold ${CATEGORY_STYLES[testCase.category] ?? ""}`}>
                {CATEGORY_LABELS[testCase.category] ?? testCase.category}
              </span>
            )}
            {testCase.origin === "recon" && (
              <span
                className="rounded-full border border-fuchsia-500/30 bg-fuchsia-500/10 px-2 py-0.5 font-semibold text-fuchsia-300"
                title="Discovered by exploring the real application"
              >
                Discovered
              </span>
            )}
          </div>
          <h3 className="mt-2 text-lg font-semibold leading-snug text-white">{testCase.goal}</h3>

          {/* Compact live status strip — visible in both collapsed and expanded */}
          {isLive && progress?.current_action && (
            <div className="mt-2 flex items-center gap-1.5 text-xs text-cyan-300/80">
              <span className="relative flex h-1.5 w-1.5 shrink-0">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-cyan-400 opacity-75" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-cyan-400" />
              </span>
              <span className="truncate">{progress.current_action}</span>
            </div>
          )}

          {/* Collapsed summary line */}
          {!expanded && (
            <div className="mt-1.5 flex items-center gap-3 text-xs text-slate-500">
              {testCase.steps.length > 0 && (
                <span>
                  {Math.min(reachedStep, testCase.steps.length)}/{testCase.steps.length} steps
                </span>
              )}
              {isLive && progress && (
                <>
                  {progress.deviations > 0 && (
                    <span className="text-amber-300/70">{progress.deviations} adapted</span>
                  )}
                </>
              )}
              {hasInterrupt && (
                <span className="font-medium text-amber-300">Input needed</span>
              )}
            </div>
          )}
        </div>

        <div className="flex shrink-0 flex-col items-end gap-2">
          <span
            className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.2em] ${
              STATUS_STYLES[status] ?? STATUS_STYLES.default
            }`}
          >
            {statusLabel}
          </span>
          <span className={`text-[10px] text-slate-600 transition-transform duration-200 ${expanded ? "rotate-180" : ""}`}>
            ▾
          </span>
        </div>
      </button>

      {/* ── expanded body ── */}
      {expanded && (
        <div className="flex flex-col gap-0 px-5 pb-5">
          {/* Meta info */}
          {(testCase.preconditions?.length > 0 || testCase.expected_result || testCase.discovery_rationale) && (
            <div className="mb-4 space-y-1">
              {testCase.preconditions?.length > 0 && (
                <p className="text-xs italic text-slate-400">Setup: {testCase.preconditions.join("; ")}</p>
              )}
              {testCase.expected_result && (
                <p className="text-xs text-slate-400">
                  <span className="font-semibold text-slate-300">Expected: </span>
                  {testCase.expected_result}
                </p>
              )}
              {testCase.discovery_rationale && (
                <p className="text-xs text-fuchsia-300/80">
                  <span className="font-semibold">Why this scenario: </span>
                  {testCase.discovery_rationale}
                </p>
              )}
            </div>
          )}

          {/* Live progress bar — turn budget */}
          {isLive && progress?.budget && progress.budget > 0 && (
            <div className="mb-4">
              <div className="mb-1 flex items-center justify-between text-[10px] text-slate-500">
                <span>Turn progress</span>
                <span>{progress.turn}/{progress.budget}</span>
              </div>
              <div className="h-1 w-full overflow-hidden rounded-full bg-slate-800">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-violet-500 transition-all duration-500"
                  style={{ width: `${Math.min(100, (progress.turn / progress.budget) * 100)}%` }}
                />
              </div>
            </div>
          )}

          {/* Live adaptation badges */}
          {isLive && progress && (progress.deviations > 0 || progress.asks > 0) && (
            <div className="mb-3 flex flex-wrap gap-2">
              {progress.deviations > 0 && (
                <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2.5 py-0.5 text-[10px] font-medium text-amber-300">
                  {progress.deviations} plan adaptation{progress.deviations !== 1 ? "s" : ""}
                </span>
              )}
              {progress.asks > 0 && (
                <span className="rounded-full border border-violet-500/30 bg-violet-500/10 px-2.5 py-0.5 text-[10px] font-medium text-violet-300">
                  {progress.asks} input request{progress.asks !== 1 ? "s" : ""} so far
                </span>
              )}
            </div>
          )}

          {/* Step list */}
          <div className="space-y-2">
            {testCase.steps.map((step, index) => {
              const stepNum   = index + 1;
              const isCurrent = !result && stepNum === reachedStep;
              const isDone    = stepNum < reachedStep || (!!result && stepNum <= reachedStep);

              return (
                <div
                  key={`${testCase.test_id}-step-${index}`}
                  className={`flex items-start gap-3 rounded-2xl border px-3 py-2 transition-all duration-300 ${
                    isCurrent
                      ? "border-cyan-400/25 bg-cyan-500/5"
                      : isDone
                        ? "border-emerald-500/10 bg-emerald-500/3"
                        : "border-white/5 bg-slate-950/35"
                  }`}
                >
                  <div
                    className={`mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold transition-all duration-300 ${
                      isDone
                        ? "bg-emerald-500/20 text-emerald-300"
                        : isCurrent
                          ? "animate-pulse bg-cyan-500/20 text-cyan-300"
                          : "bg-slate-800 text-slate-400"
                    }`}
                  >
                    {isDone ? "✓" : stepNum}
                  </div>
                  <div className="flex-1">
                    <p className={`text-sm leading-6 transition-colors duration-300 ${
                      isDone ? "text-slate-400" : isCurrent ? "text-white" : "text-slate-300"
                    }`}>
                      {step}
                    </p>
                    {/* Show current_action inline on the active step */}
                    {isCurrent && isLive && progress?.current_action && (
                      <p className="mt-0.5 flex items-center gap-1.5 text-[11px] text-cyan-400/70">
                        <span className="h-1 w-1 animate-ping rounded-full bg-cyan-400" />
                        {progress.current_action}
                      </p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* In-card interrupt */}
          {hasInterrupt && interrupt && onAnswer && (
            <InCardInterruptPanel
              interrupt={interrupt}
              onAnswer={onAnswer}
              answered={answered ?? false}
            />
          )}

          {/* Mutation timeline — live events while the test is running */}
          {!result && progress?.mutation_events && progress.mutation_events.length > 0 && (
            <MutationTimeline events={progress.mutation_events} isLive={isLive} />
          )}

          {/* Plan evolution — shown live when replanning has happened */}
          {!result && (
            <PlanEvolutionTimeline testCase={testCase} progress={progress} />
          )}

          {/* Final verdict */}
          {result && (
            <div className="mt-4 rounded-2xl border border-white/10 bg-slate-950/40 p-4">
              <div className="flex items-center justify-between">
                <div className="text-[10px] uppercase tracking-[0.25em] text-slate-400">Verdict</div>
                <span
                  className={`rounded-full border px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.15em] ${
                    STATUS_STYLES[result.status] ?? STATUS_STYLES.default
                  }`}
                >
                  {result.status}
                </span>
              </div>
              <p className="mt-2 text-sm leading-6 text-slate-200">{result.reason}</p>

              {/* Final mutation timeline (resolved events from live execution) */}
              {progress?.mutation_events && progress.mutation_events.length > 0 && (
                <MutationTimeline events={progress.mutation_events} isLive={false} />
              )}

              {/* Final plan evolution timeline (shown in verdict when replanning happened) */}
              <PlanEvolutionTimeline testCase={testCase} progress={progress} />

              {(result.deviations?.length || result.amended_steps?.length) ? (
                <div className="mt-3 rounded-xl border border-amber-500/20 bg-amber-500/5 p-3">
                  <div className="text-[10px] uppercase tracking-[0.25em] text-amber-300/80">
                    Plan amendments during execution
                  </div>
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

              {apiBase && result.screenshot_path && (
                <a
                  href={`${apiBase}/${result.screenshot_path}`}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-3 block overflow-hidden rounded-xl border border-white/10 bg-slate-950/60"
                >
                  <img
                    src={`${apiBase}/${result.screenshot_path}`}
                    alt={`Screenshot for ${testCase.test_id}`}
                    className="max-h-48 w-full object-contain"
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
        </div>
      )}
    </article>
  );
}
