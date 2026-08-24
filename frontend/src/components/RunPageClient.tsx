"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import WorkerCard, {
  Feature,
  FEATURE_PHASE_LABELS,
  FeatureProgress,
  TestCase,
  TestCategory,
  TestResult,
  WorkerProgress,
  CATEGORY_LABELS,
  CATEGORY_STYLES,
  CardInterrupt,
  CardDecision,
} from "@/components/WorkerCard";
import HumanReviewPanel, { PendingInterrupt, ResumeDecision } from "@/components/HumanReviewPanel";
import TraceViewer from "@/components/TraceViewer";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
const RECONNECT_DELAY_MS = 2000;

// ─── types ───────────────────────────────────────────────────────────────────

type RunPhase = "planning" | "recon" | "reviewing" | "executing" | "awaiting_input" | "done" | "error";

type FeatureCounts = { total: number; passed: number; failed: number; blocked?: number };

type RunSummary = {
  total: number;
  passed: number;
  failed: number;
  blocked?: number;
  by_category?: Record<TestCategory, { total: number; passed: number; failed: number; blocked?: number }>;
  by_feature?: Record<string, FeatureCounts & { name: string; description: string }>;
};

function formatElapsed(totalSeconds: number): string {
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

// ─── WorkflowPhaseBar ─────────────────────────────────────────────────────────

const PHASE_STEPS: { key: RunPhase; label: string }[] = [
  { key: "planning",  label: "Plan"    },
  { key: "recon",     label: "Analyze" },
  { key: "reviewing", label: "Review"  },
  { key: "executing", label: "Execute" },
  { key: "done",      label: "Report"  },
];
const PHASE_ORDER: RunPhase[] = ["planning", "recon", "reviewing", "executing", "done"];

function WorkflowPhaseBar({ phase }: { phase: RunPhase }) {
  // awaiting_input maps visually to the "executing" step in the stepper but uses amber styling
  const effectivePhase = phase === "awaiting_input" ? "executing" : phase;
  const currentIndex   = effectivePhase === "error" ? -1 : PHASE_ORDER.indexOf(effectivePhase);

  return (
    <div className="flex flex-wrap items-center gap-0.5">
      {PHASE_STEPS.map((step, i) => {
        const stepIndex         = PHASE_ORDER.indexOf(step.key);
        const isActive          = step.key === effectivePhase;
        const isAwaitingAtStep  = phase === "awaiting_input" && step.key === "executing";
        const isDone            = currentIndex > stepIndex && effectivePhase !== "error";

        return (
          <div key={step.key} className="flex items-center">
            {i > 0 && (
              <div
                className={`mx-1 h-px w-3 transition-colors duration-500 ${
                  isDone ? "bg-emerald-500/40" : isActive ? (isAwaitingAtStep ? "bg-amber-400/40" : "bg-cyan-400/40") : "bg-white/8"
                }`}
              />
            )}
            <span
              className={`
                inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5
                text-[10px] font-semibold uppercase tracking-[0.15em] transition-all duration-500
                ${isAwaitingAtStep
                  ? "border-amber-400/50 bg-amber-500/10 text-amber-200 animate-phase-glow"
                  : isActive
                    ? "border-cyan-400/50 bg-cyan-500/10 text-cyan-200 animate-phase-glow"
                    : isDone
                      ? "border-emerald-500/20 bg-emerald-500/5 text-emerald-400/60"
                      : "border-white/8 text-slate-600"}
              `}
            >
              {isDone
                ? <span className="text-emerald-400/70">✓</span>
                : isAwaitingAtStep
                  ? <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-400" />
                  : isActive
                    ? <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-400" />
                    : <span className="h-1.5 w-1.5 rounded-full bg-slate-700" />}
              {step.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ─── SkeletonCard ─────────────────────────────────────────────────────────────

function SkeletonCard({ index }: { index: number }) {
  return (
    <div
      className="glass-panel animate-card-in rounded-3xl p-5"
      style={{ animationDelay: `${index * 80}ms` }}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 space-y-2.5">
          <div className="skeleton h-3.5 w-20 rounded-full" />
          <div className="skeleton h-5 w-3/4 rounded-xl" />
          <div className="skeleton h-3.5 w-1/2 rounded-xl" />
        </div>
        <div className="skeleton h-5 w-14 rounded-full" />
      </div>
      <div className="mt-5 space-y-2.5">
        {[0, 1, 2].map((j) => (
          <div key={j} className="skeleton h-9 w-full rounded-2xl" style={{ opacity: 1 - j * 0.15 }} />
        ))}
      </div>
    </div>
  );
}

// ─── CrashPanel ──────────────────────────────────────────────────────────────

function CrashPanel({
  runId,
  retryError,
  onRetry,
}: {
  runId: string;
  retryError: string | null;
  onRetry: () => void;
}) {
  return (
    <div className="glass-panel animate-card-in rounded-3xl border border-rose-500/30 p-6">
      <div className="flex items-start gap-4">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-rose-500/30 bg-rose-500/10">
          <span className="text-rose-300">✕</span>
        </div>
        <div className="flex-1">
          <p className="text-xs uppercase tracking-[0.25em] text-rose-300/80">Run crashed</p>
          <h2 className="mt-1 text-lg font-semibold text-white">
            The run encountered an unrecoverable error
          </h2>
          <p className="mt-1 text-sm text-slate-400">
            The last good checkpoint is preserved. Retrying will re-execute from where it stopped.
          </p>

          {retryError && (
            <div className="mt-3 rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
              {retryError}
            </div>
          )}

          <div className="mt-5 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={onRetry}
              className="rounded-2xl bg-gradient-to-r from-rose-500 to-orange-500 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-rose-500/20 transition hover:brightness-110"
            >
              Retry run
            </button>
            <Link
              href="/"
              className="rounded-2xl border border-white/10 px-5 py-2.5 text-sm font-semibold text-slate-300 transition hover:border-white/20 hover:text-white"
            >
              Start new session
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── RunPageClient ────────────────────────────────────────────────────────────

export default function RunPageClient() {
  const runId = useSearchParams().get("id");

  const [plan, setPlan]                         = useState<TestCase[]>([]);
  const [results, setResults]                   = useState<Record<string, TestResult>>({});
  const [workerProgress, setWorkerProgress]     = useState<Record<string, WorkerProgress>>({});
  const [features, setFeatures]                 = useState<Feature[]>([]);
  const [featureProgress, setFeatureProgress]   = useState<Record<string, FeatureProgress>>({});
  const [summary, setSummary]                   = useState<RunSummary | null>(null);
  const [done, setDone]                         = useState(false);
  const [crashed, setCrashed]                   = useState(false);
  const [pendingInterrupts, setPendingInterrupts] = useState<PendingInterrupt[]>([]);
  const [resuming, setResuming]                 = useState(false);
  const [resumeError, setResumeError]           = useState<string | null>(null);
  const [retryError, setRetryError]             = useState<string | null>(null);
  // Collects per-card interrupt decisions (keyed by interrupt ID) before auto-submit.
  const [cardDecisions, setCardDecisions]       = useState<Record<string, CardDecision>>({});
  const [activeTrace, setActiveTrace]           = useState<string | null>(null);
  const [connected, setConnected]               = useState(true);
  const [lastEventAt, setLastEventAt]           = useState(() => Date.now());
  const [now, setNow]                           = useState(() => Date.now());
  // Incrementing this re-triggers the SSE useEffect, reconnecting after a retry.
  const [connectKey, setConnectKey]             = useState(0);

  const doneRef     = useRef(false);
  const crashedRef  = useRef(false); // mirrors `crashed` for use inside SSE closures
  const startedAtRef = useRef(Date.now());

  useEffect(() => {
    if (done || crashed) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [done, crashed]);

  useEffect(() => {
    if (!runId) return;
    let cancelled  = false;
    let source: EventSource | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    function connect() {
      if (cancelled) return;
      source = new EventSource(`${API_BASE}/runs/${runId}/events`);

      source.onopen = () => {
        setConnected(true);
        setLastEventAt(Date.now());
      };

      source.addEventListener("progress", (event) => {
        setLastEventAt(Date.now());
        const payload = JSON.parse((event as MessageEvent).data);
        // Do NOT clear pendingInterrupts here. The server guarantees that 'progress'
        // events are NOT emitted while any interrupt is pending (api.py event_stream),
        // so this was a no-op under normal conditions — but a race-condition footgun
        // on SSE reconnects: a stale 'progress' event could wipe a freshly-set
        // clarification interrupt, making the input form disappear before the user
        // saw it. Cleared only on 'done' (terminal) and when a 'paused' event
        // replaces it with a new set.
        if (payload.test_cases?.length)   setPlan(payload.test_cases as TestCase[]);
        if (payload.test_results?.length) {
          const incoming = payload.test_results as TestResult[];
          setResults((prev) => {
            const next = { ...prev };
            for (const r of incoming) next[r.test_id] = r;
            return next;
          });
        }
        if (payload.worker_progress)  setWorkerProgress(payload.worker_progress as Record<string, WorkerProgress>);
        if (payload.features?.length) setFeatures(payload.features as Feature[]);
        if (payload.feature_progress) setFeatureProgress(payload.feature_progress as Record<string, FeatureProgress>);
      });

      source.addEventListener("paused", (event) => {
        setLastEventAt(Date.now());
        const payload   = JSON.parse((event as MessageEvent).data);
        const interrupts = (payload.interrupts ?? []) as PendingInterrupt[];
        setPendingInterrupts(interrupts);
        setResumeError(null);
      });

      source.addEventListener("done", (event) => {
        setLastEventAt(Date.now());
        const payload = JSON.parse((event as MessageEvent).data);
        setPendingInterrupts([]);
        const incoming = payload.test_results ?? [];
        setResults((prev) => {
          const next = { ...prev };
          for (const r of incoming as TestResult[]) next[r.test_id] = r;
          return next;
        });
        setSummary((payload.summary as RunSummary) ?? null);
        if (payload.features?.length) setFeatures(payload.features as Feature[]);
        doneRef.current   = true;
        setDone(true);
        source?.close();
      });

      // Backend emits event: error when the run crashes (distinct from the SSE
      // transport error below). Without this listener the stream closes cleanly
      // and onerror fires → the client reconnects → gets another error event →
      // infinite loop. Setting doneRef breaks the reconnect cycle.
      source.addEventListener("error", () => {
        source?.close();
        if (cancelled) return;
        doneRef.current    = true;
        crashedRef.current = true;
        setCrashed(true);
        setConnected(false);
      });

      source.onerror = () => {
        source?.close();
        if (cancelled || doneRef.current) return;
        setConnected(false);
        reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
      };
    }

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      source?.close();
    };
  // connectKey intentionally included — incrementing it re-runs this effect
  // to re-open the SSE stream after a retry.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, connectKey]);

  async function submitResume(resume: Record<string, ResumeDecision>, optimisticPlan?: TestCase[]) {
    if (!runId) return;
    setResuming(true);
    setResumeError(null);
    try {
      const res = await fetch(`${API_BASE}/runs/${runId}/resume`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ resume }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Failed to submit your decision.");
      }
      if (optimisticPlan) setPlan(optimisticPlan);
      setPendingInterrupts([]);
    } catch (err) {
      setResumeError(err instanceof Error ? err.message : "Failed to submit your decision.");
    } finally {
      setResuming(false);
    }
  }

  // ── per-card interrupt routing ────────────────────────────────────────────

  // Map test_id → the pending interrupt for that specific card.
  const cardInterruptsMap = useMemo(() => {
    const map: Record<string, CardInterrupt> = {};
    for (const i of pendingInterrupts) {
      if (i.type === "risky_action" || i.type === "clarification") {
        const testId = (i.payload as { test_id: string }).test_id;
        if (testId) map[testId] = i as unknown as CardInterrupt;
      }
    }
    return map;
  }, [pendingInterrupts]);

  // Only plan_review interrupts go to HumanReviewPanel.
  const planOnlyInterrupts = useMemo(
    () => pendingInterrupts.filter((i) => i.type === "plan_review"),
    [pendingInterrupts],
  );

  // Called by each WorkerCard when the user answers its interrupt.
  //
  // The backend resume endpoint requires ALL currently-pending interrupt IDs in a
  // single command (LangGraph's Command(resume=...) dispatches each answer to the
  // matching interrupted node). So we accumulate answers locally until every
  // pending card interrupt has one, then fire a single submitResume.
  //
  // For the common case of a single interrupted test, this fires immediately.
  // For multiple simultaneous interrupts the amber banner above shows answered
  // progress so the user knows to answer the remaining cards.
  function handleCardAnswer(interruptId: string, decision: CardDecision) {
    setCardDecisions((prev) => {
      const next    = { ...prev, [interruptId]: decision };
      const cardIds = Object.values(cardInterruptsMap).map((ci) => ci.id);
      const allAnswered = cardIds.length > 0 && cardIds.every((id) => next[id] !== undefined);
      if (allAnswered) {
        const resume = Object.fromEntries(
          cardIds.map((id) => [id, next[id] as ResumeDecision]),
        );
        // Submit async; clear local decisions immediately so cards show "waiting" state
        submitResume(resume);
        return {};
      }
      // Not all answered yet — return the partial map. The amber banner will show
      // the answered count so the user knows to scroll to remaining cards.
      return next;
    });
  }

  async function retryRun() {
    if (!runId) return;
    setRetryError(null);
    try {
      const res = await fetch(`${API_BASE}/runs/${runId}/retry`, { method: "POST" });
      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error((errData as { detail?: string })?.detail || "Failed to retry run.");
      }
      // Reset crash state and re-open the SSE stream.
      setCrashed(false);
      crashedRef.current = false;
      doneRef.current    = false;
      setConnected(true);
      setLastEventAt(Date.now());
      setConnectKey((k) => k + 1);
    } catch (err) {
      setRetryError(err instanceof Error ? err.message : "Failed to retry.");
    }
  }

  // ── derived state ──────────────────────────────────────────────────────────

  const stats = useMemo(() => {
    const total     = plan.length;
    const completed = Object.keys(results).length;
    const passed    = Object.values(results).filter((r) => r.status === "Pass").length;
    const failed    = Object.values(results).filter((r) => r.status === "Fail").length;
    const blocked   = Object.values(results).filter((r) => r.status === "Blocked").length;
    return { total, completed, passed, failed, blocked };
  }, [plan, results]);

  const phase = useMemo((): RunPhase => {
    if (crashed) return "error";
    if (done)    return "done";
    if (pendingInterrupts.some((i) => i.type === "plan_review")) return "reviewing";
    // Worker-level interrupts (clarification / risky_action) mean at least one test
    // is paused waiting for the user's response. Surface this as a distinct phase so
    // the header and WorkflowPhaseBar make the pause unmistakable.
    if (pendingInterrupts.some((i) => i.type === "clarification" || i.type === "risky_action")) return "awaiting_input";
    if (plan.length > 0 && Object.values(featureProgress).some((fp) => fp.phase === "exploring")) return "recon";
    if (plan.length > 0) return "executing";
    return "planning";
  }, [crashed, done, pendingInterrupts, featureProgress, plan]);

  const workerInterruptCount = useMemo(
    () => pendingInterrupts.filter((i) => i.type === "clarification" || i.type === "risky_action").length,
    [pendingInterrupts],
  );

  const phaseStatusLabel = useMemo(() => {
    if (!connected && !crashed && !done) return "Reconnecting…";
    switch (phase) {
      case "planning":       return "Generating test plan…";
      case "recon":          return "Exploring application features…";
      case "reviewing":      return "Awaiting your review…";
      case "awaiting_input": return `Your input is needed · ${workerInterruptCount} test${workerInterruptCount === 1 ? "" : "s"} paused`;
      case "executing":      return stats.total ? `${stats.completed} / ${stats.total} tests complete` : "Executing browser checks…";
      case "done":           return "Run complete";
      case "error":          return "Run crashed";
    }
  }, [phase, connected, crashed, done, stats, workerInterruptCount]);


  const groupedPlan = useMemo(() => {
    if (features.length <= 1) return null;
    const knownIds = new Set(features.map((f) => f.feature_id));
    const groups   = features.map((feature) => ({
      feature,
      testCases: plan.filter((tc) => tc.feature_id === feature.feature_id),
    }));
    const ungrouped = plan.filter((tc) => {
      const fid = tc.feature_id;
      return !fid || !knownIds.has(fid);
    });
    return { groups, ungrouped };
  }, [features, plan]);

  // ── early return ──────────────────────────────────────────────────────────

  if (!runId) {
    return (
      <div className="glass-panel mx-auto max-w-xl rounded-3xl p-8 text-center">
        <p className="text-slate-300">No run selected — start one from the home page.</p>
      </div>
    );
  }

  // ── render ────────────────────────────────────────────────────────────────

  return (
    <div className="relative mx-auto flex w-full max-w-7xl flex-col gap-6 py-4">
      <div className="page-grid pointer-events-none absolute inset-x-0 top-0 h-96 opacity-60" />

      {/* ── header ── */}
      <header className="glass-panel animate-rise relative rounded-[2rem] p-6 md:p-8">
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div className="flex flex-col gap-3">
            <div>
              <p className="eyebrow">Run session / Live telemetry</p>
              <h1 className="mt-2 text-3xl font-semibold text-white">{runId.slice(0, 8)}</h1>
            </div>
            {/* Workflow phase stepper */}
            <WorkflowPhaseBar phase={phase} />
          </div>

          <div className="flex flex-col items-start gap-2 md:items-end">
            <div className="flex items-center gap-3">
              <span
                className={`rounded-full px-3 py-1 text-xs font-medium ${
                  done
                    ? "bg-emerald-500/15 text-emerald-300"
                    : crashed
                      ? "bg-rose-500/15 text-rose-300"
                      : phase === "awaiting_input"
                        ? "bg-amber-500/15 text-amber-300"
                        : "bg-cyan-500/15 text-cyan-300"
                }`}
              >
                {done ? "Complete" : crashed ? "Crashed" : phase === "awaiting_input" ? "Awaiting Input" : "Running"}
              </span>
              {!done && !crashed && (
                <span className="flex items-center gap-1.5 text-xs text-slate-400">
                  <span className="relative flex h-2 w-2">
                    <span
                      className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-75 ${
                        connected ? "bg-cyan-400" : "bg-rose-400"
                      }`}
                    />
                    <span
                      className={`relative inline-flex h-2 w-2 rounded-full ${
                        connected ? "bg-cyan-400" : "bg-rose-400"
                      }`}
                    />
                  </span>
                  {connected
                    ? `Live · ${Math.max(0, Math.floor((now - lastEventAt) / 1000))}s ago`
                    : "Reconnecting…"}
                  <span className="text-slate-600">·</span>
                  {formatElapsed(Math.floor((now - startedAtRef.current) / 1000))} elapsed
                </span>
              )}
            </div>
            <p className="text-sm text-slate-400">{phaseStatusLabel}</p>
          </div>
        </div>

        {/* Summary tiles — shown once done */}
        {done ? (
          <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[
              { label: "Tests",   value: stats.total,   color: "text-white",         border: "border-white/10",          bg: "bg-slate-950/40"   },
              { label: "Passed",  value: stats.passed,  color: "text-emerald-300",    border: "border-emerald-500/20",    bg: "bg-emerald-500/10" },
              { label: "Failed",  value: stats.failed,  color: "text-rose-300",       border: "border-rose-500/20",       bg: "bg-rose-500/10"    },
              { label: "Blocked", value: stats.blocked, color: "text-amber-300",      border: "border-amber-500/20",      bg: "bg-amber-500/10"   },
            ].map(({ label, value, color, border, bg }) => (
              <div key={label} className={`animate-card-in rounded-2xl border ${border} ${bg} p-4`}>
                <div className={`text-xs uppercase tracking-[0.28em] ${color === "text-white" ? "text-slate-400" : color.replace("300", "300/80")}`}>
                  {label}
                </div>
                <div className={`mt-3 text-2xl font-semibold ${color}`}>{value}</div>
              </div>
            ))}
          </div>
        ) : !crashed && stats.total > 0 ? (
          <div className="mt-6">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>
                {workerInterruptCount > 0
                  ? <span className="text-amber-300/80">
                      {workerInterruptCount === 1
                        ? "1 test paused — scroll down to answer"
                        : `${workerInterruptCount} tests paused — scroll down to answer each`}
                    </span>
                  : "Working through your tests…"}
              </span>
              <span>{stats.completed} / {stats.total} done</span>
            </div>
            <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
              <div
                className={`h-full rounded-full transition-all duration-700 ${
                  workerInterruptCount > 0
                    ? "bg-gradient-to-r from-amber-500 to-amber-400"
                    : "bg-gradient-to-r from-cyan-500 to-violet-500"
                }`}
                style={{ width: `${stats.total ? (stats.completed / stats.total) * 100 : 0}%` }}
              />
            </div>
          </div>
        ) : null}

        {/* Report link — available once the run is done */}
        {done && (
          <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-white/8 pt-4">
            <Link
              href={`/reports?id=${runId}`}
              className="inline-flex items-center gap-2 rounded-2xl border border-violet-400/30 bg-violet-500/10 px-4 py-2 text-xs font-semibold text-violet-200 transition hover:border-violet-400/50 hover:bg-violet-500/20"
            >
              View full report →
            </Link>
            <Link
              href="/history"
              className="inline-flex items-center gap-2 rounded-2xl border border-white/10 px-4 py-2 text-xs font-semibold text-slate-300 transition hover:border-white/20 hover:text-white"
            >
              Back to history
            </Link>
          </div>
        )}
      </header>

      {/* ── by-category summary ── */}
      {done && summary?.by_category && Object.keys(summary.by_category).length > 0 && (
        <section className="glass-panel animate-card-in rounded-3xl p-5 md:p-6">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">By category</h2>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {Object.entries(summary.by_category).map(([category, counts]) => (
              <div key={category} className="rounded-2xl border border-white/10 bg-slate-950/40 p-4">
                <span
                  className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.15em] ${
                    CATEGORY_STYLES[category as TestCategory] ?? ""
                  }`}
                >
                  {CATEGORY_LABELS[category as TestCategory] ?? category}
                </span>
                <div className="mt-2 text-sm text-slate-300">
                  {counts.passed}/{counts.total} passed
                  {counts.failed > 0 && <span className="text-rose-300"> · {counts.failed} failed</span>}
                  {!!counts.blocked && counts.blocked > 0 && (
                    <span className="text-amber-300"> · {counts.blocked} blocked</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── human review panel (plan_review only — per-card interrupts live inside each card) ── */}
      {planOnlyInterrupts.length > 0 && (
        <HumanReviewPanel
          interrupts={planOnlyInterrupts}
          submitting={resuming}
          error={resumeError}
          onSubmit={submitResume}
        />
      )}

      {/* ── worker input required banner ─────────────────────────────────────
           Shown whenever one or more test cases have paused and are waiting for
           the user to provide credentials, confirm a decision, or handle any
           other clarification. Appears ABOVE the card grid so it's impossible
           to miss — the per-card InCardInterruptPanel is the actual response
           surface, this banner is the prominent call-to-action. ── */}
      {workerInterruptCount > 0 && (
        <div className="animate-rise rounded-2xl border border-amber-500/40 bg-amber-500/10 px-5 py-4">
          <div className="flex flex-wrap items-start gap-3">
            <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-amber-400/40 bg-amber-500/20">
              <svg className="h-3.5 w-3.5 text-amber-300" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
              </svg>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-amber-200">
                {workerInterruptCount === 1
                  ? "Your input is needed — 1 test is paused"
                  : `Your input is needed — ${workerInterruptCount} tests are paused`}
              </p>
              <p className="mt-1 text-xs text-amber-300/70">
                {workerInterruptCount === 1
                  ? "An agent hit something it cannot safely handle on its own. Scroll down to the highlighted card, read the question, and type your response."
                  : `${workerInterruptCount} agents are each waiting for a response. Scroll down to find each amber card and answer all of them — execution resumes once every pending card has a response.`}
              </p>
              {/* answered progress when multiple interrupts are pending */}
              {workerInterruptCount > 1 && Object.keys(cardDecisions).length > 0 && (
                <p className="mt-2 text-xs font-medium text-amber-300">
                  {Object.keys(cardDecisions).length} of {workerInterruptCount} answered
                  {" — "}
                  {workerInterruptCount - Object.keys(cardDecisions).length} more to go before execution resumes
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── crash panel ── */}
      {crashed && (
        <CrashPanel runId={runId} retryError={retryError} onRetry={retryRun} />
      )}

      {/* ── test case grid ── */}
      {groupedPlan ? (
        <div className="flex flex-col gap-6">
          {groupedPlan.groups.map(({ feature, testCases }) => {
            const fProgress = featureProgress[feature.feature_id];
            const fSummary  = summary?.by_feature?.[feature.feature_id];
            return (
              <section key={feature.feature_id} className="flex flex-col gap-4">
                <div className="glass-panel rounded-2xl p-4 md:p-5">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <h2 className="text-lg font-semibold text-white">{feature.name}</h2>
                      {feature.description && (
                        <p className="mt-1 text-xs text-slate-400">{feature.description}</p>
                      )}
                    </div>
                    {fProgress?.phase === "exploring" ? (
                      <span className="flex items-center gap-1.5 rounded-full border border-fuchsia-500/30 bg-fuchsia-500/10 px-3 py-1 text-xs font-medium text-fuchsia-300">
                        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-fuchsia-400" />
                        {FEATURE_PHASE_LABELS.exploring}
                      </span>
                    ) : (
                      fSummary && (
                        <span className="text-sm text-slate-300">
                          {fSummary.passed}/{fSummary.total} passed
                          {fSummary.failed > 0 && <span className="text-rose-300"> · {fSummary.failed} failed</span>}
                          {!!fSummary.blocked && fSummary.blocked > 0 && (
                            <span className="text-amber-300"> · {fSummary.blocked} blocked</span>
                          )}
                        </span>
                      )
                    )}
                  </div>
                </div>
                {testCases.length > 0 && (
                  <div className="grid gap-5 lg:grid-cols-2 xl:grid-cols-3">
                    {testCases.map((testCase, i) => (
                      <div key={testCase.test_id} className="animate-card-in" style={{ animationDelay: `${i * 60}ms` }}>
                        <WorkerCard
                          testCase={testCase}
                          result={results[testCase.test_id]}
                          progress={workerProgress[testCase.test_id]}
                          interrupt={cardInterruptsMap[testCase.test_id]}
                          onAnswer={handleCardAnswer}
                          answered={
                            !!cardInterruptsMap[testCase.test_id] &&
                            cardDecisions[cardInterruptsMap[testCase.test_id].id] !== undefined
                          }
                          apiBase={API_BASE}
                          onViewTrace={setActiveTrace}
                        />
                      </div>
                    ))}
                  </div>
                )}
              </section>
            );
          })}
          {groupedPlan.ungrouped.length > 0 && (
            <section className="flex flex-col gap-4">
              <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">
                Other test cases
              </h2>
              <div className="grid gap-5 lg:grid-cols-2 xl:grid-cols-3">
                {groupedPlan.ungrouped.map((testCase, i) => (
                  <div key={testCase.test_id} className="animate-card-in" style={{ animationDelay: `${i * 60}ms` }}>
                    <WorkerCard
                      testCase={testCase}
                      result={results[testCase.test_id]}
                      progress={workerProgress[testCase.test_id]}
                      interrupt={cardInterruptsMap[testCase.test_id]}
                      onAnswer={handleCardAnswer}
                      answered={
                        !!cardInterruptsMap[testCase.test_id] &&
                        cardDecisions[cardInterruptsMap[testCase.test_id].id] !== undefined
                      }
                      apiBase={API_BASE}
                      onViewTrace={setActiveTrace}
                    />
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      ) : plan.length > 0 ? (
        <section className="grid gap-5 lg:grid-cols-2 xl:grid-cols-3">
          {plan.map((testCase, i) => (
            <div key={testCase.test_id} className="animate-card-in" style={{ animationDelay: `${i * 60}ms` }}>
              <WorkerCard
                testCase={testCase}
                result={results[testCase.test_id]}
                progress={workerProgress[testCase.test_id]}
                interrupt={cardInterruptsMap[testCase.test_id]}
                onAnswer={handleCardAnswer}
                answered={
                  !!cardInterruptsMap[testCase.test_id] &&
                  cardDecisions[cardInterruptsMap[testCase.test_id].id] !== undefined
                }
                apiBase={API_BASE}
                onViewTrace={setActiveTrace}
              />
            </div>
          ))}
        </section>
      ) : !crashed && pendingInterrupts.length === 0 ? (
        // Planning phase — show skeleton cards while waiting for the plan
        <section className="grid gap-5 lg:grid-cols-2 xl:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <SkeletonCard key={i} index={i} />
          ))}
        </section>
      ) : !crashed && workerInterruptCount > 0 ? (
        // Page was reloaded while a clarification interrupt was active — the plan
        // hasn't been received yet (progress events are suppressed while paused),
        // so no WorkerCards have rendered. The amber banner above already tells
        // the user what's happening; this fallback avoids a blank card area.
        <div className="glass-panel rounded-3xl p-8 text-center">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full border border-amber-400/30 bg-amber-500/10">
            <svg className="h-5 w-5 text-amber-300" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 9V5.25A2.25 2.25 0 0013.5 3h-6a2.25 2.25 0 00-2.25 2.25v13.5A2.25 2.25 0 007.5 21h6a2.25 2.25 0 002.25-2.25V15M12 9l-3 3m0 0l3 3m-3-3h12.75" />
            </svg>
          </div>
          <p className="text-sm font-semibold text-amber-200">Test cards loading…</p>
          <p className="mt-2 text-xs text-slate-400">
            The run is paused waiting for your input. Test cards will appear shortly — if they don{"'"}t, refresh the page.
          </p>
        </div>
      ) : null}

      {activeTrace && <TraceViewer traceUrl={activeTrace} onClose={() => setActiveTrace(null)} />}
    </div>
  );
}
