"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import WorkerCard, { CATEGORY_LABELS, CATEGORY_STYLES, TestCase, TestCategory, TestResult } from "@/components/WorkerCard";
import HumanReviewPanel, { PendingInterrupt, ResumeDecision } from "@/components/HumanReviewPanel";
import TraceViewer from "@/components/TraceViewer";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
const RECONNECT_DELAY_MS = 2000;

type RunSummary = {
  total: number;
  passed: number;
  failed: number;
  by_category?: Record<TestCategory, { total: number; passed: number; failed: number }>;
};

function formatElapsed(totalSeconds: number): string {
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function RunPageClient() {
  const runId = useSearchParams().get("id");
  const [plan, setPlan] = useState<TestCase[]>([]);
  const [results, setResults] = useState<Record<string, TestResult>>({});
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [done, setDone] = useState(false);
  const [status, setStatus] = useState("Planning tests…");
  const [pendingInterrupts, setPendingInterrupts] = useState<PendingInterrupt[]>([]);
  const [resuming, setResuming] = useState(false);
  const [resumeError, setResumeError] = useState<string | null>(null);
  const [activeTrace, setActiveTrace] = useState<string | null>(null);

  // Proof-of-life for the "hidden until done" view below: the backend's SSE stream
  // sends a heartbeat "progress" event roughly every second regardless of whether
  // anything actually changed (see backend/api.py's run_events), so a live pulse tied
  // to `lastEventAt` — not just a local timer — genuinely reflects the stream working,
  // not a fake animation.
  const [connected, setConnected] = useState(true);
  const [lastEventAt, setLastEventAt] = useState(() => Date.now());
  const [now, setNow] = useState(() => Date.now());
  const startedAtRef = useRef(Date.now());

  // Held in a ref (not state) so the effect below can read the current value in its
  // cleanup/reconnect closures without re-subscribing the whole EventSource on every
  // resume — resume is triggered by a plain POST, not by tearing down the stream.
  const doneRef = useRef(false);

  useEffect(() => {
    if (done) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [done]);

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
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
        setPendingInterrupts([]);
        if (payload.test_cases?.length) {
          setPlan(payload.test_cases as TestCase[]);
          setStatus("Executing browser checks…");
        }
        if (payload.test_results?.length) {
          const incoming = payload.test_results as TestResult[];
          setResults((prev) => {
            const next = { ...prev };
            for (const result of incoming) {
              next[result.test_id] = result;
            }
            return next;
          });
        }
      });

      source.addEventListener("paused", (event) => {
        setLastEventAt(Date.now());
        const payload = JSON.parse((event as MessageEvent).data);
        const interrupts = (payload.interrupts ?? []) as PendingInterrupt[];
        setPendingInterrupts(interrupts);
        setResumeError(null);
        setStatus(interrupts.length > 0 ? "Awaiting your review…" : "Paused");
      });

      source.addEventListener("done", (event) => {
        setLastEventAt(Date.now());
        const payload = JSON.parse((event as MessageEvent).data);
        const incoming = payload.test_results ?? [];
        setPendingInterrupts([]);
        setResults((prev) => {
          const next = { ...prev };
          for (const result of incoming as TestResult[]) {
            next[result.test_id] = result;
          }
          return next;
        });
        setSummary((payload.summary as RunSummary) ?? null);
        doneRef.current = true;
        setDone(true);
        setStatus("Completed");
        source?.close();
      });

      source.onerror = () => {
        source?.close();
        if (cancelled || doneRef.current) return;
        setConnected(false);
        setStatus("Connection lost — reconnecting…");
        reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
      };
    }

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      source?.close();
    };
  }, [runId]);

  async function submitResume(resume: Record<string, ResumeDecision>, optimisticPlan?: TestCase[]) {
    if (!runId) return;
    setResuming(true);
    setResumeError(null);
    try {
      const res = await fetch(`${API_BASE}/runs/${runId}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ resume }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Failed to submit your decision.");
      }
      if (optimisticPlan) {
        setPlan(optimisticPlan);
      }
      setPendingInterrupts([]);
      setStatus(optimisticPlan?.length === 0 ? "Plan rejected" : "Resuming…");
    } catch (err) {
      setResumeError(err instanceof Error ? err.message : "Failed to submit your decision.");
    } finally {
      setResuming(false);
    }
  }

  const stats = useMemo(() => {
    const total = plan.length;
    const completed = Object.keys(results).length;
    const passed = Object.values(results).filter((result) => result.status === "Pass").length;
    const failed = Object.values(results).filter((result) => result.status === "Fail").length;
    return { total, completed, passed, failed };
  }, [plan, results]);

  const awaitingApprovalTestIds = useMemo(
    () =>
      new Set(
        pendingInterrupts
          .filter(
            (i): i is Extract<PendingInterrupt, { type: "risky_action" | "clarification" }> =>
              i.type === "risky_action" || i.type === "clarification"
          )
          .map((i) => (i.payload as { test_id: string }).test_id)
      ),
    [pendingInterrupts]
  );

  if (!runId) {
    return (
      <div className="glass-panel mx-auto max-w-xl rounded-3xl p-8 text-center">
        <p className="text-slate-300">No run selected — start one from the home page.</p>
      </div>
    );
  }

  return (
    <div className="relative mx-auto flex w-full max-w-7xl flex-col gap-6 py-4">
      <div className="page-grid pointer-events-none absolute inset-x-0 top-0 h-96 opacity-60" />
      <header className="glass-panel animate-rise relative rounded-[2rem] p-6 md:p-8">
        <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="eyebrow">Run session / Live telemetry</p>
            <h1 className="mt-2 text-3xl font-semibold text-white">{runId.slice(0, 8)}</h1>
          </div>

          <div className="flex items-center gap-3">
            <span className={`rounded-full px-3 py-1 text-xs font-medium ${done ? "bg-emerald-500/15 text-emerald-300" : "bg-cyan-500/15 text-cyan-300"}`}>
              {done ? "Complete" : "Running"}
            </span>
            {!done && (
              <span className="flex items-center gap-1.5 text-xs text-slate-400">
                <span className="relative flex h-2 w-2">
                  <span
                    className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-75 ${
                      connected ? "bg-cyan-400" : "bg-rose-400"
                    }`}
                  />
                  <span className={`relative inline-flex h-2 w-2 rounded-full ${connected ? "bg-cyan-400" : "bg-rose-400"}`} />
                </span>
                {connected
                  ? `Live · checked ${Math.max(0, Math.floor((now - lastEventAt) / 1000))}s ago`
                  : "Reconnecting…"}
                <span className="text-slate-600">·</span>
                {formatElapsed(Math.floor((now - startedAtRef.current) / 1000))} elapsed
              </span>
            )}
            <span className="text-sm text-slate-300">{status}</span>
          </div>
        </div>

        {done ? (
          <div className="mt-6 grid gap-4 sm:grid-cols-3">
            <div className="rounded-2xl border border-white/10 bg-slate-950/40 p-4">
              <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Tests</div>
              <div className="mt-3 text-2xl font-semibold text-white">{stats.total}</div>
            </div>
            <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/10 p-4">
              <div className="text-xs uppercase tracking-[0.28em] text-emerald-300/80">Passed</div>
              <div className="mt-3 text-2xl font-semibold text-emerald-300">{stats.passed}</div>
            </div>
            <div className="rounded-2xl border border-rose-500/20 bg-rose-500/10 p-4">
              <div className="text-xs uppercase tracking-[0.28em] text-rose-300/80">Failed</div>
              <div className="mt-3 text-2xl font-semibold text-rose-300">{stats.failed}</div>
            </div>
          </div>
        ) : (
          stats.total > 0 && (
            // Nothing about individual tests is shown while running — just overall
            // progress. Full pass/fail detail, screenshots, and video appear once
            // everything is done, below.
            <div className="mt-6">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>Working through your tests…</span>
                <span>
                  {stats.completed} / {stats.total} done
                </span>
              </div>
              <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-slate-800">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-violet-500 transition-all duration-500"
                  style={{ width: `${stats.total ? (stats.completed / stats.total) * 100 : 0}%` }}
                />
              </div>
            </div>
          )
        )}
      </header>

      {done && summary?.by_category && Object.keys(summary.by_category).length > 0 && (
        <section className="glass-panel rounded-3xl p-5 md:p-6">
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
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {pendingInterrupts.length > 0 && (
        <HumanReviewPanel
          interrupts={pendingInterrupts}
          submitting={resuming}
          error={resumeError}
          onSubmit={submitResume}
        />
      )}

      {done ? (
        <section className="grid gap-5 lg:grid-cols-2 xl:grid-cols-3">
          {plan.map((testCase) => (
            <WorkerCard
              key={testCase.test_id}
              testCase={testCase}
              result={results[testCase.test_id]}
              awaitingApproval={awaitingApprovalTestIds.has(testCase.test_id)}
              apiBase={API_BASE}
              onViewTrace={setActiveTrace}
            />
          ))}
        </section>
      ) : (
        pendingInterrupts.length === 0 && (
          <div className="glass-panel rounded-3xl p-8 text-center text-slate-300">
            <div className="flex items-center justify-center gap-2">
              <span className="h-2 w-2 animate-pulse rounded-full bg-cyan-400" />
              {plan.length === 0 ? "Planning your QA workflow…" : "Running your tests — results will appear here when they're done."}
            </div>
            <p className="mt-2 text-xs text-slate-500">
              Still working — {formatElapsed(Math.floor((now - startedAtRef.current) / 1000))} elapsed.
            </p>
          </div>
        )
      )}

      {activeTrace && <TraceViewer traceUrl={activeTrace} onClose={() => setActiveTrace(null)} />}
    </div>
  );
}
