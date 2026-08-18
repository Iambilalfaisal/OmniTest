"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import WorkerCard, { TestCase, TestResult } from "@/components/WorkerCard";
import HumanReviewPanel, { PendingInterrupt, ResumeDecision } from "@/components/HumanReviewPanel";
import TraceViewer from "@/components/TraceViewer";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
const RECONNECT_DELAY_MS = 2000;

export default function RunPageClient() {
  const runId = useSearchParams().get("id");
  const [plan, setPlan] = useState<TestCase[]>([]);
  const [results, setResults] = useState<Record<string, TestResult>>({});
  const [done, setDone] = useState(false);
  const [status, setStatus] = useState("Planning tests…");
  const [pendingInterrupts, setPendingInterrupts] = useState<PendingInterrupt[]>([]);
  const [resuming, setResuming] = useState(false);
  const [resumeError, setResumeError] = useState<string | null>(null);
  const [activeTrace, setActiveTrace] = useState<string | null>(null);

  // Held in a ref (not state) so the effect below can read the current value in its
  // cleanup/reconnect closures without re-subscribing the whole EventSource on every
  // resume — resume is triggered by a plain POST, not by tearing down the stream.
  const doneRef = useRef(false);

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    let source: EventSource | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    function connect() {
      if (cancelled) return;
      source = new EventSource(`${API_BASE}/runs/${runId}/events`);

      source.addEventListener("progress", (event) => {
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
        const payload = JSON.parse((event as MessageEvent).data);
        const interrupts = (payload.interrupts ?? []) as PendingInterrupt[];
        setPendingInterrupts(interrupts);
        setResumeError(null);
        setStatus(interrupts.length > 0 ? "Awaiting your review…" : "Paused");
      });

      source.addEventListener("done", (event) => {
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
        doneRef.current = true;
        setDone(true);
        setStatus("Completed");
        source?.close();
      });

      source.onerror = () => {
        source?.close();
        if (cancelled || doneRef.current) return;
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
    const passed = Object.values(results).filter((result) => result.status === "Pass").length;
    const failed = Object.values(results).filter((result) => result.status === "Fail").length;
    return { total, passed, failed };
  }, [plan, results]);

  const riskyTestIds = useMemo(
    () =>
      new Set(
        pendingInterrupts
          .filter((i): i is Extract<PendingInterrupt, { type: "risky_action" }> => i.type === "risky_action")
          .map((i) => i.payload.test_id)
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
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 py-4">
      <header className="glass-panel rounded-3xl p-6 md:p-8">
        <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-cyan-300/80">Run session</p>
            <h1 className="mt-2 text-3xl font-semibold text-white">{runId.slice(0, 8)}</h1>
          </div>

          <div className="flex items-center gap-3">
            <span className={`rounded-full px-3 py-1 text-xs font-medium ${done ? "bg-emerald-500/15 text-emerald-300" : "bg-cyan-500/15 text-cyan-300"}`}>
              {done ? "Complete" : "Running"}
            </span>
            <span className="text-sm text-slate-300">{status}</span>
          </div>
        </div>

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
      </header>

      {pendingInterrupts.length > 0 && (
        <HumanReviewPanel
          interrupts={pendingInterrupts}
          submitting={resuming}
          error={resumeError}
          onSubmit={submitResume}
        />
      )}

      <section className="grid gap-5 lg:grid-cols-2 xl:grid-cols-3">
        {plan.length === 0 ? (
          pendingInterrupts.length === 0 && (
            <div className="glass-panel col-span-full rounded-3xl p-8 text-center text-slate-300">
              Planning your QA workflow…
            </div>
          )
        ) : (
          plan.map((testCase) => (
            <WorkerCard
              key={testCase.test_id}
              testCase={testCase}
              result={results[testCase.test_id]}
              awaitingApproval={riskyTestIds.has(testCase.test_id)}
              apiBase={API_BASE}
              onViewTrace={setActiveTrace}
            />
          ))
        )}
      </section>

      {activeTrace && <TraceViewer traceUrl={activeTrace} onClose={() => setActiveTrace(null)} />}
    </div>
  );
}
