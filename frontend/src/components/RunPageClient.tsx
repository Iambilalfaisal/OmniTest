"use client";

import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import WorkerCard, { TestCase, TestResult } from "@/components/WorkerCard";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export default function RunPageClient() {
  const runId = useSearchParams().get("id");
  const [plan, setPlan] = useState<TestCase[]>([]);
  const [results, setResults] = useState<Record<string, TestResult>>({});
  const [done, setDone] = useState(false);
  const [status, setStatus] = useState("Planning tests…");

  useEffect(() => {
    if (!runId) return;

    const source = new EventSource(`${API_BASE}/runs/${runId}/events`);

    source.addEventListener("progress", (event) => {
      const payload = JSON.parse((event as MessageEvent).data);
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
      const interruptCount = payload.interrupts?.length ?? 0;
      setStatus(interruptCount > 0 ? "Awaiting approval…" : "Paused");
    });

    source.addEventListener("done", (event) => {
      const payload = JSON.parse((event as MessageEvent).data);
      const incoming = payload.test_results ?? [];
      setResults((prev) => {
        const next = { ...prev };
        for (const result of incoming as TestResult[]) {
          next[result.test_id] = result;
        }
        return next;
      });
      setDone(true);
      setStatus("Completed");
      source.close();
    });

    source.onerror = () => {
      setStatus("Connection interrupted");
      source.close();
    };

    return () => source.close();
  }, [runId]);

  const stats = useMemo(() => {
    const total = plan.length;
    const passed = Object.values(results).filter((result) => result.status === "Pass").length;
    const failed = Object.values(results).filter((result) => result.status === "Fail").length;
    return { total, passed, failed };
  }, [plan, results]);

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

      <section className="grid gap-5 lg:grid-cols-2 xl:grid-cols-3">
        {plan.length === 0 ? (
          <div className="glass-panel col-span-full rounded-3xl p-8 text-center text-slate-300">
            Planning your QA workflow…
          </div>
        ) : (
          plan.map((testCase) => <WorkerCard key={testCase.test_id} testCase={testCase} result={results[testCase.test_id]} />)
        )}
      </section>
    </div>
  );
}
