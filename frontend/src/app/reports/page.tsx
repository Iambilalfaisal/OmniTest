"use client";

import { Suspense } from "react";
import { FormEvent, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import TraceViewer from "@/components/TraceViewer";
import { CATEGORY_LABELS, CATEGORY_STYLES, ClipSequencePlayer, TestCategory } from "@/components/WorkerCard";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

type TestResult = {
  test_id: string;
  // "Pass" | "Fail" | "Blocked" (backend/nodes/worker/nodes.py's Verdict).
  status: string;
  screenshot_path: string;
  trace_path?: string | null;
  // One short clip per mutating action, not one video for the whole test case — see
  // backend/nodes/worker/evidence.py's capture_mutation_clip.
  video_clips?: string[];
  reason: string;
  // What the adaptive worker had to work around, and the steps as actually executed
  // when they diverged from the written plan (backend/core/models.py's TestResult).
  deviations?: string[];
  amended_steps?: string[];
};

type RunReport = {
  summary: {
    total: number;
    passed: number;
    failed: number;
    blocked?: number;
    by_category?: Record<TestCategory, { total: number; passed: number; failed: number; blocked?: number }>;
    // Feature -> Flow -> Scenario rollup (backend/nodes/reporter.py) — by_flow omitted
    // here since this page doesn't otherwise join test_cases (feature_id/flow_id live
    // there, not on TestResult) into its flat per-result list below; the feature-level
    // pass/fail counts alone don't need that join.
    by_feature?: Record<string, { name: string; description: string; total: number; passed: number; failed: number; blocked?: number }>;
  };
  test_results: TestResult[];
  plan_approved: boolean;
};

function ReportsPageContent() {
  const searchParams = useSearchParams();
  const [runId, setRunId] = useState(searchParams.get("id") ?? "");
  const [report, setReport] = useState<RunReport | null>(null);
  const [activeTrace, setActiveTrace] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadReportById(id: string) {
    if (!id.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/runs/${id.trim()}/report`);
      if (!res.ok) throw new Error("Unable to load report for this run.");
      setReport(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  // Auto-load when navigated here with ?id=
  useEffect(() => {
    const id = searchParams.get("id");
    if (id) loadReportById(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadReport(e: FormEvent) {
    e.preventDefault();
    await loadReportById(runId);
  }

  return (
    <div className="relative mx-auto flex w-full max-w-6xl flex-col gap-6 py-4">
      <div className="page-grid pointer-events-none absolute inset-x-0 top-0 h-96 opacity-60" />
      <div className="glass-panel animate-rise relative rounded-[2rem] p-6">
        <form onSubmit={loadReport} className="flex flex-col gap-4 md:flex-row">
          <input
            value={runId}
            onChange={(e) => setRunId(e.target.value)}
            placeholder="Paste a run ID"
            className="flex-1 rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-white outline-none transition focus:border-cyan-400/70 focus:ring-2 focus:ring-cyan-500/30"
          />
          <button
            type="submit"
            disabled={loading || !runId}
            className="rounded-2xl bg-gradient-to-r from-violet-500 to-cyan-500 px-5 py-3 font-medium text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Loading…" : "Load report"}
          </button>
        </form>

        {error && <div className="mt-4 rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">{error}</div>}
      </div>

      {report && (
        <>
          <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-2xl border border-white/10 bg-slate-900/80 p-5">
              <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Total</div>
              <div className="mt-3 text-3xl font-semibold text-white">{report.summary.total}</div>
            </div>
            <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/10 p-5">
              <div className="text-xs uppercase tracking-[0.28em] text-emerald-300/80">Passed</div>
              <div className="mt-3 text-3xl font-semibold text-emerald-300">{report.summary.passed}</div>
            </div>
            <div className="rounded-2xl border border-rose-500/20 bg-rose-500/10 p-5">
              <div className="text-xs uppercase tracking-[0.28em] text-rose-300/80">Failed</div>
              <div className="mt-3 text-3xl font-semibold text-rose-300">{report.summary.failed}</div>
            </div>
            <div className="rounded-2xl border border-amber-500/20 bg-amber-500/10 p-5">
              <div className="text-xs uppercase tracking-[0.28em] text-amber-300/80">Blocked</div>
              <div className="mt-3 text-3xl font-semibold text-amber-300">{report.summary.blocked ?? 0}</div>
            </div>
          </section>

          {report.summary.by_category && Object.keys(report.summary.by_category).length > 0 && (
            <section className="glass-panel rounded-3xl p-5 md:p-6">
              <h2 className="mb-4 text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">By category</h2>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                {Object.entries(report.summary.by_category).map(([category, counts]) => (
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

          {report.summary.by_feature && Object.keys(report.summary.by_feature).length > 0 && (
            <section className="glass-panel rounded-3xl p-5 md:p-6">
              <h2 className="mb-4 text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">By feature</h2>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                {Object.entries(report.summary.by_feature).map(([featureId, counts]) => (
                  <div key={featureId} className="rounded-2xl border border-white/10 bg-slate-950/40 p-4">
                    <div className="text-sm font-semibold text-white">{counts.name}</div>
                    {counts.description && <p className="mt-1 text-xs text-slate-400">{counts.description}</p>}
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

          <section className="glass-panel rounded-3xl p-5 md:p-6">
            <div className="mb-5 flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">Test results</h2>
              <span className="text-xs uppercase tracking-[0.25em] text-slate-400">
                {report.plan_approved ? "Approved plan" : "Skipped approval"}
              </span>
            </div>

            <div className="space-y-3">
              {report.test_results.map((result) => (
                <div key={result.test_id} className="rounded-2xl border border-white/10 bg-slate-950/40 p-4">
                  <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                    <div>
                      <div className="text-sm uppercase tracking-[0.2em] text-slate-400">{result.test_id}</div>
                      <div className="mt-1 text-lg font-medium text-white">{result.status}</div>
                    </div>

                    <div className="flex items-center gap-2">
                      {result.trace_path && (
                        <button
                          type="button"
                          onClick={() => setActiveTrace(`${API_BASE}/${result.trace_path}`)}
                          className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3 py-1.5 text-xs font-medium text-cyan-200"
                        >
                          View trace
                        </button>
                      )}
                    </div>
                  </div>

                  <p className="mt-3 text-sm leading-6 text-slate-300">{result.reason}</p>

                  {(result.deviations?.length || result.amended_steps?.length) ? (
                    <div className="mt-3 rounded-xl border border-amber-500/20 bg-amber-500/5 p-3">
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

                  {result.screenshot_path && (
                    <a
                      href={`${API_BASE}/${result.screenshot_path}`}
                      target="_blank"
                      rel="noreferrer"
                      className="mt-3 block overflow-hidden rounded-xl border border-white/10 bg-slate-950/60"
                    >
                      <img
                        src={`${API_BASE}/${result.screenshot_path}`}
                        alt={`Final screenshot for ${result.test_id}`}
                        className="max-h-56 w-full object-contain"
                        loading="lazy"
                      />
                    </a>
                  )}

                  {result.video_clips && result.video_clips.length > 0 && (
                    <ClipSequencePlayer clips={result.video_clips} apiBase={API_BASE} className="mt-3" />
                  )}
                </div>
              ))}
            </div>
          </section>
        </>
      )}

      {activeTrace && <TraceViewer traceUrl={activeTrace} onClose={() => setActiveTrace(null)} />}
    </div>
  );
}

export default function ReportsPage() {
  return (
    <Suspense fallback={<div className="glass-panel mx-auto max-w-xl rounded-3xl p-8 text-center text-slate-300">Loading reports…</div>}>
      <ReportsPageContent />
    </Suspense>
  );
}
