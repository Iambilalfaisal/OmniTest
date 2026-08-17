"use client";

import { useState } from "react";
import TraceViewer from "@/components/TraceViewer";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

type TestResult = {
  test_case_id: string;
  title: string;
  status: "passed" | "failed" | "error";
  duration_ms: number;
  trace_path?: string | null;
};

type RunReport = {
  run_id: string;
  total: number;
  passed: number;
  failed: number;
  errored: number;
  duration_ms: number;
  results: TestResult[];
};

export default function ReportsPage() {
  const [runId, setRunId] = useState("");
  const [report, setReport] = useState<RunReport | null>(null);
  const [activeTrace, setActiveTrace] = useState<string | null>(null);

  async function loadReport(e: React.FormEvent) {
    e.preventDefault();
    const res = await fetch(`${API_BASE}/runs/${runId}/report`);
    if (res.ok) setReport(await res.json());
  }

  return (
    <div className="flex flex-col gap-6">
      <form onSubmit={loadReport} className="flex gap-2">
        <input
          value={runId}
          onChange={(e) => setRunId(e.target.value)}
          placeholder="Run ID"
          className="flex-1 rounded border border-neutral-700 bg-neutral-900 px-3 py-2"
        />
        <button className="rounded bg-running px-4 py-2 font-medium text-white">Load</button>
      </form>

      {report && (
        <>
          <div className="flex gap-4 text-sm">
            <span className="text-pass">{report.passed} passed</span>
            <span className="text-fail">{report.failed} failed</span>
            <span className="text-neutral-400">{report.duration_ms}ms total</span>
          </div>
          <ul className="flex flex-col divide-y divide-neutral-800">
            {report.results.map((result) => (
              <li key={result.test_case_id} className="flex items-center justify-between py-3">
                <span>{result.title}</span>
                <div className="flex items-center gap-3">
                  <span className="text-xs uppercase text-neutral-400">{result.status}</span>
                  {result.trace_path && (
                    <button
                      onClick={() => setActiveTrace(result.trace_path!)}
                      className="text-xs text-running underline"
                    >
                      view trace
                    </button>
                  )}
                </div>
              </li>
            ))}
          </ul>
        </>
      )}

      {activeTrace && (
        <TraceViewer traceUrl={`${API_BASE}/${activeTrace}`} onClose={() => setActiveTrace(null)} />
      )}
    </div>
  );
}
