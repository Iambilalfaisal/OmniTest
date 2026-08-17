"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import WorkerCard, { TestCase, TestResult } from "@/components/WorkerCard";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export default function RunPage() {
  const runId = useSearchParams().get("id");
  const [plan, setPlan] = useState<TestCase[]>([]);
  const [results, setResults] = useState<Record<string, TestResult>>({});
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!runId) return;

    const source = new EventSource(`${API_BASE}/runs/${runId}/events`);

    source.addEventListener("planner", (event) => {
      const payload = JSON.parse((event as MessageEvent).data);
      setPlan(payload.test_plan ?? []);
    });

    source.addEventListener("worker", (event) => {
      const payload = JSON.parse((event as MessageEvent).data);
      for (const result of (payload.results ?? []) as TestResult[]) {
        setResults((prev) => ({ ...prev, [result.test_case_id]: result }));
      }
    });

    source.addEventListener("reporter", () => {
      setDone(true);
      source.close();
    });

    source.onerror = () => source.close();
    return () => source.close();
  }, [runId]);

  if (!runId) {
    return <p className="text-neutral-400">No run selected — start one from the home page.</p>;
  }

  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-2xl font-semibold">
        Run {runId.slice(0, 8)} {done && <span className="text-pass">· complete</span>}
      </h1>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {plan.map((testCase) => (
          <WorkerCard key={testCase.id} testCase={testCase} result={results[testCase.id]} />
        ))}
      </div>
      {plan.length === 0 && <p className="text-neutral-400">Planning tests…</p>}
    </div>
  );
}
