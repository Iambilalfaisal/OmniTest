"use client";

import { useEffect, useState } from "react";
import type { TestCase } from "@/components/WorkerCard";

export type PlanReviewInterrupt = {
  id: string;
  type: "plan_review";
  payload: { type: "plan_review"; test_cases: TestCase[] };
};

export type RiskyActionInterrupt = {
  id: string;
  type: "risky_action";
  payload: { type: "risky_action"; test_id: string; tool: string; args: Record<string, unknown> };
};

export type PendingInterrupt =
  | PlanReviewInterrupt
  | RiskyActionInterrupt
  | { id: string; type: string; payload: Record<string, unknown> };

export type ResumeDecision = { approved: boolean; test_cases?: TestCase[]; reason?: string };

function isPlanReview(i: PendingInterrupt): i is PlanReviewInterrupt {
  return i.type === "plan_review";
}

function isRiskyAction(i: PendingInterrupt): i is RiskyActionInterrupt {
  return i.type === "risky_action";
}

export default function HumanReviewPanel({
  interrupts,
  submitting,
  error,
  onSubmit,
}: {
  interrupts: PendingInterrupt[];
  submitting: boolean;
  error: string | null;
  onSubmit: (resume: Record<string, ResumeDecision>, optimisticPlan?: TestCase[]) => void;
}) {
  const planReview = interrupts.find(isPlanReview);
  const riskyActions = interrupts.filter(isRiskyAction);

  if (planReview) {
    return <PlanReviewCard interrupt={planReview} submitting={submitting} error={error} onSubmit={onSubmit} />;
  }
  if (riskyActions.length > 0) {
    return <RiskyActionCard interrupts={riskyActions} submitting={submitting} error={error} onSubmit={onSubmit} />;
  }
  return null;
}

function PlanReviewCard({
  interrupt,
  submitting,
  error,
  onSubmit,
}: {
  interrupt: PlanReviewInterrupt;
  submitting: boolean;
  error: string | null;
  onSubmit: (resume: Record<string, ResumeDecision>, optimisticPlan?: TestCase[]) => void;
}) {
  const testCases = interrupt.payload.test_cases;
  const [selected, setSelected] = useState<Record<string, boolean>>({});

  useEffect(() => {
    setSelected(Object.fromEntries(testCases.map((tc) => [tc.test_id, true])));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [interrupt.id]);

  const chosen = testCases.filter((tc) => selected[tc.test_id]);

  function approve() {
    onSubmit({ [interrupt.id]: { approved: true, test_cases: chosen } }, chosen);
  }

  function reject() {
    onSubmit({ [interrupt.id]: { approved: false } }, []);
  }

  return (
    <div className="glass-panel rounded-3xl border border-amber-400/30 p-6">
      <p className="text-xs uppercase tracking-[0.3em] text-amber-300/80">Human review needed</p>
      <h2 className="mt-2 text-xl font-semibold text-white">Approve the test plan</h2>
      <p className="mt-1 text-sm text-slate-400">
        No browser action runs until you approve. Uncheck anything you don&apos;t want executed.
      </p>

      <div className="mt-5 space-y-3">
        {testCases.map((tc) => (
          <label
            key={tc.test_id}
            className={`flex cursor-pointer items-start gap-3 rounded-2xl border px-4 py-3 transition ${
              selected[tc.test_id]
                ? "border-cyan-400/40 bg-cyan-500/5"
                : "border-white/10 bg-slate-950/30 opacity-60"
            }`}
          >
            <input
              type="checkbox"
              checked={!!selected[tc.test_id]}
              onChange={() => setSelected((prev) => ({ ...prev, [tc.test_id]: !prev[tc.test_id] }))}
              className="mt-1 h-4 w-4 accent-cyan-500"
            />
            <div>
              <div className="text-sm font-medium text-white">{tc.goal}</div>
              <ol className="mt-2 space-y-1 text-xs text-slate-400">
                {tc.steps.map((step, i) => (
                  <li key={i}>
                    {i + 1}. {step}
                  </li>
                ))}
              </ol>
            </div>
          </label>
        ))}
      </div>

      {error && (
        <div className="mt-4 rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          {error}
        </div>
      )}

      <div className="mt-6 flex flex-wrap gap-3">
        <button
          type="button"
          disabled={submitting || chosen.length === 0}
          onClick={approve}
          className="rounded-2xl bg-gradient-to-r from-emerald-500 to-cyan-500 px-5 py-2.5 text-sm font-semibold text-white transition disabled:cursor-not-allowed disabled:opacity-50"
        >
          {submitting ? "Submitting…" : `Approve ${chosen.length}/${testCases.length} test${chosen.length === 1 ? "" : "s"}`}
        </button>
        <button
          type="button"
          disabled={submitting}
          onClick={reject}
          className="rounded-2xl border border-rose-500/30 bg-rose-500/10 px-5 py-2.5 text-sm font-semibold text-rose-200 transition disabled:cursor-not-allowed disabled:opacity-50"
        >
          Reject plan
        </button>
      </div>
    </div>
  );
}

function RiskyActionCard({
  interrupts,
  submitting,
  error,
  onSubmit,
}: {
  interrupts: RiskyActionInterrupt[];
  submitting: boolean;
  error: string | null;
  onSubmit: (resume: Record<string, ResumeDecision>) => void;
}) {
  const [decisions, setDecisions] = useState<Record<string, boolean>>({});
  const idsKey = interrupts.map((i) => i.id).join(",");

  useEffect(() => {
    const ids = new Set(interrupts.map((i) => i.id));
    setDecisions((prev) => {
      const next: Record<string, boolean> = {};
      for (const id of Object.keys(prev)) {
        if (ids.has(id)) next[id] = prev[id];
      }
      return next;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idsKey]);

  const allDecided = interrupts.every((i) => decisions[i.id] !== undefined);

  function submit() {
    const resume: Record<string, ResumeDecision> = {};
    for (const i of interrupts) {
      resume[i.id] = decisions[i.id]
        ? { approved: true }
        : { approved: false, reason: "Blocked by reviewer" };
    }
    onSubmit(resume);
  }

  return (
    <div className="glass-panel rounded-3xl border border-amber-400/30 p-6">
      <p className="text-xs uppercase tracking-[0.3em] text-amber-300/80">Human review needed</p>
      <h2 className="mt-2 text-xl font-semibold text-white">
        {interrupts.length > 1 ? "Risky actions need approval" : "Risky action needs approval"}
      </h2>
      <p className="mt-1 text-sm text-slate-400">
        The agent wants to perform an action that looks irreversible. Review before it proceeds.
      </p>

      <div className="mt-5 space-y-3">
        {interrupts.map((i) => (
          <div key={i.id} className="rounded-2xl border border-white/10 bg-slate-950/30 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <div className="text-[10px] uppercase tracking-[0.25em] text-slate-400">{i.payload.test_id}</div>
                <div className="mt-1 text-sm font-semibold text-white">{i.payload.tool}</div>
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setDecisions((prev) => ({ ...prev, [i.id]: true }))}
                  className={`rounded-full border px-3 py-1.5 text-xs font-medium transition ${
                    decisions[i.id] === true
                      ? "border-emerald-400/60 bg-emerald-500/20 text-emerald-200"
                      : "border-white/10 text-slate-300 hover:border-emerald-400/40"
                  }`}
                >
                  Allow
                </button>
                <button
                  type="button"
                  onClick={() => setDecisions((prev) => ({ ...prev, [i.id]: false }))}
                  className={`rounded-full border px-3 py-1.5 text-xs font-medium transition ${
                    decisions[i.id] === false
                      ? "border-rose-400/60 bg-rose-500/20 text-rose-200"
                      : "border-white/10 text-slate-300 hover:border-rose-400/40"
                  }`}
                >
                  Block
                </button>
              </div>
            </div>
            <pre className="mt-3 overflow-x-auto rounded-xl bg-slate-950/60 p-3 text-xs text-slate-300">
              {JSON.stringify(i.payload.args, null, 2)}
            </pre>
          </div>
        ))}
      </div>

      {error && (
        <div className="mt-4 rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          {error}
        </div>
      )}

      <div className="mt-6">
        <button
          type="button"
          disabled={submitting || !allDecided}
          onClick={submit}
          className="rounded-2xl bg-gradient-to-r from-emerald-500 to-cyan-500 px-5 py-2.5 text-sm font-semibold text-white transition disabled:cursor-not-allowed disabled:opacity-50"
        >
          {submitting ? "Submitting…" : "Submit decisions"}
        </button>
      </div>
    </div>
  );
}
