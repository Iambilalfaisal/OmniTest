"use client";

import { useEffect, useState } from "react";
import type { TestCase } from "@/components/WorkerCard";
import { CATEGORY_LABELS, CATEGORY_STYLES } from "@/components/WorkerCard";

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

export type ClarificationInterrupt = {
  id: string;
  type: "clarification";
  payload: {
    type: "clarification";
    test_id: string;
    question: string;
    context?: string | null;
    sensitive: boolean;
  };
};

export type PendingInterrupt =
  | PlanReviewInterrupt
  | RiskyActionInterrupt
  | ClarificationInterrupt
  | { id: string; type: string; payload: Record<string, unknown> };

export type ResumeDecision = { approved: boolean; test_cases?: TestCase[]; reason?: string } | { text: string };

function isPlanReview(i: PendingInterrupt): i is PlanReviewInterrupt {
  return i.type === "plan_review";
}

function isRiskyAction(i: PendingInterrupt): i is RiskyActionInterrupt {
  return i.type === "risky_action";
}

function isClarification(i: PendingInterrupt): i is ClarificationInterrupt {
  return i.type === "clarification";
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
  const clarifications = interrupts.filter(isClarification);

  // plan_review can never coexist with the other two types — it only ever occurs
  // before any worker exists — so it's safe to short-circuit alone here.
  if (planReview) {
    return <PlanReviewCard interrupt={planReview} submitting={submitting} error={error} onSubmit={onSubmit} />;
  }
  if (riskyActions.length > 0 || clarifications.length > 0) {
    return (
      <ActionableInterruptsPanel
        riskyActions={riskyActions}
        clarifications={clarifications}
        submitting={submitting}
        error={error}
        onSubmit={onSubmit}
      />
    );
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
              <div className="flex items-center gap-2">
                <div className="text-sm font-medium text-white">{tc.goal}</div>
                {tc.category && (
                  <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.15em] ${CATEGORY_STYLES[tc.category] ?? ""}`}>
                    {CATEGORY_LABELS[tc.category] ?? tc.category}
                  </span>
                )}
              </div>
              {tc.expected_result && (
                <p className="mt-1 text-xs text-slate-400">
                  <span className="font-semibold text-slate-300">Expected: </span>
                  {tc.expected_result}
                </p>
              )}
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

// Plain-English phrasing for a risky browser action — the raw tool name/JSON args are
// developer-facing and unreadable to a non-technical reviewer. Playwright MCP's action
// tools take a human-readable `element` description alongside the element ref, so we
// lead with that; the raw payload stays available behind a "technical details" toggle.
const RISKY_ACTION_VERBS: Record<string, string> = {
  browser_click: "click",
  browser_type: "type into",
  browser_fill_form: "fill out",
  browser_select_option: "choose an option in",
  browser_press_key: "press a key on",
  browser_drag: "drag",
  browser_hover: "hover over",
  browser_file_upload: "upload a file to",
  browser_navigate: "go to",
};

function humanizeToolName(tool: string): string {
  return tool.replace(/^browser_/, "").replace(/_/g, " ");
}

function describeRiskyAction(tool: string, args: Record<string, unknown>): string {
  // Arbitrary code execution can't be described as "click X" — say plainly that it's
  // running custom code and show a snippet, since this is the single riskiest action
  // type the agent can take.
  if (tool === "browser_run_code_unsafe" && typeof args.code === "string") {
    const snippet = args.code.trim().slice(0, 140);
    return `The agent wants to run custom code on the page: "${snippet}${args.code.trim().length > 140 ? "…" : ""}"`;
  }

  const verb = RISKY_ACTION_VERBS[tool];
  const target =
    typeof args.element === "string" && args.element
      ? `"${args.element}"`
      : typeof args.url === "string" && args.url
        ? args.url
        : typeof args.text === "string" && args.text
          ? `"${args.text}"`
          : typeof args.value === "string" && args.value
            ? `"${args.value}"`
            : "something on the page";

  // Even for a tool we haven't explicitly mapped, name the kind of action rather than
  // falling back to something fully generic like "do something with".
  return `The agent wants to ${verb ?? humanizeToolName(tool)} ${target}.`;
}

function ActionableInterruptsPanel({
  riskyActions,
  clarifications,
  submitting,
  error,
  onSubmit,
}: {
  riskyActions: RiskyActionInterrupt[];
  clarifications: ClarificationInterrupt[];
  submitting: boolean;
  error: string | null;
  onSubmit: (resume: Record<string, ResumeDecision>) => void;
}) {
  // Lifted here (not per-card) so one combined resume can be submitted once every
  // pending interrupt — of either type — has a decision. POST /runs/{run_id}/resume
  // requires the resume payload to cover EXACTLY the currently-pending interrupt ids,
  // so silently answering only one type would make the run permanently unresumable.
  const [riskyDecisions, setRiskyDecisions] = useState<Record<string, boolean>>({});
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const idsKey = [...riskyActions.map((i) => i.id), ...clarifications.map((i) => i.id)].join(",");

  useEffect(() => {
    const riskyIds = new Set(riskyActions.map((i) => i.id));
    const clarificationIds = new Set(clarifications.map((i) => i.id));
    setRiskyDecisions((prev) => {
      const next: Record<string, boolean> = {};
      for (const id of Object.keys(prev)) if (riskyIds.has(id)) next[id] = prev[id];
      return next;
    });
    setAnswers((prev) => {
      const next: Record<string, string> = {};
      for (const id of Object.keys(prev)) if (clarificationIds.has(id)) next[id] = prev[id];
      return next;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idsKey]);

  const allDecided =
    riskyActions.every((i) => riskyDecisions[i.id] !== undefined) &&
    clarifications.every((i) => (answers[i.id] ?? "").trim().length > 0);

  function submit() {
    const resume: Record<string, ResumeDecision> = {};
    for (const i of riskyActions) {
      resume[i.id] = riskyDecisions[i.id]
        ? { approved: true }
        : { approved: false, reason: "Blocked by reviewer" };
    }
    for (const i of clarifications) {
      resume[i.id] = { text: answers[i.id] ?? "" };
    }
    onSubmit(resume);
  }

  return (
    <div className="glass-panel rounded-3xl border border-amber-400/30 p-6">
      <p className="text-xs uppercase tracking-[0.3em] text-amber-300/80">We need your input</p>
      <h2 className="mt-2 text-xl font-semibold text-white">The test is paused</h2>
      <p className="mt-1 text-sm text-slate-400">
        Answer everything below, then press Submit. The test won&apos;t continue until you do.
      </p>

      <div className="mt-5 space-y-3">
        {riskyActions.map((i) => (
          <div key={i.id} className="rounded-2xl border border-white/10 bg-slate-950/30 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <div className="text-[10px] uppercase tracking-[0.25em] text-slate-400">Before it continues</div>
                <p className="mt-1 text-sm font-semibold text-white">{describeRiskyAction(i.payload.tool, i.payload.args)}</p>
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setRiskyDecisions((prev) => ({ ...prev, [i.id]: true }))}
                  className={`rounded-full border px-3 py-1.5 text-xs font-medium transition ${
                    riskyDecisions[i.id] === true
                      ? "border-emerald-400/60 bg-emerald-500/20 text-emerald-200"
                      : "border-white/10 text-slate-300 hover:border-emerald-400/40"
                  }`}
                >
                  Yes, go ahead
                </button>
                <button
                  type="button"
                  onClick={() => setRiskyDecisions((prev) => ({ ...prev, [i.id]: false }))}
                  className={`rounded-full border px-3 py-1.5 text-xs font-medium transition ${
                    riskyDecisions[i.id] === false
                      ? "border-rose-400/60 bg-rose-500/20 text-rose-200"
                      : "border-white/10 text-slate-300 hover:border-rose-400/40"
                  }`}
                >
                  No, skip it
                </button>
              </div>
            </div>
            <details className="mt-3">
              <summary className="cursor-pointer text-xs text-slate-500 hover:text-slate-300">Technical details</summary>
              <pre className="mt-2 overflow-x-auto rounded-xl bg-slate-950/60 p-3 text-xs text-slate-300">
                {i.payload.tool}
                {"\n"}
                {JSON.stringify(i.payload.args, null, 2)}
              </pre>
            </details>
          </div>
        ))}

        {clarifications.map((i) => (
          <div key={i.id} className="rounded-2xl border border-white/10 bg-slate-950/30 p-4">
            <div className="text-[10px] uppercase tracking-[0.25em] text-slate-400">The agent is asking</div>
            <p className="mt-1 text-sm font-semibold text-white">{i.payload.question}</p>
            {i.payload.context && <p className="mt-1 text-xs text-slate-400">{i.payload.context}</p>}
            <input
              type={i.payload.sensitive ? "password" : "text"}
              autoComplete="off"
              value={answers[i.id] ?? ""}
              onChange={(e) => setAnswers((prev) => ({ ...prev, [i.id]: e.target.value }))}
              placeholder={i.payload.sensitive ? "Your answer (hidden as you type)" : "Type your answer here"}
              className="mt-3 w-full rounded-xl border border-white/10 bg-slate-950/60 px-3 py-2 text-sm text-white outline-none transition focus:border-cyan-400/70 focus:ring-2 focus:ring-cyan-500/30"
            />
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
          {submitting ? "Submitting…" : "Submit"}
        </button>
      </div>
    </div>
  );
}
