export type TestCase = {
  test_id: string;
  goal: string;
  steps: string[];
};

export type TestResult = {
  test_id: string;
  status: string;
  screenshot_path: string;
  trace_path?: string | null;
  video_path?: string | null;
  reason: string;
};

const STATUS_STYLES: Record<string, string> = {
  running: "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
  Pass: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  Fail: "border-rose-500/30 bg-rose-500/10 text-rose-300",
  default: "border-slate-500/30 bg-slate-500/10 text-slate-300",
};

export default function WorkerCard({
  testCase,
  result,
}: {
  testCase: TestCase;
  result?: TestResult;
}) {
  const status = result?.status ?? "running";

  return (
    <article className="glass-panel rounded-3xl p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-[10px] uppercase tracking-[0.28em] text-slate-400">Test case</div>
          <h3 className="mt-2 text-xl font-semibold text-white">{testCase.goal}</h3>
        </div>
        <span className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.2em] ${STATUS_STYLES[status] ?? STATUS_STYLES.default}`}>
          {status}
        </span>
      </div>

      <div className="mt-5 space-y-3">
        {testCase.steps.map((step, index) => (
          <div key={`${testCase.test_id}-step-${index}`} className="flex items-start gap-3 rounded-2xl border border-white/5 bg-slate-950/35 px-3 py-2">
            <div className="mt-0.5 flex h-6 w-6 items-center justify-center rounded-full bg-slate-800 text-[10px] font-semibold text-slate-200">
              {index + 1}
            </div>
            <p className="text-sm leading-6 text-slate-300">{step}</p>
          </div>
        ))}
      </div>

      {result && (
        <div className="mt-5 rounded-2xl border border-white/10 bg-slate-950/40 p-3">
          <div className="text-[10px] uppercase tracking-[0.25em] text-slate-400">Verdict</div>
          <p className="mt-2 text-sm leading-6 text-slate-200">{result.reason}</p>
        </div>
      )}
    </article>
  );
}
