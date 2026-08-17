export type TestCase = { id: string; title: string; intent: string };

export type TestResult = {
  test_case_id: string;
  title: string;
  status: "passed" | "failed" | "error";
  duration_ms: number;
};

const STATUS_STYLES: Record<string, string> = {
  running: "border-running/50 bg-running/10 text-running",
  passed: "border-pass/50 bg-pass/10 text-pass",
  failed: "border-fail/50 bg-fail/10 text-fail",
  error: "border-fail/50 bg-fail/10 text-fail",
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
    <div className={`rounded-lg border p-4 ${STATUS_STYLES[status]}`}>
      <div className="flex items-center justify-between">
        <h3 className="font-medium text-neutral-100">{testCase.title}</h3>
        <span className="text-xs uppercase tracking-wide">{status}</span>
      </div>
      <p className="mt-1 text-sm text-neutral-400">{testCase.intent}</p>
      {result && <p className="mt-2 text-xs text-neutral-500">{result.duration_ms}ms</p>}
    </div>
  );
}
