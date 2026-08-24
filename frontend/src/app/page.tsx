"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useRef, useState } from "react";
import OrbitMark from "@/components/OrbitMark";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

type Mode = "explore" | "quick" | "my_plan";

const MODES: {
  id: Mode;
  label: string;
  icon: string;
  description: string;
  inputLabel: string;
  placeholder: string;
  mono?: boolean;
}[] = [
  {
    id: "explore",
    label: "Explore & plan",
    icon: "◎",
    description: "Chat with AI to shape the plan before anything runs",
    inputLabel: "Starting idea",
    placeholder:
      "e.g. Verify the checkout flow, focus on sign-up and onboarding — or leave blank and OmniTest will suggest what to test",
  },
  {
    id: "quick",
    label: "Quick start",
    icon: "⚡",
    description: "AI proposes a complete plan immediately — you approve in one click",
    inputLabel: "Focus area",
    placeholder:
      "e.g. Checkout, authentication, search — or leave blank for full site coverage",
  },
  {
    id: "my_plan",
    label: "My own plan",
    icon: "✦",
    description: "Paste your own test cases and run directly — no AI planning step",
    inputLabel: "Test cases (JSON array)",
    placeholder: `[\n  {\n    "test_id": "tc-001",\n    "goal": "User can log in with valid credentials",\n    "category": "authentication",\n    "priority": "high",\n    "requires_auth": false,\n    "preconditions": ["App is loaded"],\n    "expected_result": "Dashboard is shown",\n    "steps": ["Navigate to /login", "Enter credentials", "Click Sign In"]\n  }\n]`,
    mono: true,
  },
];

export default function HomePage() {
  const router = useRouter();
  const [url, setUrl] = useState("");
  const [mode, setMode] = useState<Mode>("explore");
  const [body, setBody] = useState("");
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const selectedMode = MODES.find((m) => m.id === mode)!;

  // Close dropdown on outside click
  useEffect(() => {
    function handler(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false);
      }
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  function selectMode(m: Mode) {
    setMode(m);
    setBody("");
    setJsonError(null);
    setError(null);
    setDropdownOpen(false);
  }

  function validateJson(): boolean {
    if (!body.trim()) return true;
    try {
      const parsed = JSON.parse(body);
      if (!Array.isArray(parsed)) {
        setJsonError("Must be a JSON array of test case objects.");
        return false;
      }
      setJsonError(null);
      return true;
    } catch {
      setJsonError("Invalid JSON — check for missing quotes, commas, or brackets.");
      return false;
    }
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!url.trim()) return;
    if (mode === "my_plan" && !validateJson()) return;
    setError(null);
    setSubmitting(true);

    try {
      if (mode === "explore") {
        const res = await fetch(`${API_BASE}/discover`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ target_url: url, starting_idea: body }),
        });
        if (!res.ok) throw new Error((await res.text()) || "Unable to start discovery.");
        const { discovery_id } = await res.json();
        router.push(`/discover?id=${discovery_id}`);
        return;
      }

      if (mode === "quick") {
        const quickIdea =
          "[QUICK_START] Explore the site and propose a comprehensive test plan covering " +
          "all major user flows, happy paths, edge cases, and error states in a single response for one-click approval." +
          (body ? `\n\nAdditional focus: ${body}` : "");
        const res = await fetch(`${API_BASE}/discover`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ target_url: url, starting_idea: quickIdea }),
        });
        if (!res.ok) throw new Error((await res.text()) || "Unable to start session.");
        const { discovery_id } = await res.json();
        router.push(`/discover?id=${discovery_id}`);
        return;
      }

      // my_plan
      const testCases = body.trim() ? JSON.parse(body) : [];
      if (testCases.length === 0) {
        setJsonError("Paste at least one test case.");
        setSubmitting(false);
        return;
      }
      const res = await fetch(`${API_BASE}/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_url: url,
          instruction: `Run ${testCases.length} user-provided test case${testCases.length !== 1 ? "s" : ""} on ${url}`,
          test_cases: testCases,
        }),
      });
      if (!res.ok) throw new Error((await res.text()) || "Unable to start run.");
      const { run_id } = await res.json();
      router.push(`/run?id=${run_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  }

  const submitLabel = {
    explore: submitting ? "Starting…" : "Start session",
    quick: submitting ? "Starting…" : "Generate plan",
    my_plan: submitting ? "Running…" : "Run tests",
  }[mode];

  return (
    <div className="relative mx-auto flex w-full max-w-3xl flex-col items-center gap-10 py-12">
      <div className="page-grid pointer-events-none absolute inset-x-0 top-0 h-[22rem] opacity-50" />

      {/* Compact branding */}
      <div className="relative flex flex-col items-center gap-3 text-center">
        <div className="signal-line mb-1 h-px w-24" />
        <div className="flex items-center gap-3">
          <OrbitMark size="lg" />
          <div className="text-left">
            <p className="eyebrow">Autonomous quality intelligence</p>
            <h1 className="text-4xl font-semibold tracking-[-0.03em] text-white md:text-5xl">
              OmniTest
            </h1>
          </div>
        </div>
        <p className="text-sm text-slate-500">
          Explore, plan, and run end-to-end tests — evidence at every step
        </p>
      </div>

      {/* Composer card */}
      <form
        onSubmit={handleSubmit}
        className="glass-panel w-full animate-rise rounded-[1.75rem] p-1"
      >
        {/* URL row */}
        <div className="flex items-center gap-2 border-b border-white/[0.07] px-5 py-3">
          <span className="shrink-0 text-xs font-medium uppercase tracking-[0.18em] text-slate-500">
            URL
          </span>
          <input
            required
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://example.com"
            className="min-w-0 flex-1 bg-transparent py-1 text-sm text-white outline-none placeholder:text-slate-600"
          />
        </div>

        {/* Body textarea */}
        <div className="px-5 py-4">
          <textarea
            value={body}
            onChange={(e) => {
              setBody(e.target.value);
              if (mode === "my_plan" && jsonError) validateJson();
            }}
            placeholder={selectedMode.placeholder}
            rows={mode === "my_plan" ? 10 : 4}
            spellCheck={mode !== "my_plan"}
            className={[
              "w-full resize-none bg-transparent text-sm text-white outline-none placeholder:text-slate-600",
              selectedMode.mono ? "font-mono text-xs leading-5" : "leading-6",
            ].join(" ")}
          />
          {jsonError && (
            <p className="mt-1 text-xs text-rose-400">{jsonError}</p>
          )}
        </div>

        {/* Bottom toolbar */}
        <div className="flex items-center justify-between gap-3 border-t border-white/[0.07] px-4 py-3">
          {/* Mode picker */}
          <div className="relative" ref={dropdownRef}>
            <button
              type="button"
              onClick={() => setDropdownOpen((o) => !o)}
              className="flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-slate-300 transition hover:border-white/20 hover:bg-white/[0.08] hover:text-white"
            >
              <span className="text-sm">{selectedMode.icon}</span>
              <span>{selectedMode.label}</span>
              <svg
                className={`h-3 w-3 text-slate-500 transition-transform ${dropdownOpen ? "rotate-180" : ""}`}
                fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            {dropdownOpen && (
              <div className="glass-panel absolute bottom-full left-0 mb-2 w-72 overflow-hidden rounded-2xl p-1 shadow-xl">
                {MODES.map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => selectMode(m.id)}
                    className={[
                      "flex w-full items-start gap-3 rounded-xl px-3 py-2.5 text-left transition",
                      mode === m.id
                        ? "bg-cyan-500/10 text-white"
                        : "text-slate-300 hover:bg-white/[0.05] hover:text-white",
                    ].join(" ")}
                  >
                    <span className={`mt-0.5 text-base shrink-0 ${mode === m.id ? "text-cyan-300" : "text-slate-400"}`}>
                      {m.icon}
                    </span>
                    <div className="min-w-0">
                      <div className="text-xs font-semibold">{m.label}</div>
                      <div className="text-[11px] leading-4 text-slate-500">{m.description}</div>
                    </div>
                    {mode === m.id && (
                      <svg className="ml-auto mt-0.5 h-3.5 w-3.5 shrink-0 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                      </svg>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Right: error + submit */}
          <div className="flex items-center gap-3">
            {error && (
              <span className="max-w-xs truncate text-xs text-rose-400">{error}</span>
            )}
            <button
              type="submit"
              disabled={submitting || (mode === "my_plan" && !!jsonError)}
              className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 via-blue-500 to-violet-500 px-4 py-2 text-xs font-semibold text-white shadow-md shadow-cyan-500/20 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {submitLabel}
              {!submitting && (
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </form>

      {/* Feature row */}
      <div className="grid w-full gap-3 sm:grid-cols-3">
        {[
          { value: "24/7", label: "Live runs", tone: "text-cyan-300" },
          { value: "MCP", label: "Browser coverage", tone: "text-violet-300" },
          { value: "Trace + video", label: "Evidence", tone: "text-lime-300" },
        ].map((f) => (
          <div
            key={f.label}
            className="glass-panel rounded-2xl px-5 py-4 transition hover:-translate-y-0.5 hover:border-white/20"
          >
            <div className={`text-lg font-semibold ${f.tone}`}>{f.value}</div>
            <div className="mt-0.5 text-[10px] uppercase tracking-[0.2em] text-slate-500">{f.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
