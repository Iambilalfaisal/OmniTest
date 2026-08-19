"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export default function HomePage() {
  const router = useRouter();
  const [url, setUrl] = useState("https://example.com");
  const [startingIdea, setStartingIdea] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function startDiscovery(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/discover`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_url: url,
          starting_idea: startingIdea,
        }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Unable to start a discovery session.");
      }

      const { discovery_id } = await res.json();
      router.push(`/discover?id=${discovery_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong while starting discovery.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 py-8">
      <section className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="glass-panel rounded-3xl p-8 md:p-10">
          <div className="mb-6 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-500 to-violet-500 text-lg font-bold text-white shadow-lg shadow-cyan-500/30">
              O
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.28em] text-cyan-300/80">AI QA platform</p>
              <h1 className="text-3xl font-semibold tracking-tight text-white md:text-4xl">OmniTest</h1>
            </div>
          </div>

          <p className="max-w-xl text-base leading-7 text-slate-300 md:text-lg">
            Give a URL and OmniTest explores the site, proposes a test plan grouped by feature — happy
            paths, edge cases, and negative cases — and talks it through with you before anything runs.
          </p>

          <div className="mt-8 grid gap-4 sm:grid-cols-3">
            {[
              { label: "Live runs", value: "24/7" },
              { label: "Browser coverage", value: "MCP" },
              { label: "Evidence", value: "Trace + video" },
            ].map((item) => (
              <div key={item.label} className="rounded-2xl border border-white/10 bg-slate-950/40 p-4">
                <div className="text-xl font-semibold text-white">{item.value}</div>
                <div className="mt-1 text-xs uppercase tracking-[0.2em] text-slate-400">{item.label}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="glass-panel rounded-3xl p-6 md:p-8">
          <div className="mb-5 flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.25em] text-slate-400">New session</p>
              <h2 className="mt-2 text-2xl font-semibold text-white">Explore &amp; plan</h2>
            </div>
            <div className="rounded-full border border-cyan-400/40 bg-cyan-500/10 px-3 py-1 text-xs font-medium text-cyan-300">
              Ready
            </div>
          </div>

          <form onSubmit={startDiscovery} className="space-y-5">
            <label className="block">
              <span className="mb-2 block text-sm text-slate-300">Target URL</span>
              <input
                required
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com"
                className="w-full rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-base text-white outline-none transition focus:border-cyan-400/70 focus:ring-2 focus:ring-cyan-500/30"
              />
            </label>

            <label className="block">
              <span className="mb-2 block text-sm text-slate-300">Starting idea (optional)</span>
              <textarea
                value={startingIdea}
                onChange={(e) => setStartingIdea(e.target.value)}
                placeholder="e.g. Verify the checkout flow — or leave blank and OmniTest will suggest what to test"
                rows={5}
                className="w-full rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-base text-white outline-none transition focus:border-cyan-400/70 focus:ring-2 focus:ring-cyan-500/30"
              />
            </label>

            {error && (
              <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={submitting}
              className="inline-flex w-full items-center justify-center rounded-2xl bg-gradient-to-r from-cyan-500 via-blue-500 to-violet-500 px-5 py-3 text-base font-semibold text-white shadow-lg shadow-cyan-500/20 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {submitting ? "Exploring the site…" : "Start exploring"}
            </button>
          </form>
        </div>
      </section>
    </div>
  );
}
