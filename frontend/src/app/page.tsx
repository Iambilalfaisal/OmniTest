"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";
import OrbitMark from "@/components/OrbitMark";

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
    <div className="relative mx-auto flex w-full max-w-7xl flex-col gap-8 py-8">
      <div className="page-grid pointer-events-none absolute inset-x-0 top-0 h-[32rem] opacity-70" />
      <section className="relative grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="glass-panel animate-rise overflow-hidden rounded-[2rem] p-8 md:p-12">
          <div className="signal-line mb-8 h-px w-40" />
          <div className="mb-6 flex items-center gap-3">
            <OrbitMark size="lg" />
            <div>
              <p className="eyebrow">Autonomous quality intelligence</p>
              <h1 className="mt-2 text-4xl font-semibold tracking-[-0.03em] text-white md:text-6xl">OmniTest</h1>
            </div>
          </div>

          <p className="max-w-2xl text-base leading-7 text-slate-300 md:text-xl md:leading-8">
            Give a URL and OmniTest explores the site, proposes a test plan grouped by feature — happy
            paths, edge cases, and negative cases — and talks it through with you before anything runs.
          </p>

          <div className="mt-8 flex items-center gap-3 text-xs uppercase tracking-[0.2em] text-slate-500">
            <span className="h-2 w-2 rounded-full bg-lime-300 shadow-[0_0_14px_#c5f36a]" />
            Exploration engine ready
          </div>

          <div className="mt-8 grid gap-4 sm:grid-cols-3">
            {[
              { label: "Live runs", value: "24/7", tone: "text-cyan-300" },
              { label: "Browser coverage", value: "MCP", tone: "text-violet-300" },
              { label: "Evidence", value: "Trace + video", tone: "text-lime-300" },
            ].map((item) => (
              <div key={item.label} className="rounded-2xl border border-white/10 bg-black/10 p-4 transition hover:-translate-y-1 hover:border-white/20">
                <div className={`text-xl font-semibold ${item.tone}`}>{item.value}</div>
                <div className="mt-1 text-xs uppercase tracking-[0.2em] text-slate-400">{item.label}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="glass-panel animate-rise-delay rounded-[2rem] p-6 md:p-8">
          <div className="mb-5 flex items-center justify-between">
            <div>
              <p className="eyebrow">Initialize / 00</p>
              <h2 className="mt-2 text-2xl font-semibold text-white">Explore &amp; plan</h2>
            </div>
            <div className="rounded-full border border-lime-300/30 bg-lime-300/10 px-3 py-1 text-xs font-medium text-lime-200">
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
