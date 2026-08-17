"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export default function HomePage() {
  const router = useRouter();
  const [url, setUrl] = useState("");
  const [objective, setObjective] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function startRun(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    try {
      const res = await fetch(`${API_BASE}/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, objective }),
      });
      const { run_id } = await res.json();
      router.push(`/run?id=${run_id}`);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={startRun} className="mx-auto flex max-w-lg flex-col gap-4">
      <h1 className="text-2xl font-semibold">Start a run</h1>
      <label className="flex flex-col gap-1 text-sm text-neutral-400">
        Target URL
        <input
          required
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://example.com"
          className="rounded border border-neutral-700 bg-neutral-900 px-3 py-2 text-neutral-100"
        />
      </label>
      <label className="flex flex-col gap-1 text-sm text-neutral-400">
        Objective
        <textarea
          required
          value={objective}
          onChange={(e) => setObjective(e.target.value)}
          placeholder="Verify the checkout flow handles an expired coupon code"
          className="rounded border border-neutral-700 bg-neutral-900 px-3 py-2 text-neutral-100"
          rows={3}
        />
      </label>
      <button
        type="submit"
        disabled={submitting}
        className="rounded bg-running px-4 py-2 font-medium text-white disabled:opacity-50"
      >
        {submitting ? "Starting…" : "Start run"}
      </button>
    </form>
  );
}
