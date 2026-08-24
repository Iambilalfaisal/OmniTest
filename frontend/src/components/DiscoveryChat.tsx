"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { CATEGORY_LABELS, CATEGORY_STYLES, TestCase } from "@/components/WorkerCard";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

type ChatMessage = { role: "user" | "assistant"; text: string };
type DiscoveryStatus = "in_progress" | "approved" | "cancelled";

export default function DiscoveryChat() {
  const router = useRouter();
  const discoveryId = useSearchParams().get("id");

  const [targetUrl, setTargetUrl] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<ChatMessage[]>([]);
  const [candidatePlan, setCandidatePlan] = useState<TestCase[]>([]);
  const [status, setStatus] = useState<DiscoveryStatus>("in_progress");
  const [turnCount, setTurnCount] = useState(0);
  const [maxTurns, setMaxTurns] = useState(20);
  const [sitePages, setSitePages] = useState(0);
  const [loading, setLoading] = useState(true);
  const [reply, setReply] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const transcriptEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!discoveryId) return;
    let cancelled = false;

    (async () => {
      try {
        const res = await fetch(`${API_BASE}/discover/${discoveryId}`);
        if (!res.ok) throw new Error("Unable to load this discovery session.");
        const data = await res.json();
        if (cancelled) return;
        setTargetUrl(data.target_url ?? null);
        setTranscript(data.transcript ?? []);
        setCandidatePlan(data.candidate_plan ?? []);
        setStatus(data.status ?? "in_progress");
        setTurnCount(data.turn_count ?? 0);
        setMaxTurns(data.max_turns ?? 20);
        setSitePages(data.site_pages_explored ?? 0);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Something went wrong.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [discoveryId]);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript, sending]);

  async function sendMessage(e: FormEvent) {
    e.preventDefault();
    if (!discoveryId || !reply.trim() || sending) return;

    const text = reply.trim();
    setTranscript((prev) => [...prev, { role: "user", text }]);
    setReply("");
    setSending(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/discover/${discoveryId}/message`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "reply", text }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Failed to send your reply.");
      }
      const data = await res.json();
      setTranscript((prev) => [...prev, { role: "assistant", text: data.assistant_message ?? "" }]);
      setCandidatePlan(data.candidate_plan ?? []);
      setTurnCount(data.turn_count ?? turnCount + 1);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send your reply.");
    } finally {
      setSending(false);
    }
  }

  async function approve() {
    if (!discoveryId || sending) return;
    setSending(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/discover/${discoveryId}/message`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "approve" }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Failed to approve the plan.");
      }
      const data = await res.json();
      setStatus("approved");
      router.push(`/run?id=${data.run_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to approve the plan.");
      setSending(false);
    }
  }

  async function cancel() {
    if (!discoveryId || sending) return;
    setSending(true);
    setError(null);
    try {
      await fetch(`${API_BASE}/discover/${discoveryId}/message`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "cancel" }),
      });
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to cancel this session.");
      setSending(false);
    }
  }

  if (!discoveryId) {
    return (
      <div className="glass-panel mx-auto max-w-xl rounded-3xl p-8 text-center">
        <p className="text-slate-300">No discovery session selected — start one from the home page.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="glass-panel mx-auto max-w-xl rounded-3xl p-8 text-center text-slate-300">
        Loading discovery session…
      </div>
    );
  }

  const atTurnLimit = turnCount >= maxTurns;
  const canReply = status === "in_progress" && !atTurnLimit;

  return (
    <div className="relative mx-auto flex w-full max-w-7xl flex-col gap-6 py-4">
      <div className="page-grid pointer-events-none absolute inset-x-0 top-0 h-96 opacity-60" />
      <header className="glass-panel animate-rise relative rounded-[2rem] p-6 md:p-8">
        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="eyebrow">Discovery session / Live</p>
            <h1 className="mt-2 text-2xl font-semibold text-white">{targetUrl}</h1>
          </div>
          <div className="flex items-center gap-4">
            {sitePages > 0 && (
              <span className="flex items-center gap-1.5 rounded-full border border-fuchsia-500/30 bg-fuchsia-500/10 px-3 py-1 text-xs font-medium text-fuchsia-300">
                <span className="h-1.5 w-1.5 rounded-full bg-fuchsia-400" />
                {sitePages} page{sitePages !== 1 ? "s" : ""} explored
              </span>
            )}
            <span className="text-xs uppercase tracking-[0.2em] text-slate-400">
              Turn {turnCount}/{maxTurns}
            </span>
          </div>
        </div>
      </header>

      <section className="grid gap-5 lg:grid-cols-[1fr_1fr]">
        <div className="glass-panel flex max-h-[32rem] flex-col rounded-3xl p-5">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">Conversation</h2>
          <div className="flex-1 space-y-3 overflow-y-auto pr-1">
            {transcript.map((m, i) => (
              <div
                key={i}
                className={`rounded-2xl px-4 py-3 text-sm leading-6 ${
                  m.role === "assistant"
                    ? "border border-cyan-400/20 bg-cyan-500/5 text-slate-200"
                    : "border border-white/10 bg-slate-950/40 text-white"
                }`}
              >
                <div className="mb-1 text-[10px] uppercase tracking-[0.2em] text-slate-500">
                  {m.role === "assistant" ? "OmniTest" : "You"}
                </div>
                {m.text}
              </div>
            ))}
            {sending && (
              <div className="flex items-center gap-2 rounded-2xl border border-cyan-400/20 bg-cyan-500/5 px-4 py-3 text-sm text-slate-400">
                <span className="flex gap-1">
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-cyan-400 [animation-delay:-0.3s]" />
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-cyan-400 [animation-delay:-0.15s]" />
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-cyan-400" />
                </span>
                OmniTest is thinking…
              </div>
            )}
            <div ref={transcriptEndRef} />
          </div>

          {error && (
            <div className="mt-3 rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
              {error}
            </div>
          )}

          {atTurnLimit && status === "in_progress" && (
            <p className="mt-3 text-xs text-amber-300">
              Turn limit reached — approve the plan or cancel to start over.
            </p>
          )}

          <form onSubmit={sendMessage} className="mt-3 flex gap-2">
            <input
              value={reply}
              onChange={(e) => setReply(e.target.value)}
              disabled={!canReply || sending}
              placeholder={canReply ? "Reply, ask a question, or suggest a test case…" : "Conversation closed"}
              className="flex-1 rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-2.5 text-sm text-white outline-none transition focus:border-cyan-400/70 focus:ring-2 focus:ring-cyan-500/30 disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={!canReply || sending || !reply.trim()}
              className="rounded-2xl bg-gradient-to-r from-cyan-500 to-violet-500 px-4 py-2.5 text-sm font-semibold text-white transition disabled:cursor-not-allowed disabled:opacity-50"
            >
              {sending ? "Sending…" : "Send"}
            </button>
          </form>
        </div>

        <div className="glass-panel flex max-h-[32rem] flex-col rounded-3xl p-5">
          <div className="mb-3 flex items-center gap-2">
            <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">
              Candidate plan ({candidatePlan.length})
            </h2>
            {sending && (
              <span className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.15em] text-cyan-300/80">
                <span className="h-3 w-3 animate-spin rounded-full border-2 border-cyan-400/30 border-t-cyan-400" />
                Updating…
              </span>
            )}
          </div>
          <div className={`flex-1 space-y-3 overflow-y-auto pr-1 transition-opacity ${sending ? "opacity-50" : ""}`}>
            {candidatePlan.length === 0 ? (
              <p className="text-sm text-slate-400">{sending ? "Building the first plan…" : "No plan proposed yet."}</p>
            ) : (
              candidatePlan.map((tc) => (
                <div key={tc.test_id} className="rounded-2xl border border-white/10 bg-slate-950/40 p-4">
                  <div className="flex items-center gap-2">
                    <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.15em] ${CATEGORY_STYLES[tc.category] ?? ""}`}>
                      {CATEGORY_LABELS[tc.category] ?? tc.category}
                    </span>
                    <span className="text-[10px] uppercase tracking-[0.15em] text-slate-500">{tc.priority}</span>
                  </div>
                  <p className="mt-2 text-sm font-medium text-white">{tc.goal}</p>
                  {tc.preconditions?.length > 0 && (
                    <p className="mt-1 text-xs italic text-slate-400">Setup: {tc.preconditions.join("; ")}</p>
                  )}
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
              ))
            )}
          </div>

          <div className="mt-4 flex flex-wrap gap-3">
            <button
              type="button"
              disabled={sending || candidatePlan.length === 0 || status !== "in_progress"}
              onClick={approve}
              className="rounded-2xl bg-gradient-to-r from-emerald-500 to-cyan-500 px-5 py-2.5 text-sm font-semibold text-white transition disabled:cursor-not-allowed disabled:opacity-50"
            >
              Approve &amp; start run
            </button>
            <button
              type="button"
              disabled={sending || status !== "in_progress"}
              onClick={cancel}
              className="rounded-2xl border border-rose-500/30 bg-rose-500/10 px-5 py-2.5 text-sm font-semibold text-rose-200 transition disabled:cursor-not-allowed disabled:opacity-50"
            >
              Cancel
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
