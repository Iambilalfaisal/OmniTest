"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Send, ClipboardList, MessageSquare } from "lucide-react";
import { CATEGORY_LABELS, CATEGORY_STYLES } from "@/components/WorkerCard";
import type { TestCase } from "@/components/WorkerCard";

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
    return () => { cancelled = true; };
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
      <div className="glass-panel mx-auto max-w-xl rounded-3xl p-8 text-center" style={{ color: "var(--text-2)" }}>
        No discovery session selected — start one from the home page.
      </div>
    );
  }

  if (loading) {
    return (
      <div className="glass-panel mx-auto max-w-xl rounded-3xl p-8 text-center" style={{ color: "var(--text-2)" }}>
        Loading discovery session…
      </div>
    );
  }

  const atTurnLimit = turnCount >= maxTurns;
  const canReply = status === "in_progress" && !atTurnLimit;
  const nearLimit = !atTurnLimit && maxTurns - turnCount <= 3;

  return (
    <div className="relative mx-auto flex w-full max-w-7xl flex-col gap-4 py-4">
      <div className="page-grid pointer-events-none absolute inset-x-0 top-0 h-96 opacity-60" />
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        className="glass-panel animate-rise relative flex min-h-[70vh] w-full overflow-hidden rounded-[1.75rem] lg:h-[calc(100vh-7rem)] lg:min-h-0 lg:flex-row"
      >
      {/* ── LEFT PANEL: Chat (45%) ────────────────────────── */}
      <div className="flex w-full flex-col border-b border-white/10 lg:w-[45%] lg:border-b-0 lg:border-r lg:border-white/10">
        {/* Chat header */}
        <div
          className="flex items-center justify-between px-5 py-3 shrink-0"
          style={{ borderBottom: "1px solid var(--border)", background: "var(--bg-surface-2)" }}
        >
          <div className="flex items-center gap-2 min-w-0">
            <MessageSquare size={14} style={{ color: "var(--primary-bright)", flexShrink: 0 }} />
            <span className="eyebrow-accent" style={{ color: "var(--primary-bright)" }}>Conversation</span>
            {targetUrl && (
              <span
                className="text-xs truncate hidden sm:block"
                style={{ color: "var(--text-3)" }}
                title={targetUrl}
              >
                — {targetUrl}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {sitePages > 0 && (
              <span className="badge badge-violet">{sitePages} pages</span>
            )}
            <span
              className="badge"
              style={
                atTurnLimit || nearLimit
                  ? { background: "var(--amber-bg)", border: "1px solid var(--amber-border)", color: "var(--amber)" }
                  : { background: "var(--bg-elevated)", border: "1px solid var(--border-md)", color: "var(--text-2)" }
              }
            >
              Turn {turnCount} / {maxTurns}
            </span>
          </div>
        </div>

        {/* Messages scrollable area */}
        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
          <AnimatePresence initial={false}>
            {transcript.map((m, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.22, ease: "easeOut" }}
                className={m.role === "assistant" ? "flex justify-start" : "flex justify-end"}
              >
                <div
                  style={{
                    maxWidth: "86%",
                    borderRadius: "10px",
                    padding: "10px 14px",
                    fontSize: "0.875rem",
                    lineHeight: "1.6",
                    ...(m.role === "assistant"
                      ? {
                          background: "var(--bg-surface-2)",
                          borderLeft: "3px solid var(--primary)",
                          color: "var(--text)",
                        }
                      : {
                          background: "var(--primary-bg)",
                          border: "1px solid var(--primary-border)",
                          color: "var(--text)",
                        }),
                  }}
                >
                  <div
                    className="eyebrow mb-1"
                    style={{ color: m.role === "assistant" ? "var(--primary-bright)" : "var(--text-3)" }}
                  >
                    {m.role === "assistant" ? "OmniTest" : "You"}
                  </div>
                  {m.text}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {/* Typing indicator */}
          {sending && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex justify-start"
            >
              <div
                style={{
                  background: "var(--bg-surface-2)",
                  borderLeft: "3px solid var(--primary)",
                  borderRadius: "10px",
                  padding: "10px 14px",
                  display: "flex",
                  flexDirection: "column",
                  gap: 6,
                }}
              >
                <div className="eyebrow" style={{ color: "var(--primary-bright)" }}>OmniTest</div>
                <div style={{ display: "flex", gap: 5, alignItems: "center" }}>
                  {[0, 0.18, 0.36].map((delay) => (
                    <motion.span
                      key={delay}
                      animate={{ y: [0, -5, 0] }}
                      transition={{ duration: 0.75, repeat: Infinity, delay, ease: "easeInOut" }}
                      style={{
                        display: "inline-block",
                        width: 7,
                        height: 7,
                        borderRadius: "50%",
                        background: "var(--primary-bright)",
                      }}
                    />
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          <div ref={transcriptEndRef} />
        </div>

        {/* Inline banners */}
        {error && (
          <div
            className="mx-4 mb-2 rounded-lg px-4 py-2 text-sm"
            style={{ background: "var(--rose-bg)", border: "1px solid var(--rose-border)", color: "var(--rose)" }}
          >
            {error}
          </div>
        )}
        {nearLimit && !atTurnLimit && (
          <div
            className="mx-4 mb-2 rounded-lg px-3 py-2 text-xs"
            style={{ background: "var(--amber-bg)", border: "1px solid var(--amber-border)", color: "var(--amber)" }}
          >
            {maxTurns - turnCount} turn{maxTurns - turnCount !== 1 ? "s" : ""} remaining — consider approving the plan soon.
          </div>
        )}
        {atTurnLimit && status === "in_progress" && (
          <div
            className="mx-4 mb-2 rounded-lg px-3 py-2 text-xs"
            style={{ background: "var(--amber-bg)", border: "1px solid var(--amber-border)", color: "var(--amber)" }}
          >
            Turn limit reached — approve the plan or cancel to start over.
          </div>
        )}

        {/* Input area */}
        <form
          onSubmit={sendMessage}
          className="flex gap-2 items-end px-4 py-4 shrink-0"
          style={{ borderTop: "1px solid var(--border)" }}
        >
          <textarea
            value={reply}
            rows={3}
            disabled={!canReply || sending}
            placeholder={canReply ? "Reply, ask a question, or suggest a test case…" : "Conversation closed"}
            onChange={(e) => setReply(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (canReply && !sending && reply.trim()) {
                  sendMessage(e as unknown as FormEvent);
                }
              }
            }}
            className="input flex-1"
            style={{ resize: "none" }}
          />
          <button
            type="submit"
            disabled={!canReply || sending || !reply.trim()}
            className="btn-primary btn-icon"
            style={{ alignSelf: "flex-end", padding: "10px" }}
          >
            <Send size={16} />
          </button>
        </form>
      </div>

      {/* ── RIGHT PANEL: Emerging Plan (55%) ─────────────── */}
      <div className="flex w-full flex-col lg:flex-1">
        {/* Plan header */}
        <div
          className="flex items-center justify-between px-5 py-3 shrink-0"
          style={{ borderBottom: "1px solid var(--border)", background: "var(--bg-surface-2)" }}
        >
          <div className="flex items-center gap-2">
            <ClipboardList size={14} style={{ color: "var(--cyan)", flexShrink: 0 }} />
            <span className="eyebrow" style={{ color: "var(--cyan)", letterSpacing: "0.3em" }}>Test Plan</span>
          </div>
          <span className="badge badge-cyan">
            {candidatePlan.length} test{candidatePlan.length !== 1 ? "s" : ""}
          </span>
        </div>

        {/* Test case cards */}
        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
          {candidatePlan.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
              <ClipboardList size={36} style={{ color: "var(--text-3)", opacity: 0.35 }} />
              <p className="text-sm" style={{ color: "var(--text-3)" }}>
                The AI will propose tests here as you chat.
              </p>
            </div>
          ) : (
            <AnimatePresence>
              {candidatePlan.map((tc, i) => (
                <motion.div
                  key={tc.test_id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.28, delay: Math.min(i * 0.04, 0.32), ease: "easeOut" }}
                  className="glass-panel p-4"
                >
                  <div className="flex items-center gap-2 flex-wrap">
                    <span
                      className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.15em] ${CATEGORY_STYLES[tc.category] ?? ""}`}
                    >
                      {CATEGORY_LABELS[tc.category] ?? tc.category}
                    </span>
                    <span
                      className="text-[10px] uppercase tracking-widest"
                      style={{ color: "var(--text-3)" }}
                    >
                      {tc.priority}
                    </span>
                    {tc.steps?.length > 0 && (
                      <span className="ml-auto text-[10px]" style={{ color: "var(--text-3)" }}>
                        {tc.steps.length} step{tc.steps.length !== 1 ? "s" : ""}
                      </span>
                    )}
                  </div>
                  <p className="mt-2 text-sm font-medium" style={{ color: "var(--text)" }}>
                    {tc.goal}
                  </p>
                  {tc.preconditions?.length > 0 && (
                    <p className="mt-1 text-xs italic" style={{ color: "var(--text-2)" }}>
                      Setup: {tc.preconditions.join("; ")}
                    </p>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>
          )}
        </div>

        {/* Approve / Cancel — fixed at bottom when plan exists */}
        {candidatePlan.length > 0 && status === "in_progress" && (
          <div
            className="flex gap-3 px-4 py-4 shrink-0"
            style={{ borderTop: "1px solid var(--border)" }}
          >
            <button
              type="button"
              disabled={sending}
              onClick={approve}
              className="btn-primary btn-lg flex-1"
            >
              Approve &amp; Run ({candidatePlan.length} test{candidatePlan.length !== 1 ? "s" : ""})
            </button>
            <button
              type="button"
              disabled={sending}
              onClick={cancel}
              className="btn-ghost"
            >
              Cancel
            </button>
          </div>
        )}
      </div>
      </motion.div>
    </div>
  );
}
