"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  KIND_LABELS,
  KIND_STYLES,
  SESSION_STATUS_LABELS,
  SESSION_STATUS_STYLES,
  SessionKind,
  SessionStatus,
} from "@/components/WorkerCard";
import HistoryTrendChart, { TrendDay } from "@/components/HistoryTrendChart";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
const PAGE_SIZE = 20;

type HistorySession = {
  id: string;
  kind: SessionKind;
  target_url: string;
  label: string;
  status: SessionStatus;
  summary?: { total: number; passed: number; failed: number } | null;
  parent_id?: string | null;
  created_at: string;
  updated_at: string;
};

type HistoryStats = {
  total_sessions: number;
  by_kind: Record<string, number>;
  by_status: Record<string, number>;
  overall_pass_rate: number | null;
  trend: TrendDay[];
};

const RUN_STATUSES: SessionStatus[] = ["running", "paused", "done", "error"];
const DISCOVERY_STATUSES: SessionStatus[] = ["in_progress", "approved", "cancelled"];

type RangeOption = { key: string; label: string; days: number | null };
const RANGE_OPTIONS: RangeOption[] = [
  { key: "7", label: "7d", days: 7 },
  { key: "30", label: "30d", days: 30 },
  { key: "90", label: "90d", days: 90 },
  { key: "all", label: "All time", days: null },
];

function timeAgo(iso: string): string {
  const diffMs = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(iso).toLocaleDateString();
}

function fillDailyTrend(trend: TrendDay[], since: Date | null, until: Date): TrendDay[] {
  if (!since) return trend; // "All time" — plot raw buckets as returned, may have gaps
  const byDate = new Map(trend.map((t) => [t.date, t]));
  const days: TrendDay[] = [];
  const cur = new Date(Date.UTC(since.getUTCFullYear(), since.getUTCMonth(), since.getUTCDate()));
  const end = new Date(Date.UTC(until.getUTCFullYear(), until.getUTCMonth(), until.getUTCDate()));
  while (cur <= end) {
    const iso = cur.toISOString().slice(0, 10);
    const existing = byDate.get(iso);
    days.push({ date: iso, passed: existing?.passed ?? 0, failed: existing?.failed ?? 0 });
    cur.setUTCDate(cur.getUTCDate() + 1);
  }
  return days;
}

export default function HistoryDashboard() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const [kind, setKind] = useState<"all" | SessionKind>((searchParams.get("kind") as SessionKind) || "all");
  const [status, setStatus] = useState(searchParams.get("status") ?? "");
  const [urlInput, setUrlInput] = useState(searchParams.get("url") ?? "");
  const [debouncedUrl, setDebouncedUrl] = useState(urlInput);
  const [range, setRange] = useState(searchParams.get("range") ?? "30");
  const [offset, setOffset] = useState(0);

  const [items, setItems] = useState<HistorySession[]>([]);
  const [total, setTotal] = useState(0);
  const [stats, setStats] = useState<HistoryStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Debounce the URL search box so every keystroke doesn't refetch.
  useEffect(() => {
    const t = setTimeout(() => setDebouncedUrl(urlInput), 300);
    return () => clearTimeout(t);
  }, [urlInput]);

  // Any filter change resets back to page 1 — otherwise "page 3" could silently
  // point past the end of a newly-narrowed result set.
  useEffect(() => {
    setOffset(0);
  }, [kind, status, debouncedUrl, range]);

  // Keep the filter state shareable via the URL.
  useEffect(() => {
    const params = new URLSearchParams();
    if (kind !== "all") params.set("kind", kind);
    if (status) params.set("status", status);
    if (debouncedUrl) params.set("url", debouncedUrl);
    if (range !== "30") params.set("range", range);
    router.replace(`/history${params.toString() ? `?${params.toString()}` : ""}`, { scroll: false });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind, status, debouncedUrl, range]);

  const { since, until } = useMemo(() => {
    const untilD = new Date();
    const opt = RANGE_OPTIONS.find((r) => r.key === range);
    if (!opt || opt.days === null) return { since: null as Date | null, until: untilD };
    const sinceD = new Date(untilD);
    sinceD.setUTCDate(sinceD.getUTCDate() - opt.days);
    return { since: sinceD, until: untilD };
  }, [range]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    const listParams = new URLSearchParams();
    if (kind !== "all") listParams.set("kind", kind);
    if (status) listParams.set("status", status);
    if (debouncedUrl) listParams.set("url", debouncedUrl);
    if (since) listParams.set("since", since.toISOString());
    listParams.set("until", until.toISOString());
    listParams.set("limit", String(PAGE_SIZE));
    listParams.set("offset", String(offset));

    const statsParams = new URLSearchParams();
    if (since) statsParams.set("since", since.toISOString());
    statsParams.set("until", until.toISOString());

    Promise.all([
      fetch(`${API_BASE}/history?${listParams.toString()}`).then((r) => {
        if (!r.ok) throw new Error("Failed to load history.");
        return r.json();
      }),
      fetch(`${API_BASE}/history/stats?${statsParams.toString()}`).then((r) => {
        if (!r.ok) throw new Error("Failed to load history stats.");
        return r.json();
      }),
    ])
      .then(([listRes, statsRes]) => {
        if (cancelled) return;
        setItems(listRes.items ?? []);
        setTotal(listRes.total ?? 0);
        setStats(statsRes as HistoryStats);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Something went wrong.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [kind, status, debouncedUrl, since, until, offset]);

  const statusOptions =
    kind === "discovery" ? DISCOVERY_STATUSES : kind === "run" ? RUN_STATUSES : [...RUN_STATUSES, ...DISCOVERY_STATUSES];

  function goTo(item: HistorySession) {
    router.push(item.kind === "discovery" ? `/discover?id=${item.id}` : `/run?id=${item.id}`);
  }

  const trendDays = stats ? fillDailyTrend(stats.trend, since, until) : [];

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-5 py-4">
      <header className="glass-panel rounded-3xl p-6">
        <p className="text-xs uppercase tracking-[0.3em] text-cyan-300/80">History</p>
        <h1 className="mt-2 text-2xl font-semibold text-white">All sessions</h1>
      </header>

      {/* Filters — one row, above everything they scope */}
      <section className="glass-panel flex flex-wrap items-center gap-3 rounded-2xl p-3">
        <div className="flex overflow-hidden rounded-full border border-white/10">
          {(["all", "run", "discovery"] as const).map((k) => (
            <button
              key={k}
              type="button"
              onClick={() => setKind(k)}
              className={`px-3 py-1.5 text-xs font-medium transition ${
                kind === k ? "bg-cyan-500/20 text-cyan-200" : "bg-slate-950/50 text-slate-300 hover:text-white"
              }`}
            >
              {k === "all" ? "All" : k === "run" ? "Runs" : "Discovery"}
            </button>
          ))}
        </div>

        <select
          value={status}
          onChange={(e) => setStatus(e.target.value)}
          className="rounded-full border border-white/10 bg-slate-950/50 px-3 py-1.5 text-xs text-slate-200 outline-none focus:border-cyan-400/60"
        >
          <option value="">All statuses</option>
          {statusOptions.map((s) => (
            <option key={s} value={s}>
              {SESSION_STATUS_LABELS[s]}
            </option>
          ))}
        </select>

        <input
          value={urlInput}
          onChange={(e) => setUrlInput(e.target.value)}
          placeholder="Search target URL…"
          className="min-w-[12rem] flex-1 rounded-full border border-white/10 bg-slate-950/50 px-3 py-1.5 text-xs text-slate-200 outline-none focus:border-cyan-400/60"
        />

        <div className="flex overflow-hidden rounded-full border border-white/10">
          {RANGE_OPTIONS.map((r) => (
            <button
              key={r.key}
              type="button"
              onClick={() => setRange(r.key)}
              className={`px-3 py-1.5 text-xs font-medium transition ${
                range === r.key ? "bg-cyan-500/20 text-cyan-200" : "bg-slate-950/50 text-slate-300 hover:text-white"
              }`}
            >
              {r.label}
            </button>
          ))}
        </div>
      </section>

      {error && (
        <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">{error}</div>
      )}

      {/* Stat tiles — reflect the selected date range, not the kind/status/url filters
          (those only narrow the list below), so the numbers here always mean
          "all completed runs in this window." */}
      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile label="Total sessions" value={stats?.total_sessions ?? "—"} />
        <StatTile
          label="Overall pass rate"
          value={stats?.overall_pass_rate != null ? `${Math.round(stats.overall_pass_rate * 100)}%` : "—"}
          accent="emerald"
        />
        <StatTile label="Done" value={stats?.by_status?.done ?? 0} accent="emerald" />
        <StatTile label="Errors" value={stats?.by_status?.error ?? 0} accent="rose" />
      </section>

      <section className="glass-panel rounded-3xl p-5">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">Daily pass/fail</h2>
        <HistoryTrendChart days={trendDays} />
      </section>

      <section className="glass-panel rounded-3xl p-3">
        {loading ? (
          <p className="p-4 text-sm text-slate-400">Loading…</p>
        ) : items.length === 0 ? (
          <p className="p-4 text-sm text-slate-400">No sessions match these filters.</p>
        ) : (
          <div className="divide-y divide-white/5">
            {items.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => goTo(item)}
                className="flex w-full flex-col gap-2 px-3 py-3 text-left transition hover:bg-white/5 sm:flex-row sm:items-center sm:justify-between"
              >
                <div className="flex min-w-0 items-center gap-3">
                  <span
                    className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.15em] ${KIND_STYLES[item.kind]}`}
                  >
                    {KIND_LABELS[item.kind]}
                  </span>
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium text-white" title={item.target_url}>
                      {item.target_url}
                    </div>
                    <div className="truncate text-xs text-slate-400" title={item.label}>
                      {item.label}
                    </div>
                  </div>
                </div>

                <div className="flex shrink-0 items-center gap-3 text-xs text-slate-400">
                  {item.summary && (
                    <span>
                      {item.summary.passed}/{item.summary.total} passed
                      {item.summary.failed > 0 && <span className="text-rose-300"> · {item.summary.failed} failed</span>}
                    </span>
                  )}
                  <span
                    className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.15em] ${
                      SESSION_STATUS_STYLES[item.status] ?? ""
                    }`}
                  >
                    {SESSION_STATUS_LABELS[item.status] ?? item.status}
                  </span>
                  <span title={item.created_at}>{timeAgo(item.created_at)}</span>
                </div>
              </button>
            ))}
          </div>
        )}

        {total > PAGE_SIZE && (
          <div className="mt-2 flex items-center justify-between border-t border-white/5 px-3 pt-3 text-xs text-slate-400">
            <span>
              Showing {Math.min(offset + 1, total)}–{Math.min(offset + PAGE_SIZE, total)} of {total}
            </span>
            <div className="flex gap-2">
              <button
                type="button"
                disabled={offset === 0}
                onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
                className="rounded-full border border-white/10 px-3 py-1.5 disabled:opacity-40"
              >
                Prev
              </button>
              <button
                type="button"
                disabled={offset + PAGE_SIZE >= total}
                onClick={() => setOffset((o) => o + PAGE_SIZE)}
                className="rounded-full border border-white/10 px-3 py-1.5 disabled:opacity-40"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

function StatTile({ label, value, accent }: { label: string; value: string | number; accent?: "emerald" | "rose" }) {
  const valueColor = accent === "emerald" ? "text-emerald-300" : accent === "rose" ? "text-rose-300" : "text-white";
  return (
    <div className="rounded-2xl border border-white/10 bg-slate-950/40 p-4">
      <div className="text-xs uppercase tracking-[0.28em] text-slate-400">{label}</div>
      <div className={`mt-3 text-2xl font-semibold ${valueColor}`}>{value}</div>
    </div>
  );
}
