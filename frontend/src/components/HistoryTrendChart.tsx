"use client";

import { useState } from "react";

export type TrendDay = { date: string; passed: number; failed: number };

const CHART_HEIGHT = 160;
const VIEW_WIDTH = 100;

function formatShortDate(iso: string): string {
  const d = new Date(iso + "T00:00:00Z");
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", timeZone: "UTC" });
}

// Stacked daily bar (passed/failed) — passed/failed wear this app's existing Pass/Fail
// status colors (emerald/rose), not a fresh categorical pair: per the dataviz method,
// a series that means good/bad wears status tokens, never arbitrary categorical hues.
export default function HistoryTrendChart({ days }: { days: TrendDay[] }) {
  const [hover, setHover] = useState<{ index: number; x: number } | null>(null);

  if (days.length === 0) {
    return <p className="text-sm text-slate-400">No completed runs in this window yet.</p>;
  }

  const maxTotal = Math.max(1, ...days.map((d) => d.passed + d.failed));
  const step = maxTotal <= 5 ? 1 : maxTotal <= 20 ? 5 : maxTotal <= 100 ? 10 : 50;
  const niceMax = Math.ceil(maxTotal / step) * step;

  const chartTop = 8;
  const chartBottom = CHART_HEIGHT - 20;
  const plotHeight = chartBottom - chartTop;
  const slotWidth = VIEW_WIDTH / days.length;
  const barWidth = Math.max(0.6, Math.min(slotWidth - 0.5, 3));
  const yTicks = [0, niceMax / 2, niceMax];
  const labelEvery = Math.max(1, Math.ceil(days.length / 6));

  return (
    <div>
      <div className="mb-2 flex items-center gap-4 text-xs text-slate-400">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-emerald-500" /> Passed
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-rose-500" /> Failed
        </span>
      </div>

      <div className="relative">
        <svg
          viewBox={`0 0 ${VIEW_WIDTH} ${CHART_HEIGHT}`}
          className="w-full"
          style={{ height: CHART_HEIGHT }}
          preserveAspectRatio="none"
        >
          {yTicks.map((t) => {
            const y = chartBottom - (t / niceMax) * plotHeight;
            return (
              <g key={t}>
                <line x1={0} x2={VIEW_WIDTH} y1={y} y2={y} stroke="rgba(255,255,255,0.08)" strokeWidth={0.3} />
                <text x={0} y={y - 1} fontSize={2.6} fill="#898781">
                  {t}
                </text>
              </g>
            );
          })}
          <line x1={0} x2={VIEW_WIDTH} y1={chartBottom} y2={chartBottom} stroke="rgba(255,255,255,0.16)" strokeWidth={0.4} />

          {days.map((d, i) => {
            const passedH = (d.passed / niceMax) * plotHeight;
            const failedH = (d.failed / niceMax) * plotHeight;
            const x = i * slotWidth + (slotWidth - barWidth) / 2;
            const passedY = chartBottom - passedH;
            // 2px-equivalent surface gap between the two stacked segments, per mark spec.
            const gap = d.passed > 0 && d.failed > 0 ? 0.6 : 0;
            const failedY = passedY - gap - failedH;
            const isHovered = hover?.index === i;
            const total = d.passed + d.failed;

            return (
              <g
                key={d.date}
                onMouseEnter={() => setHover({ index: i, x: i * slotWidth + slotWidth / 2 })}
                onMouseLeave={() => setHover((h) => (h?.index === i ? null : h))}
                style={{ cursor: total > 0 ? "pointer" : "default" }}
              >
                {/* transparent hit target, wider than the visual bar */}
                <rect x={i * slotWidth} y={chartTop} width={slotWidth} height={plotHeight} fill="transparent" />
                {d.passed > 0 && (
                  <rect
                    x={x}
                    y={passedY}
                    width={barWidth}
                    height={passedH}
                    rx={d.failed > 0 ? 0 : 0.8}
                    className={isHovered ? "fill-emerald-400" : "fill-emerald-500"}
                  />
                )}
                {d.failed > 0 && (
                  <rect
                    x={x}
                    y={failedY}
                    width={barWidth}
                    height={failedH}
                    rx={0.8}
                    className={isHovered ? "fill-rose-400" : "fill-rose-500"}
                  />
                )}
              </g>
            );
          })}
        </svg>

        {hover && (
          <div
            className="pointer-events-none absolute -translate-x-1/2 -translate-y-full rounded-xl border border-white/10 bg-slate-900/95 px-3 py-2 text-xs shadow-xl"
            style={{ left: `${(hover.x / VIEW_WIDTH) * 100}%`, top: 0 }}
          >
            <div className="font-semibold text-white">{formatShortDate(days[hover.index].date)}</div>
            <div className="text-emerald-300">{days[hover.index].passed} passed</div>
            <div className="text-rose-300">{days[hover.index].failed} failed</div>
          </div>
        )}
      </div>

      <div className="relative mt-1 h-4 text-[10px] text-slate-500">
        {days.map((d, i) =>
          i % labelEvery === 0 || i === days.length - 1 ? (
            <span
              key={d.date}
              className="absolute -translate-x-1/2"
              style={{ left: `${((i * slotWidth + slotWidth / 2) / VIEW_WIDTH) * 100}%` }}
            >
              {formatShortDate(d.date)}
            </span>
          ) : null
        )}
      </div>
    </div>
  );
}
