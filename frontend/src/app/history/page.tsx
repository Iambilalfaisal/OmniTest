import { Suspense } from "react";
import HistoryDashboard from "@/components/HistoryDashboard";

export default function HistoryPage() {
  return (
    <Suspense fallback={<div className="glass-panel mx-auto max-w-xl rounded-3xl p-8 text-center text-slate-300">Loading history…</div>}>
      <HistoryDashboard />
    </Suspense>
  );
}
