import { Suspense } from "react";
import RunPageClient from "@/components/RunPageClient";

export default function RunPage() {
  return (
    <Suspense fallback={<div className="glass-panel mx-auto max-w-xl rounded-3xl p-8 text-center text-slate-300">Loading run…</div>}>
      <RunPageClient />
    </Suspense>
  );
}
