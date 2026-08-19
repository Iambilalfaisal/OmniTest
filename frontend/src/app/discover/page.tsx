import { Suspense } from "react";
import DiscoveryChat from "@/components/DiscoveryChat";

export default function DiscoverPage() {
  return (
    <Suspense fallback={<div className="glass-panel mx-auto max-w-xl rounded-3xl p-8 text-center text-slate-300">Loading…</div>}>
      <DiscoveryChat />
    </Suspense>
  );
}
