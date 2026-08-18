export default function TraceViewer({
  traceUrl,
  onClose,
}: {
  traceUrl: string;
  onClose: () => void;
}) {
  const viewerUrl = `https://trace.playwright.dev/?trace=${encodeURIComponent(traceUrl)}`;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4 backdrop-blur-sm">
      <div className="h-[88vh] w-full max-w-6xl overflow-hidden rounded-3xl border border-white/10 bg-slate-950 shadow-2xl shadow-slate-950/50">
        <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
          <span className="text-sm uppercase tracking-[0.2em] text-slate-400">Trace viewer</span>
          <button onClick={onClose} className="rounded-full border border-white/10 px-3 py-1 text-sm text-slate-300 transition hover:border-white/20 hover:text-white">
            Close
          </button>
        </div>
        <iframe src={viewerUrl} className="h-[calc(88vh-57px)] w-full border-0" title="Playwright trace" />
      </div>
    </div>
  );
}
