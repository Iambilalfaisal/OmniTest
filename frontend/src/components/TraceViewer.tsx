export default function TraceViewer({
  traceUrl,
  onClose,
}: {
  traceUrl: string;
  onClose: () => void;
}) {
  const viewerUrl = `https://trace.playwright.dev/?trace=${encodeURIComponent(traceUrl)}`;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
      <div className="h-[85vh] w-[90vw] overflow-hidden rounded-lg border border-neutral-700 bg-neutral-950">
        <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-2">
          <span className="text-sm text-neutral-400">Trace viewer</span>
          <button onClick={onClose} className="text-sm text-neutral-400 hover:text-neutral-100">
            close
          </button>
        </div>
        <iframe src={viewerUrl} className="h-full w-full" title="Playwright trace" />
      </div>
    </div>
  );
}
