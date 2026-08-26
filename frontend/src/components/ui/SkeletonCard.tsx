export function SkeletonCard({ rows = 3 }: { rows?: number }) {
  return (
    <div className="card p-5 space-y-3">
      <div className="flex items-center gap-3">
        <div className="skeleton rounded-badge h-5 w-20" />
        <div className="skeleton rounded h-4 flex-1" />
      </div>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="skeleton rounded h-3" style={{ width: `${70 + i * 10}%` }} />
      ))}
    </div>
  )
}
