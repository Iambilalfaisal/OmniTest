interface ProgressBarProps {
  value: number      // 0-100
  running?: boolean
  className?: string
  height?: string
}

export function ProgressBar({ value, running, className = '', height = 'h-1' }: ProgressBarProps) {
  return (
    <div className={`progress-bar ${height} ${className}`}>
      <div
        className={running ? 'progress-bar-fill-running' : 'progress-bar-fill'}
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  )
}
