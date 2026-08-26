import { WORKER_PHASE_LABELS } from '@/lib/constants'

interface StatusBadgeProps {
  phase: string
  animated?: boolean
  className?: string
}

export function StatusBadge({ phase, animated = true, className = '' }: StatusBadgeProps) {
  const config = WORKER_PHASE_LABELS[phase] ?? { label: phase, badge: 'badge-queued' }
  const isLive = phase === 'running' || phase === 'awaiting_input'
  return (
    <span className={`${config.badge} ${className}`}>
      {animated && isLive && (
        <span className={phase === 'running' ? 'status-dot-running' : 'status-dot-wait'} />
      )}
      {config.label}
    </span>
  )
}
