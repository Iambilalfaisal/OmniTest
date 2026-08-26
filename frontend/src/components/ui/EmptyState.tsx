import type { FC } from 'react'

interface EmptyStateProps {
  icon: FC<{ size?: number; style?: React.CSSProperties; className?: string }>
  title: string
  description?: string
  action?: React.ReactNode
}

export function EmptyState({ icon: Icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-16 text-center">
      <div className="rounded-2xl p-4" style={{ background: 'var(--bg-surface-2)', border: '1px solid var(--border)' }}>
        <Icon size={28} style={{ color: 'var(--text-3)' }} />
      </div>
      <div>
        <p className="font-semibold text-sm" style={{ color: 'var(--text)' }}>{title}</p>
        {description && <p className="text-sm mt-1" style={{ color: 'var(--text-2)' }}>{description}</p>}
      </div>
      {action && <div>{action}</div>}
    </div>
  )
}
