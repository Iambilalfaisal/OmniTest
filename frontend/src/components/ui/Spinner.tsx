'use client'
import { cn } from '@/lib/utils'

interface SpinnerProps { size?: 'sm' | 'md' | 'lg'; className?: string }

export function Spinner({ size = 'md', className }: SpinnerProps) {
  const sizes = { sm: 'w-3 h-3', md: 'w-5 h-5', lg: 'w-8 h-8' }
  return (
    <div
      className={cn(
        'rounded-full border-2 border-t-transparent animate-spin',
        sizes[size],
        className,
      )}
      style={{ borderColor: 'var(--primary)', borderTopColor: 'transparent' }}
    />
  )
}
