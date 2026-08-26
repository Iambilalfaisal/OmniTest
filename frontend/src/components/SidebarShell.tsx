'use client'
import { useState, useEffect } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import { Compass, Globe, Clock, BarChart2, ChevronLeft, ChevronRight } from 'lucide-react'
import { OrbitMark } from './OrbitMark'

const NAV = [
  { href: '/',         icon: Compass,   label: 'Home'     },
  { href: '/discover', icon: Globe,     label: 'Discover' },
  { href: '/history',  icon: Clock,     label: 'History'  },
  { href: '/reports',  icon: BarChart2, label: 'Reports'  },
]

const STORAGE_KEY = 'omnitest_sidebar_collapsed'

export function SidebarShell() {
  const pathname = usePathname()
  const [collapsed, setCollapsed] = useState(false)
  const [mounted, setMounted] = useState(false)

  // Hydrate from localStorage after mount to avoid SSR mismatch
  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored === '1') setCollapsed(true)
    setMounted(true)
  }, [])

  function toggle() {
    setCollapsed((c) => {
      const next = !c
      localStorage.setItem(STORAGE_KEY, next ? '1' : '0')
      return next
    })
  }

  // Render a placeholder with the same width to avoid layout shift
  if (!mounted) {
    return (
      <div
        style={{
          width: 260,
          height: '100vh',
          background: 'var(--bg-surface-2)',
          borderRight: '1px solid var(--border)',
          flexShrink: 0,
        }}
      />
    )
  }

  return (
    <motion.aside
      animate={{ width: collapsed ? 72 : 260 }}
      initial={false}
      transition={{ duration: 0.25, ease: 'easeInOut' }}
      className="sticky top-0 h-screen flex flex-col overflow-hidden flex-shrink-0"
      style={{
        background: 'linear-gradient(180deg, rgba(23,27,49,0.95), rgba(11,13,24,0.94))',
        borderRight: '1px solid var(--border)',
      }}
    >
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -left-20 top-10 h-52 w-52 rounded-full bg-indigo-500/10 blur-3xl" />
        <div className="absolute -right-20 bottom-10 h-44 w-44 rounded-full bg-cyan-400/10 blur-3xl" />
        <div className="page-grid absolute inset-0 opacity-40" />
      </div>

      <div className="relative flex h-full flex-col gap-1 px-3 py-5">
        {/* Logo row */}
        <div
          className="glass-panel flex items-center gap-3 overflow-hidden rounded-2xl px-2.5 py-2 mb-4"
          style={{ minHeight: 32 }}
        >
          <div className="flex-shrink-0">
            <OrbitMark size="sm" />
          </div>
          <motion.div
            animate={{ opacity: collapsed ? 0 : 1, x: collapsed ? -8 : 0 }}
            transition={{ duration: 0.15 }}
            className="min-w-0"
          >
            <p className="eyebrow">Control</p>
            <p className="whitespace-nowrap text-sm font-semibold tracking-tight text-white">OmniTest</p>
          </motion.div>
        </div>

        {/* Nav */}
        <nav className="flex flex-1 flex-col gap-1">
          {NAV.map(({ href, icon: Icon, label }) => {
            const active = href === '/' ? pathname === '/' : pathname?.startsWith(href)
            return (
              <Link
                key={href}
                href={href}
                title={collapsed ? label : undefined}
                className={['nav-item', active ? 'nav-item-active' : ''].filter(Boolean).join(' ')}
                style={{
                  justifyContent: collapsed ? 'center' : 'flex-start',
                  paddingLeft: collapsed ? 0 : undefined,
                  paddingRight: collapsed ? 0 : undefined,
                  borderRadius: 12,
                }}
              >
                <Icon size={18} style={{ flexShrink: 0 }} />
                {!collapsed && (
                  <span className="whitespace-nowrap overflow-hidden">{label}</span>
                )}
              </Link>
            )
          })}
        </nav>

        {/* Collapse toggle */}
        <button
          onClick={toggle}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="nav-item justify-center mt-2 border border-white/10 bg-white/5"
          style={{ paddingLeft: 0, paddingRight: 0, borderRadius: 12 }}
        >
          {collapsed ? <ChevronRight size={16} /> : (
            <>
              <ChevronLeft size={16} />
              <span className="whitespace-nowrap overflow-hidden text-xs">Collapse</span>
            </>
          )}
        </button>

        {!collapsed && (
          <div className="mt-3 rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2">
            <p className="eyebrow">Status</p>
            <p className="mt-1 text-xs text-cyan-300">Neural runspace online</p>
          </div>
        )}
      </div>
    </motion.aside>
  )
}

export default SidebarShell
