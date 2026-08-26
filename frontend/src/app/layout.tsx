import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { SidebarShell } from '@/components/SidebarShell'
import { OrbitMark } from '@/components/OrbitMark'

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' })

export const metadata: Metadata = {
  title: 'OmniTest',
  description: 'AI-Powered QA Testing Platform',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body style={{ background: 'var(--bg-void)', color: 'var(--text)' }}>
        <div className="flex min-h-screen">
          {/* Sidebar — hidden on mobile */}
          <div className="hidden md:block flex-shrink-0">
            <SidebarShell />
          </div>

          {/* Mobile header */}
          <header
            className="md:hidden fixed top-0 inset-x-0 z-50 flex items-center gap-3 px-4 h-14"
            style={{
              background: 'linear-gradient(180deg, rgba(23,27,49,0.95), rgba(11,13,24,0.92))',
              borderBottom: '1px solid var(--border)',
              backdropFilter: 'blur(16px)',
              WebkitBackdropFilter: 'blur(16px)',
            }}
          >
            <OrbitMark size="sm" />
            <span className="font-semibold text-sm tracking-tight" style={{ color: 'var(--text)' }}>
              OmniTest
            </span>
          </header>

          {/* Main content */}
          <main className="flex-1 flex flex-col min-h-screen md:pt-0 pt-14">
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}
