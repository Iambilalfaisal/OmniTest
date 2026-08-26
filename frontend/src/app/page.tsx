'use client'

import { useRouter } from 'next/navigation'
import { FormEvent, useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Globe, Zap, Code2, Link as LinkIcon, ChevronDown, Check, Loader2, AlertCircle, Sparkles } from 'lucide-react'
import { OrbitMark } from '@/components/OrbitMark'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000'

type Mode = 'explore' | 'quick' | 'my_plan'

const MODES = [
  {
    id: 'explore' as Mode,
    icon: Globe,
    label: 'Explore & Plan',
    description: 'AI discovers your app and proposes a test plan',
    placeholder: 'Describe what you want tested, or leave blank to let the AI decide...',
    submitLabel: 'Start Exploring',
  },
  {
    id: 'quick' as Mode,
    icon: Zap,
    label: 'Quick Start',
    description: 'AI proposes a complete test plan in one pass for you to approve',
    placeholder: 'What should I focus on? (optional)',
    submitLabel: 'Run Now',
  },
  {
    id: 'my_plan' as Mode,
    icon: Code2,
    label: 'My Own Plan',
    description: 'Describe your own test cases in plain English (or paste JSON)',
    placeholder:
      'e.g. "Test that signing up with an email that already exists shows an error. ' +
      'Also test that logging in with the wrong password is rejected." — one or many cases, in your own words.',
    submitLabel: 'Run My Plan',
  },
]

const TILES = [
  {
    icon: Sparkles,
    title: 'Autonomous Agent',
    description: 'Explores your app like a real user',
    color: 'var(--primary)',
  },
  {
    icon: Globe,
    title: 'MCP Browser',
    description: 'Playwright-powered, fully isolated sessions',
    color: 'var(--cyan)',
  },
  {
    icon: Zap,
    title: 'Trace & Video',
    description: 'Every run captured for debugging',
    color: 'var(--violet)',
  },
]

export default function HomePage() {
  const router = useRouter()
  const [url, setUrl] = useState('')
  const [mode, setMode] = useState<Mode>('explore')
  const [body, setBody] = useState('')
  const [jsonError, setJsonError] = useState<string | null>(null)
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const selectedMode = MODES.find((m) => m.id === mode)!

  // Close dropdown on outside click
  useEffect(() => {
    function handler(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  function selectMode(m: Mode) {
    setMode(m)
    setBody('')
    setJsonError(null)
    setError(null)
    setDropdownOpen(false)
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!url.trim()) return
    if (mode === 'my_plan' && !body.trim()) {
      setJsonError('Describe at least one test case, or paste a JSON array.')
      return
    }
    setError(null)
    setSubmitting(true)

    try {
      if (mode === 'explore' || mode === 'quick') {
        // Same /discover endpoint and chat-approval flow for both — `mode` tells the
        // backend which DISCOVERY_SYSTEM_PROMPT variant to run and (once approved)
        // whether the run skips the recon subgraph (nodes/discovery.py, graph/builder.py).
        const res = await fetch(`${API_BASE}/discover`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ target_url: url, starting_idea: body, mode }),
        })
        if (!res.ok) throw new Error((await res.text()) || 'Unable to start discovery.')
        const { discovery_id } = await res.json()
        router.push(`/discover?id=${discovery_id}`)
        return
      }

      // my_plan — a JSON array of test-case objects is sent as-is (no LLM call, same as
      // before); anything else is sent as plain-English text and parsed into structured
      // test cases server-side (nodes/custom_plan.py).
      let planPayload: { test_cases: unknown[] } | { raw_plan_text: string }
      let instruction: string
      try {
        const parsed = JSON.parse(body)
        if (!Array.isArray(parsed) || parsed.length === 0) throw new Error('not a non-empty array')
        planPayload = { test_cases: parsed }
        instruction = `Run ${parsed.length} user-provided test case${parsed.length !== 1 ? 's' : ''} on ${url}`
      } catch {
        planPayload = { raw_plan_text: body }
        instruction = `Run user-described test cases on ${url}`
      }
      const res = await fetch(`${API_BASE}/runs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_url: url, instruction, ...planPayload }),
      })
      if (!res.ok) throw new Error((await res.text()) || 'Unable to start run.')
      const { run_id } = await res.json()
      router.push(`/run?id=${run_id}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="relative mx-auto flex w-full max-w-2xl flex-col items-center gap-10 py-16 px-4">

      {/* Branding */}
      <div className="flex flex-col items-center gap-4 text-center">
        <OrbitMark size="lg" />
        <div>
          <h1
            className="text-4xl font-semibold"
            style={{ color: 'var(--text)', letterSpacing: '-0.03em' }}
          >
            OmniTest
          </h1>
          <p className="mt-1.5 text-sm" style={{ color: 'var(--text-2)' }}>
            AI-Powered QA. Relentlessly Thorough.
          </p>
        </div>
      </div>

      {/* Composer card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
        className="w-full"
      >
        <form
          onSubmit={handleSubmit}
          className="panel w-full"
          style={{ borderRadius: 20, padding: 0, overflow: 'hidden' }}
        >
          {/* URL row */}
          <div
            className="flex items-center gap-3 px-4 py-3"
            style={{ borderBottom: '1px solid var(--border)' }}
          >
            <LinkIcon size={15} style={{ color: 'var(--text-3)', flexShrink: 0 }} />
            <input
              required
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com"
              style={{
                flex: 1,
                background: 'transparent',
                border: 'none',
                outline: 'none',
                color: 'var(--text)',
                fontSize: '0.875rem',
                padding: 0,
              }}
            />
          </div>

          {/* Textarea */}
          <div className="px-4 py-4">
            <textarea
              value={body}
              onChange={(e) => {
                setBody(e.target.value)
                if (jsonError) setJsonError(null)
              }}
              placeholder={selectedMode.placeholder}
              rows={mode === 'my_plan' ? 10 : 4}
              style={{
                width: '100%',
                minHeight: 120,
                resize: 'none',
                background: 'transparent',
                border: 'none',
                outline: 'none',
                color: 'var(--text)',
                fontSize: '0.875rem',
                lineHeight: '1.6',
                fontFamily: 'inherit',
              }}
            />
            {jsonError && (
              <p
                className="mt-1 flex items-center gap-1.5 text-xs"
                style={{ color: 'var(--rose)' }}
              >
                <AlertCircle size={12} />
                {jsonError}
              </p>
            )}
          </div>

          {/* Toolbar */}
          <div
            className="flex items-center justify-between gap-3"
            style={{ padding: '12px 16px', borderTop: '1px solid var(--border)' }}
          >
            {/* Mode picker */}
            <div className="relative" ref={dropdownRef}>
              <button
                type="button"
                onClick={() => setDropdownOpen((o) => !o)}
                className="btn-ghost flex items-center gap-2"
                style={{ padding: '6px 10px', fontSize: '0.75rem' }}
              >
                <selectedMode.icon size={14} />
                <span>{selectedMode.label}</span>
                <ChevronDown
                  size={12}
                  style={{
                    color: 'var(--text-3)',
                    transform: dropdownOpen ? 'rotate(180deg)' : 'none',
                    transition: 'transform 0.2s',
                  }}
                />
              </button>

              <AnimatePresence>
                {dropdownOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 6 }}
                    transition={{ duration: 0.15 }}
                    className="panel absolute bottom-full left-0 mb-2 w-72 overflow-hidden"
                    style={{ borderRadius: 14, padding: 4, zIndex: 50 }}
                  >
                    {MODES.map((m) => {
                      const MIcon = m.icon
                      const isActive = mode === m.id
                      return (
                        <button
                          key={m.id}
                          type="button"
                          onClick={() => selectMode(m.id)}
                          className="flex w-full items-start gap-3 rounded-xl px-3 py-2.5 text-left transition-colors"
                          style={{
                            background: isActive ? 'var(--primary-bg)' : 'transparent',
                            color: isActive ? 'var(--text)' : 'var(--text-2)',
                          }}
                          onMouseEnter={(e) => {
                            if (!isActive) (e.currentTarget as HTMLButtonElement).style.background = 'var(--bg-elevated)'
                          }}
                          onMouseLeave={(e) => {
                            if (!isActive) (e.currentTarget as HTMLButtonElement).style.background = 'transparent'
                          }}
                        >
                          <MIcon
                            size={16}
                            style={{
                              marginTop: 2,
                              flexShrink: 0,
                              color: isActive ? 'var(--primary-bright)' : 'var(--text-3)',
                            }}
                          />
                          <div className="min-w-0 flex-1">
                            <div className="text-xs font-semibold">{m.label}</div>
                            <div
                              className="text-[11px] leading-4"
                              style={{ color: 'var(--text-3)' }}
                            >
                              {m.description}
                            </div>
                          </div>
                          {isActive && (
                            <Check
                              size={13}
                              style={{ marginTop: 2, flexShrink: 0, color: 'var(--primary-bright)' }}
                            />
                          )}
                        </button>
                      )
                    })}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={submitting || (mode === 'my_plan' && !!jsonError)}
              className="btn-primary flex items-center gap-2"
              style={{ padding: '8px 18px', fontSize: '0.8125rem' }}
            >
              {submitting ? (
                <>
                  <Loader2 size={13} className="animate-spin" />
                  <span>Running…</span>
                </>
              ) : (
                <span>{selectedMode.submitLabel}</span>
              )}
            </button>
          </div>
        </form>

        {/* Global error */}
        {error && (
          <p
            className="mt-3 flex items-center gap-1.5 text-xs"
            style={{ color: 'var(--rose)' }}
          >
            <AlertCircle size={13} />
            {error}
          </p>
        )}
      </motion.div>

      {/* Feature tiles */}
      <div className="grid w-full gap-3 sm:grid-cols-3">
        {TILES.map((tile, i) => {
          const TIcon = tile.icon
          return (
            <motion.div
              key={tile.title}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, delay: 0.15 + i * 0.08, ease: 'easeOut' }}
              className="card p-4"
            >
              <TIcon size={18} style={{ color: tile.color, marginBottom: 8 }} />
              <div
                className="text-sm font-semibold"
                style={{ color: 'var(--text)' }}
              >
                {tile.title}
              </div>
              <div className="mt-1 text-xs" style={{ color: 'var(--text-3)' }}>
                {tile.description}
              </div>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}
