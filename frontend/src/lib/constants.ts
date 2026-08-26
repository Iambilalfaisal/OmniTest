import type { TestCategory, WorkerPhase, SessionStatus, SessionKind, FeaturePhase } from './types'

// ─── API base ─────────────────────────────────────────────────────────────────

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000'

// ─── Category styles ──────────────────────────────────────────────────────────

export const CATEGORY_STYLES: Record<string, { badge: string; dot: string; label: string }> = {
  functional:    { badge: 'badge-primary',  dot: 'bg-indigo-400',  label: 'Functional' },
  visual:        { badge: 'badge-violet',   dot: 'bg-violet-400',  label: 'Visual' },
  performance:   { badge: 'badge-cyan',     dot: 'bg-cyan-400',    label: 'Performance' },
  security:      { badge: 'badge-fail',     dot: 'bg-rose-400',    label: 'Security' },
  accessibility: { badge: 'badge-waiting',  dot: 'bg-amber-400',   label: 'Accessibility' },
  integration:   { badge: 'badge-running',  dot: 'bg-sky-400',     label: 'Integration' },
  regression:    { badge: 'badge-queued',   dot: 'bg-slate-400',   label: 'Regression' },
  usability:     { badge: 'badge-violet',   dot: 'bg-violet-400',  label: 'Usability' },
  smoke:         { badge: 'badge-pass',     dot: 'bg-emerald-400', label: 'Smoke' },
  exploratory:   { badge: 'badge-cyan',     dot: 'bg-cyan-400',    label: 'Exploratory' },
  // Legacy TestCategory values from WorkerCard
  happy_path:       { badge: 'badge-running', dot: 'bg-sky-400',    label: 'Happy path' },
  edge_case:        { badge: 'badge-violet',  dot: 'bg-violet-400', label: 'Edge case' },
  negative:         { badge: 'badge-waiting', dot: 'bg-amber-400',  label: 'Negative' },
  error_handling:   { badge: 'badge-waiting', dot: 'bg-amber-400',  label: 'Error handling' },
  state_interaction:{ badge: 'badge-primary', dot: 'bg-indigo-400', label: 'State interaction' },
}

export const CATEGORY_LABELS: Record<TestCategory, string> = {
  happy_path:        'Happy path',
  edge_case:         'Edge case',
  negative:          'Negative',
  error_handling:    'Error handling',
  security:          'Security',
  state_interaction: 'State interaction',
}

// ─── Worker / test phase styles ───────────────────────────────────────────────

export const WORKER_PHASE_LABELS: Record<string, { label: string; badge: string }> = {
  queued:          { label: 'Queued',          badge: 'badge-queued'  },
  running:         { label: 'Running',         badge: 'badge-running' },
  awaiting_input:  { label: 'Awaiting Input',  badge: 'badge-waiting' },
  grading:         { label: 'Grading',         badge: 'badge-running' },
  done:            { label: 'Done',            badge: 'badge-pass'    },
  rediscovering:   { label: 'Re-observing',    badge: 'badge-running' },
  replanning:      { label: 'Replanning',      badge: 'badge-violet'  },
  passed:          { label: 'Passed',          badge: 'badge-pass'    },
  failed:          { label: 'Failed',          badge: 'badge-fail'    },
  blocked:         { label: 'Blocked',         badge: 'badge-blocked' },
  error:           { label: 'Error',           badge: 'badge-fail'    },
  skipped:         { label: 'Skipped',         badge: 'badge-queued'  },
}

// ─── Feature phase labels ─────────────────────────────────────────────────────

export const FEATURE_PHASE_LABELS: Record<FeaturePhase, string> = {
  exploring: 'Analyzing application…',
  done:      'Discovery complete',
}

// ─── Session status styles ────────────────────────────────────────────────────

export const SESSION_STATUS_STYLES: Record<SessionStatus, string> = {
  running:     'badge-running',
  in_progress: 'badge-running',
  paused:      'badge-waiting',
  done:        'badge-pass',
  approved:    'badge-pass',
  error:       'badge-fail',
  cancelled:   'badge-queued',
}

export const SESSION_STATUS_LABELS: Record<SessionStatus, string> = {
  running:     'Running',
  in_progress: 'In progress',
  paused:      'Paused',
  done:        'Done',
  approved:    'Approved',
  error:       'Error',
  cancelled:   'Cancelled',
}

// ─── Session kind styles ──────────────────────────────────────────────────────

export const KIND_STYLES: Record<SessionKind, string> = {
  run:       'badge-primary',
  discovery: 'badge-violet',
}

export const KIND_LABELS: Record<SessionKind, string> = {
  run:       'Run',
  discovery: 'Discovery',
}

// ─── Risky action verb map ────────────────────────────────────────────────────

export const RISKY_ACTION_VERBS: Record<string, string> = {
  browser_click:         'click',
  browser_type:          'type into',
  browser_fill_form:     'fill out',
  browser_select_option: 'choose an option in',
  browser_press_key:     'press a key on',
  browser_drag:          'drag',
  browser_hover:         'hover over',
  browser_file_upload:   'upload a file to',
  browser_navigate:      'go to',
}
