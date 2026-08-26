// ─── shared model types ──────────────────────────────────────────────────────

export type TestCategory =
  | 'happy_path'
  | 'edge_case'
  | 'negative'
  | 'error_handling'
  | 'security'
  | 'state_interaction'

export type TestPriority = 'high' | 'medium' | 'low'

export type TestCase = {
  test_id: string
  goal: string
  category: TestCategory
  priority: TestPriority
  preconditions: string[]
  expected_result?: string
  steps: string[]
  feature_id?: string | null
  flow_id?: string | null
  origin?: 'planner' | 'recon'
  discovery_rationale?: string | null
}

export type Feature = {
  feature_id: string
  name: string
  description: string
}

export type FeaturePhase = 'exploring' | 'done'

export type FeatureProgress = {
  name: string
  phase: FeaturePhase
  scenario_count: number
  updated_at: number
}

export type TestResult = {
  test_id: string
  status: string
  screenshot_path: string
  trace_path?: string | null
  video_clips?: string[]
  reason: string
  deviations?: string[]
  amended_steps?: string[]
  last_step_reached?: number
}

export type WorkerPhase =
  | 'queued'
  | 'running'
  | 'awaiting_input'
  | 'grading'
  | 'done'
  | 'rediscovering'
  | 'replanning'

export type MutationEventType = 'deviation' | 'clarification' | 'risky_blocked'

export type MutationEvent = {
  type: MutationEventType
  step: number
  description: string
  user_decision: string | null
  sensitive: boolean
  timestamp: number
  resolved: boolean
}

export type PlanHistoryEntry = {
  version: number
  trigger: string
  original_steps: string[]
  new_steps: string[]
  reason: string
  replanned: boolean
}

export type WorkerProgress = {
  phase: WorkerPhase
  step_index: number
  total_steps: number
  current_action: string | null
  turn: number
  budget: number | null
  deviations: number
  asks: number
  mutation_events?: MutationEvent[]
  plan_version?: number
  plan_history?: PlanHistoryEntry[]
  updated_at: number
}

export type CardInterruptType = 'risky_action' | 'clarification'

export type CardInterrupt = {
  id: string
  type: CardInterruptType
  payload: {
    type: CardInterruptType
    test_id: string
    // risky_action fields
    tool?: string
    args?: Record<string, unknown>
    // clarification fields
    question?: string
    context?: string | null
    sensitive?: boolean
  }
}

export type CardDecision =
  | { approved: boolean; reason?: string }  // risky_action
  | { text: string }                         // clarification

export type SessionStatus =
  | 'running'
  | 'paused'
  | 'done'
  | 'error'
  | 'in_progress'
  | 'approved'
  | 'cancelled'

export type SessionKind = 'run' | 'discovery'

// ─── additional run-phase types ───────────────────────────────────────────────

export type RunPhase =
  | 'planning'
  | 'plan_review'
  | 'recon'
  | 'executing'
  | 'done'
  | 'error'
  | 'awaiting_input'

export type InterruptKind = 'plan_review' | 'risky_action' | 'clarification'
