declare module 'framer-motion' {
  import type { ComponentType, ReactNode, CSSProperties, ElementType, Ref } from 'react'

  type MotionValue = unknown

  interface AnimationProps {
    initial?: Record<string, unknown> | boolean
    animate?: Record<string, unknown>
    exit?: Record<string, unknown>
    transition?: Record<string, unknown>
    variants?: Record<string, unknown>
    whileHover?: Record<string, unknown>
    whileTap?: Record<string, unknown>
    whileFocus?: Record<string, unknown>
    layout?: boolean | string
  }

  type MotionComponentProps<T extends ElementType> = React.ComponentPropsWithRef<T> & AnimationProps

  type MotionComponent<T extends ElementType> = ComponentType<MotionComponentProps<T>>

  interface Motion {
    div: MotionComponent<'div'>
    aside: MotionComponent<'aside'>
    span: MotionComponent<'span'>
    p: MotionComponent<'p'>
    ul: MotionComponent<'ul'>
    li: MotionComponent<'li'>
    nav: MotionComponent<'nav'>
    header: MotionComponent<'header'>
    main: MotionComponent<'main'>
    section: MotionComponent<'section'>
    button: MotionComponent<'button'>
    a: MotionComponent<'a'>
    img: MotionComponent<'img'>
    h1: MotionComponent<'h1'>
    h2: MotionComponent<'h2'>
    h3: MotionComponent<'h3'>
  }

  export const motion: Motion

  export interface AnimatePresenceProps {
    children?: ReactNode
    initial?: boolean
    exitBeforeEnter?: boolean
    mode?: 'wait' | 'sync' | 'popLayout'
    onExitComplete?: () => void
  }

  export const AnimatePresence: ComponentType<AnimatePresenceProps>

  export function useMotionValue(initial: number): MotionValue
  export function useTransform(value: MotionValue, input: number[], output: unknown[]): MotionValue
  export function useSpring(value: MotionValue | number, config?: Record<string, unknown>): MotionValue
}
