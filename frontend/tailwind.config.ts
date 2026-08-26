import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        void: '#06070d',
        base: '#0b0d18',
        surface: {
          DEFAULT: '#111428',
          2: '#171b31',
        },
        elevated: '#1e2239',
        highlight: '#252a46',
        primary: {
          DEFAULT: '#6366f1',
          hover: '#818cf8',
          dim: 'rgba(99,102,241,0.12)',
        },
        status: {
          pass: '#10b981',
          fail: '#f43f5e',
          warn: '#f59e0b',
          run: '#38bdf8',
          block: '#ef4444',
          queue: '#64748b',
          wait: '#f59e0b',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      borderRadius: {
        card: '12px',
        panel: '16px',
        badge: '6px',
      },
      boxShadow: {
        card: '0 1px 3px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.06)',
        'card-hover': '0 8px 32px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.10)',
        'glow-indigo': '0 0 24px rgba(99,102,241,0.35), 0 0 0 1px rgba(99,102,241,0.25)',
        'glow-cyan': '0 0 20px rgba(34,211,238,0.30)',
        'glow-emerald': '0 0 20px rgba(16,185,129,0.30)',
        'glow-rose': '0 0 20px rgba(244,63,94,0.30)',
        'glow-amber': '0 0 16px rgba(245,158,11,0.30)',
        'inner-subtle': 'inset 0 1px 0 rgba(255,255,255,0.06)',
        panel: '0 24px 80px rgba(0,0,0,0.8), 0 0 0 1px rgba(255,255,255,0.08)',
      },
      keyframes: {
        'rise-in': {
          '0%': { opacity: '0', transform: 'translateY(16px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'card-in': {
          '0%': { opacity: '0', transform: 'translateY(8px) scale(0.98)' },
          '100%': { opacity: '1', transform: 'translateY(0) scale(1)' },
        },
        shimmer: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
        'pulse-dot': {
          '0%, 100%': { opacity: '0.35', transform: 'scale(1)' },
          '50%': { opacity: '1', transform: 'scale(1.2)' },
        },
        'pulse-ring': {
          '0%': { transform: 'scale(1)', opacity: '0.6' },
          '100%': { transform: 'scale(2.2)', opacity: '0' },
        },
        'glow-pulse': {
          '0%, 100%': { opacity: '0.5' },
          '50%': { opacity: '1' },
        },
        'scan-line': {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(200%)' },
        },
        orbit: {
          '0%': { transform: 'rotateZ(0deg) rotateX(65deg)' },
          '100%': { transform: 'rotateZ(360deg) rotateX(65deg)' },
        },
        'orbit-2': {
          '0%': { transform: 'rotateZ(0deg) rotateX(65deg) rotateY(20deg)' },
          '100%': { transform: 'rotateZ(-360deg) rotateX(65deg) rotateY(20deg)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-6px)' },
        },
        'spin-slow': {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
        'slide-in-right': {
          '0%': { transform: 'translateX(100%)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        'slide-in-left': {
          '0%': { transform: 'translateX(-100%)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        'waveform': {
          '0%, 100%': { scaleY: '0.3' },
          '50%': { scaleY: '1' },
        },
      },
      animation: {
        'rise-in': 'rise-in 0.6s ease-out forwards',
        'rise-in-delay': 'rise-in 0.6s 0.15s ease-out both',
        'fade-in': 'fade-in 0.4s ease-out forwards',
        'card-in': 'card-in 0.4s ease-out forwards',
        shimmer: 'shimmer 1.8s ease-in-out infinite',
        'pulse-dot': 'pulse-dot 2s ease-in-out infinite',
        'pulse-ring': 'pulse-ring 1.8s ease-out infinite',
        'glow-pulse': 'glow-pulse 2.4s ease-in-out infinite',
        'scan-line': 'scan-line 2s ease-in-out infinite',
        orbit: 'orbit 18s linear infinite',
        'orbit-2': 'orbit-2 12s linear infinite reverse',
        float: 'float 3.5s ease-in-out infinite',
        'spin-slow': 'spin-slow 10s linear infinite',
        'slide-in-right': 'slide-in-right 0.3s ease-out',
        'slide-in-left': 'slide-in-left 0.3s ease-out',
      },
      transitionTimingFunction: {
        smooth: 'cubic-bezier(0.16, 1, 0.3, 1)',
        bounce: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
      },
    },
  },
  plugins: [],
}

export default config
