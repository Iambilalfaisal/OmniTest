interface OrbitMarkProps { size?: 'sm' | 'md' | 'lg' }

const sizeMap = {
  sm: { wrapper: 'w-8 h-8',   core: 'w-2.5 h-2.5', ring1: 'w-7 h-7',            ring2: 'w-6 h-6',            sat: 'w-1 h-1' },
  md: { wrapper: 'w-12 h-12', core: 'w-4 h-4',      ring1: 'w-11 h-11',          ring2: 'w-9 h-9',            sat: 'w-1.5 h-1.5' },
  lg: { wrapper: 'w-20 h-20', core: 'w-6 h-6',      ring1: 'w-[76px] h-[76px]', ring2: 'w-[60px] h-[60px]', sat: 'w-2 h-2' },
}

export function OrbitMark({ size = 'md' }: OrbitMarkProps) {
  const s = sizeMap[size]
  return (
    <div className={`orbit-mark ${s.wrapper}`} aria-hidden="true">
      <span className={`orbit-mark__core ${s.core}`} />
      <span className={`orbit-mark__ring ${s.ring1}`} style={{ transform: 'rotateX(65deg)' }}>
        <span className={`orbit-mark__satellite ${s.sat}`} style={{ position: 'absolute', top: 0, left: '50%', transform: 'translateX(-50%)' }} />
      </span>
      <span className={`orbit-mark__ring orbit-mark__ring--two ${s.ring2}`} style={{ transform: 'rotateX(65deg) rotateY(20deg)' }} />
    </div>
  )
}

export default OrbitMark
