'use client'
import { useEffect, useRef, useState } from 'react'
import { DemoBadge } from '@/components/ui'
interface P { src?: string; maskSrc?: string; maskOpacity: number; demo: boolean; index: number; label: string }
export default function SliceViewer({ src, maskSrc, maskOpacity, demo, index, label }: P) {
  const ref = useRef<HTMLCanvasElement>(null), [bad, setBad] = useState(false)
  useEffect(() => { setBad(false) }, [src])
  useEffect(() => { // synthetic DEMO slice only; real slices come from the backend
    if (!demo || !ref.current) return
    const c = ref.current.getContext('2d'); if (!c) return
    c.fillStyle = '#000'; c.fillRect(0, 0, 256, 256); const k = 1 - Math.abs(index - 60) / 120
    for (let r = 100; r > 10; r -= 6) { c.fillStyle = `hsl(215 12% ${18 + (100 - r) * 0.45}%)`; c.beginPath(); c.ellipse(128, 128, r * 0.85 * k + 8, r * k + 8, 0, 0, 7); c.fill() }
    c.fillStyle = 'rgba(255,122,69,.85)'; c.beginPath(); c.arc(150, 105, 16 * k + 4, 0, 7); c.fill()
  }, [demo, index])
  return <div className="relative mx-auto aspect-square w-full max-w-[620px] overflow-hidden rounded-2xl border border-nv-line bg-black">
    {demo ? <canvas ref={ref} width={256} height={256} className="h-full w-full" aria-label={`${label} synthetic demo slice`} />
      : bad || !src ? <p className="grid h-full place-items-center p-6 text-center text-sm text-zinc-400">Slice unavailable. The backend did not return an image.</p>
      : <>{/* eslint-disable-next-line @next/next/no-img-element */}<img src={src} alt={`${label} slice ${index}`} onError={() => setBad(true)} className="h-full w-full object-contain" />
        {maskSrc && /* eslint-disable-next-line @next/next/no-img-element */ <img src={maskSrc} alt="" style={{ opacity: maskOpacity }} className="absolute inset-0 h-full w-full object-contain mix-blend-screen" />}</>}
    <div className="absolute left-3 top-3 text-xs tracking-widest text-zinc-300">{label} · {index}</div>{demo && <div className="absolute right-3 top-3"><DemoBadge /></div>}</div>
}
