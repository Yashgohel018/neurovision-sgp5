'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { api } from '@/lib/api'
import { Activity, ArrowUpRight } from 'lucide-react'

export default function LandingNavbar() {
  const [online, setOnline] = useState<boolean | null>(null)

  useEffect(() => {
    let active = true
    const check = async () => {
      try {
        const res = await api.health()
        if (active) setOnline(res.status === 'healthy')
      } catch {
        if (active) setOnline(false)
      }
    }
    check()
    const interval = setInterval(check, 10000)
    return () => {
      active = false
      clearInterval(interval)
    }
  }, [])

  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-[#03060d]/80 px-6 py-3.5 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between">
        {/* Logo and Brand */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-tr from-nv-cool to-cyan-300 shadow-[0_0_16px_rgba(77,155,255,0.4)]">
            <Activity className="h-4 w-4 text-black" />
          </div>
          <div>
            <span className="text-base font-semibold tracking-tight text-white group-hover:text-cyan-200 transition-colors">
              NeuroVision AI
            </span>
            <span className="hidden sm:inline-block ml-2 text-[10px] tracking-widest uppercase font-mono px-1.5 py-0.5 rounded border border-white/10 text-zinc-400">
              Decision Support
            </span>
          </div>
        </Link>

        {/* Navigation Links */}
        <nav className="hidden md:flex items-center gap-7 text-xs font-medium text-zinc-400">
          <a href="#multimodal" className="hover:text-white transition-colors">
            Multimodal MRI
          </a>
          <a href="#viewer" className="hover:text-white transition-colors">
            MRI Viewer
          </a>
          <a href="#pipeline" className="hover:text-white transition-colors">
            Pipeline
          </a>
          <a href="#3d" className="hover:text-white transition-colors">
            3D Tumor
          </a>
          <a href="#volumetry" className="hover:text-white transition-colors">
            Volumetry
          </a>
          <a href="#assistant" className="hover:text-white transition-colors">
            Assistant
          </a>
          <Link href="/cases" className="hover:text-white transition-colors">
            Cases
          </Link>
          <Link href="/models" className="hover:text-white transition-colors">
            Models
          </Link>
        </nav>

        {/* Right CTA and Live Health Status */}
        <div className="flex items-center gap-4">
          {/* Health Badge */}
          <div
            className={`hidden sm:flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-mono tracking-wide ${
              online === true
                ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                : online === false
                ? 'border-amber-500/40 bg-amber-500/10 text-amber-300'
                : 'border-zinc-700 bg-zinc-800 text-zinc-400'
            }`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                online === true
                  ? 'bg-emerald-400 shadow-[0_0_8px_#34d399]'
                  : online === false
                  ? 'bg-amber-400 shadow-[0_0_8px_#fbbf24]'
                  : 'bg-zinc-500'
              }`}
            />
            {online === true ? 'System Online' : online === false ? 'Backend Offline' : 'Checking…'}
          </div>

          <Link
            href="/dashboard"
            className="flex items-center gap-1.5 rounded-full border border-white/20 bg-white/10 px-4 py-1.5 text-xs font-medium text-white shadow-lg backdrop-blur hover:bg-white hover:text-black transition-all"
          >
            <span>Workspace</span>
            <ArrowUpRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      </div>
    </header>
  )
}
