'use client'

import { useRef, useState, useEffect } from 'react'
import Link from 'next/link'
import { ArrowRight, Play, Sparkles, Activity, ShieldCheck, ChevronDown } from 'lucide-react'

export default function HeroSection() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [videoLoaded, setVideoLoaded] = useState(false)
  const [videoError, setVideoError] = useState(false)

  useEffect(() => {
    const v = videoRef.current
    if (!v) return

    const markLoaded = () => {
      setVideoLoaded(true)
      setVideoError(false)
    }

    if (v.readyState >= 2) {
      markLoaded()
    }

    v.addEventListener('loadeddata', markLoaded)
    v.addEventListener('canplay', markLoaded)
    v.addEventListener('playing', markLoaded)
    v.addEventListener('timeupdate', markLoaded)

    v.play().catch(() => {
      v.muted = true
      v.play().catch(() => {})
    })

    return () => {
      v.removeEventListener('loadeddata', markLoaded)
      v.removeEventListener('canplay', markLoaded)
      v.removeEventListener('playing', markLoaded)
      v.removeEventListener('timeupdate', markLoaded)
    }
  }, [])

  return (
    <section className="relative min-h-[92vh] w-full overflow-hidden bg-[#03060d] flex items-center">
      {/* Background Video Layer (Desktop 16:9 cinematic integration) */}
      <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
        <video
          ref={videoRef}
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
          onError={() => setVideoError(true)}
          className={`h-full w-full object-cover object-center transition-opacity duration-700 ${
            videoLoaded ? 'opacity-90' : 'opacity-0'
          }`}
        >
          <source src="/video/neurovision-hero.mp4" type="video/mp4" />
          <source src="/gemini_generated_video_db98719c.mp4" type="video/mp4" />
        </video>

        {/* Cinematic Multi-stop Vignette Gradient (keeps left side dark for typography, right side showcases video) */}
        <div className="absolute inset-0 bg-gradient-to-r from-[#03060d] via-[#03060d]/70 via-50% to-transparent z-10" />
        <div className="absolute inset-0 bg-gradient-to-t from-[#03060d] via-transparent to-[#03060d]/50 z-10" />

        {/* Fallback procedural canvas glow if video loading or error */}
        {(!videoLoaded || videoError) && (
          <div className="absolute inset-0 flex items-center justify-end pr-16 bg-radial from-cyan-950/20 to-[#03060d] z-0">
            <div className="h-[480px] w-[480px] rounded-full bg-cyan-500/15 blur-3xl animate-pulse" />
          </div>
        )}
      </div>

      {/* Hero Foreground Content */}
      <div className="relative z-20 mx-auto max-w-7xl px-6 py-20 w-full">
        <div className="max-w-2xl space-y-8">
          {/* Tagline Badge */}
          <div className="inline-flex items-center gap-2 rounded-full border border-nv-cool/30 bg-nv-cool/10 px-3.5 py-1 text-xs font-medium text-cyan-300 backdrop-blur-md">
            <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
            <span>Agentic Multimodal MRI Decision Support</span>
          </div>

          {/* Master Headings */}
          <div className="space-y-3">
            <h1 className="text-5xl font-extrabold tracking-tight text-white sm:text-6xl lg:text-7xl leading-[1.05]">
              See the Brain.
              <span className="block text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-sky-300 to-[#ff7a45]">
                Understand the Tumor.
              </span>
            </h1>
            <p className="text-lg text-zinc-300 sm:text-xl font-normal leading-relaxed pt-2 max-w-xl">
              AI-powered multimodal MRI analysis for smarter clinical decision support.
              Synchronized 3D volumetry, BraTS sub-region segmentation, and explainable neural attention.
            </p>
          </div>

          {/* Action CTAs */}
          <div className="flex flex-wrap items-center gap-4 pt-2">
            <a
              href="#viewer"
              className="flex items-center gap-2.5 rounded-full bg-white px-7 py-3.5 text-sm font-semibold text-black shadow-[0_0_24px_rgba(255,255,255,0.25)] transition-all hover:bg-zinc-200 hover:scale-[1.02]"
            >
              <span>Explore NeuroVision</span>
              <ArrowRight className="h-4 w-4" />
            </a>

            <Link
              href="/analysis/CASE-2026-001"
              className="flex items-center gap-2 rounded-full border border-white/20 bg-black/40 px-6 py-3.5 text-sm font-medium text-white backdrop-blur-md transition-all hover:bg-white/10 hover:border-white/40"
            >
              <Activity className="h-4 w-4 text-nv-cool" />
              <span>View Analysis</span>
            </Link>

            <Link
              href="/dashboard"
              className="text-xs text-zinc-400 hover:text-white px-3 py-2 transition-colors"
            >
              Launch Workspace →
            </Link>
          </div>

          {/* Key Pipeline Badges */}
          <div className="grid grid-cols-3 gap-4 border-t border-white/10 pt-8 max-w-lg">
            <div>
              <div className="text-xs uppercase tracking-widest text-zinc-500 font-medium">Modalities</div>
              <div className="mt-1 font-mono text-sm font-semibold text-zinc-200">T1 · T1ce · T2 · FLAIR</div>
            </div>
            <div>
              <div className="text-xs uppercase tracking-widest text-zinc-500 font-medium">Volumetry</div>
              <div className="mt-1 font-mono text-sm font-semibold text-zinc-200">WT · TC · ET (cm³)</div>
            </div>
            <div>
              <div className="text-xs uppercase tracking-widest text-zinc-500 font-medium">Explainability</div>
              <div className="mt-1 font-mono text-sm font-semibold text-zinc-200">3D Grad-CAM</div>
            </div>
          </div>
        </div>
      </div>

      {/* Subtle Scroll Indicator */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-20 flex flex-col items-center gap-2 text-zinc-500 animate-bounce">
        <span className="text-[11px] font-mono tracking-widest uppercase text-zinc-400">
          Scroll to Explore
        </span>
        <ChevronDown className="h-4 w-4 text-zinc-400" />
      </div>
    </section>
  )
}
