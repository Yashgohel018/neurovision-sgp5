'use client'

import InteractiveWorkstation from '@/components/analysis/InteractiveWorkstation'
import { Sparkles, Eye, ShieldCheck } from 'lucide-react'

export default function InteractiveMRIViewerSection() {
  return (
    <section id="viewer" className="border-t border-nv-line bg-[#02050b] py-24 px-6">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-12">
          <div className="max-w-2xl space-y-3">
            <div className="text-xs uppercase tracking-widest text-nv-cool font-semibold flex items-center gap-1.5">
              <Eye className="h-3.5 w-3.5" /> 02 · Interactive Multi-Planar Viewer
            </div>
            <h2 className="text-3xl font-bold tracking-tight text-white sm:text-5xl">
              Precision MRI Workstation.
              <span className="block text-zinc-400">Server-Side NIfTI Streaming.</span>
            </h2>
            <p className="text-base text-zinc-400 leading-relaxed pt-1">
              Navigate synchronized orthogonal slices (Axial, Coronal, Sagittal) across all four modalities.
              Inspect sub-millimeter tissue contrast with dynamic window/level adjustments and sub-region overlays.
            </p>
          </div>

          <div className="flex items-center gap-2 rounded-2xl border border-white/10 bg-nv-panel/50 p-3 text-xs text-zinc-400 backdrop-blur">
            <ShieldCheck className="h-4 w-4 text-emerald-400 flex-shrink-0" />
            <span>Actual NIfTI Volumetric Pixel Data Streamed on Demand</span>
          </div>
        </div>

        {/* Embedded Workstation */}
        <InteractiveWorkstation
          caseId="CASE-2026-001"
          initialModality="FLAIR"
          initialPlane="axial"
          initialIndex={60}
          className="shadow-[0_0_60px_rgba(0,0,0,0.8)]"
        />
      </div>
    </section>
  )
}
