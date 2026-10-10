'use client'

import { useState } from 'react'
import { sliceUrl } from '@/lib/api'
import { Sparkles, Sliders, ShieldAlert, Eye, CheckCircle2 } from 'lucide-react'

export default function ExplainableAISection() {
  const [camOpacity, setCamOpacity] = useState(0.7)
  const [showHeatmap, setShowHeatmap] = useState(true)
  const [sliceIdx, setSliceIdx] = useState(60)

  const mriUrl = sliceUrl('CASE-2026-001', 'FLAIR', 'axial', sliceIdx)
  const camOverlayUrl = sliceUrl('CASE-2026-001', 'FLAIR', 'axial', sliceIdx, 'cam')

  return (
    <section className="border-t border-nv-line bg-[#03060d] py-24 px-6">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="max-w-2xl space-y-3">
          <div className="text-xs uppercase tracking-widest text-nv-cool font-semibold flex items-center gap-1.5">
            <Sparkles className="h-3.5 w-3.5" /> 07 · Explainable AI Architecture
          </div>
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-5xl">
            Why This Prediction?
            <span className="block text-zinc-400">3D Grad-CAM Attention Heatmaps.</span>
          </h2>
          <p className="text-base text-zinc-400 leading-relaxed pt-1">
            Black-box neural networks are unacceptable in neuro-oncology. NeuroVision AI incorporates
            Gradient-weighted Class Activation Mapping (Grad-CAM) to project layer activations back onto anatomical voxels,
            providing an auditable trail of model focus.
          </p>
        </div>

        {/* Explainability Interactive Workstation */}
        <div className="mt-14 rounded-3xl border border-nv-line bg-[#060a14] p-8">
          <div className="grid gap-8 lg:grid-cols-[1fr_360px] items-center">
            {/* Split Comparison Viewport */}
            <div className="flex flex-col items-center">
              <div className="relative aspect-square w-full max-w-[460px] overflow-hidden rounded-2xl border border-nv-line bg-black shadow-2xl">
                {/* Base MRI */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={showHeatmap ? camOverlayUrl : mriUrl}
                  alt="Grad-CAM visualization"
                  loading="lazy"
                  decoding="async"
                  className="h-full w-full object-contain"
                />

                <div className="absolute left-3 top-3 rounded bg-black/60 px-2 py-1 text-[11px] font-mono text-zinc-300 backdrop-blur">
                  FLAIR · AXIAL · SLICE {sliceIdx}
                </div>

                <div className="absolute right-3 top-3 rounded bg-orange-500/20 border border-orange-500/40 px-2 py-0.5 text-[10px] font-mono text-orange-300 backdrop-blur">
                  {showHeatmap ? 'Grad-CAM Blended' : 'Anatomy Only'}
                </div>
              </div>

              {/* Slice Slider */}
              <div className="mt-4 flex w-full max-w-[460px] items-center gap-3">
                <span className="text-xs text-zinc-400 font-mono">Slice:</span>
                <input
                  type="range"
                  min={20}
                  max={100}
                  value={sliceIdx}
                  onChange={e => setSliceIdx(Number(e.target.value))}
                  className="w-full accent-orange-500"
                />
                <span className="text-xs text-zinc-300 font-mono w-10 text-right">{sliceIdx}</span>
              </div>
            </div>

            {/* Controls and Methodology Panel */}
            <div className="space-y-6">
              <div className="space-y-2">
                <div className="text-xs uppercase tracking-widest text-zinc-500 font-semibold">
                  Attention Controls
                </div>
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => setShowHeatmap(!showHeatmap)}
                    className={`rounded-xl border px-4 py-2 text-xs font-medium transition-colors ${
                      showHeatmap
                        ? 'border-orange-500/50 bg-orange-500/20 text-orange-300'
                        : 'border-white/10 text-zinc-400 hover:text-white'
                    }`}
                  >
                    {showHeatmap ? 'Hide Activation Layer' : 'Show Activation Layer'}
                  </button>
                </div>
              </div>

              {/* Colormap Legend */}
              <div className="rounded-2xl border border-nv-line bg-black/40 p-5 space-y-3">
                <div className="text-xs font-semibold text-white">Activation Gradient (Jet Map)</div>
                <div className="h-3 w-full rounded-full bg-gradient-to-r from-blue-600 via-emerald-400 via-yellow-400 to-red-500 shadow-inner" />
                <div className="flex justify-between text-[10px] font-mono text-zinc-400">
                  <span>0.0 (Suppressed)</span>
                  <span>0.5 (Intermediate)</span>
                  <span>1.0 (High Saliency)</span>
                </div>
              </div>

              {/* Research Disclaimer Box */}
              <div className="rounded-2xl border border-amber-500/30 bg-amber-500/5 p-4 text-xs text-zinc-300 space-y-1.5">
                <div className="flex items-center gap-1.5 text-amber-400 font-semibold">
                  <ShieldAlert className="h-4 w-4" /> Explainability Disclaimer
                </div>
                <p className="leading-relaxed">
                  Grad-CAM maps depict internal gradient flow and computational saliency.
                  They represent mathematical network activations, <strong>not certified clinical or pathology findings</strong>.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
