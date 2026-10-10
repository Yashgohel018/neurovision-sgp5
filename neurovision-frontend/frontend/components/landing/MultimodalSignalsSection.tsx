'use client'

import { useState } from 'react'
import { sliceUrl } from '@/lib/api'
import type { Modality } from '@/lib/types'
import { Info, Sparkles, CheckCircle2 } from 'lucide-react'

const SEQUENCES: {
  id: Modality
  name: string
  fullName: string
  role: string
  physics: string
  tumorAppearance: string
  accentColor: string
}[] = [
  {
    id: 'T1',
    name: 'T1-Weighted',
    fullName: 'Spin-Lattice Relaxation Sequence',
    role: 'Defines anatomical boundaries & baseline tissue density',
    physics: 'Short TR, short TE. High fat signal, dark water (CSF).',
    tumorAppearance: 'Isointense to hypointense. Outlines parenchymal deformation and necrotic cores.',
    accentColor: 'border-blue-400 text-blue-400',
  },
  {
    id: 'T1ce',
    name: 'T1ce / T1Gd',
    fullName: 'Post-Contrast Gadolinium Enhanced',
    role: 'Isolates active angiogenic rim & blood-brain barrier breakdown',
    physics: 'Gadolinium shortens T1 relaxation, causing bright hyperintensity in permeable capillaries.',
    tumorAppearance: 'Vivid peripheral nodular or ring enhancement distinguishing active tumor core (ET).',
    accentColor: 'border-[#ff7a45] text-[#ff7a45]',
  },
  {
    id: 'T2',
    name: 'T2-Weighted',
    fullName: 'Spin-Spin Relaxation Sequence',
    role: 'Sensitive to abnormal free water & microstructural disruption',
    physics: 'Long TR, long TE. Water and CSF appear brightly hyperintense.',
    tumorAppearance: 'Broad hyperintense region capturing both infiltrating neoplasm and cellular swelling.',
    accentColor: 'border-cyan-400 text-cyan-400',
  },
  {
    id: 'FLAIR',
    name: 'T2-FLAIR',
    fullName: 'Fluid-Attenuated Inversion Recovery',
    role: 'Attenuates ventricular CSF to unmask peritumoral vasogenic edema',
    physics: '180° inversion pulse timed to null the signal of pure cerebrospinal fluid.',
    tumorAppearance: 'Bright perilesional halo cleanly isolating Whole Tumor (WT) edema from ventricular fluid.',
    accentColor: 'border-emerald-400 text-emerald-400',
  },
]

export default function MultimodalSignalsSection() {
  const [selectedModality, setSelectedModality] = useState<Modality>('T1ce')
  const active = SEQUENCES.find(s => s.id === selectedModality)!

  return (
    <section id="multimodal" className="border-t border-nv-line bg-[#03060d] py-24 px-6">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="max-w-2xl space-y-3">
          <div className="text-xs uppercase tracking-widest text-nv-cool font-semibold">
            01 · Multimodal MRI Synthesis
          </div>
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-5xl">
            One Scan.
            <span className="block text-zinc-400">Thousands of Signals.</span>
          </h2>
          <p className="text-base text-zinc-400 leading-relaxed pt-2">
            Gliomas are heterogeneous. A single pulse sequence cannot distinguish necrotic core from vasogenic
            edema. NeuroVision AI processes four co-registered contrasts simultaneously, deriving unified spatial intelligence.
          </p>
        </div>

        {/* 4 Modalities Interactive Grid */}
        <div className="mt-14 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {SEQUENCES.map(seq => {
            const isSelected = seq.id === selectedModality
            const previewUrl = sliceUrl('CASE-2026-001', seq.id, 'axial', 60)

            return (
              <div
                key={seq.id}
                onClick={() => setSelectedModality(seq.id)}
                className={`group cursor-pointer rounded-2xl border transition-all duration-300 p-4 bg-nv-panel/60 backdrop-blur ${
                  isSelected
                    ? 'border-nv-cool shadow-[0_0_20px_rgba(77,155,255,0.25)] bg-[#070e1c]'
                    : 'border-nv-line hover:border-white/30'
                }`}
              >
                {/* Live Slice Thumbnail */}
                <div className="relative aspect-square w-full overflow-hidden rounded-xl bg-black border border-white/5">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={previewUrl}
                    alt={`${seq.name} preview`}
                    loading="lazy"
                    decoding="async"
                    className="h-full w-full object-contain transition-transform duration-300 group-hover:scale-105"
                  />
                  <div className="absolute left-2.5 top-2.5 rounded bg-black/70 px-2 py-0.5 text-[11px] font-mono font-semibold text-white backdrop-blur">
                    {seq.id}
                  </div>
                  {isSelected && (
                    <div className="absolute right-2.5 top-2.5">
                      <span className="flex h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_8px_#38bdf8]" />
                    </div>
                  )}
                </div>

                {/* Details */}
                <div className="mt-4 space-y-1.5">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-white">{seq.name}</span>
                    <span className="text-[10px] font-mono text-zinc-500 uppercase">
                      {seq.id === 'T1ce' ? 'Contrast' : 'Standard'}
                    </span>
                  </div>
                  <p className="text-xs text-zinc-400 leading-snug line-clamp-2">
                    {seq.role}
                  </p>
                </div>
              </div>
            )
          })}
        </div>

        {/* Detailed Sequence Inspection Panel */}
        <div className="mt-8 rounded-3xl border border-nv-line bg-[#060a14] p-8">
          <div className="grid gap-8 lg:grid-cols-[1fr_360px] items-center">
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <span className={`rounded-full border px-3 py-0.5 text-xs font-mono font-semibold ${active.accentColor}`}>
                  {active.id} SEQUENCE DEEP-DIVE
                </span>
                <span className="text-sm text-zinc-400">{active.fullName}</span>
              </div>

              <h3 className="text-2xl font-bold text-white">
                {active.role}
              </h3>

              <div className="grid gap-4 sm:grid-cols-2 pt-2 text-xs">
                <div className="rounded-xl border border-nv-line/70 bg-black/30 p-4 space-y-1">
                  <div className="text-zinc-500 font-mono uppercase tracking-wider text-[10px]">
                    MR Physics & Weighting
                  </div>
                  <p className="text-zinc-300 leading-relaxed">
                    {active.physics}
                  </p>
                </div>

                <div className="rounded-xl border border-nv-line/70 bg-black/30 p-4 space-y-1">
                  <div className="text-zinc-500 font-mono uppercase tracking-wider text-[10px]">
                    Tumor & Edema Presentation
                  </div>
                  <p className="text-zinc-300 leading-relaxed">
                    {active.tumorAppearance}
                  </p>
                </div>
              </div>
            </div>

            {/* Complementary Multi-Modal Matrix */}
            <div className="rounded-2xl border border-nv-line bg-black/40 p-5 space-y-3">
              <div className="text-xs uppercase tracking-wider text-zinc-400 font-semibold flex items-center gap-2">
                <Sparkles className="h-3.5 w-3.5 text-nv-cool" /> BraTS Multi-Sequence Role
              </div>
              <ul className="space-y-2 text-xs text-zinc-400">
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 mt-0.5 flex-shrink-0" />
                  <span><strong>T1ce:</strong> Crucial for identifying <em>Enhancing Tumor (ET)</em></span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 mt-0.5 flex-shrink-0" />
                  <span><strong>FLAIR:</strong> Vital for delineating <em>Whole Tumor (WT)</em> margin</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 mt-0.5 flex-shrink-0" />
                  <span><strong>T1 & T2:</strong> Grounding anatomical reference & necrotic core (TC)</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
