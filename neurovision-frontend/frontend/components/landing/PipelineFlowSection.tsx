'use client'

import { useState } from 'react'
import {
  UploadCloud,
  Cpu,
  Layers,
  BarChart3,
  BrainCircuit,
  FileCheck2,
  ChevronRight,
  ArrowRight,
} from 'lucide-react'

const STAGES = [
  {
    step: '01',
    name: 'MRI Input',
    sub: '4 Modalities',
    icon: UploadCloud,
    color: 'from-blue-500/20 to-blue-500/5 text-blue-400 border-blue-500/30',
    desc: '4D Co-registered NIfTI series (.nii / .nii.gz) for T1, T1ce, T2, and FLAIR protocols.',
    tech: 'NiBabel 5.4 · Shape (240, 240, 155) · 1mm isotropic',
  },
  {
    step: '02',
    name: 'Preprocessing',
    sub: 'Spatial & Intensity',
    icon: Cpu,
    color: 'from-cyan-500/20 to-cyan-500/5 text-cyan-400 border-cyan-500/30',
    desc: 'Orientation alignment (RAS+), skull stripping, non-zero voxel masking, and Z-score intensity normalization.',
    tech: 'Affine Preservation · Zero-mean Unit-variance',
  },
  {
    step: '03',
    name: 'Segmentation',
    sub: 'Sub-region Masking',
    icon: Layers,
    color: 'from-emerald-500/20 to-emerald-500/5 text-emerald-400 border-emerald-500/30',
    desc: 'Multi-label voxel classification partitioning Enhancing Tumor (ET), Tumor Core (TC), and Whole Tumor (WT).',
    tech: 'BraTS 2021 convention · Discrete Label Masks (1, 2, 4)',
  },
  {
    step: '04',
    name: 'Volumetry',
    sub: 'Quantitative Metrics',
    icon: BarChart3,
    color: 'from-amber-500/20 to-amber-500/5 text-amber-400 border-amber-500/30',
    desc: 'Voxel count product integrated over physical voxel spacing dimensions, returning accurate volume in cm³.',
    tech: 'V = N_vox × (dx × dy × dz) / 1000 · No synthetic values',
  },
  {
    step: '05',
    name: '3D AI Encoder',
    sub: 'ResNet3D Backbone',
    icon: BrainCircuit,
    color: 'from-orange-500/20 to-orange-500/5 text-[#ff7a45] border-orange-500/30',
    desc: 'Volumetric feature extraction, 3D convolutions with residual skips, and Grad-CAM attention layer hooks.',
    tech: 'PyTorch · 3D Bottleneck Blocks · Softmax Distribution',
  },
  {
    step: '06',
    name: 'Clinical Intelligence',
    sub: 'Decision Support',
    icon: FileCheck2,
    color: 'from-purple-500/20 to-purple-500/5 text-purple-400 border-purple-500/30',
    desc: 'Structured clinical report generation, RAG evidence retrieval, and calibrated decision-support summaries.',
    tech: 'Research Prototype · WHO CNS 5 Taxonomy · Audit Trail',
  },
]

export default function PipelineFlowSection() {
  const [activeStage, setActiveStage] = useState(0)

  return (
    <section id="pipeline" className="border-t border-nv-line bg-[#03060d] py-24 px-6">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="max-w-2xl space-y-3">
          <div className="text-xs uppercase tracking-widest text-nv-cool font-semibold">
            03 · End-to-End Processing Architecture
          </div>
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-5xl">
            From Voxels to Intelligence.
          </h2>
          <p className="text-base text-zinc-400 leading-relaxed pt-1">
            Every step in the NeuroVision AI workflow maintains strict physical spatial geometry,
            ensuring that clinical measurements and neural representations are transparent and reproducible.
          </p>
        </div>

        {/* Pipeline Step Cards Grid */}
        <div className="mt-14 grid gap-4 sm:grid-cols-2 lg:grid-cols-6">
          {STAGES.map((st, i) => {
            const Icon = st.icon
            const isActive = i === activeStage

            return (
              <div
                key={st.step}
                onClick={() => setActiveStage(i)}
                className={`group relative cursor-pointer rounded-2xl border p-5 transition-all duration-300 bg-gradient-to-b ${st.color} ${
                  isActive
                    ? 'ring-2 ring-white/30 shadow-[0_0_24px_rgba(77,155,255,0.2)] scale-[1.02]'
                    : 'opacity-85 hover:opacity-100 hover:scale-[1.01]'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-zinc-400">
                    {st.step}
                  </span>
                  <div className="rounded-lg bg-black/40 p-2 backdrop-blur">
                    <Icon className="h-4 w-4" />
                  </div>
                </div>

                <div className="mt-4">
                  <div className="text-sm font-semibold text-white">{st.name}</div>
                  <div className="text-[11px] text-zinc-400 font-mono mt-0.5">{st.sub}</div>
                </div>

                <div className="mt-3 text-xs text-zinc-400 line-clamp-3 leading-snug">
                  {st.desc}
                </div>

                {/* Arrow to next item on large screens */}
                {i < STAGES.length - 1 && (
                  <div className="hidden lg:block absolute -right-3 top-1/2 -translate-y-1/2 z-10 text-zinc-600">
                    <ChevronRight className="h-5 w-5" />
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {/* Selected Stage Deep-Dive Banner */}
        <div className="mt-8 rounded-3xl border border-nv-line bg-[#070c17] p-6 sm:p-8">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="rounded-md bg-white/10 px-2 py-0.5 text-xs font-mono text-zinc-300">
                  STAGE {STAGES[activeStage].step}
                </span>
                <span className="text-xl font-bold text-white">
                  {STAGES[activeStage].name}
                </span>
              </div>
              <p className="text-sm text-zinc-300 leading-relaxed max-w-2xl">
                {STAGES[activeStage].desc}
              </p>
            </div>

            <div className="rounded-xl border border-nv-line bg-black/50 px-5 py-3 text-xs font-mono text-cyan-300 backdrop-blur flex-shrink-0">
              <span className="text-zinc-500 block text-[10px] uppercase tracking-wider mb-1">
                Technical Specification
              </span>
              {STAGES[activeStage].tech}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
