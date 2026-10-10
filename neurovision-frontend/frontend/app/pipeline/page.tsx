'use client'

import { useState } from 'react'
import Link from 'next/link'
import { api } from '@/lib/api'
import { useAsync } from '@/lib/hooks'
import { AppShell, ErrorState } from '@/components/ui'
import {
  Layers,
  CheckCircle2,
  FileText,
  Database,
  Sliders,
  Cpu,
  BarChart3,
  Eye,
  GitBranch,
  ArrowRight,
  ShieldCheck,
  Sparkles,
  FlaskConical
} from 'lucide-react'
import type { PipelineModuleInfo } from '@/lib/types'

export default function PipelinePage() {
  const { data: modules, error, reload } = useAsync(api.pipelineModules)
  const [selectedPhase, setSelectedPhase] = useState<string>('All')

  const phases = ['All', 'Data Ingestion', 'Data Preparation', 'Model Architecture', 'Model Training', 'Model Evaluation', 'Clinical Decision Support', 'Deployment & CDSS']

  const filteredModules = (modules ?? []).filter(
    m => selectedPhase === 'All' || m.phase === selectedPhase
  )

  return (
    <AppShell title="Scientific Pipeline & Module Architecture">
      {error ? (
        <ErrorState message={error} onRetry={reload} />
      ) : (
        <div className="space-y-8">
          {/* Header Banner */}
          <div className="rounded-3xl border border-nv-line/80 bg-gradient-to-r from-nv-panel via-nv-panel/80 to-sky-950/20 p-7 shadow-xl backdrop-blur-xl">
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
              <div className="space-y-2 max-w-3xl">
                <div className="inline-flex items-center gap-2 rounded-full border border-sky-400/30 bg-sky-500/10 px-3 py-1 text-xs font-semibold text-sky-300">
                  <FlaskConical className="h-3.5 w-3.5" />
                  <span>BraTS 2-Month Implementation Plan • MultiModal-BrainNet</span>
                </div>
                <h2 className="text-2xl font-bold tracking-tight text-white">
                  End-to-End Progression Across Modules A through L
                </h2>
                <p className="text-sm text-zinc-300 leading-relaxed">
                  NeuroVision AI unifies raw medical NIfTI imaging, preprocessing, 3D tumor ROI localization, sequence data augmentation, 
                  multimodal cross-attention deep learning, Grad-CAM explainability, and WHO CNS5 clinical decision support into one coherent ecosystem.
                </p>
              </div>

              <div className="flex flex-col sm:flex-row gap-3 w-full md:w-auto">
                <Link
                  href="/cases"
                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-sky-500 px-4 py-2.5 text-xs font-semibold text-white shadow-lg shadow-sky-500/20 hover:bg-sky-400 transition"
                >
                  <span>Explore Cohort (369 Cases)</span>
                  <ArrowRight className="h-3.5 w-3.5" />
                </Link>
                <Link
                  href="/models"
                  className="inline-flex items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-xs font-semibold text-zinc-200 hover:bg-white/10 transition"
                >
                  <BarChart3 className="h-3.5 w-3.5 text-sky-400" />
                  <span>Benchmark Zoo</span>
                </Link>
              </div>
            </div>
          </div>

          {/* Phase Filter Tabs */}
          <div className="flex items-center gap-2 overflow-x-auto pb-2 border-b border-nv-line/60">
            <span className="text-xs text-zinc-400 font-mono uppercase mr-2">Filter Phase:</span>
            {phases.map(phase => (
              <button
                key={phase}
                onClick={() => setSelectedPhase(phase)}
                className={`rounded-xl px-3.5 py-1.5 text-xs font-semibold whitespace-nowrap transition-all border ${
                  selectedPhase === phase
                    ? 'border-sky-400 bg-sky-500/20 text-sky-200 shadow-sm'
                    : 'border-white/5 bg-white/[0.02] text-zinc-400 hover:text-white hover:bg-white/5'
                }`}
              >
                {phase}
              </button>
            ))}
          </div>

          {/* Module Grid */}
          <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-3">
            {filteredModules.map(mod => (
              <div
                key={mod.id}
                className="group relative rounded-2xl border border-nv-line/80 bg-nv-panel/70 p-6 backdrop-blur shadow-lg hover:border-sky-500/40 transition-all flex flex-col justify-between"
              >
                <div className="space-y-4">
                  {/* Card Header */}
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="rounded-lg bg-sky-500/20 border border-sky-500/30 px-2 py-0.5 font-mono text-xs font-bold text-sky-300">
                          {mod.code}
                        </span>
                        <span className="text-[11px] font-mono text-zinc-400">
                          {mod.phase}
                        </span>
                      </div>
                      <h3 className="text-base font-bold text-white group-hover:text-sky-300 transition-colors">
                        {mod.title}
                      </h3>
                    </div>

                    <span className="flex items-center gap-1 rounded-full bg-emerald-500/15 border border-emerald-500/30 px-2.5 py-0.5 text-[11px] font-semibold text-emerald-300 whitespace-nowrap">
                      <CheckCircle2 className="h-3 w-3" />
                      <span>{mod.status}</span>
                    </span>
                  </div>

                  {/* Summary */}
                  <p className="text-xs text-zinc-300 leading-relaxed">
                    {mod.summary}
                  </p>

                  {/* Key Metrics */}
                  <div className="rounded-xl border border-white/5 bg-black/30 p-3 space-y-1.5">
                    <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">
                      Key Verified Parameters:
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      {Object.entries(mod.key_metrics).map(([k, v]) => (
                        <div key={k} className="flex flex-col">
                          <span className="text-[11px] text-zinc-400">{k.replace(/_/g, ' ')}:</span>
                          <span className="font-mono font-semibold text-sky-200">
                            {typeof v === 'boolean' ? (v ? 'Enabled' : 'Disabled') : String(v)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Deliverables List */}
                <div className="mt-4 pt-3 border-t border-white/5 space-y-1.5">
                  <div className="text-[10px] font-mono uppercase text-zinc-400">
                    Generated Research Artifacts:
                  </div>
                  <div className="space-y-1">
                    {mod.deliverables.map(del => (
                      <div key={del} className="flex items-center gap-1.5 text-[11px] font-mono text-zinc-300">
                        <FileText className="h-3 w-3 text-sky-400 flex-shrink-0" />
                        <span className="truncate">{del}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </AppShell>
  )
}
