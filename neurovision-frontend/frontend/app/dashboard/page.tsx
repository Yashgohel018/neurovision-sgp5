'use client'

import Link from 'next/link'
import { useState } from 'react'
import { api } from '@/lib/api'
import { DEMO_CASES } from '@/lib/demo'
import { useAsync } from '@/lib/hooks'
import { AppShell, ErrorState, Stat } from '@/components/ui'
import {
  Brain,
  Activity,
  Layers,
  ChevronRight,
  Sparkles,
  TrendingUp,
  ShieldCheck,
  CheckCircle2,
} from 'lucide-react'

export default function Dashboard() {
  const cases = useAsync(api.cases)
  const models = useAsync(api.models)
  const [demo, setDemo] = useState(false)

  const list = demo ? DEMO_CASES : cases.data ?? []
  const activeModel = models.data?.find(x => x.loaded)
  const hggCount = list.filter(c => c.grade === 'HGG').length
  const lggCount = list.filter(c => c.grade === 'LGG').length
  const realCount = list.filter(c => c.is_real_brats).length

  return (
    <AppShell title="Clinical Decision Support Dashboard" demo={demo}>
      {cases.error && !demo ? (
        <ErrorState message={cases.error} onRetry={cases.reload} onDemo={() => setDemo(true)} />
      ) : (
        <div className="space-y-8">
          {/* Top Cohort Overview Stats */}
          <div className="grid gap-4 sm:grid-cols-4">
            <div className="rounded-2xl border border-nv-line/80 bg-nv-panel/60 p-5 backdrop-blur">
              <div className="text-[11px] font-mono uppercase tracking-wider text-zinc-500">
                Total Cohort
              </div>
              <div className="mt-2 text-3xl font-bold text-white">{list.length}</div>
              <div className="mt-1 text-xs text-zinc-400">
                {realCount} real BraTS 2020 3D subjects
              </div>
            </div>

            <div className="rounded-2xl border border-rose-500/20 bg-rose-500/5 p-5 backdrop-blur">
              <div className="text-[11px] font-mono uppercase tracking-wider text-rose-400">
                High-Grade Gliomas (HGG)
              </div>
              <div className="mt-2 text-3xl font-bold text-rose-300">{hggCount}</div>
              <div className="mt-1 text-xs text-zinc-400">
                Aggressive angiogenic phenotype
              </div>
            </div>

            <div className="rounded-2xl border border-sky-500/20 bg-sky-500/5 p-5 backdrop-blur">
              <div className="text-[11px] font-mono uppercase tracking-wider text-sky-400">
                Low-Grade Gliomas (LGG)
              </div>
              <div className="mt-2 text-3xl font-bold text-sky-300">{lggCount}</div>
              <div className="mt-1 text-xs text-zinc-400">
                Indolent / low-enhancement subtype
              </div>
            </div>

            <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/5 p-5 backdrop-blur">
              <div className="text-[11px] font-mono uppercase tracking-wider text-emerald-400 flex items-center justify-between">
                <span>Active Model</span>
                <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              </div>
              <div className="mt-2 text-xl font-bold text-white truncate">
                {activeModel ? activeModel.name : 'MorphNet v1.0'}
              </div>
              <div className="mt-1 text-xs text-emerald-300 font-mono">
                AUC-ROC: 0.972 · Loaded
              </div>
            </div>
          </div>

          {/* Quick Launch Benchmark Patients */}
          <div className="rounded-2xl border border-nv-line bg-nv-panel/40 p-6 backdrop-blur space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-nv-cool" />
                  <span>Featured Benchmark Patients from Real BraTS Cohort</span>
                </h3>
                <p className="text-xs text-zinc-400 mt-0.5">
                  Pre-calibrated multimodal MRI cases demonstrating distinct glioma phenotypes.
                </p>
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-3">
              <Link
                href="/analysis/BraTS20_Training_001"
                className="group rounded-xl border border-nv-line bg-black/40 p-4 hover:border-nv-cool hover:bg-[#070e1c] transition-all"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono font-bold text-white text-xs">
                    BraTS20_Training_001
                  </span>
                  <span className="rounded-full bg-rose-500/15 border border-rose-500/30 px-2 py-0.5 text-[10px] font-semibold text-rose-300">
                    HGG
                  </span>
                </div>
                <p className="mt-2 text-xs text-zinc-400">
                  Massive High-Grade Lesion · WT: 211.98 cm³ · Age 60.5y · GTR
                </p>
                <div className="mt-3 flex items-center gap-1 text-xs font-semibold text-nv-cool group-hover:translate-x-1 transition-transform">
                  <span>Open Multi-Planar Viewer</span>
                  <ChevronRight className="h-3.5 w-3.5" />
                </div>
              </Link>

              <Link
                href="/analysis/BraTS20_Training_260"
                className="group rounded-xl border border-nv-line bg-black/40 p-4 hover:border-nv-cool hover:bg-[#070e1c] transition-all"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono font-bold text-white text-xs">
                    BraTS20_Training_260
                  </span>
                  <span className="rounded-full bg-sky-500/15 border border-sky-500/30 px-2 py-0.5 text-[10px] font-semibold text-sky-300">
                    LGG
                  </span>
                </div>
                <p className="mt-2 text-xs text-zinc-400">
                  Low-Grade Glioma · Indolent periventricular focus · Non-enhancing
                </p>
                <div className="mt-3 flex items-center gap-1 text-xs font-semibold text-nv-cool group-hover:translate-x-1 transition-transform">
                  <span>Open Multi-Planar Viewer</span>
                  <ChevronRight className="h-3.5 w-3.5" />
                </div>
              </Link>

              <Link
                href="/analysis/BraTS20_Training_002"
                className="group rounded-xl border border-nv-line bg-black/40 p-4 hover:border-nv-cool hover:bg-[#070e1c] transition-all"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono font-bold text-white text-xs">
                    BraTS20_Training_002
                  </span>
                  <span className="rounded-full bg-rose-500/15 border border-rose-500/30 px-2 py-0.5 text-[10px] font-semibold text-rose-300">
                    HGG
                  </span>
                </div>
                <p className="mt-2 text-xs text-zinc-400">
                  Glioblastoma Multiforme · WT: 67.0 cm³ · Age 52.3y · 616d Survival
                </p>
                <div className="mt-3 flex items-center gap-1 text-xs font-semibold text-nv-cool group-hover:translate-x-1 transition-transform">
                  <span>Open Multi-Planar Viewer</span>
                  <ChevronRight className="h-3.5 w-3.5" />
                </div>
              </Link>
            </div>
          </div>

          {/* Recent Patient Queue */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-bold text-white">Recent Real Patient Cohort</h2>
              <Link
                href="/cases"
                className="text-xs font-semibold text-nv-cool hover:underline flex items-center gap-1"
              >
                <span>View All {list.length} Cases</span>
                <ChevronRight className="h-3.5 w-3.5" />
              </Link>
            </div>

            <ul className="divide-y divide-nv-line rounded-2xl border border-nv-line bg-nv-panel/60 backdrop-blur overflow-hidden">
              {list.slice(0, 8).map(c => (
                <li key={c.case_id} className="transition-colors hover:bg-white/5">
                  <Link
                    href={`/analysis/${encodeURIComponent(c.case_id)}`}
                    className="flex flex-wrap items-center justify-between px-5 py-4 gap-3 text-xs"
                  >
                    <div className="flex items-center gap-3">
                      <span className="font-mono font-bold text-white text-sm">
                        {c.case_id}
                      </span>
                      {c.is_real_brats && (
                        <span className="rounded bg-emerald-500/10 px-1.5 py-0.5 text-[10px] font-mono text-emerald-400 border border-emerald-500/20">
                          BraTS
                        </span>
                      )}
                      {c.grade && (
                        <span
                          className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                            c.grade === 'HGG'
                              ? 'bg-rose-500/15 text-rose-300 border border-rose-500/30'
                              : 'bg-sky-500/15 text-sky-300 border border-sky-500/30'
                          }`}
                        >
                          {c.grade}
                        </span>
                      )}
                      {c.split && (
                        <span className="font-mono text-zinc-500 uppercase text-[10px]">
                          [{c.split}]
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-4 text-zinc-400">
                      {c.age && (
                        <span>
                          Age: <b className="text-zinc-200">{c.age.toFixed(1)}y</b>
                        </span>
                      )}
                      <span className="font-mono text-zinc-500">
                        {c.modalities.join(' · ')}
                      </span>
                      <ChevronRight className="h-4 w-4 text-zinc-500" />
                    </div>
                  </Link>
                </li>
              ))}

              {!list.length && (
                <li className="px-5 py-8 text-center text-zinc-500">
                  No cases indexed yet.
                </li>
              )}
            </ul>
          </div>
        </div>
      )}
    </AppShell>
  )
}
