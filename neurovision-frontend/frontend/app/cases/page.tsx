'use client'

import Link from 'next/link'
import { useState, useMemo } from 'react'
import { api, validNifti, MAX_UPLOAD_MB } from '@/lib/api'
import { DEMO_CASES } from '@/lib/demo'
import { useAsync } from '@/lib/hooks'
import type { Modality, CaseSummary } from '@/lib/types'
import { AppShell, ErrorState } from '@/components/ui'
import {
  Search,
  Filter,
  Brain,
  UploadCloud,
  Trash2,
  ChevronRight,
  ChevronLeft,
  Activity,
  Layers,
  Sparkles,
  PlusCircle,
  Clock,
  User,
} from 'lucide-react'

const MODS: (Modality | 'Seg')[] = ['T1', 'T1ce', 'T2', 'FLAIR', 'Seg']
const ITEMS_PER_PAGE = 12

export default function Cases() {
  const cases = useAsync(api.cases)
  const [demo, setDemo] = useState(false)
  const [msg, setMsg] = useState('')
  const [search, setSearch] = useState('')
  const [gradeFilter, setGradeFilter] = useState<'ALL' | 'HGG' | 'LGG' | 'CUSTOM'>('ALL')
  const [splitFilter, setSplitFilter] = useState<'ALL' | 'train' | 'val' | 'test'>('ALL')
  const [page, setPage] = useState(1)
  const [selectedMod, setSelectedMod] = useState<Modality | 'Seg'>('T1')
  const [isUploading, setIsUploading] = useState<string | null>(null)

  const act = async (fn: () => Promise<unknown>) => {
    setMsg('')
    try {
      await fn()
      cases.reload()
    } catch (e) {
      setMsg((e as Error).message)
    }
  }

  const rawList: CaseSummary[] = demo ? DEMO_CASES : cases.data ?? []

  const filteredList = useMemo(() => {
    return rawList.filter(c => {
      // Search filter
      const matchesSearch =
        !search ||
        c.case_id.toLowerCase().includes(search.toLowerCase()) ||
        (c.grade && c.grade.toLowerCase().includes(search.toLowerCase()))

      // Grade filter
      let matchesGrade = true
      if (gradeFilter === 'HGG') matchesGrade = c.grade === 'HGG'
      else if (gradeFilter === 'LGG') matchesGrade = c.grade === 'LGG'
      else if (gradeFilter === 'CUSTOM') matchesGrade = !c.is_real_brats

      // Split filter
      let matchesSplit = true
      if (splitFilter !== 'ALL') matchesSplit = c.split === splitFilter

      return matchesSearch && matchesGrade && matchesSplit
    })
  }, [rawList, search, gradeFilter, splitFilter])

  const totalPages = Math.ceil(filteredList.length / ITEMS_PER_PAGE) || 1
  const displayedList = useMemo(() => {
    const start = (page - 1) * ITEMS_PER_PAGE
    return filteredList.slice(start, start + ITEMS_PER_PAGE)
  }, [filteredList, page])

  const hggCount = rawList.filter(c => c.grade === 'HGG').length
  const lggCount = rawList.filter(c => c.grade === 'LGG').length
  const realCount = rawList.filter(c => c.is_real_brats).length

  return (
    <AppShell title="Patient Cohort & BraTS Cases" demo={demo}>
      {cases.error && !demo ? (
        <ErrorState message={cases.error} onRetry={cases.reload} onDemo={() => setDemo(true)} />
      ) : (
        <div className="space-y-6">
          {/* Header Metric Banners */}
          <div className="grid gap-4 sm:grid-cols-4">
            <div className="rounded-2xl border border-nv-line/80 bg-nv-panel/60 p-4 backdrop-blur">
              <div className="text-[11px] font-mono uppercase tracking-wider text-zinc-500">
                Total Cohort Size
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="text-2xl font-bold text-white">{rawList.length}</span>
                <span className="text-xs text-zinc-400">cases indexed</span>
              </div>
            </div>

            <div className="rounded-2xl border border-rose-500/20 bg-rose-500/5 p-4 backdrop-blur">
              <div className="text-[11px] font-mono uppercase tracking-wider text-rose-400">
                High-Grade Gliomas (HGG)
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="text-2xl font-bold text-rose-300">{hggCount}</span>
                <span className="text-xs text-zinc-400">Glioblastoma IV / Anaplastic</span>
              </div>
            </div>

            <div className="rounded-2xl border border-sky-500/20 bg-sky-500/5 p-4 backdrop-blur">
              <div className="text-[11px] font-mono uppercase tracking-wider text-sky-400">
                Low-Grade Gliomas (LGG)
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="text-2xl font-bold text-sky-300">{lggCount}</span>
                <span className="text-xs text-zinc-400">Oligodendroglioma / Astro II-III</span>
              </div>
            </div>

            <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/5 p-4 backdrop-blur">
              <div className="text-[11px] font-mono uppercase tracking-wider text-emerald-400">
                Real BraTS 2020 Data
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="text-2xl font-bold text-emerald-300">{realCount}</span>
                <span className="text-xs text-zinc-400">3D NIfTI subjects</span>
              </div>
            </div>
          </div>

          {/* Action Bar & Search / Filter Controls */}
          <div className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-nv-line bg-nv-panel/40 p-4 backdrop-blur">
            <div className="flex flex-1 flex-wrap items-center gap-3 min-w-[280px]">
              {/* Search Bar */}
              <div className="relative flex-1 min-w-[220px]">
                <Search className="absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
                <input
                  type="text"
                  placeholder="Search by ID (e.g. BraTS20_Training_001)..."
                  value={search}
                  onChange={e => {
                    setSearch(e.target.value)
                    setPage(1)
                  }}
                  className="w-full rounded-xl border border-nv-line bg-black/60 pl-10 pr-4 py-2 text-xs text-white placeholder-zinc-500 focus:border-nv-cool focus:outline-none"
                />
              </div>

              {/* Grade Filter Chips */}
              <div className="flex rounded-xl border border-nv-line bg-black/40 p-1 text-xs">
                {(['ALL', 'HGG', 'LGG', 'CUSTOM'] as const).map(g => (
                  <button
                    key={g}
                    onClick={() => {
                      setGradeFilter(g)
                      setPage(1)
                    }}
                    className={`rounded-lg px-3 py-1 font-medium transition-all ${
                      gradeFilter === g
                        ? 'bg-nv-cool text-white shadow-sm'
                        : 'text-zinc-400 hover:text-white'
                    }`}
                  >
                    {g === 'ALL' ? 'All Grades' : g}
                  </button>
                ))}
              </div>

              {/* Split Filter Chips */}
              <div className="flex rounded-xl border border-nv-line bg-black/40 p-1 text-xs">
                {(['ALL', 'train', 'val', 'test'] as const).map(s => (
                  <button
                    key={s}
                    onClick={() => {
                      setSplitFilter(s)
                      setPage(1)
                    }}
                    className={`rounded-lg px-2.5 py-1 font-medium uppercase text-[11px] transition-all ${
                      splitFilter === s
                        ? 'bg-zinc-700 text-white'
                        : 'text-zinc-400 hover:text-white'
                    }`}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>

            {/* Create Case Button */}
            {!demo && (
              <button
                onClick={() => act(api.createCase)}
                className="flex items-center gap-2 rounded-xl bg-white px-4 py-2 text-xs font-semibold text-black hover:bg-zinc-200 transition-colors shadow-sm"
              >
                <PlusCircle className="h-4 w-4" />
                <span>New Clinical Case</span>
              </button>
            )}
          </div>

          {msg && (
            <div role="alert" className="rounded-xl border border-nv-acc/40 bg-nv-acc/10 p-3 text-xs text-nv-acc">
              {msg}
            </div>
          )}

          {/* Patient Cards Grid */}
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {displayedList.map(c => (
              <div
                key={c.case_id}
                className="group relative flex flex-col justify-between overflow-hidden rounded-2xl border border-nv-line/80 bg-nv-panel/70 p-5 backdrop-blur transition-all hover:border-nv-cool/50 hover:shadow-[0_4px_24px_rgba(77,155,255,0.08)]"
              >
                <div>
                  {/* Top Badges */}
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-bold text-white text-sm">
                          {c.case_id}
                        </span>
                        {c.is_real_brats && (
                          <span className="rounded bg-emerald-500/10 px-1.5 py-0.5 text-[10px] font-mono text-emerald-400 border border-emerald-500/20">
                            BraTS
                          </span>
                        )}
                        {c.split && (
                          <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] font-mono uppercase text-zinc-400 border border-white/5">
                            {c.split}
                          </span>
                        )}
                      </div>
                      <div className="text-[11px] text-zinc-500 font-mono mt-0.5">
                        {c.created_at.slice(0, 10)}
                      </div>
                    </div>

                    {/* Grade Pill */}
                    {c.grade ? (
                      <span
                        className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
                          c.grade === 'HGG'
                            ? 'bg-rose-500/15 text-rose-300 border border-rose-500/30'
                            : 'bg-sky-500/15 text-sky-300 border border-sky-500/30'
                        }`}
                      >
                        {c.grade}
                      </span>
                    ) : (
                      <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-[11px] text-zinc-400">
                        {c.status}
                      </span>
                    )}
                  </div>

                  {/* Clinical Details */}
                  {(c.age !== null && c.age !== undefined) || c.extent_of_resection ? (
                    <div className="mt-3.5 flex flex-wrap gap-2 text-[11px] text-zinc-400 border-t border-white/5 pt-3">
                      {c.age !== null && c.age !== undefined && (
                        <span className="rounded-md bg-black/40 px-2 py-1 font-mono border border-nv-line">
                          Age: <b className="text-zinc-200">{c.age.toFixed(1)}y</b>
                        </span>
                      )}
                      {c.survival_days !== null && c.survival_days !== undefined && (
                        <span className="rounded-md bg-black/40 px-2 py-1 font-mono border border-nv-line">
                          Surv: <b className="text-zinc-200">{Math.round(c.survival_days)}d</b>
                        </span>
                      )}
                      {c.extent_of_resection && (
                        <span className="rounded-md bg-black/40 px-2 py-1 font-mono border border-nv-line">
                          Resection: <b className="text-zinc-200">{c.extent_of_resection}</b>
                        </span>
                      )}
                    </div>
                  ) : null}

                  {/* Modality Badges */}
                  <div className="mt-3 flex flex-wrap items-center gap-1.5">
                    {MODS.map(m => {
                      const present = c.modalities.includes(m)
                      return (
                        <span
                          key={m}
                          className={`rounded px-1.5 py-0.5 text-[10px] font-mono ${
                            present
                              ? 'bg-nv-cool/15 text-nv-cool border border-nv-cool/30'
                              : 'bg-zinc-900 text-zinc-600 border border-transparent'
                          }`}
                        >
                          {m}
                        </span>
                      )
                    })}
                  </div>
                </div>

                {/* Bottom Actions */}
                <div className="mt-5 flex items-center justify-between border-t border-white/5 pt-4">
                  <Link
                    href={`/analysis/${encodeURIComponent(c.case_id)}`}
                    className="flex items-center gap-1.5 rounded-full border border-nv-cool/40 bg-nv-cool/10 px-4 py-1.5 text-xs font-semibold text-white hover:bg-nv-cool hover:text-white transition-all shadow-sm"
                  >
                    <span>Inspect 3D MRI</span>
                    <ChevronRight className="h-3.5 w-3.5" />
                  </Link>

                  {!demo && !c.is_real_brats && (
                    <div className="flex items-center gap-2">
                      <select
                        aria-label="Modality"
                        value={selectedMod}
                        onChange={e => setSelectedMod(e.target.value as Modality | 'Seg')}
                        className="rounded-lg border border-nv-line bg-black px-2 py-1 text-[11px] text-zinc-300"
                      >
                        {MODS.map(m => (
                          <option key={m}>{m}</option>
                        ))}
                      </select>

                      <label className="cursor-pointer rounded-lg border border-nv-line bg-black/60 px-2 py-1 text-[11px] text-zinc-400 hover:text-white transition-colors">
                        Upload
                        <input
                          type="file"
                          accept=".nii,.nii.gz"
                          className="sr-only"
                          onChange={e => {
                            const f = e.target.files?.[0]
                            if (!f) return
                            if (!validNifti(f)) {
                              return setMsg(`Only .nii or .nii.gz up to ${MAX_UPLOAD_MB} MB supported.`)
                            }
                            act(() => api.upload(c.case_id, selectedMod, f))
                          }}
                        />
                      </label>

                      <button
                        onClick={() => act(() => api.deleteCase(c.case_id))}
                        title="Delete custom case"
                        className="text-zinc-500 hover:text-rose-400 transition-colors p-1"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {displayedList.length === 0 && (
            <div className="rounded-2xl border border-nv-line bg-nv-panel/40 p-12 text-center text-zinc-500">
              No cases matching filter criteria. Clear search or reset filters.
            </div>
          )}

          {/* Pagination Strip */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between border-t border-nv-line pt-4 text-xs text-zinc-400">
              <div>
                Showing {(page - 1) * ITEMS_PER_PAGE + 1}–
                {Math.min(page * ITEMS_PER_PAGE, filteredList.length)} of {filteredList.length} cases
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPage(Math.max(1, page - 1))}
                  disabled={page <= 1}
                  className="flex items-center gap-1 rounded-xl border border-nv-line bg-nv-panel px-3 py-1.5 hover:bg-white/5 disabled:opacity-40"
                >
                  <ChevronLeft className="h-3.5 w-3.5" />
                  <span>Previous</span>
                </button>

                <span className="font-mono text-white px-2">
                  Page {page} of {totalPages}
                </span>

                <button
                  onClick={() => setPage(Math.min(totalPages, page + 1))}
                  disabled={page >= totalPages}
                  className="flex items-center gap-1 rounded-xl border border-nv-line bg-nv-panel px-3 py-1.5 hover:bg-white/5 disabled:opacity-40"
                >
                  <span>Next</span>
                  <ChevronRight className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </AppShell>
  )
}
