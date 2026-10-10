'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { api } from '@/lib/api'
import type { CaseSummary, ModelInfo } from '@/lib/types'
import { FolderGit2, Plus, ArrowUpRight, CheckCircle2, Clock, Cpu } from 'lucide-react'

export default function CaseManagementPreviewSection() {
  const [cases, setCases] = useState<CaseSummary[]>([])
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [createMsg, setCreateMsg] = useState('')

  const loadData = async () => {
    try {
      const [cList, mList] = await Promise.all([api.cases(), api.models()])
      setCases(cList)
      setModels(mList)
    } catch {
      // Fallback if initial load fails
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  const handleCreate = async () => {
    try {
      setCreateMsg('Creating case…')
      await api.createCase()
      await loadData()
      setCreateMsg('New case created successfully.')
      setTimeout(() => setCreateMsg(''), 3000)
    } catch (e) {
      setCreateMsg((e as Error).message)
    }
  }

  const processedCount = cases.filter(c => c.status === 'ready').length
  const pendingCount = cases.filter(c => c.status !== 'ready').length
  const loadedModel = models.find(m => m.loaded)

  return (
    <section className="border-t border-nv-line bg-[#03060d] py-24 px-6">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-12">
          <div className="max-w-2xl space-y-3">
            <div className="text-xs uppercase tracking-widest text-nv-cool font-semibold flex items-center gap-1.5">
              <FolderGit2 className="h-3.5 w-3.5" /> 09 · Case Management & Clinical Dashboard
            </div>
            <h2 className="text-3xl font-bold tracking-tight text-white sm:text-5xl">
              Live Clinical Case Registry.
            </h2>
            <p className="text-base text-zinc-400 leading-relaxed pt-1">
              Synchronized directly with the FastAPI SQLite backend. Inspect uploaded multi-modal scans,
              create new patient cases, and transition seamlessly into the detailed analysis workspace.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={handleCreate}
              className="flex items-center gap-2 rounded-full bg-white px-5 py-2 text-xs font-semibold text-black hover:bg-zinc-200 transition-colors"
            >
              <Plus className="h-4 w-4" />
              <span>Create New Case</span>
            </button>
            <Link
              href="/cases"
              className="flex items-center gap-1.5 rounded-full border border-white/20 bg-black/40 px-5 py-2 text-xs font-medium text-white hover:bg-white/10 transition-colors"
            >
              <span>View All Cases</span>
              <ArrowUpRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </div>

        {createMsg && (
          <div className="mb-6 rounded-xl border border-nv-cool/30 bg-nv-cool/10 p-3 text-xs text-cyan-300">
            {createMsg}
          </div>
        )}

        {/* 4 Dashboard Metric Cards (Calculated from real backend data) */}
        <div className="grid gap-4 sm:grid-cols-4 mb-8">
          <div className="rounded-2xl border border-nv-line bg-nv-panel/60 p-5 backdrop-blur">
            <div className="text-[11px] font-mono uppercase tracking-wider text-zinc-500">
              Total Registered Cases
            </div>
            <div className="mt-2 text-3xl font-extrabold text-white">
              {loading ? '…' : cases.length}
            </div>
          </div>

          <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/5 p-5 backdrop-blur">
            <div className="text-[11px] font-mono uppercase tracking-wider text-emerald-400">
              Processed / Ready
            </div>
            <div className="mt-2 text-3xl font-extrabold text-white">
              {loading ? '…' : processedCount}
            </div>
          </div>

          <div className="rounded-2xl border border-amber-500/20 bg-amber-500/5 p-5 backdrop-blur">
            <div className="text-[11px] font-mono uppercase tracking-wider text-amber-400">
              Pending Sequences
            </div>
            <div className="mt-2 text-3xl font-extrabold text-white">
              {loading ? '…' : pendingCount}
            </div>
          </div>

          <div className="rounded-2xl border border-blue-500/20 bg-blue-500/5 p-5 backdrop-blur">
            <div className="text-[11px] font-mono uppercase tracking-wider text-blue-400">
              Active Checkpoint
            </div>
            <div className="mt-2 text-lg font-bold text-white truncate">
              {loadedModel ? loadedModel.name : 'Unmounted'}
            </div>
          </div>
        </div>

        {/* Live Case Table */}
        <div className="overflow-hidden rounded-2xl border border-nv-line bg-nv-panel/40 backdrop-blur">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-nv-line bg-black/40 text-[11px] font-mono uppercase text-zinc-500">
              <tr>
                <th className="px-6 py-3.5">Case ID</th>
                <th className="px-6 py-3.5">Registered Date</th>
                <th className="px-6 py-3.5">Available Modalities</th>
                <th className="px-6 py-3.5">Status</th>
                <th className="px-6 py-3.5 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-nv-line/50">
              {cases.slice(0, 5).map(c => (
                <tr key={c.case_id} className="hover:bg-white/5 transition-colors">
                  <td className="px-6 py-4 font-mono font-semibold text-white">
                    {c.case_id}
                  </td>
                  <td className="px-6 py-4 text-zinc-400">
                    {new Date(c.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-6 py-4 font-mono text-zinc-300">
                    {c.modalities.length > 0 ? (
                      <span className="flex gap-1.5 flex-wrap">
                        {c.modalities.map(m => (
                          <span
                            key={m}
                            className="rounded bg-white/10 px-1.5 py-0.5 text-[10px] text-zinc-300"
                          >
                            {m}
                          </span>
                        ))}
                      </span>
                    ) : (
                      <span className="text-zinc-500 italic">None uploaded</span>
                    )}
                  </td>
                  <td className="px-6 py-4">
                    <span
                      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[10px] font-mono capitalize ${
                        c.status === 'ready'
                          ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                          : 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
                      }`}
                    >
                      {c.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <Link
                      href={`/analysis/${c.case_id}`}
                      className="rounded-full border border-white/20 bg-white/5 px-3 py-1 text-xs text-zinc-300 hover:bg-white hover:text-black transition-colors"
                    >
                      Open Analysis →
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}
