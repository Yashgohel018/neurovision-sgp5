'use client'

import { useState } from 'react'
import { api } from '@/lib/api'
import type { RagAnswer } from '@/lib/types'
import { AppShell, DemoBadge } from '@/components/ui'
import {
  MessageSquare,
  Sparkles,
  BookOpen,
  Send,
  ExternalLink,
  ShieldAlert,
  HelpCircle,
} from 'lucide-react'

const SUGGESTED_QUERIES = [
  { label: 'Summarize BraTS20_Training_001', caseId: 'BraTS20_Training_001', query: 'Summarize clinical findings and tumor volumetrics.' },
  { label: 'Explain WT, TC, and ET metrics', query: 'What are the BraTS tumor sub-regions (WT, TC, ET) and how are volumes calculated?' },
  { label: 'Diagnostic value of T1ce vs FLAIR', query: 'What complementary clinical information do T1ce and FLAIR sequences provide in gliomas?' },
  { label: 'WHO CNS5 Glioma Grading', query: 'How does the 2021 WHO CNS5 classification define diffuse astrocytic and oligodendroglial tumors?' },
]

export default function Assistant() {
  const [q, setQ] = useState('Summarize clinical findings and tumor volumetrics.')
  const [caseId, setCaseId] = useState('BraTS20_Training_001')
  const [a, setA] = useState<RagAnswer | null>(null)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)

  const ask = async (customQ?: string, customCase?: string) => {
    const activeQuery = customQ ?? q
    const activeCase = customCase !== undefined ? customCase : caseId
    if (!activeQuery.trim()) return

    setBusy(true)
    setErr('')
    try {
      setA(await api.rag(activeQuery, activeCase ? activeCase.trim() : undefined))
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <AppShell title="Clinical Decision Support Assistant">
      <div className="max-w-3xl space-y-6">
        <div>
          <h2 className="text-sm font-semibold text-white flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-nv-cool" />
            <span>Neuroradiology Intelligence & Literature Retrieval</span>
          </h2>
          <p className="mt-1 text-xs text-zinc-400 leading-relaxed">
            Synthesizes structured patient imaging metrics, multi-parametric MRI findings, and WHO CNS5 oncology guidelines.
          </p>
        </div>

        {/* Suggested Queries Strip */}
        <div className="space-y-2">
          <div className="text-[11px] font-mono uppercase tracking-wider text-zinc-500">
            Suggested Clinical Queries:
          </div>
          <div className="flex flex-wrap gap-2">
            {SUGGESTED_QUERIES.map(item => (
              <button
                key={item.label}
                onClick={() => {
                  setQ(item.query)
                  if (item.caseId) setCaseId(item.caseId)
                  ask(item.query, item.caseId ?? '')
                }}
                className="rounded-xl border border-nv-line bg-nv-panel/60 px-3 py-1.5 text-xs text-zinc-300 hover:border-nv-cool hover:text-white transition-all text-left"
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        {/* Input Form */}
        <div className="rounded-2xl border border-nv-line bg-nv-panel/70 p-5 backdrop-blur space-y-3">
          <div className="flex gap-3">
            <input
              aria-label="Patient Case ID (optional)"
              placeholder="Target Case ID (e.g. BraTS20_Training_001)..."
              value={caseId}
              onChange={e => setCaseId(e.target.value)}
              className="w-1/3 rounded-xl border border-nv-line bg-black/60 px-3.5 py-2 text-xs font-mono text-white placeholder-zinc-500 focus:border-nv-cool focus:outline-none"
            />
            <div className="text-[11px] text-zinc-500 self-center">
              Optional: Leave empty for general literature retrieval.
            </div>
          </div>

          <textarea
            aria-label="Clinical Question"
            value={q}
            onChange={e => setQ(e.target.value)}
            rows={3}
            placeholder="Ask a clinical decision-support question..."
            className="w-full rounded-xl border border-nv-line bg-black/60 px-3.5 py-2.5 text-xs text-white placeholder-zinc-500 focus:border-nv-cool focus:outline-none"
          />

          <div className="flex items-center justify-between pt-1">
            <span className="text-[11px] text-zinc-500">
              Retrieval-Augmented Generation across WHO CNS & BraTS Benchmark protocols
            </span>
            <button
              onClick={() => ask()}
              disabled={busy || !q.trim()}
              className="flex items-center gap-1.5 rounded-full bg-white px-5 py-2 text-xs font-semibold text-black hover:bg-zinc-200 transition-colors disabled:opacity-50 shadow-sm"
            >
              <Send className="h-3.5 w-3.5" />
              <span>{busy ? 'Synthesizing…' : 'Ask Assistant'}</span>
            </button>
          </div>
        </div>

        {err && (
          <div role="alert" className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-400">
            {err}
          </div>
        )}

        {/* Answer Display */}
        {a && (
          <div className="rounded-2xl border border-nv-line bg-nv-panel/80 p-6 backdrop-blur space-y-4">
            <div className="flex items-center justify-between border-b border-nv-line/60 pb-3">
              <span className="text-xs font-semibold uppercase tracking-wider text-nv-cool flex items-center gap-1.5">
                <BookOpen className="h-4 w-4" />
                <span>Evidence-Based Synthesis</span>
              </span>
              {a.is_demo && <DemoBadge />}
            </div>

            <p className="whitespace-pre-wrap text-xs leading-relaxed text-zinc-200 font-sans">
              {a.answer}
            </p>

            {a.sources.length > 0 && (
              <div className="border-t border-nv-line/60 pt-4 space-y-2">
                <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-500">
                  Referenced Literature & Citations:
                </div>
                <ul className="space-y-1.5 text-xs text-zinc-400">
                  {a.sources.map((s, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="text-nv-cool">•</span>
                      <div>
                        {s.url ? (
                          <a
                            href={s.url}
                            target="_blank"
                            rel="noreferrer"
                            className="font-medium text-zinc-300 hover:text-white underline inline-flex items-center gap-1"
                          >
                            <span>{s.title}</span>
                            <ExternalLink className="h-3 w-3" />
                          </a>
                        ) : (
                          <span className="font-medium text-zinc-300">{s.title}</span>
                        )}
                        {s.relevance && (
                          <p className="text-[11px] text-zinc-500">{s.relevance}</p>
                        )}
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-2.5 text-[11px] text-zinc-400 flex items-center gap-2">
              <ShieldAlert className="h-4 w-4 text-amber-400 flex-shrink-0" />
              <span>AI decision support information. Must be validated by qualified neuroradiologists.</span>
            </div>
          </div>
        )}
      </div>
    </AppShell>
  )
}
