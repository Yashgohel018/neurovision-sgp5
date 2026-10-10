'use client'

import { useState } from 'react'
import { api } from '@/lib/api'
import type { RagAnswer } from '@/lib/types'
import { MessageSquare, Send, BookOpen, ShieldAlert, Sparkles, ExternalLink } from 'lucide-react'

const SUGGESTED_QUERIES = [
  'Explain BraTS WT, TC, and ET volumetric label definitions',
  'What complementary information do T1ce and FLAIR provide?',
  'Summarize findings for case CASE-2026-001',
  'How does 3D Grad-CAM compute class activation gradients?',
]

export default function ClinicalAssistantSection() {
  const [query, setQuery] = useState('')
  const [caseId, setCaseId] = useState('CASE-2026-001')
  const [answer, setAnswer] = useState<RagAnswer | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleAsk = async (qText?: string) => {
    const textToSend = qText ?? query
    if (!textToSend.trim()) return
    setLoading(true)
    setError('')
    try {
      const res = await api.rag(textToSend, caseId || undefined)
      setAnswer(res)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section id="assistant" className="border-t border-nv-line bg-[#02050b] py-24 px-6">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="max-w-2xl space-y-3">
          <div className="text-xs uppercase tracking-widest text-nv-cool font-semibold flex items-center gap-1.5">
            <MessageSquare className="h-3.5 w-3.5" /> 08 · Clinical AI Assistant (RAG Architecture)
          </div>
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-5xl">
            Evidence-Grounded Reasoning.
            <span className="block text-zinc-400">Scientific Reference Retrieval.</span>
          </h2>
          <p className="text-base text-zinc-400 leading-relaxed pt-1">
            Query multiparametric MRI findings, BraTS protocols, or patient case summaries.
            Responses are linked directly to peer-reviewed neuro-oncological literature and clear clinical disclaimers.
          </p>
        </div>

        {/* Assistant Chat Interface Card */}
        <div className="mt-14 max-w-4xl mx-auto rounded-3xl border border-nv-line bg-[#050914] p-8 shadow-2xl space-y-6">
          {/* Preset Prompts Strip */}
          <div className="space-y-2">
            <div className="text-xs font-mono uppercase text-zinc-500 tracking-wider">
              Quick Research Prompts:
            </div>
            <div className="flex flex-wrap gap-2">
              {SUGGESTED_QUERIES.map(sq => (
                <button
                  key={sq}
                  onClick={() => {
                    setQuery(sq)
                    handleAsk(sq)
                  }}
                  className="rounded-full border border-white/10 bg-white/5 px-3.5 py-1.5 text-xs text-zinc-300 hover:border-nv-cool hover:text-white transition-colors text-left"
                >
                  {sq}
                </button>
              ))}
            </div>
          </div>

          {/* Input Box */}
          <div className="space-y-3 pt-2">
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Target Case ID (e.g. CASE-2026-001)"
                value={caseId}
                onChange={e => setCaseId(e.target.value)}
                className="w-48 rounded-xl border border-nv-line bg-black/60 px-3.5 py-2.5 text-xs font-mono text-white placeholder-zinc-600 focus:outline-none focus:border-nv-cool"
              />
              <div className="relative flex-1">
                <input
                  type="text"
                  placeholder="Ask the clinical research assistant about MRI findings or protocols…"
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleAsk()}
                  className="w-full rounded-xl border border-nv-line bg-black/60 px-4 py-2.5 text-xs text-white placeholder-zinc-500 focus:outline-none focus:border-nv-cool pr-12"
                />
                <button
                  onClick={() => handleAsk()}
                  disabled={loading || !query.trim()}
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded-lg bg-white p-1.5 text-black hover:bg-zinc-200 transition-colors disabled:opacity-40"
                >
                  <Send className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          </div>

          {error && (
            <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-400">
              {error}
            </div>
          )}

          {/* Response Box */}
          {answer && (
            <div className="rounded-2xl border border-nv-line bg-black/40 p-6 space-y-4">
              <div className="flex items-center justify-between border-b border-white/5 pb-3">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-cyan-400" />
                  <span className="text-xs font-semibold text-white">Assistant Structured Response</span>
                </div>
                <span className="text-[10px] font-mono text-zinc-500 uppercase">
                  Grounded in RAG Knowledge Base
                </span>
              </div>

              <div className="text-xs text-zinc-200 leading-relaxed whitespace-pre-line">
                {answer.answer}
              </div>

              {/* Retrieved Sources */}
              {answer.sources && answer.sources.length > 0 && (
                <div className="border-t border-white/5 pt-4 space-y-2">
                  <div className="flex items-center gap-1.5 text-[11px] font-mono uppercase tracking-wider text-zinc-400">
                    <BookOpen className="h-3.5 w-3.5" /> Supporting Literature Citations:
                  </div>
                  <ul className="space-y-1.5 text-xs">
                    {answer.sources.map((s, idx) => (
                      <li key={idx} className="flex items-center justify-between rounded-lg bg-white/5 p-2 text-zinc-300">
                        <span>{s.title}</span>
                        {s.url && (
                          <a
                            href={s.url}
                            target="_blank"
                            rel="noreferrer"
                            className="flex items-center gap-1 text-[11px] font-mono text-cyan-400 hover:underline"
                          >
                            <span>DOI</span>
                            <ExternalLink className="h-3 w-3" />
                          </a>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Disclaimer */}
              <div className="flex items-center gap-2 rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-[11px] text-zinc-400">
                <ShieldAlert className="h-4 w-4 text-amber-400 flex-shrink-0" />
                <span>{answer.disclaimer}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
