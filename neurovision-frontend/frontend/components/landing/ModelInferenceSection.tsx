'use client'

import { useState } from 'react'
import { api } from '@/lib/api'
import { DEMO_RESULT } from '@/lib/demo'
import type { InferenceResult } from '@/lib/types'
import { Cpu, AlertCircle, Sparkles, Play, ShieldAlert, CheckCircle2 } from 'lucide-react'

export default function ModelInferenceSection() {
  const [demoMode, setDemoMode] = useState(false)
  const [result, setResult] = useState<InferenceResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleRunInference = async () => {
    setLoading(true)
    setError('')
    try {
      if (demoMode) {
        // Calibrated synthetic demonstration
        const res = DEMO_RESULT('CASE-2026-001')
        setResult(res)
      } else {
        // Real API inference call to FastAPI backend
        const res = await api.inference('CASE-2026-001')
        setResult(res)
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="border-t border-nv-line bg-[#02050b] py-24 px-6">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="max-w-2xl space-y-3">
          <div className="text-xs uppercase tracking-widest text-nv-cool font-semibold flex items-center gap-1.5">
            <Cpu className="h-3.5 w-3.5" /> 06 · Multimodal Neural Inference
          </div>
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-5xl">
            AI / ML Inference Architecture.
          </h2>
          <p className="text-base text-zinc-400 leading-relaxed pt-1">
            Multimodal 3D deep learning models accept 4-channel NIfTI inputs.
            In strict compliance with clinical research ethics, if no trained weights are mounted in{' '}
            <code className="text-xs font-mono text-cyan-300">models/</code>, the system displays an honest
            unloaded status rather than synthesizing unverified patient predictions.
          </p>
        </div>

        {/* Inference Card & Controls */}
        <div className="mt-14 rounded-3xl border border-nv-line bg-[#050914] p-8 shadow-2xl">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-nv-line/70 pb-6">
            <div className="flex items-center gap-3">
              <span className="font-mono text-sm text-zinc-300 font-semibold">
                Input Target: CASE-2026-001
              </span>
              <span className="rounded-full bg-white/10 px-2.5 py-0.5 text-xs text-zinc-400 font-mono">
                T1 + T1ce + T2 + FLAIR
              </span>
            </div>

            {/* Real vs Demo Mode Toggle */}
            <div className="flex items-center gap-3">
              <span className="text-xs text-zinc-400">Environment:</span>
              <div className="flex rounded-xl border border-nv-line bg-black/60 p-1">
                <button
                  onClick={() => {
                    setDemoMode(false)
                    setResult(null)
                  }}
                  className={`rounded-lg px-3 py-1 text-xs font-medium transition-colors ${
                    !demoMode ? 'bg-white/15 text-white' : 'text-zinc-400 hover:text-white'
                  }`}
                >
                  Live Pipeline
                </button>
                <button
                  onClick={() => {
                    setDemoMode(true)
                    setResult(null)
                  }}
                  className={`rounded-lg px-3 py-1 text-xs font-medium transition-colors ${
                    demoMode
                      ? 'bg-orange-500/20 text-orange-300 border border-orange-500/40'
                      : 'text-zinc-400 hover:text-white'
                  }`}
                >
                  Preview DEMO Mode
                </button>
              </div>
            </div>
          </div>

          {/* Action Button & Status Output */}
          <div className="mt-6 flex flex-wrap items-center gap-4">
            <button
              onClick={handleRunInference}
              disabled={loading}
              className="flex items-center gap-2 rounded-full bg-white px-6 py-2.5 text-xs font-semibold text-black hover:bg-zinc-200 transition-colors disabled:opacity-50"
            >
              <Play className="h-3.5 w-3.5 fill-current" />
              <span>{loading ? 'Running 3D Model…' : 'Run Model Inference'}</span>
            </button>

            {demoMode && (
              <span className="rounded-full border border-orange-500/50 bg-orange-500/10 px-3 py-1 text-[11px] font-mono font-semibold text-[#ff7a45]">
                DEMO DATA · Calibrated Research Demonstration Only
              </span>
            )}
          </div>

          {error && (
            <div className="mt-4 flex items-center gap-2 text-xs text-red-400">
              <AlertCircle className="h-4 w-4" />
              <span>{error}</span>
            </div>
          )}

          {/* Result Inspection */}
          {result && (
            <div className="mt-8 rounded-2xl border border-nv-line bg-black/40 p-6 space-y-6">
              {/* Honest Model Not Loaded Message */}
              {!result.prediction ? (
                <div className="flex items-start gap-4 rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 text-xs text-zinc-300">
                  <AlertCircle className="h-5 w-5 text-amber-400 flex-shrink-0 mt-0.5" />
                  <div className="space-y-1">
                    <strong className="text-amber-400 font-semibold block">
                      Model Checkpoint Not Loaded
                    </strong>
                    <p className="leading-relaxed">
                      {result.message || 'Model not available — inference requires a trained checkpoint in models/.'}
                    </p>
                    <p className="text-zinc-400 pt-1">
                      To activate live prediction, mount a PyTorch checkpoint (e.g.{' '}
                      <code className="text-zinc-300 font-mono">backend/models/NeuroVision-ResNet3D.pt</code>).
                      Switch to <strong>Preview DEMO Mode</strong> above to inspect sample prediction layouts.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center justify-between border-b border-white/5 pb-3">
                    <div>
                      <div className="text-[11px] font-mono uppercase tracking-widest text-zinc-500">
                        Classification Prediction
                      </div>
                      <div className="text-2xl font-bold text-white mt-0.5">
                        {result.prediction}
                      </div>
                    </div>
                    {demoMode && (
                      <span className="rounded bg-orange-500/20 px-2 py-0.5 text-[10px] font-mono text-orange-300">
                        DEMO PREDICTION
                      </span>
                    )}
                  </div>

                  {/* Probabilities Distribution */}
                  {result.probabilities && (
                    <div className="space-y-2">
                      <div className="text-xs text-zinc-400">Softmax Confidence Distribution:</div>
                      {Object.entries(result.probabilities).map(([cls, prob]) => (
                        <div key={cls} className="space-y-1">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-zinc-300">{cls}</span>
                            <span className="text-cyan-400">{(prob * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-2 w-full rounded-full bg-white/10 overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-nv-cool to-cyan-300 rounded-full"
                              style={{ width: `${prob * 100}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="flex items-center justify-between text-[11px] font-mono text-zinc-500 pt-2 border-t border-white/5">
                    <span>Model: {result.model_version ?? 'Calibrated Demo'}</span>
                    <span>Latency: {result.inference_time ? `${result.inference_time}s` : '0.12s'}</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
