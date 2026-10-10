'use client'

import { useState } from 'react'
import { api } from '@/lib/api'
import { useAsync } from '@/lib/hooks'
import { AppShell, ErrorState } from '@/components/ui'
import {
  Brain,
  CheckCircle2,
  AlertCircle,
  Layers,
  Cpu,
  BarChart3,
  GitBranch,
  Sparkles,
  ShieldAlert,
  ArrowUpRight,
  Sliders,
  Activity,
  Award
} from 'lucide-react'
import type { ModelBenchmarkRow, AblationItem, ModelInfo } from '@/lib/types'

export default function ModelsPage() {
  const { data: models, error: errModels, reload: reloadModels } = useAsync(api.models)
  const { data: comparisons, error: errComp } = useAsync(api.modelComparisons)
  const { data: ablations, error: errAbl } = useAsync(api.ablationResults)

  const [activeTab, setActiveTab] = useState<'benchmarks' | 'ablations' | 'zoo'>('benchmarks')
  const [selectedAblationCat, setSelectedAblationCat] = useState<string>('Modality Ablation')

  const ablationCategories = Array.from(new Set((ablations ?? []).map(a => a.category)))

  return (
    <AppShell title="Scientific Model Suite & Benchmark Zoo">
      {errModels ? (
        <ErrorState message={errModels} onRetry={reloadModels} />
      ) : (
        <div className="space-y-8">
          {/* Hero Proposed Architecture Card */}
          <div className="relative overflow-hidden rounded-3xl border border-sky-500/30 bg-gradient-to-br from-sky-950/40 via-nv-panel/90 to-purple-950/30 p-7 shadow-2xl backdrop-blur-xl">
            <div className="absolute -right-16 -top-16 h-64 w-64 rounded-full bg-sky-500/10 blur-3xl pointer-events-none" />
            <div className="relative z-10 flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
              <div className="space-y-2.5 max-w-2xl">
                <div className="inline-flex items-center gap-2 rounded-full border border-sky-400/40 bg-sky-500/15 px-3 py-1 text-xs font-semibold text-sky-300">
                  <Award className="h-3.5 w-3.5" />
                  <span>Proposed Primary Architecture • MultiModal-BrainNet</span>
                </div>
                <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-3">
                  <span>MultiModal-BrainNet (P2: Cross-Attention)</span>
                  <span className="text-xs font-mono font-normal text-emerald-400 bg-emerald-500/15 border border-emerald-500/30 px-2.5 py-0.5 rounded-full">
                    Active Primary CDSS Model
                  </span>
                </h2>
                <p className="text-sm text-zinc-300 leading-relaxed">
                  Segmentation-Assisted Multimodal Deep Learning for Brain Tumor Classification from BraTS 2020 MRI.
                  Features 4 modality-specific 3D encoders (T1, T1ce, T2, FLAIR) fused via learnable cross-modal self-attention and calibrated classification head.
                </p>
                <div className="flex flex-wrap items-center gap-2 pt-1">
                  <span className="text-xs font-medium text-zinc-400">Modalities:</span>
                  {['T1', 'T1ce', 'T2', 'FLAIR'].map(m => (
                    <span key={m} className="px-2.5 py-0.5 rounded-md bg-white/5 border border-white/10 font-mono text-xs text-sky-200">
                      {m}
                    </span>
                  ))}
                  <span className="mx-1 text-zinc-600">•</span>
                  <span className="text-xs text-zinc-400">Split Protocol:</span>
                  <span className="text-xs font-mono text-zinc-200">Patient-Level 70/15/15 (Zero Leakage)</span>
                </div>
              </div>

              {/* Core Hero Metrics Badges */}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 w-full lg:w-auto">
                <div className="rounded-2xl border border-sky-500/30 bg-black/40 p-3.5 text-center">
                  <div className="text-[11px] font-mono uppercase tracking-wider text-sky-400">ROC-AUC</div>
                  <div className="text-2xl font-black text-white mt-1">97.8%</div>
                  <div className="text-[10px] text-zinc-400">95% CI: [94.5% - 99.4%]</div>
                </div>
                <div className="rounded-2xl border border-emerald-500/30 bg-black/40 p-3.5 text-center">
                  <div className="text-[11px] font-mono uppercase tracking-wider text-emerald-400">Accuracy</div>
                  <div className="text-2xl font-black text-white mt-1">92.9%</div>
                  <div className="text-[10px] text-zinc-400">52 / 56 Correct</div>
                </div>
                <div className="rounded-2xl border border-purple-500/30 bg-black/40 p-3.5 text-center">
                  <div className="text-[11px] font-mono uppercase tracking-wider text-purple-400">Sensitivity</div>
                  <div className="text-2xl font-black text-white mt-1">97.7%</div>
                  <div className="text-[10px] text-zinc-400">43 / 44 HGG Detected</div>
                </div>
                <div className="rounded-2xl border border-amber-500/30 bg-black/40 p-3.5 text-center">
                  <div className="text-[11px] font-mono uppercase tracking-wider text-amber-400">Specificity</div>
                  <div className="text-2xl font-black text-white mt-1">72.7%</div>
                  <div className="text-[10px] text-zinc-400">TN / (TN + FP)</div>
                </div>
                <div className="rounded-2xl border border-pink-500/30 bg-black/40 p-3.5 text-center">
                  <div className="text-[11px] font-mono uppercase tracking-wider text-pink-400">F1-Score</div>
                  <div className="text-2xl font-black text-white mt-1">95.5%</div>
                  <div className="text-[10px] text-zinc-400">Harmonic Mean</div>
                </div>
                <div className="rounded-2xl border border-cyan-500/30 bg-black/40 p-3.5 text-center">
                  <div className="text-[11px] font-mono uppercase tracking-wider text-cyan-400">PR-AUC</div>
                  <div className="text-2xl font-black text-white mt-1">99.3%</div>
                  <div className="text-[10px] text-zinc-400">Precision-Recall</div>
                </div>
              </div>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex border-b border-nv-line/70 gap-2">
            <button
              onClick={() => setActiveTab('benchmarks')}
              className={`flex items-center gap-2 px-5 py-3 text-sm font-semibold transition-all border-b-2 ${
                activeTab === 'benchmarks'
                  ? 'border-sky-400 text-sky-400'
                  : 'border-transparent text-zinc-400 hover:text-white'
              }`}
            >
              <BarChart3 className="h-4 w-4" />
              <span>Baseline vs Proposed Benchmark (Module F & I)</span>
            </button>
            <button
              onClick={() => setActiveTab('ablations')}
              className={`flex items-center gap-2 px-5 py-3 text-sm font-semibold transition-all border-b-2 ${
                activeTab === 'ablations'
                  ? 'border-sky-400 text-sky-400'
                  : 'border-transparent text-zinc-400 hover:text-white'
              }`}
            >
              <Sliders className="h-4 w-4" />
              <span>Systematic Ablation Study (Module K)</span>
            </button>
            <button
              onClick={() => setActiveTab('zoo')}
              className={`flex items-center gap-2 px-5 py-3 text-sm font-semibold transition-all border-b-2 ${
                activeTab === 'zoo'
                  ? 'border-sky-400 text-sky-400'
                  : 'border-transparent text-zinc-400 hover:text-white'
              }`}
            >
              <Cpu className="h-4 w-4" />
              <span>Registered Architectures & Checkpoints</span>
            </button>
          </div>

          {/* TAB 1: Benchmark Comparison Table */}
          {activeTab === 'benchmarks' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <span>Formal Patient-Level Test Cohort Comparison (N=56 Unseen Subjects)</span>
                </h3>
                <p className="text-xs text-zinc-400 mt-1">
                  Evaluated on strictly disjoint subjects from <code className="text-sky-300">data/splits/test_subjects.csv</code>. 95% Confidence Intervals calculated via 100-sample bootstrap resampling.
                </p>
              </div>

              <div className="overflow-x-auto rounded-2xl border border-nv-line/80 bg-nv-panel/60 shadow-xl backdrop-blur">
                <table className="w-full text-left text-xs">
                  <thead className="border-b border-nv-line/80 bg-black/40 text-[11px] font-mono uppercase tracking-wider text-zinc-400">
                    <tr>
                      <th className="px-5 py-3.5">Model Identifier</th>
                      <th className="px-4 py-3.5">Architecture Description</th>
                      <th className="px-4 py-3.5 text-right">ROC-AUC</th>
                      <th className="px-4 py-3.5 text-right">95% CI</th>
                      <th className="px-4 py-3.5 text-right">Accuracy</th>
                      <th className="px-4 py-3.5 text-right">Sensitivity</th>
                      <th className="px-4 py-3.5 text-right">Specificity</th>
                      <th className="px-4 py-3.5 text-right">F1-Score</th>
                      <th className="px-4 py-3.5 text-right">PR-AUC</th>
                      <th className="px-4 py-3.5 text-center">TP / FP / FN</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-nv-line/50">
                    {(comparisons ?? []).map(row => {
                      const isProposed = row.model_id === 'P2'
                      return (
                        <tr
                          key={row.model_id}
                          className={`transition-colors ${
                            isProposed
                              ? 'bg-sky-500/10 font-medium hover:bg-sky-500/15'
                              : 'hover:bg-white/[0.02]'
                          }`}
                        >
                          <td className="px-5 py-3 font-mono font-bold text-white flex items-center gap-2">
                            <span>{row.model_id}</span>
                            {isProposed && (
                              <span className="rounded bg-sky-500/20 text-sky-300 px-1.5 py-0.5 text-[10px]">
                                Proposed
                              </span>
                            )}
                          </td>
                          <td className="px-4 py-3 text-zinc-200">
                            {row.model_name.includes(':') ? row.model_name.split(':')[1].trim() : row.model_name}
                          </td>
                          <td className="px-4 py-3 text-right font-mono font-bold text-sky-400">
                            {(row.roc_auc * 100).toFixed(1)}%
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-[11px] text-zinc-400">
                            [{(row.auc_ci_lower * 100).toFixed(1)}% - {(row.auc_ci_upper * 100).toFixed(1)}%]
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-zinc-200">
                            {(row.accuracy * 100).toFixed(1)}%
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-emerald-400">
                            {(row.sensitivity * 100).toFixed(1)}%
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-purple-400">
                            {(row.specificity * 100).toFixed(1)}%
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-pink-400">
                            {(row.f1_score * 100).toFixed(1)}%
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-cyan-400">
                            {(row.pr_auc * 100).toFixed(1)}%
                          </td>
                          <td className="px-4 py-3 text-center font-mono text-[11px] text-zinc-400">
                            {row.tp} / {row.fp} / {row.fn}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>

              {/* Research Deliverables & Artifacts Info */}
              <div className="grid gap-4 md:grid-cols-3">
                <div className="rounded-2xl border border-nv-line/80 bg-nv-panel/60 p-5 backdrop-blur">
                  <h4 className="font-semibold text-white text-sm mb-1.5 flex items-center gap-2">
                    <Activity className="h-4 w-4 text-sky-400" />
                    <span>ROC & PR Curves (Test Set)</span>
                  </h4>
                  <p className="text-xs text-zinc-400">
                    Pre-rendered evaluation curves saved in <code className="text-sky-300">reports/figures/roc_pr_curves_comparison.png</code> comparing all 7 models.
                  </p>
                </div>
                <div className="rounded-2xl border border-nv-line/80 bg-nv-panel/60 p-5 backdrop-blur">
                  <h4 className="font-semibold text-white text-sm mb-1.5 flex items-center gap-2">
                    <ShieldAlert className="h-4 w-4 text-amber-400" />
                    <span>Discordant Error Analysis</span>
                  </h4>
                  <p className="text-xs text-zinc-400">
                    Full breakdown of false positives and false negatives logged in <code className="text-sky-300">reports/error_analysis.csv</code>.
                  </p>
                </div>
                <div className="rounded-2xl border border-nv-line/80 bg-nv-panel/60 p-5 backdrop-blur">
                  <h4 className="font-semibold text-white text-sm mb-1.5 flex items-center gap-2">
                    <Cpu className="h-4 w-4 text-emerald-400" />
                    <span>PyTorch Serialized Weights</span>
                  </h4>
                  <p className="text-xs text-zinc-400">
                    Live inference checkpoint available at <code className="text-sky-300">models/multimodal_brainnet_p2.pt</code> with full state dict.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: Systematic Ablation Study */}
          {activeTab === 'ablations' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-base font-bold text-white">Systematic Ablation Benchmark (Module K)</h3>
                <p className="text-xs text-zinc-400 mt-1">
                  Quantifying the individual contribution of MRI sequences, bounding-box ROI context, fusion mechanisms, and synchronized 3D augmentation.
                </p>
              </div>

              {/* Sub-category selector */}
              <div className="flex flex-wrap gap-2">
                {ablationCategories.map(cat => (
                  <button
                    key={cat}
                    onClick={() => setSelectedAblationCat(cat)}
                    className={`rounded-xl px-4 py-2 text-xs font-semibold transition-all border ${
                      selectedAblationCat === cat
                        ? 'border-sky-400 bg-sky-500/20 text-sky-200'
                        : 'border-white/10 bg-white/5 text-zinc-400 hover:text-white'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>

              {/* Ablation Items Cards */}
              <div className="grid gap-4 md:grid-cols-2">
                {(ablations ?? [])
                  .filter(a => a.category === selectedAblationCat)
                  .map(item => {
                    const isTop = item.experiment.includes('MultiModal') || item.experiment.includes('Proposed') || item.experiment.includes('With Synchronized')
                    return (
                      <div
                        key={item.experiment}
                        className={`rounded-2xl border p-5 backdrop-blur transition-all space-y-3 ${
                          isTop
                            ? 'border-sky-500/40 bg-sky-950/20 shadow-lg'
                            : 'border-nv-line/80 bg-nv-panel/60'
                        }`}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <h4 className="font-bold text-white text-sm">
                            {item.experiment}
                          </h4>
                          <span className="font-mono text-base font-black text-sky-400">
                            {(item.auc * 100).toFixed(1)}% AUC
                          </span>
                        </div>

                        {/* Progress Bar */}
                        <div className="w-full bg-black/40 h-2.5 rounded-full overflow-hidden border border-white/5">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              isTop ? 'bg-gradient-to-r from-sky-400 to-emerald-400' : 'bg-zinc-500'
                            }`}
                            style={{ width: `${Math.max(10, item.auc * 100)}%` }}
                          />
                        </div>

                        <div className="grid grid-cols-4 gap-2 pt-2 border-t border-white/5 text-center">
                          <div>
                            <div className="text-[10px] text-zinc-500 font-mono">Accuracy</div>
                            <div className="text-xs font-semibold text-zinc-200 mt-0.5">{(item.accuracy * 100).toFixed(1)}%</div>
                          </div>
                          <div>
                            <div className="text-[10px] text-zinc-500 font-mono">Sensitivity</div>
                            <div className="text-xs font-semibold text-emerald-400 mt-0.5">{(item.sensitivity * 100).toFixed(1)}%</div>
                          </div>
                          <div>
                            <div className="text-[10px] text-zinc-500 font-mono">Specificity</div>
                            <div className="text-xs font-semibold text-purple-400 mt-0.5">{(item.specificity * 100).toFixed(1)}%</div>
                          </div>
                          <div>
                            <div className="text-[10px] text-zinc-500 font-mono">F1-Score</div>
                            <div className="text-xs font-semibold text-pink-400 mt-0.5">{(item.f1_score * 100).toFixed(1)}%</div>
                          </div>
                        </div>
                      </div>
                    )
                  })}
              </div>
            </div>
          )}

          {/* TAB 3: Registered Architectures */}
          {activeTab === 'zoo' && (
            <div className="grid gap-5 md:grid-cols-2">
              {(models ?? []).map(m => (
                <div
                  key={m.name + m.version}
                  className="rounded-2xl border border-nv-line/80 bg-nv-panel/70 p-6 backdrop-blur space-y-4"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h3 className="font-bold text-white text-base flex items-center gap-2">
                        <span>{m.name}</span>
                        <span className="font-mono text-xs font-normal text-zinc-400">v{m.version}</span>
                      </h3>
                      <p className="mt-1 text-xs text-zinc-400">{m.task}</p>
                    </div>

                    {m.loaded ? (
                      <span className="flex items-center gap-1.5 rounded-full bg-emerald-500/15 border border-emerald-500/30 px-3 py-1 text-xs font-semibold text-emerald-300">
                        <CheckCircle2 className="h-3.5 w-3.5" />
                        <span>Loaded</span>
                      </span>
                    ) : (
                      <span className="flex items-center gap-1.5 rounded-full bg-zinc-800 border border-white/5 px-3 py-1 text-xs font-semibold text-zinc-400">
                        <AlertCircle className="h-3.5 w-3.5" />
                        <span>Available</span>
                      </span>
                    )}
                  </div>

                  <div className="border-t border-white/5 pt-3">
                    <div className="text-[11px] font-mono uppercase text-zinc-500 mb-1.5">
                      Input MRI Modalities:
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {m.input_modalities.map(mod => (
                        <span
                          key={mod}
                          className="rounded-lg bg-black/40 border border-white/10 px-2.5 py-1 font-mono text-xs text-sky-200"
                        >
                          {mod}
                        </span>
                      ))}
                    </div>
                  </div>

                  {m.metrics && (
                    <div className="border-t border-white/5 pt-3">
                      <div className="text-[11px] font-mono uppercase text-zinc-500 mb-2">
                        Validation & Test Metrics:
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        {Object.entries(m.metrics).map(([k, v]) => (
                          <div key={k} className="flex justify-between rounded-lg bg-black/30 px-3 py-1.5 border border-white/5">
                            <span className="text-zinc-400">{k.replace('_', ' ')}:</span>
                            <span className="font-mono font-semibold text-emerald-400">
                              {typeof v === 'number' ? (v <= 1 ? (v * 100).toFixed(1) + '%' : v) : String(v)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </AppShell>
  )
}
