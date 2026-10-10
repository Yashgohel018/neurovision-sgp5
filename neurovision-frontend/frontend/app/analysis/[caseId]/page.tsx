'use client'

import { useParams } from 'next/navigation'
import { useState, useEffect } from 'react'
import Link from 'next/link'
import { api, sliceUrl } from '@/lib/api'
import { DEMO_RESULT } from '@/lib/demo'
import type { InferenceResult, Modality, Plane, Volumes, CaseSummary, AnalysisReport } from '@/lib/types'
import { AppShell, Stat, DemoBadge } from '@/components/ui'
import InteractiveWorkstation from '@/components/analysis/InteractiveWorkstation'
import ThreeBrainScene from '@/components/visualization/ThreeBrainScene'
import ProbabilityChart from '@/components/charts/ProbabilityChart'
import {
  Layers,
  Box,
  Eye,
  Activity,
  FileText,
  Sparkles,
  ChevronLeft,
  ChevronRight,
  ShieldAlert,
  ArrowLeft,
  User,
  Clock,
  Scissors,
  CheckCircle2,
  X,
} from 'lucide-react'

const MODS: Modality[] = ['T1', 'T1ce', 'T2', 'FLAIR']

export default function Analysis() {
  const { caseId } = useParams<{ caseId: string }>()
  const isDemo = caseId.startsWith('DEMO')

  const [activeTab, setActiveTab] = useState<'mri' | '3d'>('mri')
  const [selectedMod, setSelectedMod] = useState<Modality>('FLAIR')
  const [caseData, setCaseData] = useState<CaseSummary | null>(null)
  const [res, setRes] = useState<InferenceResult | null>(null)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  const [volData, setVolData] = useState<Volumes | null>(null)
  const [report, setReport] = useState<AnalysisReport | null>(null)
  const [showReportModal, setShowReportModal] = useState(false)

  useEffect(() => {
    // Attempt to load volumes and patient metadata immediately
    if (!isDemo) {
      api.volume(caseId).then(v => setVolData(v)).catch(() => {})
      api.case(caseId).then(c => setCaseData(c)).catch(() => {})
    } else {
      setVolData({ WT_volume_cm3: 52.4, TC_volume_cm3: 24.8, ET_volume_cm3: 11.2 })
      setCaseData({
        case_id: caseId,
        created_at: '2026-10-06T09:30:00Z',
        modalities: ['T1', 'T1ce', 'T2', 'FLAIR', 'Seg'],
        status: 'ready',
        grade: 'HGG',
        age: 60.5,
        survival_days: 289,
        extent_of_resection: 'GTR',
        is_real_brats: true,
      })
    }
  }, [caseId, isDemo])

  const runAnalysis = async () => {
    setBusy(true)
    setErr('')
    try {
      const result = isDemo ? DEMO_RESULT(caseId) : await api.inference(caseId)
      setRes(result)
      if (result.tumor) setVolData(result.tumor)
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const loadReport = async () => {
    try {
      const rep = await api.report(caseId)
      setReport(rep)
      setShowReportModal(true)
    } catch (e) {
      setErr('Failed to load clinical report.')
    }
  }

  const v = res?.tumor ?? volData

  return (
    <AppShell title={`Clinical Workstation · ${caseId}`} demo={isDemo}>
      <div className="space-y-6">
        {/* Navigation Breadcrumb & Clinical Demographics Banner */}
        <div className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-nv-line/80 bg-nv-panel/60 p-4 backdrop-blur">
          <div className="flex items-center gap-3">
            <Link
              href="/cases"
              className="flex items-center gap-1.5 rounded-xl border border-nv-line bg-black/40 px-3 py-1.5 text-xs text-zinc-400 hover:text-white transition-colors"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              <span>Back to Cohort</span>
            </Link>

            <div className="h-4 w-px bg-nv-line" />

            <div className="flex items-center gap-2">
              <span className="font-mono font-bold text-white text-sm">{caseId}</span>
              {caseData?.is_real_brats && (
                <span className="rounded bg-emerald-500/15 px-2 py-0.5 text-[10px] font-mono font-semibold text-emerald-400 border border-emerald-500/30">
                  BraTS 2020 Medical Data
                </span>
              )}
              {caseData?.grade && (
                <span
                  className={`rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${
                    caseData.grade === 'HGG'
                      ? 'bg-rose-500/15 text-rose-300 border border-rose-500/30'
                      : 'bg-sky-500/15 text-sky-300 border border-sky-500/30'
                  }`}
                >
                  Grade: {caseData.grade}
                </span>
              )}
            </div>
          </div>

          {/* Clinical Covariates Pill Strip */}
          <div className="flex flex-wrap items-center gap-2 text-xs">
            {caseData?.age !== null && caseData?.age !== undefined && (
              <span className="rounded-lg bg-black/50 px-2.5 py-1 text-zinc-300 font-mono border border-nv-line">
                Age: <b className="text-white">{caseData.age.toFixed(1)}y</b>
              </span>
            )}
            {caseData?.survival_days !== null && caseData?.survival_days !== undefined && (
              <span className="rounded-lg bg-black/50 px-2.5 py-1 text-zinc-300 font-mono border border-nv-line">
                Survival: <b className="text-white">{Math.round(caseData.survival_days)}d</b>
              </span>
            )}
            {caseData?.extent_of_resection && (
              <span className="rounded-lg bg-black/50 px-2.5 py-1 text-zinc-300 font-mono border border-nv-line">
                Resection: <b className="text-white">{caseData.extent_of_resection}</b>
              </span>
            )}
            {caseData?.split && (
              <span className="rounded-lg bg-zinc-800 px-2.5 py-1 text-zinc-400 font-mono uppercase text-[11px] border border-white/5">
                Split: {caseData.split}
              </span>
            )}
          </div>
        </div>

        {/* Workspace Top Tab Bar (Multi-planar Viewer vs 3D Volumetric Mesh) */}
        <div className="flex flex-wrap items-center justify-between border-b border-nv-line/70 pb-4">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setActiveTab('mri')}
              className={`flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-semibold transition-all ${
                activeTab === 'mri'
                  ? 'bg-nv-cool text-white shadow-[0_0_16px_rgba(77,155,255,0.4)]'
                  : 'text-zinc-400 hover:text-white bg-nv-panel/60 border border-nv-line'
              }`}
            >
              <Eye className="h-4 w-4" />
              <span>Multi-Planar MRI Viewer (Axial / Coronal / Sagittal)</span>
            </button>

            <button
              onClick={() => setActiveTab('3d')}
              className={`flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-semibold transition-all ${
                activeTab === '3d'
                  ? 'bg-cyan-500 text-black shadow-[0_0_16px_rgba(56,189,248,0.4)]'
                  : 'text-zinc-400 hover:text-white bg-nv-panel/60 border border-nv-line'
              }`}
            >
              <Box className="h-4 w-4" />
              <span>Interactive 3D Tumor Geometry</span>
            </button>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={runAnalysis}
              disabled={busy}
              className="flex items-center gap-2 rounded-full bg-white px-5 py-2 text-xs font-semibold text-black hover:bg-zinc-200 transition-colors disabled:opacity-50 shadow-sm"
            >
              <Activity className="h-3.5 w-3.5" />
              <span>{busy ? 'Running Inference…' : 'Run Neural Inference'}</span>
            </button>
          </div>
        </div>

        {/* Central Workspace Layout */}
        <div className="grid gap-6 xl:grid-cols-[1fr_340px]">
          {/* Main Visual Display (Center) */}
          <div className="space-y-4">
            {activeTab === 'mri' ? (
              <InteractiveWorkstation
                caseId={caseId}
                initialModality={selectedMod}
                initialPlane="axial"
                initialIndex={67}
              />
            ) : (
              <ThreeBrainScene height="640px" initialCaseId={caseId} />
            )}

            {/* BOTTOM: MRI Modality Synchronized Quick Strip */}
            <div className="rounded-2xl border border-nv-line bg-nv-panel/50 p-4 backdrop-blur">
              <div className="mb-2 text-xs font-mono uppercase tracking-wider text-zinc-500">
                Synchronized Sequence Strip (Click to Switch):
              </div>
              <div className="grid grid-cols-4 gap-4">
                {MODS.map(m => {
                  const url = sliceUrl(caseId, m, 'axial', 67)
                  const isCurrent = selectedMod === m
                  return (
                    <button
                      key={m}
                      onClick={() => {
                        setSelectedMod(m)
                        setActiveTab('mri')
                      }}
                      className={`group relative overflow-hidden rounded-xl border p-2 text-left transition-all ${
                        isCurrent
                          ? 'border-nv-cool ring-1 ring-nv-cool bg-[#070e1c]'
                          : 'border-nv-line bg-black/40 hover:border-white/20'
                      }`}
                    >
                      <div className="aspect-video w-full overflow-hidden rounded-lg bg-black">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={url}
                          alt={`${m} preview`}
                          className="h-full w-full object-contain group-hover:scale-105 transition-transform"
                        />
                      </div>
                      <div className="mt-2 flex items-center justify-between text-xs">
                        <span className="font-bold text-white">{m}</span>
                        <span className="text-[10px] text-zinc-500 font-mono">155 Slices</span>
                      </div>
                    </button>
                  )
                })}
              </div>
            </div>
          </div>

          {/* RIGHT: Results & Volumetric Quantifications */}
          <aside className="space-y-6" aria-label="Analysis Results">
            {err && (
              <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-400">
                {err}
              </div>
            )}

            {/* Tumor Volumetric Card */}
            <div className="rounded-2xl border border-nv-line bg-nv-panel/70 p-5 space-y-4 backdrop-blur">
              <div className="text-xs uppercase tracking-widest text-zinc-400 font-semibold flex items-center justify-between">
                <span>Tumor Volumetry</span>
                <span className="text-[10px] font-mono text-emerald-400">BraTS Standard</span>
              </div>

              {v ? (
                <div className="space-y-3">
                  <div className="grid grid-cols-3 gap-2">
                    <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-3">
                      <div className="text-[10px] font-mono uppercase text-emerald-400">WT</div>
                      <div className="text-lg font-bold text-white mt-1">
                        {v.WT_volume_cm3.toFixed(1)} <span className="text-[10px] text-zinc-500">cm³</span>
                      </div>
                      <div className="text-[9px] text-zinc-500 mt-0.5">Whole Tumor</div>
                    </div>

                    <div className="rounded-xl border border-blue-500/30 bg-blue-500/5 p-3">
                      <div className="text-[10px] font-mono uppercase text-blue-400">TC</div>
                      <div className="text-lg font-bold text-white mt-1">
                        {v.TC_volume_cm3.toFixed(1)} <span className="text-[10px] text-zinc-500">cm³</span>
                      </div>
                      <div className="text-[9px] text-zinc-500 mt-0.5">Tumor Core</div>
                    </div>

                    <div className="rounded-xl border border-orange-500/30 bg-orange-500/5 p-3">
                      <div className="text-[10px] font-mono uppercase text-orange-400">ET</div>
                      <div className="text-lg font-bold text-white mt-1">
                        {v.ET_volume_cm3.toFixed(1)} <span className="text-[10px] text-zinc-500">cm³</span>
                      </div>
                      <div className="text-[9px] text-zinc-500 mt-0.5">Enhancing Rim</div>
                    </div>
                  </div>

                  {v.WT_volume_cm3 > 0 && (
                    <div className="rounded-xl border border-white/5 bg-black/40 p-2.5 text-[11px] font-mono space-y-1">
                      <div className="flex justify-between text-zinc-400">
                        <span>Active Enhancing Fraction:</span>
                        <b className="text-orange-400">
                          {((v.ET_volume_cm3 / v.WT_volume_cm3) * 100).toFixed(1)}%
                        </b>
                      </div>
                      <div className="flex justify-between text-zinc-400">
                        <span>Resectable Core Fraction:</span>
                        <b className="text-blue-400">
                          {((v.TC_volume_cm3 / v.WT_volume_cm3) * 100).toFixed(1)}%
                        </b>
                      </div>
                    </div>
                  )}

                  <div className="text-[10px] font-mono text-zinc-500">
                    Formula: Voxel count × (dx × dy × dz) / 1000 cm³
                  </div>
                </div>
              ) : (
                <p className="text-xs text-zinc-500 leading-relaxed">
                  Volumetric metrics pending segmentation mask calculation.
                </p>
              )}
            </div>

            {/* Neural Classification Result */}
            <div className="rounded-2xl border border-nv-line bg-nv-panel/70 p-5 space-y-4 backdrop-blur">
              <div className="text-xs uppercase tracking-widest text-zinc-400 font-semibold flex items-center justify-between">
                <span>Neural Diagnosis</span>
                <span className="text-[10px] font-mono text-sky-400">MultiModal-BrainNet</span>
              </div>

              {res?.prediction && res.probabilities ? (
                <div className="space-y-3.5">
                  <div className="flex items-center justify-between">
                    <span
                      className={`text-lg font-bold ${
                        res.prediction.includes('HGG') ? 'text-rose-400' : 'text-sky-400'
                      }`}
                    >
                      {res.prediction}
                    </span>
                    {res.confidence_score && (
                      <span className="font-mono text-xs font-semibold text-zinc-300">
                        {(res.confidence_score * 100).toFixed(1)}% Conf
                      </span>
                    )}
                  </div>
                  <ProbabilityChart probabilities={res.probabilities} />

                  {/* Modality Cross-Attention Attribution */}
                  {res.modality_attention && (
                    <div className="rounded-xl border border-white/5 bg-black/40 p-3 space-y-2">
                      <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-400 flex justify-between">
                        <span>Sequence Cross-Attention:</span>
                        <span className="text-sky-400">P2 Layer Weights</span>
                      </div>
                      <div className="grid grid-cols-4 gap-2 text-center">
                        {Object.entries(res.modality_attention).map(([mod, weight]) => {
                          const isHigh = weight > 0.3
                          return (
                            <div key={mod} className="space-y-1">
                              <div className="text-[10px] font-mono text-zinc-400">{mod}</div>
                              <div
                                className={`text-xs font-bold font-mono ${
                                  mod === 'T1ce'
                                    ? 'text-rose-400'
                                    : mod === 'FLAIR'
                                    ? 'text-emerald-400'
                                    : mod === 'T2'
                                    ? 'text-amber-400'
                                    : 'text-sky-400'
                                }`}
                              >
                                {(weight * 100).toFixed(1)}%
                              </div>
                              <div className="w-full bg-white/5 h-1.5 rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${
                                    mod === 'T1ce'
                                      ? 'bg-rose-400'
                                      : mod === 'FLAIR'
                                      ? 'bg-emerald-400'
                                      : mod === 'T2'
                                      ? 'bg-amber-400'
                                      : 'bg-sky-400'
                                  }`}
                                  style={{ width: `${Math.min(100, weight * 100)}%` }}
                                />
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )}

                  {/* Scientific Rationale */}
                  {res.rationale && (
                    <div className="rounded-xl border border-sky-500/20 bg-sky-500/5 p-3 text-xs text-zinc-300 leading-relaxed">
                      <div className="text-[10px] font-mono uppercase text-sky-400 mb-1">
                        Cross-Modality Evidence:
                      </div>
                      <p>{res.rationale}</p>
                    </div>
                  )}

                  <div className="text-[10px] font-mono text-zinc-500 border-t border-white/5 pt-2">
                    {res.model_version} · Latency: {res.inference_time ?? 0.02}s
                  </div>
                </div>
              ) : (
                <div className="rounded-xl border border-white/5 bg-black/40 p-3 text-xs text-zinc-400 space-y-2">
                  <p>
                    {res?.message ??
                      'Click "Run Neural Inference" to classify tumor grade and evaluate calibrated probabilities.'}
                  </p>
                </div>
              )}
            </div>

            {/* Clinical Decision Report Action */}
            <div className="rounded-2xl border border-nv-line bg-nv-panel/50 p-4 flex items-center justify-between">
              <div className="text-xs text-zinc-300">
                <FileText className="inline h-4 w-4 mr-1 text-nv-cool" />
                Diagnostic Report
              </div>
              <button
                onClick={loadReport}
                className="rounded-full border border-white/20 bg-white/5 px-3 py-1 text-[11px] text-white hover:bg-white hover:text-black transition-colors"
              >
                Generate Report →
              </button>
            </div>

            {/* Clinical Safety Disclaimer */}
            <div className="flex items-start gap-2 rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-[11px] text-zinc-400">
              <ShieldAlert className="h-4 w-4 text-amber-400 flex-shrink-0 mt-0.5" />
              <span>Research clinical decision-support system. Requires qualified radiological review.</span>
            </div>
          </aside>
        </div>

        {/* Modal: Clinical Decision Support Report */}
        {showReportModal && report && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
            <div className="relative max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded-3xl border border-nv-line bg-[#040813] p-6 shadow-2xl">
              <div className="flex items-center justify-between border-b border-nv-line pb-4">
                <div>
                  <h3 className="text-base font-bold text-white">Clinical Decision Support Report</h3>
                  <p className="text-xs font-mono text-zinc-400">Case ID: {report.case_id} · Generated: {report.generated_at}</p>
                </div>
                <button
                  onClick={() => setShowReportModal(false)}
                  className="rounded-lg p-1.5 text-zinc-400 hover:bg-white/10 hover:text-white"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              <div className="mt-5 space-y-5 text-xs text-zinc-300 leading-relaxed">
                <div>
                  <h4 className="font-semibold uppercase tracking-wider text-nv-cool mb-2">Radiological Findings</h4>
                  <ul className="space-y-1.5 list-disc pl-4 text-zinc-300">
                    {report.radiological_findings.map((f, i) => (
                      <li key={i}>{f}</li>
                    ))}
                  </ul>
                </div>

                {report.tumor_volumes && (
                  <div>
                    <h4 className="font-semibold uppercase tracking-wider text-nv-cool mb-2">Quantified Volumetrics</h4>
                    <div className="grid grid-cols-3 gap-2 font-mono">
                      <div className="rounded-lg border border-nv-line bg-black/40 p-2">
                        WT: <b>{report.tumor_volumes.WT_volume_cm3} cm³</b>
                      </div>
                      <div className="rounded-lg border border-nv-line bg-black/40 p-2">
                        TC: <b>{report.tumor_volumes.TC_volume_cm3} cm³</b>
                      </div>
                      <div className="rounded-lg border border-nv-line bg-black/40 p-2">
                        ET: <b>{report.tumor_volumes.ET_volume_cm3} cm³</b>
                      </div>
                    </div>
                  </div>
                )}

                {report.prediction && (
                  <div>
                    <h4 className="font-semibold uppercase tracking-wider text-nv-cool mb-2">Model Prediction & Sub-typing</h4>
                    <div className="rounded-xl border border-nv-line bg-black/40 p-3 font-mono">
                      <div className="text-sm font-bold text-white">{report.prediction}</div>
                      {report.probabilities && (
                        <div className="mt-1 text-zinc-400">
                          HGG: {(report.probabilities.HGG * 100).toFixed(1)}% · LGG: {(report.probabilities.LGG * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                  </div>
                )}

                <div>
                  <h4 className="font-semibold uppercase tracking-wider text-nv-cool mb-1">Methodology</h4>
                  <p className="text-zinc-400">{report.methodology}</p>
                </div>

                <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-[11px] text-zinc-400">
                  <b>Clinical Disclaimer:</b> {report.disclaimer}
                </div>
              </div>

              <div className="mt-6 flex justify-end">
                <button
                  onClick={() => setShowReportModal(false)}
                  className="rounded-full bg-white px-5 py-2 text-xs font-semibold text-black hover:bg-zinc-200"
                >
                  Close Report
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </AppShell>
  )
}
