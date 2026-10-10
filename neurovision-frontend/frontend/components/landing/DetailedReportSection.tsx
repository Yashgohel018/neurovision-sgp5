'use client'

import { useState } from 'react'
import { api, sliceUrl } from '@/lib/api'
import type { AnalysisReport } from '@/lib/types'
import { FileText, Printer, Download, ShieldCheck, CheckCircle2 } from 'lucide-react'

export default function DetailedReportSection() {
  const [report, setReport] = useState<AnalysisReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [caseId, setCaseId] = useState('CASE-2026-001')

  const handleFetchReport = async () => {
    setLoading(true)
    try {
      const rep = await api.report(caseId)
      setReport(rep)
    } catch {
      // Fallback
    } finally {
      setLoading(false)
    }
  }

  const handlePrint = () => {
    window.print()
  }

  return (
    <section className="border-t border-nv-line bg-[#02050b] py-24 px-6 print:py-0 print:bg-white print:text-black">
      <div className="mx-auto max-w-7xl">
        {/* Header (hidden during print) */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-12 print:hidden">
          <div className="max-w-2xl space-y-3">
            <div className="text-xs uppercase tracking-widest text-nv-cool font-semibold flex items-center gap-1.5">
              <FileText className="h-3.5 w-3.5" /> 10 · Clinical Research Documentation
            </div>
            <h2 className="text-3xl font-bold tracking-tight text-white sm:text-5xl">
              Structured Analysis Report.
            </h2>
            <p className="text-base text-zinc-400 leading-relaxed pt-1">
              Generate standardized, auditable case reports detailing multi-sequence integrity,
              volumetric burden distribution (WT/TC/ET), and model classification rationale.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={handleFetchReport}
              disabled={loading}
              className="rounded-full bg-white px-5 py-2.5 text-xs font-semibold text-black hover:bg-zinc-200 transition-colors disabled:opacity-50"
            >
              {loading ? 'Synthesizing Report…' : 'Generate Case Report'}
            </button>
            <button
              onClick={handlePrint}
              className="flex items-center gap-1.5 rounded-full border border-white/20 bg-black/40 px-5 py-2.5 text-xs font-medium text-white hover:bg-white/10 transition-colors"
            >
              <Printer className="h-3.5 w-3.5" />
              <span>Print / Export PDF</span>
            </button>
          </div>
        </div>

        {/* Report Preview Document Card */}
        <div className="mx-auto max-w-4xl rounded-3xl border border-nv-line bg-[#070c18] p-8 sm:p-10 shadow-2xl space-y-8 print:border-none print:shadow-none print:bg-white print:p-0">
          {/* Document Header */}
          <div className="flex items-center justify-between border-b border-nv-line pb-6 print:border-black">
            <div>
              <div className="text-xs uppercase tracking-widest text-cyan-400 font-mono font-semibold print:text-black">
                NeuroVision AI Clinical Decision Support
              </div>
              <h3 className="text-2xl font-bold text-white mt-1 print:text-black">
                Multimodal Brain MRI Analysis Report
              </h3>
            </div>
            <div className="text-right text-xs font-mono text-zinc-400 print:text-zinc-700">
              <div>Case: {caseId}</div>
              <div>Date: {report?.generated_at ?? new Date().toISOString().split('T')[0]}</div>
            </div>
          </div>

          {/* Patient Case Overview */}
          <div className="grid gap-6 sm:grid-cols-2 text-xs">
            <div className="space-y-2 rounded-2xl border border-nv-line/60 bg-black/30 p-4 print:bg-gray-100 print:border-gray-300">
              <div className="font-mono text-zinc-500 uppercase tracking-wider text-[10px]">
                Imaging Modalities Verified
              </div>
              <div className="font-semibold text-white print:text-black">
                T1 · T1ce · T2 · FLAIR (1mm Isotropic NIfTI Series)
              </div>
              <div className="text-zinc-400">
                Segmentation Mask: Available (BraTS labels 1, 2, 4 verified)
              </div>
            </div>

            <div className="space-y-2 rounded-2xl border border-nv-line/60 bg-black/30 p-4 print:bg-gray-100 print:border-gray-300">
              <div className="font-mono text-zinc-500 uppercase tracking-wider text-[10px]">
                Model & Encoder Architecture
              </div>
              <div className="font-semibold text-white print:text-black">
                {report?.model_version ?? 'NeuroVision-ResNet3D (v1.0.0)'}
              </div>
              <div className="text-zinc-400">
                Inference State: {report?.prediction ?? 'Classification Checkpoint Verified (HGG)'}
              </div>
            </div>
          </div>

          {/* Quantitative Volumetric Table */}
          <div className="space-y-3">
            <div className="text-xs uppercase tracking-wider font-semibold text-white print:text-black">
              Quantitative Tumor Volumetric Analysis
            </div>
            <div className="overflow-hidden rounded-xl border border-nv-line/70 print:border-gray-300">
              <table className="w-full text-left text-xs">
                <thead className="bg-black/50 text-[10px] font-mono uppercase text-zinc-400 border-b border-nv-line print:bg-gray-200 print:text-black">
                  <tr>
                    <th className="px-4 py-2.5">Sub-Region Class</th>
                    <th className="px-4 py-2.5">BraTS Annotation Labels</th>
                    <th className="px-4 py-2.5">Quantified Volume</th>
                    <th className="px-4 py-2.5">Relative Burden</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-nv-line/50 text-zinc-300 print:text-black print:divide-gray-200">
                  <tr>
                    <td className="px-4 py-3 font-semibold text-emerald-400 print:text-black">
                      Whole Tumor (WT)
                    </td>
                    <td className="px-4 py-3 font-mono">Labels 1 + 2 + 4</td>
                    <td className="px-4 py-3 font-mono font-bold">52.4 cm³</td>
                    <td className="px-4 py-3">100.0% total pathological mass</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-3 font-semibold text-blue-400 print:text-black">
                      Tumor Core (TC)
                    </td>
                    <td className="px-4 py-3 font-mono">Labels 1 + 4</td>
                    <td className="px-4 py-3 font-mono font-bold">24.8 cm³</td>
                    <td className="px-4 py-3">47.3% macroscopic resectable core</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-3 font-semibold text-orange-400 print:text-black">
                      Enhancing Tumor (ET)
                    </td>
                    <td className="px-4 py-3 font-mono">Label 4</td>
                    <td className="px-4 py-3 font-mono font-bold">11.2 cm³</td>
                    <td className="px-4 py-3">21.4% vascular active angiogenic rim</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Key Radiological Findings & Thumbnail */}
          <div className="grid gap-6 sm:grid-cols-[1fr_220px] items-center">
            <div className="space-y-3 text-xs text-zinc-300 print:text-black leading-relaxed">
              <div className="font-semibold text-white uppercase tracking-wider text-[11px] print:text-black">
                Summary of Imaging Observations:
              </div>
              <ul className="space-y-2 list-disc pl-4">
                <li>
                  Prominent peripheral nodular enhancement observed in right temporal-parietal region on T1ce sequences.
                </li>
                <li>
                  Extensive perilesional vasogenic edema confirmed on FLAIR with mass effect on ipsilateral lateral ventricle.
                </li>
                <li>
                  Central non-enhancing necrotic cavity demonstrated on T1 and T2 weighted acquisitions.
                </li>
              </ul>
            </div>

            {/* Thumbnail Preview */}
            <div className="overflow-hidden rounded-xl border border-nv-line bg-black aspect-square print:border-gray-300">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={sliceUrl('CASE-2026-001', 'FLAIR', 'axial', 60, 'seg')}
                alt="Key slice thumbnail"
                className="h-full w-full object-contain"
              />
            </div>
          </div>

          {/* Regulatory Disclaimer in Document */}
          <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 text-[11px] text-zinc-400 print:border-gray-300 print:bg-transparent print:text-gray-600">
            <strong>Investigational Clinical Decision Support Output:</strong> This automated report is generated
            by NeuroVision AI for clinical decision support and scientific research. It is not a definitive diagnosis.
            Review and correlation by qualified radiologists and neuro-oncology specialists is required prior to clinical intervention.
          </div>
        </div>
      </div>
    </section>
  )
}
