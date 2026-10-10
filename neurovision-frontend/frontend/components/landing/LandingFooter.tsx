import Link from 'next/link'
import { ShieldAlert, Terminal, FileText, Database } from 'lucide-react'

export default function LandingFooter() {
  return (
    <footer className="border-t border-nv-line bg-[#020409] text-zinc-400">
      {/* Medical Disclaimer Banner */}
      <div className="border-b border-nv-line/60 bg-amber-500/5 px-6 py-4">
        <div className="mx-auto flex max-w-7xl items-start gap-3">
          <ShieldAlert className="h-5 w-5 flex-shrink-0 text-amber-400 mt-0.5" />
          <p className="text-xs leading-relaxed text-zinc-300">
            <strong className="text-amber-400 font-semibold uppercase tracking-wider">
              Medical Research Disclaimer:
            </strong>{' '}
            NeuroVision AI is an AI-assisted research and clinical decision-support prototype. It does not replace
            clinical diagnosis, pathology verification, or judgment by board-certified radiologists and neuro-oncologists.
            All model predictions, Grad-CAM attention overlays, and volumetric calculations require human expert oversight.
          </p>
        </div>
      </div>

      {/* Main Footer Links */}
      <div className="mx-auto max-w-7xl px-6 py-12">
        <div className="grid gap-8 sm:grid-cols-2 md:grid-cols-4">
          <div className="space-y-3">
            <div className="text-sm font-semibold text-white">NeuroVision AI</div>
            <p className="text-xs text-zinc-500 leading-relaxed">
              Multimodal 3D brain MRI segmentation, quantitative tumor volumetry, and explainable deep learning decision support.
            </p>
            <div className="pt-2 text-[11px] font-mono text-zinc-600">
              Target Desktop: 1920×1080 / 1440×900
            </div>
          </div>

          <div>
            <div className="mb-3 text-xs uppercase tracking-widest text-zinc-500 font-semibold">
              Clinical Workspace
            </div>
            <ul className="space-y-2 text-xs">
              <li>
                <Link href="/dashboard" className="hover:text-white transition-colors">
                  System Dashboard
                </Link>
              </li>
              <li>
                <Link href="/cases" className="hover:text-white transition-colors">
                  Case Repository
                </Link>
              </li>
              <li>
                <Link href="/analysis/CASE-2026-001" className="hover:text-white transition-colors">
                  Multi-planar Analysis
                </Link>
              </li>
              <li>
                <Link href="/assistant" className="hover:text-white transition-colors">
                  Clinical AI Assistant (RAG)
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <div className="mb-3 text-xs uppercase tracking-widest text-zinc-500 font-semibold">
              Standards & Pipelines
            </div>
            <ul className="space-y-2 text-xs">
              <li className="flex items-center gap-1.5">
                <Database className="h-3 w-3 text-nv-cool" /> BraTS Benchmark Standards
              </li>
              <li className="flex items-center gap-1.5">
                <FileText className="h-3 w-3 text-nv-cool" /> NIfTI-1 / NIfTI-2 (.nii.gz)
              </li>
              <li className="flex items-center gap-1.5">
                <Terminal className="h-3 w-3 text-nv-cool" /> Grad-CAM Attention Heatmaps
              </li>
              <li className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-emerald-400" /> WHO 2021 CNS Classification
              </li>
            </ul>
          </div>

          <div>
            <div className="mb-3 text-xs uppercase tracking-widest text-zinc-500 font-semibold">
              API & Environment
            </div>
            <ul className="space-y-2 text-xs">
              <li>
                <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" className="hover:text-white transition-colors underline">
                  FastAPI Swagger Docs
                </a>
              </li>
              <li>
                <a href="http://localhost:8000/api/health" target="_blank" rel="noreferrer" className="hover:text-white transition-colors underline">
                  Backend Health Endpoint
                </a>
              </li>
              <li className="text-zinc-500">FastAPI · PyTorch · Three.js · Next.js 14</li>
            </ul>
          </div>
        </div>

        <div className="mt-12 flex flex-wrap items-center justify-between border-t border-nv-line/60 pt-6 text-[11px] text-zinc-600">
          <div>© 2026 NeuroVision AI Project. Research & Clinical Decision Support.</div>
          <div className="font-mono">v1.0.0-production-prototype</div>
        </div>
      </div>
    </footer>
  )
}
