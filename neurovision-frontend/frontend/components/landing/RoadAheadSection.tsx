'use client'

import { Milestone, CheckCircle2, Clock, GitBranch, ShieldCheck } from 'lucide-react'

const ROADMAP = [
  {
    phase: 'Phase 1 · Foundation',
    title: 'Baseline 3D ResNet & NIfTI Engine',
    status: 'Ready & Operational',
    statusBadge: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300',
    desc: '4-modality NIfTI ingestion, 3D slice rendering, BraTS sub-region volumetry (WT/TC/ET), and baseline classification inference.',
  },
  {
    phase: 'Phase 2 · Advanced Encoders',
    title: 'Multimodal Hierarchical SwinUNETR',
    status: 'In Development',
    statusBadge: 'border-cyan-500/40 bg-cyan-500/10 text-cyan-300',
    desc: 'Self-attention vision transformers with shifted windows for multi-scale voxel feature extraction and boundary refinement.',
  },
  {
    phase: 'Phase 3 · Cross-Attention',
    title: 'Late Attention Cross-Modal Fusion',
    status: 'Planned',
    statusBadge: 'border-amber-500/40 bg-amber-500/10 text-amber-300',
    desc: 'Cross-attention matrices between T1ce and FLAIR sequences to weight vascular enhancement against vasogenic edema automatically.',
  },
  {
    phase: 'Phase 4 · Agentic Reasoning',
    title: 'RAG & Autonomous Clinical Decision Agent',
    status: 'Active Prototype',
    statusBadge: 'border-purple-500/40 bg-purple-500/10 text-purple-300',
    desc: 'Direct integration with medical vector databases, PubMed retrieval, and automated multi-disciplinary tumor board summaries.',
  },
  {
    phase: 'Phase 5 · Edge Deployment',
    title: 'Hospital PACS & DICOM-Router Edge Nodes',
    status: 'Planned',
    statusBadge: 'border-zinc-500/40 bg-zinc-500/10 text-zinc-400',
    desc: 'On-premise zero-trust Docker/Kubernetes nodes interfacing with hospital PACS systems with hardware-accelerated TensorRT inference.',
  },
]

export default function RoadAheadSection() {
  return (
    <section className="border-t border-nv-line bg-[#03060d] py-24 px-6">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="max-w-2xl space-y-3">
          <div className="text-xs uppercase tracking-widest text-nv-cool font-semibold flex items-center gap-1.5">
            <GitBranch className="h-3.5 w-3.5" /> 11 · Architectural Roadmap
          </div>
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-5xl">
            The Road Ahead.
          </h2>
          <p className="text-base text-zinc-400 leading-relaxed pt-1">
            A transparent view of our model evolution and engineering milestones.
            Future systems are clearly designated as In Development or Planned to preserve scientific integrity.
          </p>
        </div>

        {/* Roadmap Timeline */}
        <div className="mt-14 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {ROADMAP.map((item, idx) => (
            <div
              key={idx}
              className="rounded-3xl border border-nv-line bg-nv-panel/50 p-6 backdrop-blur flex flex-col justify-between space-y-4 hover:border-white/20 transition-all"
            >
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[11px] text-zinc-500">{item.phase}</span>
                  <span
                    className={`rounded-full border px-2.5 py-0.5 text-[10px] font-mono tracking-wide ${item.statusBadge}`}
                  >
                    {item.status}
                  </span>
                </div>
                <h3 className="text-lg font-bold text-white leading-snug">{item.title}</h3>
                <p className="text-xs text-zinc-400 leading-relaxed">{item.desc}</p>
              </div>

              <div className="pt-3 border-t border-white/5 text-[10px] font-mono text-zinc-500">
                Milestone 0{idx + 1}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
