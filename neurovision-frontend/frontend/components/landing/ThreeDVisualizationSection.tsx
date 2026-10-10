'use client'

import ThreeBrainScene from '@/components/visualization/ThreeBrainScene'
import { Box, Sparkles, Orbit } from 'lucide-react'

export default function ThreeDVisualizationSection() {
  return (
    <section id="3d" className="border-t border-nv-line bg-[#02050b] py-24 px-6">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-12">
          <div className="max-w-2xl space-y-3">
            <div className="text-xs uppercase tracking-widest text-nv-cool font-semibold flex items-center gap-1.5">
              <Box className="h-3.5 w-3.5" /> 04 · Volumetric Surface Reconstruction
            </div>
            <h2 className="text-3xl font-bold tracking-tight text-white sm:text-5xl">
              Interactive 3D Tumor Geometry.
              <span className="block text-zinc-400">BraTS Sub-Region Spatial Layers.</span>
            </h2>
            <p className="text-base text-zinc-400 leading-relaxed pt-1">
              Inspect the complex 3D topography of the neoplasm embedded within the parenchymal cortex.
              Toggle between the anatomical brain frame, enhancing tumor core (ET), peritumoral vasogenic edema (WT), and calibrated ROI bounding box.
            </p>
          </div>

          <div className="flex items-center gap-2 rounded-2xl border border-white/10 bg-nv-panel/50 p-3 text-xs text-zinc-400 backdrop-blur">
            <Orbit className="h-4 w-4 text-cyan-400 flex-shrink-0 animate-spin" />
            <span>WebGL 3D Engine · Full Orbit, Pan & Layer Toggles</span>
          </div>
        </div>

        {/* 3D Scene Container */}
        <ThreeBrainScene height="620px" className="shadow-[0_0_80px_rgba(0,0,0,0.9)]" />

        {/* Scientific Annotation Cards Below 3D Scene */}
        <div className="mt-8 grid gap-4 sm:grid-cols-3">
          <div className="rounded-2xl border border-nv-line bg-nv-panel/40 p-5 space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-cyan-300">
              <span className="h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_8px_#38bdf8]" />
              Anatomical Brain Envelope
            </div>
            <p className="text-xs text-zinc-400 leading-relaxed">
              Synthesized from non-zero cortical brain voxels, establishing anatomical spatial context and orientation relative to surrounding gyri and sulci.
            </p>
          </div>

          <div className="rounded-2xl border border-nv-line bg-nv-panel/40 p-5 space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-orange-300">
              <span className="h-2 w-2 rounded-full bg-[#ff7a45] shadow-[0_0_8px_#ff7a45]" />
              Enhancing Tumor Core (ET)
            </div>
            <p className="text-xs text-zinc-400 leading-relaxed">
              Derived from BraTS Label 4 (post-contrast hyperintensity). Visualized in restrained copper/orange with surface deformation capturing vascular permeability.
            </p>
          </div>

          <div className="rounded-2xl border border-nv-line bg-nv-panel/40 p-5 space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-emerald-300">
              <span className="h-2 w-2 rounded-full bg-[#59d98c] shadow-[0_0_8px_#59d98c]" />
              Peritumoral Edema Halo (WT)
            </div>
            <p className="text-xs text-zinc-400 leading-relaxed">
              Semi-transparent green/cyan shell encompassing vasogenic edema (Label 2) and necrotic core (Label 1), mapping the full infiltrating pathological margin.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
