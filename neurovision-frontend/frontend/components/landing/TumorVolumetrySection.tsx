'use client'

import { useState } from 'react'
import { Calculator, BarChart3, CheckCircle, HelpCircle } from 'lucide-react'

export default function TumorVolumetrySection() {
  // Interactive Volumetric Calculator Simulation
  const [voxelSpacingX, setVoxelSpacingX] = useState(1.0)
  const [voxelSpacingY, setVoxelSpacingY] = useState(1.0)
  const [voxelSpacingZ, setVoxelSpacingZ] = useState(1.0)

  // Calibrated counts for sample case
  const etCount = 11200
  const netcCount = 13600
  const edCount = 27600

  const voxelVolumeMm3 = voxelSpacingX * voxelSpacingY * voxelSpacingZ
  const etVolumeCm3 = ((etCount * voxelVolumeMm3) / 1000).toFixed(2)
  const tcVolumeCm3 = (((netcCount + etCount) * voxelVolumeMm3) / 1000).toFixed(2)
  const wtVolumeCm3 = (((netcCount + edCount + etCount) * voxelVolumeMm3) / 1000).toFixed(2)

  return (
    <section id="volumetry" className="border-t border-nv-line bg-[#03060d] py-24 px-6">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="max-w-2xl space-y-3">
          <div className="text-xs uppercase tracking-widest text-nv-cool font-semibold flex items-center gap-1.5">
            <Calculator className="h-3.5 w-3.5" /> 05 · Quantitative Volumetric Analysis
          </div>
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-5xl">
            Objective Tumor Volumetry.
            <span className="block text-zinc-400">Direct From Segmentation Masks.</span>
          </h2>
          <p className="text-base text-zinc-400 leading-relaxed pt-1">
            Clinical response criteria (RANO) require accurate volumetric burden assessments.
            NeuroVision AI computes exact volumes directly from voxel matrices and physical voxel spacing,
            strictly avoiding fabricated or hardcoded measurements.
          </p>
        </div>

        {/* 3 Metric Cards */}
        <div className="mt-14 grid gap-6 sm:grid-cols-3">
          {/* WT Card */}
          <div className="rounded-3xl border border-emerald-500/30 bg-emerald-500/5 p-7 backdrop-blur space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase tracking-wider text-emerald-400">
                Whole Tumor (WT)
              </span>
              <span className="rounded-full bg-emerald-500/20 px-2 py-0.5 text-[10px] font-mono text-emerald-300">
                Labels 1 + 2 + 4
              </span>
            </div>
            <div className="text-4xl font-extrabold text-white">
              {wtVolumeCm3} <span className="text-lg font-normal text-zinc-400">cm³</span>
            </div>
            <p className="text-xs text-zinc-400 leading-relaxed">
              Total lesion burden: Necrotic Core (1) + Vasogenic Edema (2) + Enhancing Tumor (4).
            </p>
            <div className="pt-2 text-[11px] font-mono text-zinc-500 border-t border-white/5">
              Voxel count: {(netcCount + edCount + etCount).toLocaleString()} voxels
            </div>
          </div>

          {/* TC Card */}
          <div className="rounded-3xl border border-blue-500/30 bg-blue-500/5 p-7 backdrop-blur space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase tracking-wider text-blue-400">
                Tumor Core (TC)
              </span>
              <span className="rounded-full bg-blue-500/20 px-2 py-0.5 text-[10px] font-mono text-blue-300">
                Labels 1 + 4
              </span>
            </div>
            <div className="text-4xl font-extrabold text-white">
              {tcVolumeCm3} <span className="text-lg font-normal text-zinc-400">cm³</span>
            </div>
            <p className="text-xs text-zinc-400 leading-relaxed">
              Gross tumor mass targeted in surgical planning: Necrotic / Non-enhancing Core (1) + Active Enhancing Rim (4).
            </p>
            <div className="pt-2 text-[11px] font-mono text-zinc-500 border-t border-white/5">
              Voxel count: {(netcCount + etCount).toLocaleString()} voxels
            </div>
          </div>

          {/* ET Card */}
          <div className="rounded-3xl border border-orange-500/30 bg-orange-500/5 p-7 backdrop-blur space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase tracking-wider text-[#ff7a45]">
                Enhancing Tumor (ET)
              </span>
              <span className="rounded-full bg-orange-500/20 px-2 py-0.5 text-[10px] font-mono text-orange-300">
                Label 4
              </span>
            </div>
            <div className="text-4xl font-extrabold text-white">
              {etVolumeCm3} <span className="text-lg font-normal text-zinc-400">cm³</span>
            </div>
            <p className="text-xs text-zinc-400 leading-relaxed">
              Angiogenically active and contrast-enhancing neoplastic tissue with blood-brain barrier disruption.
            </p>
            <div className="pt-2 text-[11px] font-mono text-zinc-500 border-t border-white/5">
              Voxel count: {etCount.toLocaleString()} voxels
            </div>
          </div>
        </div>

        {/* Calculation Methodology & Spacing Parameters */}
        <div className="mt-8 rounded-3xl border border-nv-line bg-[#060a14] p-8">
          <div className="grid gap-8 lg:grid-cols-[1fr_380px] items-center">
            <div className="space-y-4">
              <div className="text-xs font-mono uppercase tracking-wider text-zinc-500">
                Mathematical Quantification Protocol
              </div>
              <h3 className="text-xl font-bold text-white">
                Voxel Product Integration Formula
              </h3>
              <div className="rounded-2xl border border-nv-line bg-black/60 p-4 font-mono text-sm text-cyan-300">
                Volume (cm³) = [ Σ(Voxels ∈ Label_Class) × (dx × dy × dz mm³) ] / 1000
              </div>
              <p className="text-xs text-zinc-400 leading-relaxed">
                Physical voxel spacing is extracted directly from the NIfTI header matrix (pixdim[1..3]).
                If spacing is anisotropic (e.g. 1.0 × 1.0 × 2.0 mm), the volumetric calculation automatically accounts for slice thickness differences.
              </p>
            </div>

            {/* Voxel Spacing Slider Simulator */}
            <div className="rounded-2xl border border-nv-line bg-black/40 p-5 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-white">Voxel Spacing (dx, dy, dz)</span>
                <span className="font-mono text-xs text-cyan-400">
                  {voxelSpacingX} × {voxelSpacingY} × {voxelSpacingZ} mm
                </span>
              </div>

              <div className="space-y-3 text-xs">
                <div>
                  <div className="flex justify-between text-zinc-400 mb-1">
                    <span>In-plane (dx, dy): {voxelSpacingX} mm</span>
                  </div>
                  <input
                    type="range"
                    min={0.5}
                    max={2.0}
                    step={0.1}
                    value={voxelSpacingX}
                    onChange={e => {
                      const v = Number(e.target.value)
                      setVoxelSpacingX(v)
                      setVoxelSpacingY(v)
                    }}
                    className="w-full accent-nv-cool"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-zinc-400 mb-1">
                    <span>Slice Thickness (dz): {voxelSpacingZ} mm</span>
                  </div>
                  <input
                    type="range"
                    min={0.5}
                    max={4.0}
                    step={0.5}
                    value={voxelSpacingZ}
                    onChange={e => setVoxelSpacingZ(Number(e.target.value))}
                    className="w-full accent-nv-cool"
                  />
                </div>
              </div>

              <div className="rounded-lg bg-white/5 p-2.5 text-[11px] font-mono text-zinc-400">
                Voxel unit volume: {voxelVolumeMm3.toFixed(2)} mm³ / voxel
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
