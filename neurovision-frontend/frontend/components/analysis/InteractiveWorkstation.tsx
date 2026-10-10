'use client'

import { useState, useRef, useEffect } from 'react'
import { sliceUrl } from '@/lib/api'
import type { Modality, Plane } from '@/lib/types'
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Crosshair,
  Sliders,
  Layers,
  Sparkles,
  Maximize2,
  ChevronLeft,
  ChevronRight,
  Sun,
  Activity,
} from 'lucide-react'

interface InteractiveWorkstationProps {
  caseId?: string
  initialModality?: Modality
  initialPlane?: Plane
  initialIndex?: number
  className?: string
  showAllControls?: boolean
}

const MODALITIES: { id: Modality; label: string; desc: string }[] = [
  { id: 'T1', label: 'T1', desc: 'Anatomical Structure' },
  { id: 'T1ce', label: 'T1ce', desc: 'Active Enhancing Core' },
  { id: 'T2', label: 'T2', desc: 'Edema & Tissue Pathology' },
  { id: 'FLAIR', label: 'FLAIR', desc: 'CSF-Suppressed Infiltration' },
]

const PLANES: { id: Plane; label: string }[] = [
  { id: 'axial', label: 'Axial' },
  { id: 'coronal', label: 'Coronal' },
  { id: 'sagittal', label: 'Sagittal' },
]

export default function InteractiveWorkstation({
  caseId = 'CASE-2026-001',
  initialModality = 'FLAIR',
  initialPlane = 'axial',
  initialIndex = 60,
  className = '',
  showAllControls = true,
}: InteractiveWorkstationProps) {
  const [modality, setModality] = useState<Modality>(initialModality)
  const [plane, setPlane] = useState<Plane>(initialPlane)
  const [index, setIndex] = useState(initialIndex)
  const [overlay, setOverlay] = useState<'none' | 'seg' | 'cam'>('seg')
  const [opacity, setOpacity] = useState(0.65)
  const [zoom, setZoom] = useState(1.0)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [isPanning, setIsPanning] = useState(false)
  const panStartRef = useRef({ x: 0, y: 0 })
  const [brightness, setBrightness] = useState(100)
  const [contrast, setContrast] = useState(100)
  const [showCrosshair, setShowCrosshair] = useState(true)
  const [crosshairPos, setCrosshairPos] = useState({ x: 152, y: 108 })
  const [imageError, setImageError] = useState(false)

  const viewerRef = useRef<HTMLDivElement>(null)
  const canvasFallbackRef = useRef<HTMLCanvasElement>(null)

  const currentSrc = sliceUrl(
    caseId,
    modality,
    plane,
    index,
    overlay === 'none' ? undefined : overlay
  )

  useEffect(() => {
    setImageError(false)
  }, [currentSrc])

  // Mouse wheel zoom listener directly inside viewer
  useEffect(() => {
    const el = viewerRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const factor = e.deltaY < 0 ? 1.15 : 0.87
      setZoom((prev) => Math.max(0.8, Math.min(6.0, Math.round(prev * factor * 10) / 10)))
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  // Fallback synthetic high-res MRI rendering if offline or image fails
  useEffect(() => {
    if (!imageError || !canvasFallbackRef.current) return
    const canvas = canvasFallbackRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.fillStyle = '#03060d'
    ctx.fillRect(0, 0, 320, 320)

    const rel = Math.max(0.2, 1 - Math.abs(index - 60) / 110)
    const cx = 160, cy = 160
    const rx = 115 * rel, ry = 135 * rel

    // Skull outline
    ctx.strokeStyle = '#555e70'
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.ellipse(cx, cy, rx + 8, ry + 8, 0, 0, Math.PI * 2)
    ctx.stroke()

    // Brain tissue
    ctx.fillStyle = modality === 'T1' ? '#252a36' : modality === 'T2' ? '#3c4659' : '#1e2430'
    ctx.beginPath()
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2)
    ctx.fill()

    // Ventricles
    ctx.fillStyle = modality === 'T2' ? '#8fa3c7' : '#0a0d14'
    ctx.beginPath()
    ctx.ellipse(cx - 18 * rel, cy, 10 * rel, 35 * rel, 0, 0, Math.PI * 2)
    ctx.ellipse(cx + 18 * rel, cy, 10 * rel, 35 * rel, 0, 0, Math.PI * 2)
    ctx.fill()

    // Lesion
    const tx = 190, ty = 135
    if (overlay === 'seg') {
      // WT edema
      ctx.fillStyle = `rgba(89, 217, 140, ${opacity * 0.7})`
      ctx.beginPath()
      ctx.arc(tx, ty, 34 * rel, 0, Math.PI * 2)
      ctx.fill()

      // TC Core
      ctx.fillStyle = `rgba(77, 155, 255, ${opacity * 0.85})`
      ctx.beginPath()
      ctx.arc(tx, ty, 22 * rel, 0, Math.PI * 2)
      ctx.fill()

      // ET Enhancing
      ctx.fillStyle = `rgba(255, 122, 69, ${opacity})`
      ctx.beginPath()
      ctx.arc(tx + 4, ty - 2, 14 * rel, 0, Math.PI * 2)
      ctx.fill()
    } else if (overlay === 'cam') {
      const grad = ctx.createRadialGradient(tx, ty, 2, tx, ty, 36)
      grad.addColorStop(0, `rgba(255, 50, 50, ${opacity})`)
      grad.addColorStop(0.5, `rgba(255, 220, 0, ${opacity * 0.7})`)
      grad.addColorStop(1, 'transparent')
      ctx.fillStyle = grad
      ctx.beginPath()
      ctx.arc(tx, ty, 38, 0, Math.PI * 2)
      ctx.fill()
    }
  }, [imageError, modality, plane, index, overlay, opacity])

  const handleViewerClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!viewerRef.current || isPanning) return
    const rect = viewerRef.current.getBoundingClientRect()
    const x = Math.round(e.clientX - rect.left)
    const y = Math.round(e.clientY - rect.top)
    setCrosshairPos({ x, y })
  }

  const handlePointerDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (zoom > 1.0 || e.button === 1 || e.button === 2) {
      setIsPanning(true)
      panStartRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y }
    }
  }

  const handlePointerMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (isPanning) {
      setPan({ x: e.clientX - panStartRef.current.x, y: e.clientY - panStartRef.current.y })
    }
  }

  const handlePointerUp = () => {
    setIsPanning(false)
  }

  const handleResetAdjustments = () => {
    setZoom(1.0)
    setPan({ x: 0, y: 0 })
    setBrightness(100)
    setContrast(100)
    setOpacity(0.65)
  }

  return (
    <div className={`overflow-hidden rounded-3xl border border-nv-line bg-[#040813] ${className}`}>
      {/* Top Bar: Modality Selector & Plane Tabs */}
      <div className="flex flex-wrap items-center justify-between border-b border-nv-line/70 bg-nv-panel/60 px-6 py-4 backdrop-blur-md">
        {/* Modality Selector */}
        <div className="flex items-center gap-2">
          <span className="text-xs uppercase tracking-wider text-zinc-500 mr-1">Modality:</span>
          <div className="flex rounded-xl border border-nv-line bg-black/40 p-1">
            {MODALITIES.map(m => (
              <button
                key={m.id}
                onClick={() => setModality(m.id)}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
                  modality === m.id
                    ? 'bg-nv-cool text-white shadow-[0_0_12px_rgba(77,155,255,0.4)]'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                <span>{m.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Orthogonal Plane Selector */}
        <div className="flex items-center gap-2">
          <span className="text-xs uppercase tracking-wider text-zinc-500 mr-1">View:</span>
          <div className="flex rounded-xl border border-nv-line bg-black/40 p-1">
            {PLANES.map(p => (
              <button
                key={p.id}
                onClick={() => setPlane(p.id)}
                className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
                  plane === p.id
                    ? 'bg-white/15 text-white'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {/* Overlay Mode Selector */}
        <div className="flex items-center gap-2">
          <span className="text-xs uppercase tracking-wider text-zinc-500 mr-1">Layer:</span>
          <div className="flex rounded-xl border border-nv-line bg-black/40 p-1">
            <button
              onClick={() => setOverlay('none')}
              className={`rounded-lg px-2.5 py-1.5 text-xs transition-colors ${
                overlay === 'none' ? 'bg-white/15 text-white' : 'text-zinc-400 hover:text-white'
              }`}
            >
              Anatomy
            </button>
            <button
              onClick={() => setOverlay('seg')}
              className={`flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs transition-colors ${
                overlay === 'seg'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              <Layers className="h-3 w-3" /> Segmentation
            </button>
            <button
              onClick={() => setOverlay('cam')}
              className={`flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs transition-colors ${
                overlay === 'cam'
                  ? 'bg-orange-500/20 text-orange-300 border border-orange-500/40'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              <Sparkles className="h-3 w-3" /> Grad-CAM
            </button>
          </div>
        </div>
      </div>

      {/* Main Interactive Grid */}
      <div className="grid gap-6 p-6 lg:grid-cols-[1fr_320px]">
        {/* Slice Canvas Viewport */}
        <div className="flex flex-col items-center">
          <div
            ref={viewerRef}
            onClick={handleViewerClick}
            onMouseDown={handlePointerDown}
            onMouseMove={handlePointerMove}
            onMouseUp={handlePointerUp}
            onMouseLeave={handlePointerUp}
            className={`group relative aspect-square w-full max-w-[540px] overflow-hidden rounded-2xl border border-nv-line bg-black shadow-2xl ${
              zoom > 1.0 ? (isPanning ? 'cursor-grabbing' : 'cursor-grab') : 'cursor-crosshair'
            }`}
          >
            {/* Image Slice */}
            {!imageError ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={currentSrc}
                alt={`${modality} ${plane} slice ${index}`}
                decoding="async"
                onError={() => setImageError(true)}
                style={{
                  transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                  transformOrigin: 'center center',
                  filter: `brightness(${brightness}%) contrast(${contrast}%)`,
                }}
                className="h-full w-full object-contain transition-transform duration-75 select-none"
              />
            ) : (
              <canvas
                ref={canvasFallbackRef}
                width={320}
                height={320}
                style={{
                  transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                  transformOrigin: 'center center',
                  filter: `brightness(${brightness}%) contrast(${contrast}%)`,
                }}
                className="h-full w-full object-contain"
              />
            )}

            {/* Crosshair Overlay */}
            {showCrosshair && (
              <div className="pointer-events-none absolute inset-0">
                <div
                  className="absolute left-0 right-0 border-t border-cyan-400/40"
                  style={{ top: `${crosshairPos.y}px` }}
                />
                <div
                  className="absolute bottom-0 top-0 border-l border-cyan-400/40"
                  style={{ left: `${crosshairPos.x}px` }}
                />
                <div
                  className="absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border border-cyan-300"
                  style={{ left: `${crosshairPos.x}px`, top: `${crosshairPos.y}px` }}
                />
              </div>
            )}

            {/* Viewport Info Tags */}
            <div className="absolute left-3 top-3 flex items-center gap-2">
              <span className="rounded bg-black/60 px-2 py-1 text-[11px] font-mono text-zinc-300 backdrop-blur">
                {modality} · {plane.toUpperCase()} · SLICE {index}/{plane === 'axial' ? 154 : 239}
              </span>
            </div>

            <div className="absolute right-3 top-3 flex items-center gap-2">
              <span className="rounded bg-black/60 px-2 py-1 text-[11px] font-mono text-cyan-300 backdrop-blur">
                POS: [{crosshairPos.x}, {crosshairPos.y}, {index}] mm
              </span>
            </div>

            {/* Quick Zoom Indicator */}
            {zoom !== 1.0 && (
              <div className="absolute bottom-3 left-3 rounded bg-black/60 px-2 py-1 text-[11px] font-mono text-zinc-400 backdrop-blur">
                ZOOM: {Math.round(zoom * 100)}%
              </div>
            )}
          </div>

          {/* Synchronized Slice Slider Strip */}
          {(() => {
            const maxSlice = plane === 'axial' ? 154 : 239
            return (
              <div className="mt-4 flex w-full max-w-[540px] items-center gap-3">
                <button
                  onClick={() => setIndex(Math.max(0, index - 1))}
                  disabled={index <= 0}
                  className="rounded-lg border border-nv-line bg-nv-panel p-2 text-zinc-400 hover:text-white disabled:opacity-30"
                >
                  <ChevronLeft className="h-4 w-4" />
                </button>

                <div className="flex-1">
                  <input
                    type="range"
                    min={0}
                    max={maxSlice}
                    value={Math.min(index, maxSlice)}
                    onChange={e => setIndex(Number(e.target.value))}
                    className="w-full accent-nv-cool cursor-pointer"
                  />
                </div>

                <button
                  onClick={() => setIndex(Math.min(maxSlice, index + 1))}
                  disabled={index >= maxSlice}
                  className="rounded-lg border border-nv-line bg-nv-panel p-2 text-zinc-400 hover:text-white disabled:opacity-30"
                >
                  <ChevronRight className="h-4 w-4" />
                </button>

                <span className="w-16 text-right font-mono text-xs text-zinc-300">
                  {Math.min(index, maxSlice)} / {maxSlice}
                </span>
              </div>
            )
          })()}
        </div>

        {/* Side Controls & Metric Readouts */}
        <div className="space-y-5 rounded-2xl border border-nv-line/70 bg-nv-panel/40 p-5">
          <div className="flex items-center justify-between border-b border-nv-line pb-3">
            <div className="text-xs uppercase tracking-wider text-zinc-400 font-semibold">
              Viewer Adjustments
            </div>
            <button
              onClick={handleResetAdjustments}
              title="Reset Adjustments"
              className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-300"
            >
              <RotateCcw className="h-3 w-3" /> Reset
            </button>
          </div>

          {/* Overlay Opacity */}
          {overlay !== 'none' && (
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs text-zinc-400">
                <span>Overlay Opacity</span>
                <span className="font-mono text-zinc-200">{Math.round(opacity * 100)}%</span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={opacity}
                onChange={e => setOpacity(Number(e.target.value))}
                className="w-full accent-emerald-500 cursor-pointer"
              />
            </div>
          )}

          {/* Window / Level (Brightness & Contrast) */}
          <div className="space-y-3 pt-2">
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs text-zinc-400">
                <span className="flex items-center gap-1">
                  <Sun className="h-3 w-3" /> Brightness (Window Level)
                </span>
                <span className="font-mono text-zinc-200">{brightness}%</span>
              </div>
              <input
                type="range"
                min={50}
                max={200}
                value={brightness}
                onChange={e => setBrightness(Number(e.target.value))}
                className="w-full accent-nv-cool cursor-pointer"
              />
            </div>

            <div className="space-y-1.5">
              <div className="flex justify-between text-xs text-zinc-400">
                <span className="flex items-center gap-1">
                  <Sliders className="h-3 w-3" /> Contrast (Window Width)
                </span>
                <span className="font-mono text-zinc-200">{contrast}%</span>
              </div>
              <input
                type="range"
                min={50}
                max={200}
                value={contrast}
                onChange={e => setContrast(Number(e.target.value))}
                className="w-full accent-nv-cool cursor-pointer"
              />
            </div>
          </div>

          {/* Advanced Zoom and Pan Inspection Tools */}
          <div className="space-y-2 pt-2 border-t border-nv-line/50">
            <div className="flex justify-between items-center text-xs text-zinc-400">
              <span className="font-semibold text-zinc-200">Slice Magnification</span>
              <span className="font-mono text-cyan-300 font-bold bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-500/30">
                {Math.round(zoom * 100)}% ({zoom.toFixed(1)}x)
              </span>
            </div>

            <input
              type="range"
              min={0.8}
              max={6.0}
              step={0.1}
              value={zoom}
              onChange={(e) => setZoom(Number(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer h-1.5 rounded-lg bg-zinc-700"
              aria-label="Slice Magnification Level"
            />

            {/* Quick Zoom Presets */}
            <div className="grid grid-cols-5 gap-1 pt-1">
              {[
                { label: '1x', val: 1.0 },
                { label: '1.5x', val: 1.5 },
                { label: '2.5x', val: 2.5 },
                { label: '4x', val: 4.0 },
                { label: '6x', val: 6.0 },
              ].map((p) => (
                <button
                  key={p.label}
                  onClick={() => setZoom(p.val)}
                  className={`rounded-md py-1 text-[11px] font-mono transition-colors ${
                    Math.abs(zoom - p.val) < 0.25
                      ? 'bg-cyan-500/25 text-cyan-300 border border-cyan-500/40'
                      : 'bg-white/5 text-zinc-400 hover:text-white'
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-2 pt-1">
              <button
                onClick={() => setZoom((z) => Math.max(0.8, Math.round((z - 0.3) * 10) / 10))}
                className="flex-1 flex items-center justify-center gap-1 rounded-lg border border-nv-line bg-black/40 py-1.5 text-xs text-zinc-300 hover:text-white"
              >
                <ZoomOut className="h-3.5 w-3.5" /> Out
              </button>
              <button
                onClick={() => setZoom((z) => Math.min(6.0, Math.round((z + 0.3) * 10) / 10))}
                className="flex-1 flex items-center justify-center gap-1 rounded-lg border border-nv-line bg-black/40 py-1.5 text-xs text-zinc-300 hover:text-white"
              >
                <ZoomIn className="h-3.5 w-3.5" /> In
              </button>

              {(pan.x !== 0 || pan.y !== 0) && (
                <button
                  onClick={() => setPan({ x: 0, y: 0 })}
                  title="Recenter Slice Position"
                  className="rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-2 py-1.5 text-[11px] text-cyan-300 hover:bg-cyan-500/20"
                >
                  Recenter
                </button>
              )}

              <button
                onClick={() => setShowCrosshair(!showCrosshair)}
                title="Toggle Target Crosshair"
                className={`rounded-lg border p-1.5 transition-colors ${
                  showCrosshair
                    ? 'border-cyan-500/50 bg-cyan-500/20 text-cyan-300'
                    : 'border-nv-line bg-black/40 text-zinc-500'
                }`}
              >
                <Crosshair className="h-4 w-4" />
              </button>
            </div>

            {zoom > 1.0 && (
              <p className="text-[10px] text-zinc-500 font-mono text-center pt-0.5">
                Drag slice to pan · Scroll wheel to zoom
              </p>
            )}
          </div>

          {/* Sub-region Color Key (when overlay is seg) */}
          {overlay === 'seg' && (
            <div className="rounded-xl border border-nv-line/50 bg-black/30 p-3 space-y-2 text-xs">
              <div className="font-semibold text-zinc-300">BraTS Segmentation Key</div>
              <div className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full bg-[#ff7a45]" />
                <span className="text-zinc-300">Enhancing Tumor (ET) · Label 4</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full bg-[#4d9bff]" />
                <span className="text-zinc-300">Tumor Core (TC) · Labels 1 + 4</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full bg-[#59d98c]" />
                <span className="text-zinc-300">Whole Tumor (WT) · Labels 1 + 2 + 4</span>
              </div>
            </div>
          )}

          {overlay === 'cam' && (
            <div className="rounded-xl border border-nv-line/50 bg-black/30 p-3 space-y-1.5 text-xs">
              <div className="font-semibold text-orange-300">3D Grad-CAM Attention Map</div>
              <p className="text-zinc-400 leading-relaxed">
                Visualizes spatial activations in final layer feature maps. Hotter colors indicate higher model attention weights.
              </p>
              <div className="h-2 w-full rounded-full bg-gradient-to-r from-blue-600 via-yellow-400 to-red-500 mt-2" />
              <div className="flex justify-between text-[10px] text-zinc-500">
                <span>Low Saliency</span>
                <span>Max Attention</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
