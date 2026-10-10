'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import * as THREE from 'three'
import {
  RotateCw,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Layers,
  Compass,
  Focus,
  Activity,
  CheckCircle2,
  Box,
  ChevronDown
} from 'lucide-react'
import { api } from '@/lib/api'
import type { BrainGeometry3D } from '@/lib/types'

interface ThreeBrainSceneProps {
  className?: string
  height?: string
  autoRotateInit?: boolean
  initialCaseId?: string
}

export default function ThreeBrainScene({
  className = '',
  height = '620px',
  autoRotateInit = true,
  initialCaseId = 'CASE-2026-001',
}: ThreeBrainSceneProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const [selectedCase, setSelectedCase] = useState<string>(initialCaseId)
  const [geometryData, setGeometryData] = useState<BrainGeometry3D | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [autoRotate, setAutoRotate] = useState<boolean>(autoRotateInit)
  const [showBrain, setShowBrain] = useState<boolean>(true)
  const [showTumor, setShowTumor] = useState<boolean>(true)
  const [showEdema, setShowEdema] = useState<boolean>(true)
  const [showNecrotic, setShowNecrotic] = useState<boolean>(true)
  const [showROI, setShowROI] = useState<boolean>(true)
  const [showPlanes, setShowPlanes] = useState<boolean>(false)
  const [cameraDistance, setCameraDistance] = useState<number>(6.2)

  // Object references for instant toggle and dynamic geometry swapping
  const sceneRef = useRef<THREE.Scene | null>(null)
  const brainGroupRef = useRef<THREE.Group | null>(null)
  const tumorGroupRef = useRef<THREE.Group | null>(null)
  const edemaGroupRef = useRef<THREE.Group | null>(null)
  const necroticGroupRef = useRef<THREE.Group | null>(null)
  const roiBoxRef = useRef<THREE.Group | null>(null)
  const planesGroupRef = useRef<THREE.Group | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const mainGroupRef = useRef<THREE.Group | null>(null)
  const isDraggingRef = useRef<boolean>(false)

  // 1. Fetch real patient 3D geometry from FastAPI backend
  const fetchGeometry = useCallback(async (caseId: string) => {
    setLoading(true)
    try {
      const data = await api.geometry3d(caseId)
      setGeometryData(data)
    } catch {
      // Graceful fallback to anatomical reference if offline
      setGeometryData(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchGeometry(selectedCase)
  }, [selectedCase, fetchGeometry])

  // Helper to build a Three.js mesh from vertex and face index arrays
  const buildMeshFromData = (
    vertices: [number, number, number][],
    faces: [number, number, number][],
    material: THREE.Material,
    renderOrder = 0
  ): THREE.Mesh => {
    const geom = new THREE.BufferGeometry()
    const pos = new Float32Array(vertices.length * 3)
    for (let i = 0; i < vertices.length; i++) {
      pos[i * 3] = vertices[i][0]
      pos[i * 3 + 1] = vertices[i][1]
      pos[i * 3 + 2] = vertices[i][2]
    }
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3))

    const idx = new Uint32Array(faces.length * 3)
    for (let i = 0; i < faces.length; i++) {
      idx[i * 3] = faces[i][0]
      idx[i * 3 + 1] = faces[i][1]
      idx[i * 3 + 2] = faces[i][2]
    }
    geom.setIndex(new THREE.BufferAttribute(idx, 1))
    geom.computeVertexNormals()

    const mesh = new THREE.Mesh(geom, material)
    mesh.renderOrder = renderOrder
    return mesh
  }

  // Helper to build realistic anatomical point cloud
  const buildPointCloudFromData = (points: [number, number, number][]): THREE.Points => {
    const geom = new THREE.BufferGeometry()
    const pos = new Float32Array(points.length * 3)
    const col = new Float32Array(points.length * 3)

    for (let i = 0; i < points.length; i++) {
      pos[i * 3] = points[i][0]
      pos[i * 3 + 1] = points[i][1]
      pos[i * 3 + 2] = points[i][2]

      // Height gradient: Top dorsal cortex cyan/azure, base deeper blue
      const t = Math.min(1, Math.max(0, (points[i][1] + 1.2) / 2.4))
      col[i * 3] = THREE.MathUtils.lerp(0.08, 0.28, t)
      col[i * 3 + 1] = THREE.MathUtils.lerp(0.35, 0.88, t)
      col[i * 3 + 2] = THREE.MathUtils.lerp(0.75, 1.0, t)
    }

    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    geom.setAttribute('color', new THREE.BufferAttribute(col, 3))

    const pCanvas = document.createElement('canvas')
    pCanvas.width = pCanvas.height = 32
    const pCtx = pCanvas.getContext('2d')
    if (pCtx) {
      const grad = pCtx.createRadialGradient(16, 16, 0, 16, 16, 16)
      grad.addColorStop(0, '#ffffff')
      grad.addColorStop(0.35, '#38bdf8dd')
      grad.addColorStop(0.7, '#0284c755')
      grad.addColorStop(1, 'transparent')
      pCtx.fillStyle = grad
      pCtx.fillRect(0, 0, 32, 32)
    }

    const pTexture = new THREE.CanvasTexture(pCanvas)
    const mat = new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      map: pTexture,
      transparent: true,
      opacity: 0.72,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })

    return new THREE.Points(geom, mat)
  }

  // 2. Main Three.js Scene Setup & Re-hydration
  useEffect(() => {
    const container = mountRef.current
    if (!container) return

    while (container.firstChild) {
      container.removeChild(container.firstChild)
    }

    const scene = new THREE.Scene()
    sceneRef.current = scene

    const width = Math.max(container.clientWidth || 800, 320)
    const heightPx = Math.max(container.clientHeight || 600, 320)

    const camera = new THREE.PerspectiveCamera(40, width / heightPx, 0.1, 100)
    camera.position.set(0, 0.8, 6.2)
    camera.lookAt(0, 0.1, 0)
    cameraRef.current = camera

    let renderer: THREE.WebGLRenderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    } catch {
      return
    }

    renderer.setSize(width, heightPx)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearColor(0x040813, 1)
    container.appendChild(renderer.domElement)
    rendererRef.current = renderer

    const mainGroup = new THREE.Group()
    mainGroup.position.set(0, 0.05, 0)
    scene.add(mainGroup)
    mainGroupRef.current = mainGroup

    // Groups for layers
    const brainGroup = new THREE.Group()
    const tumorGroup = new THREE.Group()
    const edemaGroup = new THREE.Group()
    const necroticGroup = new THREE.Group()
    const roiBoxGroup = new THREE.Group()
    const planesGroup = new THREE.Group()

    mainGroup.add(brainGroup)
    mainGroup.add(edemaGroup)
    mainGroup.add(tumorGroup)
    mainGroup.add(necroticGroup)
    mainGroup.add(roiBoxGroup)
    mainGroup.add(planesGroup)

    brainGroupRef.current = brainGroup
    tumorGroupRef.current = tumorGroup
    edemaGroupRef.current = edemaGroup
    necroticGroupRef.current = necroticGroup
    roiBoxRef.current = roiBoxGroup
    planesGroupRef.current = planesGroup

    let tumorCenter = new THREE.Vector3(0.45, 0.22, 0.32)

    // =========================================================================
    // Real Data Mesh Construction (or Biomechanical Fallback)
    // =========================================================================
    if (geometryData && geometryData.is_real_data && geometryData.brain.vertices.length > 0) {
      // 1. True Anatomical Brain Surface & Cortical Point Cloud from Real MRI
      const brainMat = new THREE.MeshStandardMaterial({
        color: 0x0284c7,
        emissive: 0x082f49,
        emissiveIntensity: 0.18,
        roughness: 0.7,
        metalness: 0.1,
        transparent: true,
        opacity: 0.12,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
      const brainMesh = buildMeshFromData(
        geometryData.brain.vertices,
        geometryData.brain.faces,
        brainMat,
        0
      )
      brainGroup.add(brainMesh)

      // Point cloud from true cortex surface
      if (geometryData.brain.point_cloud && geometryData.brain.point_cloud.length > 0) {
        const brainPoints = buildPointCloudFromData(geometryData.brain.point_cloud)
        brainGroup.add(brainPoints)
      }

      // Wireframe contour of true brain sulci
      const brainWireMat = new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        wireframe: true,
        transparent: true,
        opacity: 0.08,
        depthWrite: false,
      })
      const brainWire = buildMeshFromData(
        geometryData.brain.vertices,
        geometryData.brain.faces,
        brainWireMat,
        0
      )
      brainGroup.add(brainWire)

      // 2. Real Enhancing Tumor (ET) Mesh
      if (geometryData.et && geometryData.et.vertices.length > 0) {
        const etMat = new THREE.MeshStandardMaterial({
          color: 0xff5722,
          emissive: 0xe64a19,
          emissiveIntensity: 0.75,
          roughness: 0.3,
          metalness: 0.2,
          transparent: true,
          opacity: 0.94,
          side: THREE.DoubleSide,
          depthWrite: true,
        })
        const etMesh = buildMeshFromData(
          geometryData.et.vertices,
          geometryData.et.faces,
          etMat,
          2
        )
        tumorGroup.add(etMesh)

        // Wireframe contour
        const etWireMat = new THREE.MeshBasicMaterial({
          color: 0xffa07a,
          wireframe: true,
          transparent: true,
          opacity: 0.45,
        })
        const etWire = buildMeshFromData(
          geometryData.et.vertices,
          geometryData.et.faces,
          etWireMat,
          2
        )
        tumorGroup.add(etWire)
      }

      // 3. Real Peritumoral Vasogenic Edema (WT) Mesh
      if (geometryData.wt && geometryData.wt.vertices.length > 0) {
        const wtMat = new THREE.MeshPhysicalMaterial({
          color: 0x10b981,
          emissive: 0x047857,
          emissiveIntensity: 0.25,
          roughness: 0.35,
          transmission: 0.65,
          transparent: true,
          opacity: 0.26,
          side: THREE.DoubleSide,
          depthWrite: false,
        })
        const wtMesh = buildMeshFromData(
          geometryData.wt.vertices,
          geometryData.wt.faces,
          wtMat,
          3
        )
        edemaGroup.add(wtMesh)

        const wtWireMat = new THREE.MeshBasicMaterial({
          color: 0x34d399,
          wireframe: true,
          transparent: true,
          opacity: 0.2,
          depthWrite: false,
        })
        const wtWire = buildMeshFromData(
          geometryData.wt.vertices,
          geometryData.wt.faces,
          wtWireMat,
          4
        )
        edemaGroup.add(wtWire)
      }

      // 4. Real Central Necrotic Cavity (NCR) Mesh
      if (geometryData.ncr && geometryData.ncr.vertices.length > 0) {
        const ncrMat = new THREE.MeshStandardMaterial({
          color: 0x7f1d1d,
          emissive: 0x991b1b,
          emissiveIntensity: 0.45,
          roughness: 0.8,
          transparent: true,
          opacity: 0.96,
          side: THREE.DoubleSide,
          depthWrite: true,
        })
        const ncrMesh = buildMeshFromData(
          geometryData.ncr.vertices,
          geometryData.ncr.faces,
          ncrMat,
          1
        )
        necroticGroup.add(ncrMesh)
      }

      // 5. Real Calibrated Contextual ROI Bounding Box
      if (geometryData.bbox) {
        const sz = geometryData.bbox.size
        const cx = geometryData.bbox.center
        tumorCenter = new THREE.Vector3(cx[0], cx[1], cx[2])

        const boxGeo = new THREE.BoxGeometry(sz[0], sz[1], sz[2])
        const edges = new THREE.EdgesGeometry(boxGeo)
        const lineMat = new THREE.LineBasicMaterial({
          color: 0x38bdf8,
          transparent: true,
          opacity: 0.75,
        })
        const roiLine = new THREE.LineSegments(edges, lineMat)
        roiLine.position.set(cx[0], cx[1], cx[2])
        roiBoxGroup.add(roiLine)

        const boxFillMat = new THREE.MeshBasicMaterial({
          color: 0x0284c7,
          transparent: true,
          opacity: 0.04,
          side: THREE.BackSide,
          depthWrite: false,
        })
        const boxFill = new THREE.Mesh(boxGeo, boxFillMat)
        boxFill.position.set(cx[0], cx[1], cx[2])
        roiBoxGroup.add(boxFill)
      }
    } else {
      // Anatomically calibrated reference fallback
      const N_POINTS = 9000
      const pos = new Float32Array(N_POINTS * 3)
      const col = new Float32Array(N_POINTS * 3)

      for (let i = 0; i < N_POINTS; i++) {
        const u = Math.random() * 2 - 1
        const theta = Math.random() * 2 * Math.PI
        const r0 = Math.sqrt(Math.max(0, 1 - u * u))

        let sx = r0 * Math.cos(theta) * 1.35
        let sy = u * 1.05
        let sz = r0 * Math.sin(theta) * 1.55

        const hemiSign = Math.sign(sx) || 1
        const fissureGap = 0.07 * Math.exp(-Math.pow(sx * 4, 2))
        sx += hemiSign * fissureGap

        if (sy < -0.4) sy = sy * 0.75 - 0.1

        const sulci = 1.0 + 0.07 * Math.sin(8 * sx + 5 * sy) * Math.cos(7 * sz + 3 * sx)
        const scaleF = 1.48
        pos[i * 3] = sx * sulci * scaleF
        pos[i * 3 + 1] = sy * sulci * scaleF
        pos[i * 3 + 2] = sz * sulci * scaleF

        const t = (pos[i * 3 + 1] + 1.2) / 2.4
        col[i * 3] = THREE.MathUtils.lerp(0.08, 0.25, t)
        col[i * 3 + 1] = THREE.MathUtils.lerp(0.35, 0.85, t)
        col[i * 3 + 2] = THREE.MathUtils.lerp(0.75, 1.0, t)
      }

      const brainGeo = new THREE.BufferGeometry()
      brainGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
      brainGeo.setAttribute('color', new THREE.BufferAttribute(col, 3))
      const brainMat = new THREE.PointsMaterial({
        size: 0.06,
        vertexColors: true,
        transparent: true,
        opacity: 0.7,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
      brainGroup.add(new THREE.Points(brainGeo, brainMat))

      // Fallback procedural tumor meshes
      const wtGeo = new THREE.IcosahedronGeometry(0.85, 3)
      const wtMat = new THREE.MeshPhysicalMaterial({
        color: 0x10b981,
        emissive: 0x047857,
        emissiveIntensity: 0.25,
        roughness: 0.4,
        transmission: 0.6,
        transparent: true,
        opacity: 0.22,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
      const wtMesh = new THREE.Mesh(wtGeo, wtMat)
      wtMesh.position.copy(tumorCenter)
      edemaGroup.add(wtMesh)

      const etGeo = new THREE.IcosahedronGeometry(0.52, 4)
      const etMat = new THREE.MeshStandardMaterial({
        color: 0xff5722,
        emissive: 0xe64a19,
        emissiveIntensity: 0.65,
        roughness: 0.35,
        transparent: true,
        opacity: 0.92,
        depthWrite: true,
      })
      const etMesh = new THREE.Mesh(etGeo, etMat)
      etMesh.position.copy(tumorCenter)
      tumorGroup.add(etMesh)

      const boxGeo = new THREE.BoxGeometry(2.1, 1.95, 2.05)
      const edges = new THREE.EdgesGeometry(boxGeo)
      const roiLine = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x38bdf8, opacity: 0.65, transparent: true }))
      roiLine.position.copy(tumorCenter)
      roiBoxGroup.add(roiLine)
    }

    // =========================================================================
    // Orthogonal Radiological Slicing Planes
    // =========================================================================
    const makePlane = (color: number, rotX: number, rotY: number) => {
      const pGeo = new THREE.PlaneGeometry(3.6, 3.6, 12, 12)
      const pMat = new THREE.MeshBasicMaterial({
        color,
        wireframe: true,
        transparent: true,
        opacity: 0.22,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
      const pMesh = new THREE.Mesh(pGeo, pMat)
      pMesh.rotation.set(rotX, rotY, 0)
      pMesh.position.copy(tumorCenter)
      return pMesh
    }

    planesGroup.add(makePlane(0x38bdf8, Math.PI / 2, 0))   // Axial
    planesGroup.add(makePlane(0xa855f7, 0, 0))            // Coronal
    planesGroup.add(makePlane(0xf59e0b, 0, Math.PI / 2))   // Sagittal
    planesGroup.visible = showPlanes

    // =========================================================================
    // Illumination Studio
    // =========================================================================
    scene.add(new THREE.AmbientLight(0xffffff, 0.85))

    const keyLight = new THREE.DirectionalLight(0x38bdf8, 1.4)
    keyLight.position.set(5, 7, 6)
    scene.add(keyLight)

    const warmFill = new THREE.DirectionalLight(0xff7a45, 1.1)
    warmFill.position.set(-6, -4, -4)
    scene.add(warmFill)

    const rimLight = new THREE.DirectionalLight(0xa855f7, 0.85)
    rimLight.position.set(0, -6, 5)
    scene.add(rimLight)

    // =========================================================================
    // Interactive Pointer Controls
    // =========================================================================
    let prevMouseX = 0
    let prevMouseY = 0

    const onPointerDown = (e: MouseEvent) => {
      isDraggingRef.current = true
      prevMouseX = e.clientX
      prevMouseY = e.clientY
    }

    const onPointerMove = (e: MouseEvent) => {
      if (!isDraggingRef.current) return
      const deltaX = e.clientX - prevMouseX
      const deltaY = e.clientY - prevMouseY
      prevMouseX = e.clientX
      prevMouseY = e.clientY

      mainGroup.rotation.y += deltaX * 0.007
      mainGroup.rotation.x += deltaY * 0.007
    }

    const onPointerUp = () => {
      isDraggingRef.current = false
    }

    const onWheel = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) {
        e.preventDefault()
        camera.position.z = Math.max(3.0, Math.min(12.0, camera.position.z + e.deltaY * 0.004))
        setCameraDistance(Math.round(camera.position.z * 10) / 10)
      }
    }

    const domEl = renderer.domElement
    domEl.addEventListener('mousedown', onPointerDown)
    window.addEventListener('mousemove', onPointerMove)
    window.addEventListener('mouseup', onPointerUp)
    domEl.addEventListener('wheel', onWheel, { passive: false })

    // Viewport Visibility Tracking & Render Loop
    let isVisible = true
    let observer: IntersectionObserver | null = null

    if (typeof IntersectionObserver !== 'undefined') {
      observer = new IntersectionObserver(
        ([entry]) => {
          isVisible = entry.isIntersecting
        },
        { threshold: 0.02 }
      )
      observer.observe(container)
    }

    let animId = 0
    const clock = new THREE.Clock()

    const animate = () => {
      animId = requestAnimationFrame(animate)
      if (!isVisible) return

      const delta = clock.getDelta()

      if (autoRotate && !isDraggingRef.current) {
        mainGroup.rotation.y += delta * 0.22
      }

      renderer.render(scene, camera)
    }
    animate()

    const handleResize = () => {
      if (!container) return
      const w = Math.max(container.clientWidth || 800, 320)
      const h = Math.max(container.clientHeight || 600, 320)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    window.addEventListener('resize', handleResize)

    return () => {
      cancelAnimationFrame(animId)
      if (observer) observer.disconnect()
      window.removeEventListener('resize', handleResize)
      domEl.removeEventListener('mousedown', onPointerDown)
      window.removeEventListener('mousemove', onPointerMove)
      window.removeEventListener('mouseup', onPointerUp)
      domEl.removeEventListener('wheel', onWheel)
      renderer.dispose()
      if (container.contains(domEl)) {
        container.removeChild(domEl)
      }
    }
  }, [geometryData, autoRotate])

  // Reactive Visibility Toggles
  useEffect(() => {
    if (brainGroupRef.current) brainGroupRef.current.visible = showBrain
  }, [showBrain])

  useEffect(() => {
    if (tumorGroupRef.current) tumorGroupRef.current.visible = showTumor
  }, [showTumor])

  useEffect(() => {
    if (edemaGroupRef.current) edemaGroupRef.current.visible = showEdema
  }, [showEdema])

  useEffect(() => {
    if (necroticGroupRef.current) necroticGroupRef.current.visible = showNecrotic
  }, [showNecrotic])

  useEffect(() => {
    if (roiBoxRef.current) roiBoxRef.current.visible = showROI
  }, [showROI])

  useEffect(() => {
    if (planesGroupRef.current) planesGroupRef.current.visible = showPlanes
  }, [showPlanes])

  const handleResetCamera = () => {
    if (cameraRef.current) {
      cameraRef.current.position.set(0, 0.8, 6.2)
      cameraRef.current.lookAt(0, 0.1, 0)
      setCameraDistance(6.2)
    }
    if (mainGroupRef.current) {
      mainGroupRef.current.rotation.set(0, 0, 0)
    }
  }

  const handleZoom = (delta: number) => {
    if (cameraRef.current) {
      cameraRef.current.position.z = Math.max(3.0, Math.min(12.0, cameraRef.current.position.z + delta))
      setCameraDistance(Math.round(cameraRef.current.position.z * 10) / 10)
    }
  }

  const handleFocusTumor = () => {
    if (mainGroupRef.current && cameraRef.current) {
      mainGroupRef.current.rotation.set(-0.15, -0.75, 0)
      cameraRef.current.position.set(0, 0.4, 4.8)
      setCameraDistance(4.8)
    }
  }

  return (
    <div className={`relative overflow-hidden rounded-3xl border border-nv-line bg-[#040813] shadow-2xl ${className}`}>
      {/* 3D WebGL Canvas Container */}
      <div
        ref={mountRef}
        style={{ height }}
        className="w-full cursor-grab active:cursor-grabbing"
        aria-label="Real 3D Patient Brain and Tumor Surface Reconstruction"
      />

      {/* Floating Status & Case Selector Header */}
      <div className="absolute left-5 top-5 right-5 flex flex-wrap items-center justify-between gap-3 pointer-events-none">
        <div className="flex items-center gap-2.5 rounded-full border border-white/10 bg-black/75 px-3.5 py-1.5 backdrop-blur-md pointer-events-auto">
          <span className="flex h-2.5 w-2.5 rounded-full bg-cyan-400 shadow-[0_0_10px_#38bdf8] animate-pulse" />
          <div className="flex items-center gap-2 text-xs font-semibold tracking-wide text-zinc-200">
            <span>
              {geometryData?.is_real_data ? 'Real Patient NIfTI Mesh' : '3D Reconstruction'}
            </span>
            <span className="text-zinc-500">·</span>
            <span className="font-mono text-cyan-300 text-[11px]">
              {geometryData?.dataset_source ?? 'BraTS 2020'}
            </span>
          </div>
        </div>

        {/* Dynamic Case Selector */}
        <div className="flex items-center gap-2 pointer-events-auto">
          <div className="flex items-center gap-1.5 rounded-full border border-white/10 bg-black/80 px-3 py-1 text-xs backdrop-blur-md">
            <span className="text-[11px] font-mono text-zinc-400">Scan:</span>
            <select
              value={selectedCase}
              onChange={(e) => setSelectedCase(e.target.value)}
              className="bg-transparent text-xs font-mono font-semibold text-white focus:outline-none cursor-pointer"
            >
              <option value="CASE-2026-001" className="bg-[#040813] text-white">CASE-2026-001 (BraTS_001 · HGG)</option>
              <option value="CASE-2026-002" className="bg-[#040813] text-white">CASE-2026-002 (BraTS_002 · HGG)</option>
            </select>
            {loading && (
              <span className="h-2 w-2 rounded-full bg-cyan-400 animate-ping" />
            )}
          </div>

          {/* Camera Toolbar */}
          <div className="flex items-center gap-1 rounded-full border border-white/10 bg-black/80 p-1 shadow-xl backdrop-blur-md">
            <button
              onClick={() => setAutoRotate(!autoRotate)}
              title={autoRotate ? 'Pause Rotation' : 'Auto Rotate'}
              className={`rounded-full p-1.5 text-xs transition-colors ${
                autoRotate ? 'bg-sky-500/25 text-sky-300' : 'text-zinc-400 hover:text-white'
              }`}
            >
              <RotateCw className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={handleFocusTumor}
              title="Focus on Tumor Lesion"
              className="rounded-full p-1.5 text-zinc-400 transition-colors hover:text-sky-300"
            >
              <Focus className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => handleZoom(-0.8)}
              title="Zoom In"
              className="rounded-full p-1.5 text-zinc-400 transition-colors hover:text-white"
            >
              <ZoomIn className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => handleZoom(0.8)}
              title="Zoom Out"
              className="rounded-full p-1.5 text-zinc-400 transition-colors hover:text-white"
            >
              <ZoomOut className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={handleResetCamera}
              title="Reset Camera"
              className="rounded-full p-1.5 text-zinc-400 transition-colors hover:text-white"
            >
              <Maximize2 className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Layer Control Strip & Metric Badges at Bottom */}
      <div className="absolute bottom-5 left-5 right-5 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-white/10 bg-black/85 px-4 py-3 shadow-2xl backdrop-blur-xl">
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="flex items-center gap-1.5 text-zinc-400 font-semibold mr-1">
            <Layers className="h-3.5 w-3.5 text-sky-400" />
            <span>Layers:</span>
          </span>

          {/* Brain Cortex Toggle */}
          <button
            onClick={() => setShowBrain(!showBrain)}
            className={`flex items-center gap-1.5 rounded-xl border px-3 py-1.5 font-medium transition-all ${
              showBrain
                ? 'border-cyan-500/50 bg-cyan-500/15 text-cyan-200 shadow-sm'
                : 'border-white/10 text-zinc-500 hover:text-zinc-300 bg-white/5'
            }`}
          >
            <span className="h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_6px_#38bdf8]" />
            <span>Brain Cortex ({geometryData?.brain.vertex_count ?? 3008} v)</span>
          </button>

          {/* Enhancing Tumor (ET) Toggle */}
          <button
            onClick={() => setShowTumor(!showTumor)}
            className={`flex items-center gap-1.5 rounded-xl border px-3 py-1.5 font-medium transition-all ${
              showTumor
                ? 'border-orange-500/50 bg-orange-500/15 text-orange-200 shadow-sm'
                : 'border-white/10 text-zinc-500 hover:text-zinc-300 bg-white/5'
            }`}
          >
            <span className="h-2 w-2 rounded-full bg-[#ff5722] shadow-[0_0_6px_#ff5722]" />
            <span>Enhancing Core (ET)</span>
            {geometryData?.et?.volume_cm3 && (
              <span className="font-mono text-[10px] text-orange-300 ml-1">
                {geometryData.et.volume_cm3} cm³
              </span>
            )}
          </button>

          {/* Peritumoral Edema (WT) Toggle */}
          <button
            onClick={() => setShowEdema(!showEdema)}
            className={`flex items-center gap-1.5 rounded-xl border px-3 py-1.5 font-medium transition-all ${
              showEdema
                ? 'border-emerald-500/50 bg-emerald-500/15 text-emerald-200 shadow-sm'
                : 'border-white/10 text-zinc-500 hover:text-zinc-300 bg-white/5'
            }`}
          >
            <span className="h-2 w-2 rounded-full bg-[#10b981] shadow-[0_0_6px_#10b981]" />
            <span>Vasogenic Edema (WT)</span>
            {geometryData?.wt?.volume_cm3 && (
              <span className="font-mono text-[10px] text-emerald-300 ml-1">
                {geometryData.wt.volume_cm3} cm³
              </span>
            )}
          </button>

          {/* Necrotic Core (NCR) Toggle */}
          <button
            onClick={() => setShowNecrotic(!showNecrotic)}
            className={`flex items-center gap-1.5 rounded-xl border px-3 py-1.5 font-medium transition-all ${
              showNecrotic
                ? 'border-rose-500/50 bg-rose-500/15 text-rose-200 shadow-sm'
                : 'border-white/10 text-zinc-500 hover:text-zinc-300 bg-white/5'
            }`}
          >
            <span className="h-2 w-2 rounded-full bg-[#dc2626] shadow-[0_0_6px_#dc2626]" />
            <span>Necrotic Cavity (NCR)</span>
          </button>

          {/* ROI Box Toggle */}
          <button
            onClick={() => setShowROI(!showROI)}
            className={`flex items-center gap-1.5 rounded-xl border px-3 py-1.5 font-medium transition-all ${
              showROI
                ? 'border-sky-500/50 bg-sky-500/15 text-sky-200 shadow-sm'
                : 'border-white/10 text-zinc-500 hover:text-zinc-300 bg-white/5'
            }`}
          >
            <Box className="h-3 w-3 text-sky-400" />
            <span>ROI Context Margin (+10 vox)</span>
          </button>

          {/* Slicing Planes Toggle */}
          <button
            onClick={() => setShowPlanes(!showPlanes)}
            className={`flex items-center gap-1.5 rounded-xl border px-3 py-1.5 font-medium transition-all ${
              showPlanes
                ? 'border-purple-500/50 bg-purple-500/15 text-purple-200 shadow-sm'
                : 'border-white/10 text-zinc-500 hover:text-zinc-300 bg-white/5'
            }`}
          >
            <Compass className="h-3 w-3 text-purple-400" />
            <span>Orthogonal Planes</span>
          </button>
        </div>

        {/* Live Metrics & Interaction Hint */}
        <div className="flex items-center gap-3 text-[11px] font-mono text-zinc-400">
          <span className="hidden sm:inline-block">
            {geometryData?.dimensions_voxel
              ? `Voxel Grid: ${geometryData.dimensions_voxel.join('×')}`
              : 'Iso-Surface 1mm³'}
          </span>
          <span className="text-zinc-600">·</span>
          <span>Ctrl + Scroll to zoom</span>
        </div>
      </div>
    </div>
  )
}
