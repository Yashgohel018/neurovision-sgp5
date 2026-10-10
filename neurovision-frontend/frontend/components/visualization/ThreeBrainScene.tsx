'use client'

import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { RotateCw, ZoomIn, ZoomOut, Maximize2, Layers, Compass, Focus } from 'lucide-react'

interface ThreeBrainSceneProps {
  className?: string
  height?: string
  autoRotateInit?: boolean
}

export default function ThreeBrainScene({
  className = '',
  height = '600px',
  autoRotateInit = true,
}: ThreeBrainSceneProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const [autoRotate, setAutoRotate] = useState(autoRotateInit)
  const [showBrain, setShowBrain] = useState(true)
  const [showTumor, setShowTumor] = useState(true)
  const [showEdema, setShowEdema] = useState(true)
  const [showNecrotic, setShowNecrotic] = useState(true)
  const [showROI, setShowROI] = useState(true)
  const [showPlanes, setShowPlanes] = useState(false)
  const [cameraDistance, setCameraDistance] = useState(6.2)

  // Object references for instant toggle without re-instantiating scene
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

  useEffect(() => {
    const container = mountRef.current
    if (!container) return

    // Clean any prior canvas
    while (container.firstChild) {
      container.removeChild(container.firstChild)
    }

    // 1. Scene setup
    const scene = new THREE.Scene()
    sceneRef.current = scene

    const width = Math.max(container.clientWidth || 800, 320)
    const heightPx = Math.max(container.clientHeight || 600, 320)

    // Perspective camera properly centered at world origin
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

    // Master group centered vertically and horizontally
    const mainGroup = new THREE.Group()
    mainGroup.position.set(0, 0.05, 0)
    scene.add(mainGroup)
    mainGroupRef.current = mainGroup

    // =========================================================================
    // 2. Anatomically Modeled Dual-Hemisphere Brain Cortex Point Cloud
    // =========================================================================
    const brainGroup = new THREE.Group()
    mainGroup.add(brainGroup)
    brainGroupRef.current = brainGroup

    const N_POINTS = 11000
    const positions = new Float32Array(N_POINTS * 3)
    const colors = new Float32Array(N_POINTS * 3)

    let pIdx = 0
    while (pIdx < N_POINTS) {
      // Generate spherical coordinates with hemispheric longitudinal fissure
      const u = Math.random() * 2 - 1
      const theta = Math.random() * 2 * Math.PI
      const r0 = Math.sqrt(Math.max(0, 1 - u * u))

      let x = r0 * Math.cos(theta)
      let y = u
      let z = r0 * Math.sin(theta)

      // Anatomic brain scaling:
      // X = lateral/width (1.35), Y = vertical/height (1.05), Z = anterior-posterior/depth (1.55)
      let sx = x * 1.35
      let sy = y * 1.05
      let sz = z * 1.55

      // Add interhemispheric longitudinal fissure (groove between left & right hemispheres)
      const hemiSign = Math.sign(sx) || 1
      const fissureGap = 0.07 * Math.exp(-Math.pow(sx * 4, 2))
      sx += hemiSign * fissureGap

      // Flatten inferior base (ventral skull base) and round dorsal cortex
      if (sy < -0.4) {
        sy = sy * 0.75 - 0.1
      }

      // Add anatomical gyri and sulci wave convolutions
      const sulci = 1.0 + 0.07 * Math.sin(8 * sx + 5 * sy) * Math.cos(7 * sz + 3 * sx)
                    + 0.04 * Math.sin(14 * sy + 9 * sz)
                    + 0.03 * Math.cos(18 * sz + 12 * sx)

      const scaleFactor = 1.48
      const px = sx * sulci * scaleFactor
      const py = sy * sulci * scaleFactor
      const pz = sz * sulci * scaleFactor

      positions[pIdx * 3] = px
      positions[pIdx * 3 + 1] = py
      positions[pIdx * 3 + 2] = pz

      // Color gradient: Electric cyan / azure on dorsal top, deeper blue on base
      const t = (py + 1.2) / 2.4
      colors[pIdx * 3] = THREE.MathUtils.lerp(0.08, 0.25, t)     // R
      colors[pIdx * 3 + 1] = THREE.MathUtils.lerp(0.35, 0.85, t) // G
      colors[pIdx * 3 + 2] = THREE.MathUtils.lerp(0.75, 1.0, t)  // B

      pIdx++
    }

    const brainGeo = new THREE.BufferGeometry()
    brainGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    brainGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3))

    // High-performance circular soft radial particle texture
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
    const brainMat = new THREE.PointsMaterial({
      size: 0.065,
      vertexColors: true,
      map: pTexture,
      transparent: true,
      opacity: 0.72,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })

    const brainPoints = new THREE.Points(brainGeo, brainMat)
    brainGroup.add(brainPoints)

    // Faint holographic outer brain boundary silhouette
    const brainShellGeo = new THREE.SphereGeometry(1.9, 24, 18)
    brainShellGeo.scale(1.25, 0.92, 1.4)
    const brainShellMat = new THREE.MeshBasicMaterial({
      color: 0x0284c7,
      wireframe: true,
      transparent: true,
      opacity: 0.06,
    })
    const brainShell = new THREE.Mesh(brainShellGeo, brainShellMat)
    brainGroup.add(brainShell)

    // =========================================================================
    // 3. Multi-Compartment BraTS Lesion (Edema WT + Enhancing Core ET + Necrotic NCR)
    // Centered anatomically in frontoparietal white matter: (0.45, 0.22, 0.35)
    // =========================================================================
    const tumorCenter = new THREE.Vector3(0.48, 0.22, 0.32)

    // A. Peritumoral Vasogenic Edema (WT / ED - Outer Translucent Emerald Halo)
    const edemaGroup = new THREE.Group()
    mainGroup.add(edemaGroup)
    edemaGroupRef.current = edemaGroup

    const wtGeo = new THREE.IcosahedronGeometry(0.88, 3)
    const wtPos = wtGeo.attributes.position
    for (let i = 0; i < wtPos.count; i++) {
      const vx = wtPos.getX(i), vy = wtPos.getY(i), vz = wtPos.getZ(i)
      const noise = 1 + 0.22 * Math.sin(vx * 4.5 + vy * 3.5) * Math.sin(vz * 4.0)
      wtPos.setXYZ(i, vx * noise, vy * noise, vz * noise)
    }
    wtGeo.computeVertexNormals()

    // Key fix: depthWrite: false + renderOrder: 3 ensures Edema never occludes inner tumor!
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
    wtMesh.renderOrder = 3
    edemaGroup.add(wtMesh)

    // Delicate emerald contour cage for edema shape readability
    const wtWireMat = new THREE.MeshBasicMaterial({
      color: 0x34d399,
      wireframe: true,
      transparent: true,
      opacity: 0.16,
      depthWrite: false,
    })
    const wtWireMesh = new THREE.Mesh(wtGeo, wtWireMat)
    wtWireMesh.position.copy(tumorCenter)
    wtWireMesh.renderOrder = 4
    edemaGroup.add(wtWireMesh)

    // B. Enhancing Active Tumor Core (ET - Fiery Copper/Orange Hypervascular Neoplasm)
    const tumorGroup = new THREE.Group()
    mainGroup.add(tumorGroup)
    tumorGroupRef.current = tumorGroup

    const etGeo = new THREE.IcosahedronGeometry(0.55, 4)
    const etPos = etGeo.attributes.position
    for (let i = 0; i < etPos.count; i++) {
      const vx = etPos.getX(i), vy = etPos.getY(i), vz = etPos.getZ(i)
      const noise = 1 + 0.18 * Math.sin(vx * 6 + vy * 5) * Math.cos(vz * 6)
      etPos.setXYZ(i, vx * noise, vy * noise, vz * noise)
    }
    etGeo.computeVertexNormals()

    const etMat = new THREE.MeshStandardMaterial({
      color: 0xff5722,
      emissive: 0xe64a19,
      emissiveIntensity: 0.65,
      roughness: 0.35,
      metalness: 0.25,
      transparent: true,
      opacity: 0.92,
      depthWrite: true,
    })
    const etMesh = new THREE.Mesh(etGeo, etMat)
    etMesh.position.copy(tumorCenter)
    etMesh.renderOrder = 2
    tumorGroup.add(etMesh)

    // Glowing ET perimeter wireframe
    const etWireMat = new THREE.MeshBasicMaterial({
      color: 0xffa07a,
      wireframe: true,
      transparent: true,
      opacity: 0.45,
    })
    const etWireMesh = new THREE.Mesh(etGeo, etWireMat)
    etWireMesh.position.copy(tumorCenter)
    etWireMesh.renderOrder = 2
    tumorGroup.add(etWireMesh)

    // C. Central Necrotic Cavity (NCR / NET - Dark Crimson Non-enhancing Core)
    const necroticGroup = new THREE.Group()
    mainGroup.add(necroticGroup)
    necroticGroupRef.current = necroticGroup

    const ncrGeo = new THREE.IcosahedronGeometry(0.32, 3)
    const ncrMat = new THREE.MeshStandardMaterial({
      color: 0x7f1d1d,
      emissive: 0x991b1b,
      emissiveIntensity: 0.4,
      roughness: 0.8,
      transparent: true,
      opacity: 0.95,
      depthWrite: true,
    })
    const ncrMesh = new THREE.Mesh(ncrGeo, ncrMat)
    ncrMesh.position.copy(tumorCenter)
    ncrMesh.renderOrder = 1
    necroticGroup.add(ncrMesh)

    // =========================================================================
    // 4. 3D Contextual ROI Bounding Box with Corner Accents
    // =========================================================================
    const roiBoxGroup = new THREE.Group()
    mainGroup.add(roiBoxGroup)
    roiBoxRef.current = roiBoxGroup

    const boxSize = new THREE.Vector3(2.1, 1.95, 2.05)
    const roiBoxGeo = new THREE.BoxGeometry(boxSize.x, boxSize.y, boxSize.z)
    const roiEdges = new THREE.EdgesGeometry(roiBoxGeo)
    const roiLineMat = new THREE.LineBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.65,
    })
    const roiBox = new THREE.LineSegments(roiEdges, roiLineMat)
    roiBox.position.copy(tumorCenter)
    roiBoxGroup.add(roiBox)

    // Faint contextual box fill
    const roiFillMat = new THREE.MeshBasicMaterial({
      color: 0x0284c7,
      transparent: true,
      opacity: 0.04,
      side: THREE.BackSide,
      depthWrite: false,
    })
    const roiFill = new THREE.Mesh(roiBoxGeo, roiFillMat)
    roiFill.position.copy(tumorCenter)
    roiBoxGroup.add(roiFill)

    // =========================================================================
    // 5. Orthogonal Radiological Slicing Planes
    // =========================================================================
    const planesGroup = new THREE.Group()
    mainGroup.add(planesGroup)
    planesGroupRef.current = planesGroup
    planesGroup.visible = false

    // Axial (Cyan), Coronal (Purple), Sagittal (Amber)
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

    const axPlane = makePlane(0x38bdf8, Math.PI / 2, 0)   // Axial plane
    const corPlane = makePlane(0xa855f7, 0, 0)            // Coronal plane
    const sagPlane = makePlane(0xf59e0b, 0, Math.PI / 2)   // Sagittal plane

    planesGroup.add(axPlane)
    planesGroup.add(corPlane)
    planesGroup.add(sagPlane)

    // =========================================================================
    // 6. Illumination & Lighting Studio
    // =========================================================================
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.75)
    scene.add(ambientLight)

    const keyLight = new THREE.DirectionalLight(0x38bdf8, 1.4)
    keyLight.position.set(5, 7, 6)
    scene.add(keyLight)

    const warmFill = new THREE.DirectionalLight(0xff7a45, 1.1)
    warmFill.position.set(-6, -4, -4)
    scene.add(warmFill)

    const rimLight = new THREE.DirectionalLight(0xa855f7, 0.8)
    rimLight.position.set(0, -6, 5)
    scene.add(rimLight)

    // =========================================================================
    // 7. Interactive Pointer Controls (Rotate, Pan, Zoom)
    // =========================================================================
    let isDragging = false
    let prevMouseX = 0
    let prevMouseY = 0

    const onPointerDown = (e: MouseEvent) => {
      isDragging = true
      prevMouseX = e.clientX
      prevMouseY = e.clientY
    }

    const onPointerMove = (e: MouseEvent) => {
      if (!isDragging) return
      const deltaX = e.clientX - prevMouseX
      const deltaY = e.clientY - prevMouseY
      prevMouseX = e.clientX
      prevMouseY = e.clientY

      mainGroup.rotation.y += deltaX * 0.007
      mainGroup.rotation.x += deltaY * 0.007
    }

    const onPointerUp = () => {
      isDragging = false
    }

    const onWheel = (e: WheelEvent) => {
      // Only zoom if user explicitly holds Ctrl/Meta key (standard map/3D viewport convention)
      // This prevents trapping the user's page scroll when browsing down the page!
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

    // =========================================================================
    // 8. Viewport Visibility Tracking & Optimized Animation Loop
    // =========================================================================
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

      // When scrolled out of view, pause rendering to save 100% GPU/CPU for smooth page scrolling!
      if (!isVisible) return

      const delta = clock.getDelta()
      const elapsed = clock.getElapsedTime()

      if (autoRotate && !isDragging) {
        mainGroup.rotation.y += delta * 0.22
      }

      // Subtle biological pulsation of active neoplastic core
      const pulse = 1.0 + 0.035 * Math.sin(elapsed * 2.2)
      etMesh.scale.set(pulse, pulse, pulse)
      etWireMesh.scale.set(pulse, pulse, pulse)

      // Slight shimmering of cortical particle cloud
      brainMat.size = 0.065 + 0.005 * Math.sin(elapsed * 1.5)

      renderer.render(scene, camera)
    }
    animate()

    // Responsive Canvas Resize
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
      brainGeo.dispose()
      brainMat.dispose()
      wtGeo.dispose()
      wtMat.dispose()
      etGeo.dispose()
      etMat.dispose()
      ncrGeo.dispose()
      ncrMat.dispose()
      if (container.contains(domEl)) {
        container.removeChild(domEl)
      }
    }
  }, [autoRotate])

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
        aria-label="Interactive 3D Brain and Tumor Visualization"
      />

      {/* Header Floating Status */}
      <div className="pointer-events-none absolute left-5 top-5 flex items-center gap-2.5 rounded-full border border-white/10 bg-black/60 px-3.5 py-1.5 backdrop-blur-md">
        <span className="flex h-2.5 w-2.5 rounded-full bg-cyan-400 shadow-[0_0_10px_#38bdf8] animate-pulse" />
        <span className="text-xs font-semibold tracking-wider uppercase text-zinc-200">
          3D Volumetric Mesh Reconstruction · BraTS Coordinate Space
        </span>
      </div>

      {/* Camera Toolbar Top-Right */}
      <div className="absolute right-5 top-5 flex items-center gap-1.5 rounded-full border border-white/10 bg-black/70 p-1.5 shadow-xl backdrop-blur-md">
        <button
          onClick={() => setAutoRotate(!autoRotate)}
          title={autoRotate ? 'Pause Rotation' : 'Auto Rotate'}
          className={`rounded-full p-2 text-xs transition-colors ${
            autoRotate ? 'bg-sky-500/25 text-sky-300' : 'text-zinc-400 hover:text-white'
          }`}
        >
          <RotateCw className="h-4 w-4" />
        </button>
        <button
          onClick={handleFocusTumor}
          title="Focus on Tumor Lesion"
          className="rounded-full p-2 text-zinc-400 transition-colors hover:text-sky-300"
        >
          <Focus className="h-4 w-4" />
        </button>
        <button
          onClick={() => handleZoom(-0.8)}
          title="Zoom In"
          className="rounded-full p-2 text-zinc-400 transition-colors hover:text-white"
        >
          <ZoomIn className="h-4 w-4" />
        </button>
        <button
          onClick={() => handleZoom(0.8)}
          title="Zoom Out"
          className="rounded-full p-2 text-zinc-400 transition-colors hover:text-white"
        >
          <ZoomOut className="h-4 w-4" />
        </button>
        <button
          onClick={handleResetCamera}
          title="Reset Camera Orientation"
          className="rounded-full p-2 text-zinc-400 transition-colors hover:text-white"
        >
          <Maximize2 className="h-4 w-4" />
        </button>
      </div>

      {/* Layer Control Strip at Bottom */}
      <div className="absolute bottom-5 left-5 right-5 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-white/10 bg-black/80 px-4 py-3 shadow-2xl backdrop-blur-xl">
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
            <span>Brain Cortex</span>
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
            <span>Enhancing Tumor (ET)</span>
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
            <span>Edema (WT)</span>
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
            <span>Necrotic Core (NCR)</span>
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
            <span className="h-2 w-2 rounded-full border border-sky-400" />
            <span>ROI Bounding Box</span>
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

        {/* Camera Info Tag */}
        <div className="text-[11px] font-mono text-zinc-400">
          Dist: {cameraDistance}m · Drag to rotate · Ctrl + Scroll to zoom
        </div>
      </div>
    </div>
  )
}
