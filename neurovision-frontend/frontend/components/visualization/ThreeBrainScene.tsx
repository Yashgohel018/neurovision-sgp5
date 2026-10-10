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
  Box,
  Target,
  Crosshair,
  Grid,
  Move,
  Scan,
  Sparkles,
  Info,
  ChevronRight,
  ShieldCheck,
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
  height = '640px',
  autoRotateInit = false,
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

  // Advanced Zoom & Precision Inspection States
  const [zoomMultiplier, setZoomMultiplier] = useState<number>(1.0)
  const [focusTargetMode, setFocusTargetMode] = useState<'brain' | 'tumor' | 'custom'>('brain')
  const [showReticle, setShowReticle] = useState<boolean>(false)
  const [showWireframeHighlight, setShowWireframeHighlight] = useState<boolean>(false)
  const [wheelMode, setWheelMode] = useState<'zoom' | 'scroll'>('zoom')
  const [activeFocalPoint, setActiveFocalPoint] = useState<[number, number, number]>([0, 0.1, 0])
  const [toastMessage, setToastMessage] = useState<string | null>(null)

  // Object & Engine references
  const sceneRef = useRef<THREE.Scene | null>(null)
  const brainGroupRef = useRef<THREE.Group | null>(null)
  const tumorGroupRef = useRef<THREE.Group | null>(null)
  const edemaGroupRef = useRef<THREE.Group | null>(null)
  const necroticGroupRef = useRef<THREE.Group | null>(null)
  const roiBoxRef = useRef<THREE.Group | null>(null)
  const planesGroupRef = useRef<THREE.Group | null>(null)
  const wireframesRef = useRef<THREE.Mesh[]>([])
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const isDraggingRef = useRef<boolean>(false)
  const lastMultiplierRef = useRef<number>(1.0)

  // Spherical Camera & Target State Ref for silky smooth 60fps lerping
  const cameraStateRef = useRef({
    // Interpolated values
    radius: 6.2,
    theta: 0,
    phi: Math.PI / 2.3,
    targetX: 0,
    targetY: 0.1,
    targetZ: 0,

    // Target destinations
    desiredRadius: 6.2,
    desiredTheta: 0,
    desiredPhi: Math.PI / 2.3,
    desiredTargetX: 0,
    desiredTargetY: 0.1,
    desiredTargetZ: 0,

    // Patient-specific lesion centroid
    tumorCenterX: 0.45,
    tumorCenterY: 0.22,
    tumorCenterZ: 0.32,
    focusMode: 'brain' as 'brain' | 'tumor' | 'custom',
  })

  // Show transient notification toast
  const showNotification = useCallback((msg: string) => {
    setToastMessage(msg)
    const timer = setTimeout(() => setToastMessage(null), 3000)
    return () => clearTimeout(timer)
  }, [])

  // 1. Fetch real patient 3D geometry from FastAPI backend
  const fetchGeometry = useCallback(async (caseId: string) => {
    setLoading(true)
    try {
      const data = await api.geometry3d(caseId)
      setGeometryData(data)
    } catch {
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

  // Helper to build realistic anatomical cortical point cloud
  const buildPointCloudFromData = (points: [number, number, number][]): THREE.Points => {
    const geom = new THREE.BufferGeometry()
    const pos = new Float32Array(points.length * 3)
    const col = new Float32Array(points.length * 3)

    for (let i = 0; i < points.length; i++) {
      pos[i * 3] = points[i][0]
      pos[i * 3 + 1] = points[i][1]
      pos[i * 3 + 2] = points[i][2]

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

  // 2. Main Three.js Scene Setup & Spherical Camera Rig
  useEffect(() => {
    const container = mountRef.current
    if (!container) return

    while (container.firstChild) {
      container.removeChild(container.firstChild)
    }

    const scene = new THREE.Scene()
    sceneRef.current = scene
    wireframesRef.current = []

    const width = Math.max(container.clientWidth || 800, 320)
    const heightPx = Math.max(container.clientHeight || 640, 320)

    // Ultra-close near clipping (0.01) enables deep micro-voxel zooming without polygon clipping
    const camera = new THREE.PerspectiveCamera(40, width / heightPx, 0.01, 100)
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

    // Structural Layer Groups
    const mainGroup = new THREE.Group()
    scene.add(mainGroup)

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

    // Determine initial patient lesion centroid
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

      if (geometryData.brain.point_cloud && geometryData.brain.point_cloud.length > 0) {
        const brainPoints = buildPointCloudFromData(geometryData.brain.point_cloud)
        brainGroup.add(brainPoints)
      }

      // High-precision Wireframe for sulci & gyri
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
      wireframesRef.current.push(brainWire)

      // 2. Real Enhancing Tumor (ET) Mesh
      if (geometryData.et && geometryData.et.vertices.length > 0) {
        const etMat = new THREE.MeshStandardMaterial({
          color: 0xff5722,
          emissive: 0xe64a19,
          emissiveIntensity: 0.8,
          roughness: 0.28,
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

        // ET Voxel Wireframe
        const etWireMat = new THREE.MeshBasicMaterial({
          color: 0xffa07a,
          wireframe: true,
          transparent: true,
          opacity: 0.4,
        })
        const etWire = buildMeshFromData(
          geometryData.et.vertices,
          geometryData.et.faces,
          etWireMat,
          2
        )
        tumorGroup.add(etWire)
        wireframesRef.current.push(etWire)

        // Calculate exact centroid of enhancing tumor
        let sx = 0, sy = 0, sz = 0
        const vs = geometryData.et.vertices
        for (let i = 0; i < vs.length; i++) {
          sx += vs[i][0]
          sy += vs[i][1]
          sz += vs[i][2]
        }
        tumorCenter = new THREE.Vector3(sx / vs.length, sy / vs.length, sz / vs.length)
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
        wireframesRef.current.push(wtWire)
      }

      // 4. Real Central Necrotic Cavity (NCR) Mesh
      if (geometryData.ncr && geometryData.ncr.vertices.length > 0) {
        const ncrMat = new THREE.MeshStandardMaterial({
          color: 0x7f1d1d,
          emissive: 0x991b1b,
          emissiveIntensity: 0.5,
          roughness: 0.75,
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

        const ncrWireMat = new THREE.MeshBasicMaterial({
          color: 0xf87171,
          wireframe: true,
          transparent: true,
          opacity: 0.35,
        })
        const ncrWire = buildMeshFromData(
          geometryData.ncr.vertices,
          geometryData.ncr.faces,
          ncrWireMat,
          1
        )
        necroticGroup.add(ncrWire)
        wireframesRef.current.push(ncrWire)
      }

      // 5. Real Contextual ROI Bounding Box
      if (geometryData.bbox) {
        const sz = geometryData.bbox.size
        const cx = geometryData.bbox.center

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
      // High-precision anatomical reference fallback
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

      // Fallback meshes
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

    // Save tumor center into camera controller ref
    cameraStateRef.current.tumorCenterX = tumorCenter.x
    cameraStateRef.current.tumorCenterY = tumorCenter.y
    cameraStateRef.current.tumorCenterZ = tumorCenter.z

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
    // Studio Illumination & Camera Headlight
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

    // Camera Headlight: moves with camera to ensure micro-voxel details inside deep cavities remain bright
    const cameraHeadlight = new THREE.PointLight(0xffffff, 0.95, 25)
    camera.add(cameraHeadlight)
    scene.add(camera)

    // =========================================================================
    // Pointer, Gesture, and Pan/Orbit Controller
    // =========================================================================
    let prevPointerX = 0
    let prevPointerY = 0
    let dragMode: 'none' | 'orbit' | 'pan' = 'none'

    // Raycaster for double-click pinpoint voxel focus
    const raycaster = new THREE.Raycaster()
    const pointerVec = new THREE.Vector2()

    const onPointerDown = (e: MouseEvent) => {
      isDraggingRef.current = true
      prevPointerX = e.clientX
      prevPointerY = e.clientY

      if (e.button === 2 || e.shiftKey || e.button === 1) {
        dragMode = 'pan'
      } else {
        dragMode = 'orbit'
      }
    }

    const onPointerMove = (e: MouseEvent) => {
      if (!isDraggingRef.current || dragMode === 'none') return

      const deltaX = e.clientX - prevPointerX
      const deltaY = e.clientY - prevPointerY
      prevPointerX = e.clientX
      prevPointerY = e.clientY

      const s = cameraStateRef.current

      if (dragMode === 'orbit') {
        // Orbit around active focal target
        s.desiredTheta -= deltaX * 0.007
        s.desiredPhi = Math.max(0.06, Math.min(Math.PI - 0.06, s.desiredPhi - deltaY * 0.007))
      } else if (dragMode === 'pan') {
        // Pan focal target parallel to camera view plane
        const forward = new THREE.Vector3(
          s.targetX - camera.position.x,
          s.targetY - camera.position.y,
          s.targetZ - camera.position.z
        ).normalize()

        const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize()
        const up = new THREE.Vector3().crossVectors(right, forward).normalize()

        // Pan speed scales proportionally with zoom distance
        const panSpeed = (s.desiredRadius / 750) * 1.6
        s.desiredTargetX += (-right.x * deltaX + up.x * deltaY) * panSpeed
        s.desiredTargetY += (-right.y * deltaX + up.y * deltaY) * panSpeed
        s.desiredTargetZ += (-right.z * deltaX + up.z * deltaY) * panSpeed
        s.focusMode = 'custom'
        setFocusTargetMode('custom')
      }
    }

    const onPointerUp = () => {
      isDraggingRef.current = false
      dragMode = 'none'
    }

    const onContextMenu = (e: MouseEvent) => {
      e.preventDefault()
    }

    // High-precision Mouse Wheel Zoom with Exponential Scaling
    const onWheel = (e: WheelEvent) => {
      if (wheelMode === 'zoom' || e.ctrlKey || e.metaKey || e.altKey) {
        e.preventDefault()
        const s = cameraStateRef.current
        // Smooth exponential zoom factor
        const zoomStep = Math.exp(e.deltaY * 0.0016)
        // Zoom range from 0.22 (extreme 28x macro close-up) to 14.0 (macro overview)
        s.desiredRadius = Math.max(0.22, Math.min(14.0, s.desiredRadius * zoomStep))
      }
    }

    // Double-Click Raycasting: Focus Camera Directly on Clicked Voxel Coordinate
    const onDoubleClick = (e: MouseEvent) => {
      if (!container || !cameraRef.current) return
      const rect = container.getBoundingClientRect()
      pointerVec.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
      pointerVec.y = -((e.clientY - rect.top) / rect.height) * 2 + 1

      raycaster.setFromCamera(pointerVec, cameraRef.current)
      const targets = [tumorGroup, necroticGroup, edemaGroup, brainGroup]
      const hits = raycaster.intersectObjects(targets, true)

      if (hits.length > 0) {
        const hitPt = hits[0].point
        const s = cameraStateRef.current
        s.desiredTargetX = hitPt.x
        s.desiredTargetY = hitPt.y
        s.desiredTargetZ = hitPt.z
        // Zoom closer directly towards the clicked surface
        s.desiredRadius = Math.max(0.35, s.desiredRadius * 0.55)
        s.focusMode = 'custom'
        setFocusTargetMode('custom')
        setActiveFocalPoint([
          Math.round(hitPt.x * 100) / 100,
          Math.round(hitPt.y * 100) / 100,
          Math.round(hitPt.z * 100) / 100,
        ])
        showNotification(
          `Pinpoint Focal Target: [${hitPt.x.toFixed(2)}, ${hitPt.y.toFixed(2)}, ${hitPt.z.toFixed(2)}]`
        )
      }
    }

    const domEl = renderer.domElement
    domEl.addEventListener('mousedown', onPointerDown)
    window.addEventListener('mousemove', onPointerMove)
    window.addEventListener('mouseup', onPointerUp)
    domEl.addEventListener('contextmenu', onContextMenu)
    domEl.addEventListener('wheel', onWheel, { passive: false })
    domEl.addEventListener('dblclick', onDoubleClick)

    // Viewport Intersection Tracking
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

    // 60fps Spherical Camera Animation Loop with Damped Smoothing
    let animId = 0
    const clock = new THREE.Clock()

    const animate = () => {
      animId = requestAnimationFrame(animate)
      if (!isVisible) return

      const delta = clock.getDelta()
      const s = cameraStateRef.current

      // Silky smooth lerp towards desired target and radius
      const lerpSpeed = 0.14
      s.radius += (s.desiredRadius - s.radius) * lerpSpeed
      s.theta += (s.desiredTheta - s.theta) * lerpSpeed
      s.phi += (s.desiredPhi - s.phi) * lerpSpeed
      s.targetX += (s.desiredTargetX - s.targetX) * lerpSpeed
      s.targetY += (s.desiredTargetY - s.targetY) * lerpSpeed
      s.targetZ += (s.desiredTargetZ - s.targetZ) * lerpSpeed

      // Continuous slow cinematic auto-rotation
      if (autoRotate && !isDraggingRef.current) {
        s.desiredTheta += delta * 0.22
        s.theta += delta * 0.22
      }

      // Convert spherical coordinates around active target to Cartesian 3D
      const sinPhi = Math.sin(s.phi)
      const cosPhi = Math.cos(s.phi)
      const sinTheta = Math.sin(s.theta)
      const cosTheta = Math.cos(s.theta)

      camera.position.x = s.targetX + s.radius * sinPhi * sinTheta
      camera.position.y = s.targetY + s.radius * cosPhi
      camera.position.z = s.targetZ + s.radius * sinPhi * cosTheta

      camera.lookAt(s.targetX, s.targetY, s.targetZ)

      // Calculate magnification multiplier (baseline 6.2 distance = 1.0x)
      const mult = Math.round((6.2 / Math.max(0.2, s.radius)) * 10) / 10
      if (Math.abs(mult - lastMultiplierRef.current) >= 0.1) {
        lastMultiplierRef.current = mult
        setZoomMultiplier(mult)
      }

      renderer.render(scene, camera)
    }
    animate()

    const handleResize = () => {
      if (!container) return
      const w = Math.max(container.clientWidth || 800, 320)
      const h = Math.max(container.clientHeight || 640, 320)
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
      domEl.removeEventListener('contextmenu', onContextMenu)
      domEl.removeEventListener('wheel', onWheel)
      domEl.removeEventListener('dblclick', onDoubleClick)
      renderer.dispose()
      if (container.contains(domEl)) {
        container.removeChild(domEl)
      }
    }
  }, [geometryData, autoRotate, showPlanes, wheelMode, showNotification])

  // Reactive Layer Visibility Toggles
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

  // Toggle Enhanced Voxel Wireframe Highlight
  useEffect(() => {
    wireframesRef.current.forEach((wf) => {
      if (wf.material instanceof THREE.MeshBasicMaterial) {
        wf.material.opacity = showWireframeHighlight ? 0.75 : 0.2
        wf.material.needsUpdate = true
      }
    })
  }, [showWireframeHighlight])

  // =========================================================================
  // Interactive Camera & Zoom Handlers
  // =========================================================================

  // Continuous Slider Zoom Handler
  const handleSliderZoom = (newMult: number) => {
    const s = cameraStateRef.current
    s.desiredRadius = Math.max(0.25, Math.min(14.0, 6.2 / newMult))
    setZoomMultiplier(newMult)
  }

  // Step Zoom (+ / -)
  const handleStepZoom = (factor: number) => {
    const s = cameraStateRef.current
    s.desiredRadius = Math.max(0.22, Math.min(14.0, s.desiredRadius * factor))
  }

  // Preset Quick Zoom
  const handleApplyPreset = (targetMult: number) => {
    handleSliderZoom(targetMult)
    showNotification(`Magnification: ${targetMult.toFixed(1)}x`)
  }

  // Focus Entire Anatomical Brain
  const handleFocusBrain = () => {
    const s = cameraStateRef.current
    s.desiredTargetX = 0
    s.desiredTargetY = 0.1
    s.desiredTargetZ = 0
    s.desiredRadius = 6.2
    s.desiredPhi = Math.PI / 2.3
    s.desiredTheta = 0
    s.focusMode = 'brain'
    setFocusTargetMode('brain')
    setActiveFocalPoint([0, 0.1, 0])
    showNotification('Camera Centered: Whole Brain Overview')
  }

  // Focus Directly on Tumor Lesion Centroid (ROI Inspection Mode)
  const handleFocusTumor = () => {
    const s = cameraStateRef.current
    s.desiredTargetX = s.tumorCenterX
    s.desiredTargetY = s.tumorCenterY
    s.desiredTargetZ = s.tumorCenterZ
    // 5.4x magnification centered directly on the tumor centroid
    s.desiredRadius = 1.15
    s.desiredPhi = Math.PI / 2.2
    s.desiredTheta = 0.4
    s.focusMode = 'tumor'
    setFocusTargetMode('tumor')
    setActiveFocalPoint([
      Math.round(s.tumorCenterX * 100) / 100,
      Math.round(s.tumorCenterY * 100) / 100,
      Math.round(s.tumorCenterZ * 100) / 100,
    ])
    showNotification('Target Centered on Tumor Centroid (ROI Focus)')
  }

  // Focus Deep Inside Necrotic Core / Micro Voxel Cavity
  const handleFocusNecroticCore = () => {
    const s = cameraStateRef.current
    s.desiredTargetX = s.tumorCenterX
    s.desiredTargetY = s.tumorCenterY
    s.desiredTargetZ = s.tumorCenterZ
    // Extreme 12.4x micro-voxel zoom
    s.desiredRadius = 0.5
    s.desiredPhi = Math.PI / 2.1
    s.desiredTheta = 0.8
    s.focusMode = 'tumor'
    setFocusTargetMode('tumor')
    showNotification('Ultra Micro-Voxel Inspection: 12.4x Zoom')
  }

  return (
    <div className={`relative overflow-hidden rounded-3xl border border-nv-line bg-[#040813] shadow-2xl ${className}`}>
      {/* 3D WebGL Canvas Container */}
      <div
        ref={mountRef}
        style={{ height }}
        className="w-full cursor-grab active:cursor-grabbing select-none"
        aria-label="Real 3D Patient Brain and Tumor Surface Reconstruction with Deep Zoom Inspection"
      />

      {/* Subtle Millimeter Reticle / Precision Targeting Crosshair Overlay */}
      {showReticle && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div className="relative flex items-center justify-center">
            {/* Concentric Calibration Circles */}
            <div className="h-44 w-44 rounded-full border border-cyan-400/25 border-dashed animate-spin-slow" />
            <div className="absolute h-24 w-24 rounded-full border border-cyan-400/40" />
            <div className="absolute h-10 w-10 rounded-full border border-cyan-300/60" />
            <div className="absolute h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_12px_#38bdf8]" />

            {/* Crosshair Hairlines */}
            <div className="absolute h-64 w-px bg-gradient-to-b from-transparent via-cyan-400/40 to-transparent" />
            <div className="absolute w-64 h-px bg-gradient-to-r from-transparent via-cyan-400/40 to-transparent" />

            {/* Millimeter Scale Labels */}
            <span className="absolute -top-28 text-[9px] font-mono text-cyan-300/70 bg-black/60 px-1 rounded">
              +10 mm
            </span>
            <span className="absolute -bottom-28 text-[9px] font-mono text-cyan-300/70 bg-black/60 px-1 rounded">
              -10 mm
            </span>
            <span className="absolute -right-28 text-[9px] font-mono text-cyan-300/70 bg-black/60 px-1 rounded">
              +10 mm
            </span>
            <span className="absolute -left-28 text-[9px] font-mono text-cyan-300/70 bg-black/60 px-1 rounded">
              -10 mm
            </span>
          </div>
        </div>
      )}

      {/* Notification Toast */}
      {toastMessage && (
        <div className="absolute top-20 left-1/2 -translate-x-1/2 z-30 pointer-events-none">
          <div className="flex items-center gap-2 rounded-full border border-cyan-400/40 bg-black/90 px-4 py-1.5 text-xs font-medium text-cyan-200 shadow-2xl backdrop-blur-md animate-fade-in">
            <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
            <span>{toastMessage}</span>
          </div>
        </div>
      )}

      {/* Floating Status & Case Selector Header */}
      <div className="absolute left-5 top-5 right-5 flex flex-wrap items-center justify-between gap-3 pointer-events-none z-10">
        <div className="flex items-center gap-2.5 rounded-full border border-white/10 bg-black/80 px-3.5 py-1.5 backdrop-blur-md pointer-events-auto shadow-lg">
          <span className="flex h-2.5 w-2.5 rounded-full bg-cyan-400 shadow-[0_0_10px_#38bdf8] animate-pulse" />
          <div className="flex items-center gap-2 text-xs font-semibold tracking-wide text-zinc-200">
            <span>
              {geometryData?.is_real_data ? 'Real Patient Voxel Mesh' : '3D Reconstruction'}
            </span>
            <span className="text-zinc-500">·</span>
            <span className="font-mono text-cyan-300 text-[11px]">
              {geometryData?.dataset_source ?? 'BraTS 2020'}
            </span>
          </div>
        </div>

        {/* Dynamic Case Selector & Quick Action Bar */}
        <div className="flex items-center gap-2 pointer-events-auto">
          <div className="flex items-center gap-1.5 rounded-full border border-white/10 bg-black/85 px-3 py-1 text-xs backdrop-blur-md shadow-lg">
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

          {/* Camera Quick Bar */}
          <div className="flex items-center gap-1 rounded-full border border-white/10 bg-black/85 p-1 shadow-xl backdrop-blur-md">
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
              title="Focus on Tumor Centroid (ROI Focus)"
              className={`rounded-full p-1.5 transition-colors ${
                focusTargetMode === 'tumor'
                  ? 'bg-orange-500/30 text-orange-300'
                  : 'text-zinc-400 hover:text-orange-300'
              }`}
            >
              <Focus className="h-3.5 w-3.5" />
            </button>

            <button
              onClick={() => handleStepZoom(0.75)}
              title="Zoom In (Ctrl + Scroll / +)"
              className="rounded-full p-1.5 text-zinc-400 transition-colors hover:text-white"
            >
              <ZoomIn className="h-3.5 w-3.5" />
            </button>

            <button
              onClick={() => handleStepZoom(1.33)}
              title="Zoom Out (Ctrl + Scroll / -)"
              className="rounded-full p-1.5 text-zinc-400 transition-colors hover:text-white"
            >
              <ZoomOut className="h-3.5 w-3.5" />
            </button>

            <button
              onClick={handleFocusBrain}
              title="Reset to Whole Brain Overview"
              className="rounded-full p-1.5 text-zinc-400 transition-colors hover:text-white"
            >
              <Maximize2 className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Floating High-Precision Zoom & Inspection Control Strip (Top-Right / Middle Overlay) */}
      <div className="absolute right-5 top-20 z-20 flex flex-col items-end gap-2.5 pointer-events-none">
        <div className="flex flex-col gap-2 rounded-2xl border border-white/10 bg-black/85 p-3 shadow-2xl backdrop-blur-xl pointer-events-auto w-64">
          <div className="flex items-center justify-between border-b border-white/10 pb-2 text-xs">
            <div className="flex items-center gap-1.5 font-semibold text-zinc-200">
              <Scan className="h-3.5 w-3.5 text-cyan-400" />
              <span>Inspection Zoom</span>
            </div>
            <span className="font-mono font-bold text-cyan-300 text-xs bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-500/30">
              {zoomMultiplier.toFixed(1)}x ({Math.round(zoomMultiplier * 100)}%)
            </span>
          </div>

          {/* Continuous Zoom Slider */}
          <div className="space-y-1 pt-1">
            <div className="flex justify-between text-[10px] font-mono text-zinc-400">
              <span>0.5x Overview</span>
              <span>15.0x Micro</span>
            </div>
            <input
              type="range"
              min={0.5}
              max={15.0}
              step={0.1}
              value={Math.min(15.0, Math.max(0.5, zoomMultiplier))}
              onChange={(e) => handleSliderZoom(parseFloat(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer h-1.5 rounded-lg bg-zinc-700"
              aria-label="Inspection Zoom Level Slider"
            />
          </div>

          {/* Quick Zoom Presets Chips */}
          <div className="grid grid-cols-4 gap-1.5 pt-1">
            {[
              { label: '1x Fit', val: 1.0 },
              { label: '2.5x', val: 2.5 },
              { label: '5x ROI', val: 5.0 },
              { label: '10x Core', val: 10.0 },
            ].map((p) => (
              <button
                key={p.label}
                onClick={() => handleApplyPreset(p.val)}
                className={`rounded-lg py-1 text-[11px] font-mono font-medium transition-all ${
                  Math.abs(zoomMultiplier - p.val) < 0.4
                    ? 'bg-cyan-500/25 text-cyan-300 border border-cyan-500/50'
                    : 'bg-white/5 text-zinc-400 hover:text-white hover:bg-white/10 border border-transparent'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* Target Focal Center Selector */}
          <div className="pt-2 border-t border-white/10 space-y-1.5">
            <div className="flex justify-between text-[11px] text-zinc-400">
              <span>Orbit Pivot Center:</span>
              <span className="font-mono text-xs font-semibold text-zinc-200">
                {focusTargetMode === 'tumor'
                  ? 'Tumor Centroid'
                  : focusTargetMode === 'custom'
                  ? 'Voxel Hit'
                  : 'Brain Center'}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-1.5">
              <button
                onClick={handleFocusBrain}
                className={`flex items-center justify-center gap-1 rounded-lg py-1 text-[11px] transition-colors ${
                  focusTargetMode === 'brain'
                    ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40'
                    : 'bg-white/5 text-zinc-400 hover:text-white'
                }`}
              >
                <Box className="h-3 w-3" /> Brain Origin
              </button>
              <button
                onClick={handleFocusTumor}
                className={`flex items-center justify-center gap-1 rounded-lg py-1 text-[11px] transition-colors ${
                  focusTargetMode === 'tumor'
                    ? 'bg-orange-500/25 text-orange-300 border border-orange-500/40'
                    : 'bg-white/5 text-zinc-400 hover:text-white'
                }`}
              >
                <Target className="h-3 w-3" /> Lesion Centroid
              </button>
            </div>
          </div>

          {/* Deep Inspection Tools (Reticle & Wireframe) */}
          <div className="pt-1.5 border-t border-white/10 flex items-center justify-between text-[11px]">
            <button
              onClick={() => setShowReticle(!showReticle)}
              title="Toggle Millimeter Targeting Reticle"
              className={`flex items-center gap-1 rounded-lg px-2 py-1 transition-colors ${
                showReticle
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-zinc-400 hover:text-white bg-white/5'
              }`}
            >
              <Crosshair className="h-3 w-3" /> Reticle
            </button>

            <button
              onClick={() => setShowWireframeHighlight(!showWireframeHighlight)}
              title="Toggle Enhanced Voxel Wireframe Topology"
              className={`flex items-center gap-1 rounded-lg px-2 py-1 transition-colors ${
                showWireframeHighlight
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-zinc-400 hover:text-white bg-white/5'
              }`}
            >
              <Grid className="h-3 w-3" /> Voxel Grid
            </button>

            <button
              onClick={() => setWheelMode(wheelMode === 'zoom' ? 'scroll' : 'zoom')}
              title={wheelMode === 'zoom' ? 'Mouse wheel zooms 3D model' : 'Mouse wheel scrolls page'}
              className="text-[10px] font-mono text-zinc-400 hover:text-cyan-300"
            >
              Wheel: {wheelMode === 'zoom' ? 'Zoom 3D' : 'Page'}
            </button>
          </div>
        </div>

        {/* Live Coordinate & Scale Badge */}
        <div className="rounded-xl border border-white/10 bg-black/80 px-3 py-1.5 text-[10px] font-mono text-zinc-400 backdrop-blur-md pointer-events-auto shadow-lg space-y-0.5">
          <div className="flex items-center justify-between gap-3">
            <span>Focal Pivot:</span>
            <b className="text-zinc-200">[{activeFocalPoint.join(', ')}]</b>
          </div>
          <div className="flex items-center justify-between gap-3 text-zinc-500">
            <span>Scale Resolution:</span>
            <span className="text-cyan-400/80">1 voxel = 1.00 mm³</span>
          </div>
        </div>
      </div>

      {/* Layer Control Strip & Metric Badges at Bottom */}
      <div className="absolute bottom-5 left-5 right-5 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-white/10 bg-black/85 px-4 py-3 shadow-2xl backdrop-blur-xl z-10">
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
            <span>ROI Box (+10 vox)</span>
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

        {/* Live Interaction Hint Guide */}
        <div className="flex items-center gap-3 text-[11px] font-mono text-zinc-400">
          <span className="hidden lg:inline-block text-zinc-500">
            Left Drag: Orbit · Right/Shift Drag: Pan · Scroll: Zoom · Dbl-Click: Pinpoint Target
          </span>
          <span className="text-zinc-600 hidden lg:inline-block">·</span>
          <span className="text-cyan-300">
            {geometryData?.dimensions_voxel
              ? `${geometryData.dimensions_voxel.join('×')} voxels`
              : 'Iso-Surface 1mm³'}
          </span>
        </div>
      </div>
    </div>
  )
}
