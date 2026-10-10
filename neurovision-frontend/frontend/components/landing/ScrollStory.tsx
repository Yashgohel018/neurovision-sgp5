'use client'
import { useEffect } from 'react'
import Link from 'next/link'
import './story.css'
import { initScrollStory } from '@/lib/scrollStory'
const L = [['T1', 'anatomy'], ['T1ce', 'enhancing tumor'], ['T2', 'tissue abnormality'], ['FLAIR', 'edema / fluid signal']]
export default function ScrollStory() {
  useEffect(() => initScrollStory(), [])
  return <div className="story">
    <canvas id="gl" aria-hidden="true" /><div className="fade" />
    <div className="nogl"><p>WebGL is unavailable in this browser, so the 3D story cannot render.</p></div>
    <nav className="story-nav" aria-label="Primary"><b>NeuroVision AI</b><span><a href="#" data-k="1">Explore</a><a href="#" data-k="3">Technology</a><a href="#" data-k="4">Analysis</a><Link href="/cases">Cases</Link><Link href="/dashboard" className="go">Launch Workspace</Link></span></nav>
    <section className="ch c0"><p className="eye">Agentic multimodal MRI decision support</p>
      <h1><span id="l1" style={{ color: 'var(--fg)' }}>See the Brain.</span><span id="l2">Understand the Tumor.</span></h1>
      <p className="t">Multimodal brain MRI, turned into an interactive, explainable decision-support experience.</p>
      <div className="cta"><button data-k="1">Explore NeuroVision</button><button data-k="4">View the Science</button></div></section>
    <section className="ch"><p className="eye">01 · Multimodal MRI</p><h2>One Scan.<br />Thousands of Signals.</h2><p className="t">T1 shows structure, T1ce enhancing tumor, T2 tissue abnormality, FLAIR edema. One slice slider drives all four.</p></section>
    <section className="ch"><p className="eye">02 · 3D Volume</p><h2>From Voxels<br />to Volume.</h2><p className="t">The mask becomes a surface inside a bounding box. Volume is voxel count × voxel spacing, computed from the segmentation, never typed in.</p></section>
    <section className="ch"><p className="eye">03 · Explainability</p><h2>Why This<br />Prediction?</h2><p className="t">Model attention is overlaid on the scan. It is an attention visualisation, not a clinical finding.</p></section>
    <section className="ch"><p className="eye">04 · Analysis Workspace</p><h2>Everything.<br />In One Place.</h2><p className="t">Slices, volumes, probabilities and a clinical summary side by side. HGG / LGG: <b style={{ color: 'var(--fg)' }}>model not loaded</b> until a trained checkpoint is attached.</p></section>
    <section className="ch"><p className="eye">NeuroVision AI</p><h2>Ready when<br />your model is.</h2><p className="t">Research and decision-support output — not a diagnosis.</p><div className="cta" style={{ justifyContent: 'flex-start' }}><Link href="/dashboard"><button>Launch Workspace</button></Link><button data-k="0">Replay the story</button></div></section>
    {L.map(([a, b]) => <div className="lb" key={a}><span>{a}</span><small>{b}</small></div>)}
    <div className="spark" aria-hidden="true">✦</div><div className="rail" role="tablist" aria-label="Chapters" /><div className="hint" id="hint">Scroll</div>
    <div className="badge"><b>DEMO DATA</b> · procedural visualisation, not patient data. Research / decision-support output — not a diagnosis.</div>
    <div style={{ height: '700vh' }} /></div>
}
