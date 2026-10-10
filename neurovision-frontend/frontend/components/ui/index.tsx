import Link from 'next/link'
const NAV = [['Dashboard', '/dashboard'], ['Cases', '/cases'], ['Pipeline', '/pipeline'], ['Models', '/models'], ['Assistant', '/assistant']]
export const DISCLAIMER = 'NeuroVision AI is a research and clinical decision-support system. It does not replace diagnosis or clinical judgment by qualified healthcare professionals.'
export function DemoBadge({ label = 'DEMO DATA' }: { label?: string }) { return <span className="rounded-full border border-nv-acc/60 px-2.5 py-0.5 text-[11px] font-semibold tracking-widest text-nv-acc">{label}</span> }
export function ErrorState({ message, onRetry, onDemo }: { message: string; onRetry?: () => void; onDemo?: () => void }) {
  return <div role="alert" className="rounded-2xl border border-nv-line bg-nv-panel p-6"><p className="text-sm text-zinc-300">{message}</p>
    <div className="mt-4 flex gap-3 text-sm">{onRetry && <button onClick={onRetry} className="rounded-full border border-white/20 px-4 py-1.5 hover:bg-white/10">Retry</button>}{onDemo && <button onClick={onDemo} className="rounded-full bg-white px-4 py-1.5 text-black">Load DEMO DATA</button>}</div></div>
}
export function AppShell({ title, children, demo }: { title: string; children: React.ReactNode; demo?: boolean }) {
  return <div className="min-h-screen bg-nv-bg">
    <header className="sticky top-0 z-10 flex items-center justify-between border-b border-nv-line bg-nv-bg/80 px-6 py-3 backdrop-blur">
      <Link href="/" className="font-semibold tracking-tight">NeuroVision AI</Link>
      <nav aria-label="App" className="flex gap-5 text-sm text-zinc-400">{NAV.map(([n, h]) => <Link key={h} href={h} className="hover:text-white">{n}</Link>)}</nav></header>
    <main className="mx-auto max-w-7xl px-6 py-10"><div className="mb-8 flex items-center gap-3"><h1 className="text-4xl font-semibold tracking-tight">{title}</h1>{demo && <DemoBadge />}</div>{children}</main>
    <footer className="border-t border-nv-line px-6 py-5 text-xs text-zinc-500">Research / Decision Support Output — Not a Diagnosis. {DISCLAIMER}</footer></div>
}
export const Stat = ({ label, value }: { label: string; value: React.ReactNode }) => <div className="rounded-2xl border border-nv-line bg-nv-panel p-5"><div className="text-xs uppercase tracking-widest text-zinc-500">{label}</div><div className="mt-2 text-3xl font-semibold">{value}</div></div>
