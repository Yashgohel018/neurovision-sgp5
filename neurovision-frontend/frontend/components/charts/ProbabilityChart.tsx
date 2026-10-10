'use client'
import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis, Tooltip } from 'recharts'
export default function ProbabilityChart({ probabilities }: { probabilities: Record<string, number> }) {
  const data = Object.entries(probabilities).map(([name, p]) => ({ name, pct: +(p * 100).toFixed(1) }))
  return <div className="h-32" role="img" aria-label="Class probabilities"><ResponsiveContainer><BarChart data={data} layout="vertical" margin={{ left: 8, right: 16 }}>
    <XAxis type="number" domain={[0, 100]} stroke="#52525b" fontSize={11} /><YAxis type="category" dataKey="name" stroke="#a1a1aa" fontSize={12} width={40} /><Tooltip formatter={(v: number) => `${v}%`} contentStyle={{ background: '#0a0f1a', border: '1px solid #1b2536' }} />
    <Bar dataKey="pct" fill="#ff7a45" radius={4} /></BarChart></ResponsiveContainer></div>
}
