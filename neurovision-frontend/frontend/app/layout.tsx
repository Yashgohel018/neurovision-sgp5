import './globals.css'
import type { Metadata, Viewport } from 'next'
export const metadata: Metadata = { title: 'NeuroVision AI', description: 'Multimodal brain MRI clinical decision support (research / educational).' }
export const viewport: Viewport = { width: 'device-width', initialScale: 1, viewportFit: 'cover' }
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>
}
