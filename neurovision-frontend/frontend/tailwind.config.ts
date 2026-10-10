import type { Config } from 'tailwindcss'
export default { content: ['./app/**/*.{ts,tsx}','./components/**/*.{ts,tsx}'], theme: { extend: { colors: { nv: { bg:'#03060d', panel:'#0a0f1a', line:'#1b2536', acc:'#ff7a45', cool:'#4d9bff' } } } }, plugins: [] } satisfies Config
