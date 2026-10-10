# NeuroVision AI — Frontend (Next.js 14, TypeScript strict, Tailwind)

```
npm install
cp .env.example .env     # set NEXT_PUBLIC_API_URL to your FastAPI backend
npm run dev              # http://localhost:3000
npm run typecheck && npm run build
```

## Where you plug things in
All backend calls live in `lib/api.ts`; response shapes are in `lib/types.ts`. Nothing else talks to the network.

| Endpoint | Used by | Notes |
|---|---|---|
| GET/POST /api/cases, DELETE /api/cases/{id} | Cases, Dashboard | |
| POST /api/cases/{id}/upload (multipart: modality, file) | Cases | `.nii` / `.nii.gz` only |
| GET /api/cases/{id}/slice?modality&plane&index[&overlay=seg or cam] | Analysis | return a PNG (slice extracted server-side) |
| POST /api/inference `{case_id}` | Analysis | `prediction` / `probabilities` may be `null` -> UI shows "model not loaded" |
| GET /api/models | Models, Dashboard | |
| POST /api/rag/query `{query, case_id?}` | Assistant | put your agent API behind this; keep keys server-side |

Errors are mapped to friendly messages in `api.ts`; raw server text is never shown.

## Demo mode
If the backend is down, Dashboard and Cases offer **Load DEMO DATA**. Demo views carry a visible DEMO DATA badge and never show a prediction.

## Landing page
`components/landing/ScrollStory.tsx` (markup) + `lib/scrollStory.ts` (Three.js engine, `// @ts-nocheck`, procedural DEMO visuals). Chapter shapes are built in `build()`; swap them for real slices / tumor mesh points from your backend. Honors `prefers-reduced-motion`.

## Not included yet
Backend, model code, auth, PDF report export, interactive 3D tumor viewer page.
