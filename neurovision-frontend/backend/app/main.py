from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import CORS_ORIGINS
from app.api import health, cases, inference, models, assistant, reports

app = FastAPI(
    title="NeuroVision AI — Clinical Decision Support API",
    description="Multimodal Brain MRI Analysis, Segmentation, Volumetry, and Decision Support API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(cases.router)
app.include_router(inference.router)
app.include_router(models.router)
app.include_router(assistant.router)
app.include_router(reports.router)

@app.get("/")
def root():
    return {
        "project": "NeuroVision AI",
        "tagline": "See the Brain. Understand the Tumor.",
        "status": "online",
        "docs_url": "/docs",
    }
