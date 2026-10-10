from fastapi import APIRouter
from app.services.inference_service import inference_service

router = APIRouter()

@router.get("/api/health")
def health_check():
    models = inference_service.get_registered_models()
    loaded_count = sum(1 for m in models if m.loaded)
    return {
        "status": "healthy",
        "service": "NeuroVision AI Backend",
        "version": "1.0.0",
        "registered_models": len(models),
        "loaded_models": loaded_count,
    }
