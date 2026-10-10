from typing import List
from fastapi import APIRouter
from app.schemas.inference import (
    ModelInfo,
    ModelBenchmarkRow,
    AblationItem,
    PipelineModuleInfo
)
from app.services.inference_service import inference_service

router = APIRouter()

@router.get("/api/models", response_model=List[ModelInfo])
def list_models():
    return inference_service.get_registered_models()

@router.get("/api/models/comparison", response_model=List[ModelBenchmarkRow])
def get_model_comparisons():
    return inference_service.get_benchmark_comparison()

@router.get("/api/models/ablation", response_model=List[AblationItem])
def get_ablation_results():
    return inference_service.get_ablation_results()

@router.get("/api/pipeline/modules", response_model=List[PipelineModuleInfo])
def get_pipeline_modules():
    return inference_service.get_pipeline_modules()
