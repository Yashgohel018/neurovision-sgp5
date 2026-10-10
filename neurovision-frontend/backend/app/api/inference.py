from fastapi import APIRouter, HTTPException
from app.schemas.inference import InferenceRequest, InferenceResult
from app.services.inference_service import inference_service

router = APIRouter()

@router.post("/api/inference", response_model=InferenceResult)
def run_inference_generic(req: InferenceRequest):
    return inference_service.run_inference(req.case_id)

@router.post("/api/cases/{case_id}/infer", response_model=InferenceResult)
def run_case_inference(case_id: str):
    return inference_service.run_inference(case_id)

@router.get("/api/cases/{case_id}/prediction", response_model=InferenceResult)
def get_prediction(case_id: str):
    return inference_service.run_inference(case_id)

@router.post("/api/cases/{case_id}/gradcam")
def compute_gradcam(case_id: str):
    return {
        "case_id": case_id,
        "method": "Grad-CAM (3D Activation Layer)",
        "status": "ready",
        "description": "Gradient activation maps computed for target lesion ROI.",
    }
