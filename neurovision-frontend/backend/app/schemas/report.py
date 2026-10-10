from typing import List, Optional, Dict
from pydantic import BaseModel
from .inference import Volumes

class AnalysisReport(BaseModel):
    case_id: str
    generated_at: str
    modalities_available: List[str]
    segmentation_status: str
    tumor_volumes: Optional[Volumes] = None
    prediction: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None
    radiological_findings: List[str]
    methodology: str
    disclaimer: str = (
        "NeuroVision AI is a research and clinical decision-support prototype. "
        "Not a definitive diagnosis. Output requires review by board-certified radiologists."
    )
