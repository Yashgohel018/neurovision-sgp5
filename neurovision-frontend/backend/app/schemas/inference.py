from typing import Dict, Optional, List, Any
from pydantic import BaseModel, Field

class Volumes(BaseModel):
    WT_volume_cm3: float = Field(..., description="Whole Tumor (labels 1 + 2 + 4)")
    TC_volume_cm3: float = Field(..., description="Tumor Core (labels 1 + 4)")
    ET_volume_cm3: float = Field(..., description="Enhancing Tumor (label 4)")

class InferenceRequest(BaseModel):
    case_id: str

class InferenceResult(BaseModel):
    case_id: str
    prediction: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    confidence_score: Optional[float] = None
    modality_attention: Optional[Dict[str, float]] = None
    rationale: Optional[str] = None
    tumor: Optional[Volumes] = None
    segmentation_available: bool = False
    model_version: Optional[str] = None
    inference_time: Optional[float] = None
    is_demo: bool = False
    message: Optional[str] = None

class ModelInfo(BaseModel):
    name: str
    version: str
    task: str
    loaded: bool
    input_modalities: List[str]
    metrics: Optional[Dict[str, float]] = None

class ModelBenchmarkRow(BaseModel):
    model_id: str
    model_name: str
    roc_auc: float
    accuracy: float
    sensitivity: float
    specificity: float
    f1_score: float
    pr_auc: float
    auc_ci_lower: float
    auc_ci_upper: float
    acc_ci_lower: float
    acc_ci_upper: float
    tp: int
    tn: int
    fp: int
    fn: int

class AblationItem(BaseModel):
    category: str
    experiment: str
    auc: float
    accuracy: float
    sensitivity: float
    specificity: float
    f1_score: float

class PipelineModuleInfo(BaseModel):
    id: str
    code: str
    title: str
    phase: str
    status: str
    deliverables: List[str]
    key_metrics: Dict[str, Any]
    summary: str
