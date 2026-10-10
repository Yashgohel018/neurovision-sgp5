from typing import List, Optional, Literal
from pydantic import BaseModel, Field

ModalityType = Literal["T1", "T1ce", "T2", "FLAIR", "Seg"]
PlaneType = Literal["axial", "coronal", "sagittal"]
CaseStatusType = Literal["pending", "processing", "ready", "error"]

class CaseSummary(BaseModel):
    case_id: str
    created_at: str
    modalities: List[str] = Field(default_factory=list)
    status: CaseStatusType = "pending"
    model_version: Optional[str] = None
    is_demo: bool = False
    grade: Optional[str] = None
    age: Optional[float] = None
    survival_days: Optional[float] = None
    extent_of_resection: Optional[str] = None
    split: Optional[str] = None
    is_real_brats: bool = False
    slice_count: Optional[int] = None

class CreateCaseResponse(BaseModel):
    case_id: str
    created_at: str
    modalities: List[str] = Field(default_factory=list)
    status: CaseStatusType = "pending"
    is_demo: bool = False

class DeleteCaseResponse(BaseModel):
    deleted: bool
    case_id: str
