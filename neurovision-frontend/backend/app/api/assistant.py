from typing import List
from fastapi import APIRouter
from app.schemas.assistant import RagRequest, RagAnswer, RagSource
from app.services.assistant_service import assistant_service

router = APIRouter()

@router.post("/api/rag/query", response_model=RagAnswer)
def query_clinical_assistant(req: RagRequest):
    return assistant_service.answer_query(query=req.query, case_id=req.case_id)

@router.get("/api/assistant/references", response_model=List[RagSource])
def get_references():
    return [
        RagSource(
            title="Bakas et al. (2021) 'The Brain Tumor Segmentation (BraTS) Challenge'",
            url="https://doi.org/10.1038/s41597-021-01015-8",
            relevance="Standard benchmark protocols for multi-modal brain tumor MRI"
        ),
        RagSource(
            title="Louis et al. (2021) 'The 2021 WHO Classification of Tumors of the Central Nervous System'",
            url="https://doi.org/10.1093/neuonc/noab106",
            relevance="WHO CNS5 taxonomy and molecular diagnostics"
        ),
        RagSource(
            title="Selvaraju et al. (2017) 'Grad-CAM: Visual Explanations from Deep Networks'",
            url="https://doi.org/10.1109/ICCV.2017.74",
            relevance="Explainability and activation mapping in convolutional networks"
        )
    ]
