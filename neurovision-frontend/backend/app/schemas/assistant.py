from typing import List, Optional
from pydantic import BaseModel, Field

class RagSource(BaseModel):
    title: str
    url: Optional[str] = None
    relevance: Optional[str] = None

class RagRequest(BaseModel):
    query: str
    case_id: Optional[str] = None

class RagAnswer(BaseModel):
    answer: str
    sources: List[RagSource] = Field(default_factory=list)
    is_demo: bool = False
    disclaimer: str = (
        "NeuroVision AI is a research and clinical decision-support system. "
        "It does not replace diagnosis or clinical judgment by qualified healthcare professionals."
    )
