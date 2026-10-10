from fastapi import APIRouter
from app.schemas.report import AnalysisReport
from app.services.report_service import report_service

router = APIRouter()

@router.get("/api/cases/{case_id}/report", response_model=AnalysisReport)
def get_report(case_id: str):
    return report_service.generate_report(case_id)

@router.post("/api/cases/{case_id}/report", response_model=AnalysisReport)
def create_report(case_id: str):
    return report_service.generate_report(case_id)
