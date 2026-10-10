import io
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Response
from fastapi.responses import Response

from app.schemas.case import CaseSummary, CreateCaseResponse, DeleteCaseResponse
from app.services.case_service import case_service
from app.services.nifti_service import nifti_service
from app.services.volumetry_service import volumetry_service
from app.services.geometry_service import geometry_service
from app.config import CASES_DIR

router = APIRouter()

@router.get("/api/cases", response_model=List[CaseSummary])
def list_cases(
    limit: Optional[int] = None,
    offset: int = 0,
    query: Optional[str] = None,
    grade: Optional[str] = None,
    split: Optional[str] = None,
    include_brats: bool = True,
):
    return case_service.get_all_cases(
        limit=limit,
        offset=offset,
        query=query,
        grade=grade,
        split=split,
        include_brats=include_brats,
    )

@router.post("/api/cases", response_model=CaseSummary)
def create_case():
    return case_service.create_case()

@router.get("/api/cases/{case_id}", response_model=CaseSummary)
def get_case(case_id: str):
    c = case_service.get_case(case_id)
    if not c:
        raise HTTPException(status_code=404, detail="Case not found.")
    return c

@router.delete("/api/cases/{case_id}", response_model=DeleteCaseResponse)
def delete_case(case_id: str):
    success = case_service.delete_case(case_id)
    if not success:
        raise HTTPException(status_code=404, detail="Case not found.")
    return DeleteCaseResponse(deleted=True, case_id=case_id)

@router.post("/api/cases/{case_id}/upload", response_model=CaseSummary)
async def upload_modality(
    case_id: str,
    modality: str = Form(...),
    file: UploadFile = File(...),
):
    valid_mods = ["T1", "T1ce", "T2", "FLAIR", "Seg"]
    if modality not in valid_mods:
        raise HTTPException(status_code=400, detail=f"Invalid modality. Must be one of: {valid_mods}")

    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(status_code=415, detail="Only .nii or .nii.gz files are supported.")

    case_folder = CASES_DIR / case_id
    case_folder.mkdir(parents=True, exist_ok=True)

    ext = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"
    target_path = case_folder / f"{modality}{ext}"

    content = await file.read()
    if len(content) > 500 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds 500MB maximum size limit.")

    with open(target_path, "wb") as f:
        f.write(content)

    updated = case_service.add_modality(case_id, modality)
    if not updated:
        raise HTTPException(status_code=404, detail="Case not found.")
    return updated

@router.get("/api/cases/{case_id}/modalities")
def get_modalities(case_id: str):
    c = case_service.get_case(case_id)
    if not c:
        raise HTTPException(status_code=404, detail="Case not found.")
    return {"case_id": case_id, "modalities": c.modalities}

@router.get("/api/cases/{case_id}/slice")
def get_slice(
    case_id: str,
    modality: str = "FLAIR",
    plane: str = "axial",
    index: int = 60,
    overlay: Optional[str] = None,
):
    png_bytes = nifti_service.render_slice_png(
        case_id=case_id,
        modality=modality,
        plane=plane,
        index=index,
        overlay=overlay,
    )
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400, stale-while-revalidate=3600"}
    )

@router.get("/api/cases/{case_id}/volume")
def get_volume(case_id: str):
    v = volumetry_service.calculate_volumes(case_id)
    if not v:
        raise HTTPException(status_code=404, detail="No segmentation mask available for volumetric quantification.")
    return v

@router.get("/api/cases/{case_id}/geometry3d")
def get_geometry_3d(case_id: str):
    return geometry_service.get_case_geometry_3d(case_id)
