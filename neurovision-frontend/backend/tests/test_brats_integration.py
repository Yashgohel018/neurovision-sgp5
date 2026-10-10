from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_brats_cases_listing():
    res = client.get("/api/cases?limit=10")
    assert res.status_code == 200
    cases = res.json()
    assert len(cases) >= 10
    ids = [c["case_id"] for c in cases]
    assert any("BraTS20_Training_" in cid for cid in ids) or "CASE-2026-001" in ids

def test_brats_single_case_retrieval():
    res = client.get("/api/cases/BraTS20_Training_001")
    assert res.status_code == 200
    data = res.json()
    assert data["case_id"] == "BraTS20_Training_001"
    assert data["grade"] == "HGG"
    assert data["is_real_brats"] is True
    assert set(["T1", "T1ce", "T2", "FLAIR", "Seg"]).issubset(set(data["modalities"]))

def test_brats_real_slice_rendering():
    res = client.get("/api/cases/BraTS20_Training_001/slice?modality=FLAIR&plane=axial&index=67&overlay=seg")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert len(res.content) > 1000  # Real PNG byte stream

def test_brats_real_slice_cam():
    res = client.get("/api/cases/BraTS20_Training_001/slice?modality=T1ce&plane=axial&index=67&overlay=cam")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert len(res.content) > 1000

def test_brats_real_volumetry():
    res = client.get("/api/cases/BraTS20_Training_001/volume")
    assert res.status_code == 200
    data = res.json()
    assert data["WT_volume_cm3"] > 200.0
    assert data["TC_volume_cm3"] > 40.0
    assert data["ET_volume_cm3"] > 20.0

def test_brats_real_inference_hgg():
    res = client.post("/api/inference", json={"case_id": "BraTS20_Training_001"})
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] == "High-Grade Glioma (HGG)"
    assert data["probabilities"]["HGG"] > 0.8
    assert data["segmentation_available"] is True

def test_brats_real_inference_lgg():
    res = client.post("/api/inference", json={"case_id": "BraTS20_Training_262"})
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] == "Low-Grade Glioma (LGG)"
    assert data["probabilities"]["LGG"] > 0.6

def test_brats_real_report():
    res = client.get("/api/cases/BraTS20_Training_001/report")
    assert res.status_code == 200
    data = res.json()
    assert data["case_id"] == "BraTS20_Training_001"
    assert any("60.5" in f for f in data["radiological_findings"])
    assert data["prediction"] == "High-Grade Glioma (HGG)"
