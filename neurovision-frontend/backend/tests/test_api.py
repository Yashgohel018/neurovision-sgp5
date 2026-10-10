from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "service" in data

def test_cases_crud():
    # List cases
    res = client.get("/api/cases")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # Create new case
    res_create = client.post("/api/cases")
    assert res_create.status_code == 200
    created = res_create.json()
    case_id = created["case_id"]
    assert case_id.startswith("CASE-")

    # Get single case
    res_get = client.get(f"/api/cases/{case_id}")
    assert res_get.status_code == 200
    assert res_get.json()["case_id"] == case_id

    # Delete case
    res_del = client.delete(f"/api/cases/{case_id}")
    assert res_del.status_code == 200
    assert res_del.json()["deleted"] is True

def test_slice_rendering():
    res = client.get("/api/cases/CASE-2026-001/slice?modality=FLAIR&plane=axial&index=60")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert len(res.content) > 100

def test_volumetry():
    res = client.get("/api/cases/CASE-2026-001/volume")
    assert res.status_code == 200
    data = res.json()
    assert "WT_volume_cm3" in data
    assert "TC_volume_cm3" in data
    assert "ET_volume_cm3" in data

def test_inference_honest_model_state():
    res = client.post("/api/inference", json={"case_id": "CASE-2026-001"})
    assert res.status_code == 200
    data = res.json()
    assert data["case_id"] == "CASE-2026-001"
    # When no model checkpoint is trained, it should not fabricate predictions
    assert data["prediction"] is None or isinstance(data["prediction"], str)

def test_models_registry():
    res = client.get("/api/models")
    assert res.status_code == 200
    models = res.json()
    assert len(models) >= 1
    assert "name" in models[0]
    assert "metrics" in models[0]

def test_models_comparison():
    res = client.get("/api/models/comparison")
    assert res.status_code == 200
    rows = res.json()
    assert len(rows) >= 5
    assert any(r["model_id"] == "P2" for r in rows)

def test_models_ablation():
    res = client.get("/api/models/ablation")
    assert res.status_code == 200
    items = res.json()
    assert len(items) >= 4
    assert any(i["category"] == "Modality Ablation" for i in items)

def test_pipeline_modules():
    res = client.get("/api/pipeline/modules")
    assert res.status_code == 200
    mods = res.json()
    assert len(mods) == 12
    assert any(m["id"] == "A" for m in mods)
    assert any(m["id"] == "L" for m in mods)

def test_assistant_rag():
    res = client.post("/api/rag/query", json={"query": "Explain WT and TC volume metrics"})
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert len(data["sources"]) > 0
    assert "disclaimer" in data

def test_report_generation():
    res = client.get("/api/cases/CASE-2026-001/report")
    assert res.status_code == 200
    data = res.json()
    assert data["case_id"] == "CASE-2026-001"
    assert "disclaimer" in data
    assert "radiological_findings" in data
