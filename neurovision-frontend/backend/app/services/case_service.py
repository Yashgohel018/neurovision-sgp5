import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import pandas as pd

from app.config import (
    CASES_DIR,
    DATA_DIR,
    BRATS_DATA_DIR,
    DATASET_METADATA_CSV,
    SPLITS_DIR,
    TUMOR_STATS_CSV,
)
from app.schemas.case import CaseSummary

INDEX_FILE = DATA_DIR / "cases_index.json"

DEFAULT_CASES = [
    {
        "case_id": "CASE-2026-001",
        "created_at": "2026-10-06T09:30:00Z",
        "modalities": ["T1", "T1ce", "T2", "FLAIR", "Seg"],
        "status": "ready",
        "model_version": "NeuroVision-MorphNet v1.0",
        "is_demo": False,
        "grade": "HGG",
        "age": 60.5,
        "survival_days": 289.0,
        "extent_of_resection": "GTR",
        "split": "train",
        "is_real_brats": True,
        "slice_count": 155,
    },
    {
        "case_id": "CASE-2026-002",
        "created_at": "2026-10-06T14:15:00Z",
        "modalities": ["T1", "T1ce", "T2", "FLAIR", "Seg"],
        "status": "ready",
        "model_version": "NeuroVision-MorphNet v1.0",
        "is_demo": False,
        "grade": "HGG",
        "age": 52.3,
        "survival_days": 616.0,
        "extent_of_resection": "GTR",
        "split": "train",
        "is_real_brats": True,
        "slice_count": 155,
    },
]

class CaseService:
    def __init__(self):
        self._ensure_index()
        self._brats_cases_cache: List[Dict[str, Any]] = []
        self._brats_lookup: Dict[str, Dict[str, Any]] = {}
        self._load_brats_registry()

    def _ensure_index(self):
        if not INDEX_FILE.exists():
            with open(INDEX_FILE, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CASES, f, indent=2)

    def _read_index(self) -> List[Dict[str, Any]]:
        self._ensure_index()
        try:
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_CASES

    def _write_index(self, cases: List[Dict[str, Any]]):
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2)

    def _load_brats_registry(self):
        """Loads and indexes the 369 real BraTS 2020 patient cases from learnNeuro."""
        if not DATASET_METADATA_CSV.exists():
            return

        try:
            df = pd.read_csv(DATASET_METADATA_CSV)
            
            # Load splits if available
            split_map: Dict[str, str] = {}
            if SPLITS_DIR.exists():
                for s_name in ["train", "val", "test"]:
                    s_file = SPLITS_DIR / f"{s_name}_subjects.csv"
                    if s_file.exists():
                        s_df = pd.read_csv(s_file)
                        for sid in s_df["subject_id"]:
                            split_map[str(sid)] = s_name

            cases_list = []
            import re
            for _, row in df.iterrows():
                sid = str(row["subject_id"])
                
                # Safely parse age
                age_val = None
                raw_age = row.get("age")
                if pd.notna(raw_age):
                    try:
                        age_val = float(raw_age)
                    except (ValueError, TypeError):
                        m = re.search(r'[\d\.]+', str(raw_age))
                        age_val = float(m.group()) if m else None

                # Safely parse survival days
                surv_val = None
                raw_surv = row.get("survival_days")
                if pd.notna(raw_surv):
                    try:
                        surv_val = float(raw_surv)
                    except (ValueError, TypeError):
                        m = re.search(r'\d+', str(raw_surv))
                        surv_val = float(m.group()) if m else None

                resect = str(row["extent_of_resection"]) if pd.notna(row.get("extent_of_resection")) else None
                grade_val = str(row["grade"]) if pd.notna(row.get("grade")) else "Unknown"

                # Check available modalities
                mods = ["T1", "T1ce", "T2", "FLAIR"]
                if pd.notna(row.get("seg_path")):
                    mods.append("Seg")

                item = {
                    "case_id": sid,
                    "created_at": "2020-06-01T00:00:00Z",
                    "modalities": mods,
                    "status": "ready",
                    "model_version": "NeuroVision-MorphNet v1.0",
                    "is_demo": False,
                    "grade": grade_val,
                    "age": age_val,
                    "survival_days": surv_val,
                    "extent_of_resection": resect,
                    "split": split_map.get(sid, "train"),
                    "is_real_brats": True,
                    "slice_count": int(row.get("shape_z", 155)),
                }
                cases_list.append(item)
                self._brats_lookup[sid] = item

            self._brats_cases_cache = cases_list
        except Exception as e:
            print(f"[CaseService] Warning: Failed to load BraTS registry: {e}")

    def get_all_cases(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        query: Optional[str] = None,
        grade: Optional[str] = None,
        split: Optional[str] = None,
        include_brats: bool = True,
    ) -> List[CaseSummary]:
        # Custom uploaded cases first
        custom_data = self._read_index()

        all_cases = list(custom_data)
        if include_brats:
            all_cases.extend(self._brats_cases_cache)

        # Filters
        if query:
            q_clean = query.strip().lower()
            all_cases = [c for c in all_cases if q_clean in c["case_id"].lower()]

        if grade:
            g_clean = grade.strip().upper()
            all_cases = [c for c in all_cases if c.get("grade") == g_clean]

        if split:
            s_clean = split.strip().lower()
            all_cases = [c for c in all_cases if c.get("split") == s_clean]

        # Pagination
        if offset > 0:
            all_cases = all_cases[offset:]
        if limit is not None and limit > 0:
            all_cases = all_cases[:limit]

        return [CaseSummary(**c) for c in all_cases]

    def get_case(self, case_id: str) -> Optional[CaseSummary]:
        # Alias CASE-2026-001 -> BraTS20_Training_001
        lookup_id = case_id
        if case_id == "CASE-2026-001" and "BraTS20_Training_001" in self._brats_lookup:
            base = dict(self._brats_lookup["BraTS20_Training_001"])
            base["case_id"] = "CASE-2026-001"
            return CaseSummary(**base)
        if case_id == "CASE-2026-002" and "BraTS20_Training_002" in self._brats_lookup:
            base = dict(self._brats_lookup["BraTS20_Training_002"])
            base["case_id"] = "CASE-2026-002"
            return CaseSummary(**base)

        # Check custom cases
        for c in self._read_index():
            if c["case_id"] == case_id:
                return CaseSummary(**c)

        # Check real BraTS registry
        if case_id in self._brats_lookup:
            return CaseSummary(**self._brats_lookup[case_id])

        return None

    def create_case(self, case_id: Optional[str] = None) -> CaseSummary:
        cases = self._read_index()
        now = datetime.now(timezone.utc).isoformat()
        if not case_id:
            case_id = f"CASE-{datetime.now().year}-{len(cases) + 1:04d}"

        new_case = {
            "case_id": case_id,
            "created_at": now,
            "modalities": [],
            "status": "pending",
            "model_version": None,
            "is_demo": False,
            "grade": None,
            "age": None,
            "survival_days": None,
            "extent_of_resection": None,
            "split": None,
            "is_real_brats": False,
            "slice_count": 155,
        }
        cases.insert(0, new_case)
        self._write_index(cases)

        case_folder = CASES_DIR / case_id
        case_folder.mkdir(parents=True, exist_ok=True)

        return CaseSummary(**new_case)

    def delete_case(self, case_id: str) -> bool:
        cases = self._read_index()
        filtered = [c for c in cases if c["case_id"] != case_id]
        if len(filtered) == len(cases):
            return False
        self._write_index(filtered)

        case_folder = CASES_DIR / case_id
        if case_folder.exists():
            import shutil
            shutil.rmtree(case_folder, ignore_errors=True)
        return True

    def add_modality(self, case_id: str, modality: str) -> Optional[CaseSummary]:
        cases = self._read_index()
        for c in cases:
            if c["case_id"] == case_id:
                if modality not in c["modalities"]:
                    c["modalities"].append(modality)
                if len(c["modalities"]) >= 4:
                    c["status"] = "ready"
                self._write_index(cases)
                return CaseSummary(**c)
        return None

case_service = CaseService()
