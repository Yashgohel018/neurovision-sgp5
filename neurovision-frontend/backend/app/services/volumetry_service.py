from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import nibabel as nib

from app.config import CASES_DIR, BRATS_DATA_DIR, TUMOR_STATS_CSV
from app.schemas.inference import Volumes

class VolumetryService:
    def __init__(self):
        self._stats_cache: Dict[str, Dict[str, float]] = {}
        self._load_tumor_statistics()

    def _load_tumor_statistics(self):
        if not TUMOR_STATS_CSV.exists():
            return
        try:
            df = pd.read_csv(TUMOR_STATS_CSV)
            for _, row in df.iterrows():
                sid = str(row["subject_id"])
                self._stats_cache[sid] = {
                    "WT": round(float(row.get("wt_volume_cm3", 0.0)), 2),
                    "TC": round(float(row.get("tc_volume_cm3", 0.0)), 2),
                    "ET": round(float(row.get("et_volume_cm3", 0.0)), 2),
                    "ED": round(float(row.get("ed_volume_cm3", 0.0)), 2),
                    "NCR": round(float(row.get("ncr_volume_cm3", 0.0)), 2),
                }
        except Exception as e:
            print(f"[VolumetryService] Warning: Failed to load tumor statistics: {e}")

    def calculate_volumes(self, case_id: str) -> Optional[Volumes]:
        effective_id = case_id
        if case_id == "CASE-2026-001":
            effective_id = "BraTS20_Training_001"
        elif case_id == "CASE-2026-002":
            effective_id = "BraTS20_Training_002"

        # 1. Check precomputed real statistics from learnNeuro Module B
        if effective_id in self._stats_cache:
            v = self._stats_cache[effective_id]
            return Volumes(
                WT_volume_cm3=v["WT"],
                TC_volume_cm3=v["TC"],
                ET_volume_cm3=v["ET"],
            )

        # 2. Check for real NIfTI segmentation file
        seg_file = None
        # Check custom uploaded case folder
        case_dir = CASES_DIR / case_id
        if case_dir.exists():
            for name in ["Seg.nii.gz", "Seg.nii", "seg.nii.gz", "seg.nii"]:
                p = case_dir / name
                if p.exists():
                    seg_file = p
                    break

        # Check real BraTS directory
        if not seg_file:
            brats_dir = BRATS_DATA_DIR / effective_id
            if brats_dir.exists():
                for ext in [".nii", ".nii.gz"]:
                    candidate = brats_dir / f"{effective_id}_seg{ext}"
                    if candidate.exists():
                        seg_file = candidate
                        break

        if seg_file and seg_file.exists():
            try:
                img = nib.load(str(seg_file))
                data = img.get_fdata().astype(np.int32)
                header = img.header
                zooms = header.get_zooms()[:3]  # (dx, dy, dz) in mm
                voxel_volume_mm3 = float(zooms[0] * zooms[1] * zooms[2])
                voxel_volume_cm3 = voxel_volume_mm3 / 1000.0

                # BraTS label convention:
                # 1 = Necrotic / non-enhancing tumor core (NETC)
                # 2 = Peritumoral edema (ED)
                # 4 = Enhancing tumor (ET)
                # WT = 1 + 2 + 4, TC = 1 + 4, ET = 4
                count_1 = int(np.sum(data == 1))
                count_2 = int(np.sum(data == 2))
                count_4 = int(np.sum(data == 4))

                et_count = count_4
                tc_count = count_1 + count_4
                wt_count = count_1 + count_2 + count_4

                return Volumes(
                    WT_volume_cm3=round(wt_count * voxel_volume_cm3, 2),
                    TC_volume_cm3=round(tc_count * voxel_volume_cm3, 2),
                    ET_volume_cm3=round(et_count * voxel_volume_cm3, 2),
                )
            except Exception:
                pass

        return None

volumetry_service = VolumetryService()
