"""
NeuroVision AI: Module B - Multi-Sequence Intensity Profiler
Calculates non-zero intensity distributions, percentiles, dynamic ranges, and signal-to-noise
proxies across T1, T1ce, T2, and FLAIR modalities to configure Module C normalization.
"""
from typing import Dict, Any, List, Optional
import os
import numpy as np
import nibabel as nib
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("IntensityProfiler")


class IntensityProfiler:
    """
    Computes intensity statistics for non-zero (brain tissue) voxels across multi-sequence MRI.
    Informs Module C's non-zero z-score and percentile clipping thresholds.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        prof_cfg = self.config.get("intensity_profiling", {})
        self.modalities = prof_cfg.get("modalities", ["t1", "t1ce", "t2", "flair"])
        self.percentiles = prof_cfg.get("percentiles", [1, 5, 25, 50, 75, 95, 99])

    def profile_volume(self, file_path: str) -> Dict[str, float]:
        """
        Extracts non-zero voxel intensities from a single NIfTI volume and computes summary metrics.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"MRI volume file not found: {file_path}")

        img = nib.load(file_path)
        data = img.get_fdata(dtype=np.float32)

        # Extract strictly non-zero tissue voxels
        brain_voxels = data[data > 0]

        if len(brain_voxels) == 0:
            return {
                "num_nonzero_voxels": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "p1": 0.0,
                "p5": 0.0,
                "p25": 0.0,
                "p75": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "iqr": 0.0,
            }

        pct_values = np.percentile(brain_voxels, self.percentiles)
        pct_dict = {f"p{p}": round(float(val), 2) for p, val in zip(self.percentiles, pct_values)}

        mean_val = float(np.mean(brain_voxels))
        std_val = float(np.std(brain_voxels))
        median_val = float(pct_dict.get("p50", np.median(brain_voxels)))
        p25 = pct_dict.get("p25", float(np.percentile(brain_voxels, 25)))
        p75 = pct_dict.get("p75", float(np.percentile(brain_voxels, 75)))
        iqr_val = round(p75 - p25, 2)

        stats = {
            "num_nonzero_voxels": int(len(brain_voxels)),
            "min": round(float(np.min(brain_voxels)), 2),
            "max": round(float(np.max(brain_voxels)), 2),
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "median": round(median_val, 2),
            "iqr": iqr_val,
        }
        stats.update(pct_dict)
        return stats

    def profile_subject(self, subject_row: pd.Series) -> Dict[str, Any]:
        """
        Profiles all 4 MRI sequences for a single patient record.
        """
        result = {"subject_id": subject_row.get("subject_id", "unknown")}
        for mod in self.modalities:
            path_col = f"{mod}_path"
            file_path = subject_row.get(path_col)
            if file_path and os.path.isfile(file_path):
                mod_stats = self.profile_volume(file_path)
                for k, v in mod_stats.items():
                    result[f"{mod}_{k}"] = v
            else:
                logger.warning(f"Modality {mod} path missing or invalid for {result['subject_id']}")
        return result

    def aggregate_cohort_profiles(self, profile_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Aggregates intensity statistics across an entire patient cohort to inform Module C.
        """
        df_prof = pd.DataFrame(profile_records)
        aggregated: Dict[str, Dict[str, float]] = {}

        for mod in self.modalities:
            mean_col = f"{mod}_mean"
            std_col = f"{mod}_std"
            p1_col = f"{mod}_p1"
            p99_col = f"{mod}_p99"
            max_col = f"{mod}_max"

            if mean_col in df_prof.columns:
                aggregated[mod] = {
                    "cohort_mean": round(float(df_prof[mean_col].mean()), 2),
                    "cohort_std": round(float(df_prof[std_col].mean()), 2),
                    "suggested_clip_p1": round(float(df_prof[p1_col].mean()), 2),
                    "suggested_clip_p99": round(float(df_prof[p99_col].mean()), 2),
                    "cohort_max": round(float(df_prof[max_col].max()), 2),
                }

        return aggregated
