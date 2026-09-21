"""
NeuroVision AI: Module B - Volumetric Quantification Engine
Computes voxel counts, physical volumes (cm³), physiological ratios, 3D centroids,
and bounding boxes for tumor sub-regions (WT, TC, ET, ED, NCR) across BraTS 2020 cohorts.
"""
from typing import Dict, Any, Optional, Tuple, List
import os
import numpy as np
import nibabel as nib
from src.utils.logger import setup_logger

logger = setup_logger("TumorVolumetrics")


class TumorVolumetricAnalyzer:
    """
    High-performance vectorized tumor sub-region volumetric and spatial analyzer.
    Extracts physical sub-region burdens and 3D spatial properties for Agent 1 and Agent 2.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        outlier_cfg = self.config.get("outliers", {})
        self.micro_threshold = outlier_cfg.get("micro_tumor_threshold_cm3", 5.0)
        self.massive_threshold = outlier_cfg.get("massive_tumor_threshold_cm3", 150.0)
        self.zero_et_threshold = outlier_cfg.get("zero_et_threshold_cm3", 0.0)

    @staticmethod
    def compute_voxel_volume_cm3(spacing: Tuple[float, float, float]) -> float:
        """
        Calculates physical volume of a single voxel in cm³.
        Formula: (spacing_x * spacing_y * spacing_z) / 1000.0 (since 1 cm³ = 1000 mm³)
        """
        sx, sy, sz = spacing
        return (sx * sy * sz) / 1000.0

    def analyze_mask_array(
        self,
        mask: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Dict[str, Any]:
        """
        Quantifies tumor sub-regions, centroids, and bounding boxes directly from a 3D numpy array.

        BraTS 2020 Ground Truth Label Definitions:
          0: Background (healthy brain tissue)
          1: Necrotic and Non-Enhancing Tumor Core (NCR/NET)
          2: Peritumoral Vasogenic Edema (ED)
          4: GD-Enhancing Tumor (ET)

        Composite Sub-regions:
          Whole Tumor (WT): NCR + ED + ET (labels 1, 2, 4)
          Tumor Core (TC): NCR + ET (labels 1, 4)
          Enhancing Tumor (ET): label 4
          Peritumoral Edema (ED): label 2
          Necrotic Core (NCR): label 1
        """
        voxel_vol_cm3 = self.compute_voxel_volume_cm3(spacing)

        # Fast bincount on flattened int array
        max_label = int(np.max(mask)) if mask.size > 0 else 0
        minlength = max(5, max_label + 1)
        bincounts = np.bincount(mask.ravel().astype(np.int64), minlength=minlength)

        ncr_voxels = int(bincounts[1]) if len(bincounts) > 1 else 0
        ed_voxels = int(bincounts[2]) if len(bincounts) > 2 else 0
        et_voxels = int(bincounts[4]) if len(bincounts) > 4 else 0

        # Composites
        tc_voxels = ncr_voxels + et_voxels
        wt_voxels = tc_voxels + ed_voxels

        # Physical volumes in cm³
        ncr_vol_cm3 = round(ncr_voxels * voxel_vol_cm3, 4)
        ed_vol_cm3 = round(ed_voxels * voxel_vol_cm3, 4)
        et_vol_cm3 = round(et_voxels * voxel_vol_cm3, 4)
        tc_vol_cm3 = round(tc_voxels * voxel_vol_cm3, 4)
        wt_vol_cm3 = round(wt_voxels * voxel_vol_cm3, 4)

        # Physiological ratios (Biomarkers of infiltrative malignancy)
        et_to_wt_ratio = round(et_vol_cm3 / wt_vol_cm3, 4) if wt_vol_cm3 > 0 else 0.0
        tc_to_wt_ratio = round(tc_vol_cm3 / wt_vol_cm3, 4) if wt_vol_cm3 > 0 else 0.0
        ed_to_wt_ratio = round(ed_vol_cm3 / wt_vol_cm3, 4) if wt_vol_cm3 > 0 else 0.0

        # Spatial properties: Centroid and 3D Bounding Box
        tumor_mask = np.isin(mask, [1, 2, 4])
        coords = np.argwhere(tumor_mask)

        if len(coords) > 0:
            centroid_x = round(float(np.mean(coords[:, 0])), 2)
            centroid_y = round(float(np.mean(coords[:, 1])), 2)
            centroid_z = round(float(np.mean(coords[:, 2])), 2)

            min_x, min_y, min_z = [int(v) for v in np.min(coords, axis=0)]
            max_x, max_y, max_z = [int(v) for v in np.max(coords, axis=0)]

            span_x = max_x - min_x + 1
            span_y = max_y - min_y + 1
            span_z = max_z - min_z + 1
        else:
            centroid_x = centroid_y = centroid_z = 0.0
            min_x = max_x = span_x = 0
            min_y = max_y = span_y = 0
            min_z = max_z = span_z = 0

        # Outlier and clinical edge case detection
        is_micro = wt_vol_cm3 < self.micro_threshold
        is_massive = wt_vol_cm3 > self.massive_threshold
        is_non_enhancing = et_vol_cm3 <= self.zero_et_threshold

        outlier_flags: List[str] = []
        if is_micro:
            outlier_flags.append("micro_tumor")
        if is_massive:
            outlier_flags.append("massive_tumor")
        if is_non_enhancing:
            outlier_flags.append("non_enhancing")

        return {
            "wt_voxels": wt_voxels,
            "wt_volume_cm3": wt_vol_cm3,
            "tc_voxels": tc_voxels,
            "tc_volume_cm3": tc_vol_cm3,
            "et_voxels": et_voxels,
            "et_volume_cm3": et_vol_cm3,
            "ed_voxels": ed_voxels,
            "ed_volume_cm3": ed_vol_cm3,
            "ncr_voxels": ncr_voxels,
            "ncr_volume_cm3": ncr_vol_cm3,
            "et_to_wt_ratio": et_to_wt_ratio,
            "tc_to_wt_ratio": tc_to_wt_ratio,
            "ed_to_wt_ratio": ed_to_wt_ratio,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "centroid_z": centroid_z,
            "bbox_min_x": min_x,
            "bbox_max_x": max_x,
            "bbox_span_x": span_x,
            "bbox_min_y": min_y,
            "bbox_max_y": max_y,
            "bbox_span_y": span_y,
            "bbox_min_z": min_z,
            "bbox_max_z": max_z,
            "bbox_span_z": span_z,
            "is_micro_tumor": is_micro,
            "is_massive_tumor": is_massive,
            "is_non_enhancing": is_non_enhancing,
            "outlier_flags": ";".join(outlier_flags) if outlier_flags else "normal",
        }

    def analyze_nifti_file(
        self,
        seg_path: str,
        spacing: Optional[Tuple[float, float, float]] = None
    ) -> Dict[str, Any]:
        """
        Loads NIfTI segmentation file from disk and calculates all volumetric metrics.
        """
        if not os.path.isfile(seg_path):
            raise FileNotFoundError(f"Segmentation mask not found at: {seg_path}")

        img = nib.load(seg_path)
        if spacing is None:
            zooms = img.header.get_zooms()[:3]
            spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

        mask = img.get_fdata().astype(np.int16)
        return self.analyze_mask_array(mask, spacing=spacing)
