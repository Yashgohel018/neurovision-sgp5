"""
NeuroVision AI: Module D - 3D ROI Morphometric & Volumetric Metadata Generator
Computes structured quantitative geometric descriptors, sub-region ratios,
and bounding envelope dimensions across the cohort.
"""
from typing import Dict, Any, List, Optional, Tuple
import os
import pandas as pd
import numpy as np

from src.module_d_roi.bbox_extractor import TumorBoundingBox
from src.utils.logger import setup_logger

logger = setup_logger("ROIMetadataGenerator")


class ROIMetadataGenerator:
    """
    Computes and aggregates clinical morphometric features from extracted tumor ROIs.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        roi_cfg = config.get("roi_extraction", {}) if config else {}
        self.target_shape = tuple(roi_cfg.get("target_roi_shape", (96, 96, 96)))
        caching_cfg = config.get("caching", {}) if config else {}
        self.output_csv = caching_cfg.get("metadata_output", "data/roi_metadata.csv")

    @staticmethod
    def compute_subject_record(
        subject_id: str,
        bbox: TumorBoundingBox,
        mask: Optional[np.ndarray],
        image_4d: Optional[np.ndarray] = None,
        grade: Optional[str] = None,
        label: Optional[int] = None,
        split: Optional[str] = None,
        cached_file: Optional[str] = None,
        target_roi_shape: Optional[Tuple[int, int, int]] = (96, 96, 96)
    ) -> Dict[str, Any]:
        """
        Compute comprehensive morphometric and volumetric record for a subject.
        """
        # Sub-region voxel counts
        wt_voxels = int(bbox.tumor_voxels)
        tc_voxels = 0
        et_voxels = 0
        ed_voxels = 0
        ncr_voxels = 0

        if mask is not None and not bbox.is_empty:
            tc_voxels = int(np.sum((mask == 1) | (mask == 4)))
            et_voxels = int(np.sum(mask == 4))
            ed_voxels = int(np.sum(mask == 2))
            ncr_voxels = int(np.sum(mask == 1))

        et_to_wt = round(et_voxels / wt_voxels, 4) if wt_voxels > 0 else 0.0
        tc_to_wt = round(tc_voxels / wt_voxels, 4) if wt_voxels > 0 else 0.0
        ed_to_wt = round(ed_voxels / wt_voxels, 4) if wt_voxels > 0 else 0.0

        # Brain volume estimation
        brain_voxels = 0
        if image_4d is not None:
            if image_4d.ndim == 4:
                brain_voxels = int(np.sum(np.any(image_4d > 0, axis=0)))
            else:
                brain_voxels = int(np.sum(image_4d > 0))

        tumor_to_brain_ratio = round(wt_voxels / brain_voxels, 4) if brain_voxels > 0 else 0.0

        # Aspect ratios of the tumor bounding box
        aspect_zy = round(bbox.span_z / max(1, bbox.span_y), 3)
        aspect_yx = round(bbox.span_y / max(1, bbox.span_x), 3)

        return {
            "subject_id": subject_id,
            "grade": grade,
            "label": label,
            "split": split,
            # Bounding box bounds (raw)
            "bbox_min_z": bbox.z_min,
            "bbox_max_z": bbox.z_max,
            "bbox_span_z": bbox.span_z,
            "bbox_min_y": bbox.y_min,
            "bbox_max_y": bbox.y_max,
            "bbox_span_y": bbox.span_y,
            "bbox_min_x": bbox.x_min,
            "bbox_max_x": bbox.x_max,
            "bbox_span_x": bbox.span_x,
            "bbox_volume_voxels": bbox.bbox_volume_voxels,
            # Centroid
            "centroid_z": bbox.centroid_z,
            "centroid_y": bbox.centroid_y,
            "centroid_x": bbox.centroid_x,
            # Context margin & Cropping bounds
            "margin_applied": bbox.margin_applied,
            "crop_min_z": bbox.crop_z_min,
            "crop_max_z": bbox.crop_z_max,
            "crop_span_z": bbox.crop_span_z,
            "crop_min_y": bbox.crop_y_min,
            "crop_max_y": bbox.crop_y_max,
            "crop_span_y": bbox.crop_span_y,
            "crop_min_x": bbox.crop_x_min,
            "crop_max_x": bbox.crop_x_max,
            "crop_span_x": bbox.crop_span_x,
            "crop_volume_voxels": bbox.crop_volume_voxels,
            # Sub-region Volumetrics
            "wt_voxels": wt_voxels,
            "tc_voxels": tc_voxels,
            "et_voxels": et_voxels,
            "ed_voxels": ed_voxels,
            "ncr_voxels": ncr_voxels,
            "et_to_wt_ratio": et_to_wt,
            "tc_to_wt_ratio": tc_to_wt,
            "ed_to_wt_ratio": ed_to_wt,
            # Intracranial relationship
            "brain_voxels": brain_voxels,
            "tumor_to_brain_ratio": tumor_to_brain_ratio,
            "aspect_ratio_zy": aspect_zy,
            "aspect_ratio_yx": aspect_yx,
            # ROI properties
            "target_roi_shape": str(target_roi_shape),
            "is_empty": bbox.is_empty,
            "cached_file": cached_file
        }

    def export_metadata(
        self,
        records: List[Dict[str, Any]],
        output_csv: Optional[str] = None,
        base_dir: str = "."
    ) -> pd.DataFrame:
        """
        Convert records list to DataFrame and export to CSV.
        
        Args:
            records: List of subject morphometric dictionaries.
            output_csv: Optional destination CSV path.
            base_dir: Project root directory.
            
        Returns:
            Saved pandas DataFrame.
        """
        out_path = os.path.join(base_dir, output_csv if output_csv is not None else self.output_csv)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        df = pd.DataFrame(records)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved ROI morphometric metadata ({len(df)} records) to: {out_path}")
        return df
