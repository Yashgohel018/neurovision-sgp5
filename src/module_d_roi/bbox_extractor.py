"""
NeuroVision AI: Module D - 3D Tumor Bounding Box Extractor
Computes tight 3D bounding coordinates, morphological centroids,
and margin-expanded peritumoral context envelopes from segmentation masks.
"""
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional, Union, Dict, Any
import numpy as np


@dataclass
class TumorBoundingBox:
    """
    Structured representation of a 3D tumor bounding box with contextual margin.
    """
    # Raw bounding box bounds (0-indexed, inclusive)
    z_min: int
    z_max: int
    y_min: int
    y_max: int
    x_min: int
    x_max: int

    # Bounding box spans
    span_z: int
    span_y: int
    span_x: int
    bbox_volume_voxels: int

    # Morphological centroid of the tumor
    centroid_z: float
    centroid_y: float
    centroid_x: float

    # Margin-expanded cropping bounds (inclusive)
    margin_applied: int
    crop_z_min: int
    crop_z_max: int
    crop_y_min: int
    crop_y_max: int
    crop_x_min: int
    crop_x_max: int

    # Margin-expanded spans
    crop_span_z: int
    crop_span_y: int
    crop_span_x: int
    crop_volume_voxels: int

    # Tissue voxel count
    tumor_voxels: int
    is_empty: bool = False

    def to_slice_tuple(self) -> Tuple[slice, slice, slice]:
        """
        Convert margin-expanded bounds into Python slice objects for array indexing.
        
        Returns:
            Tuple of (slice(crop_z_min, crop_z_max + 1),
                      slice(crop_y_min, crop_y_max + 1),
                      slice(crop_x_min, crop_x_max + 1))
        """
        return (
            slice(self.crop_z_min, self.crop_z_max + 1),
            slice(self.crop_y_min, self.crop_y_max + 1),
            slice(self.crop_x_min, self.crop_x_max + 1)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert bounding box to dictionary representation."""
        return asdict(self)


class TumorBoundingBoxExtractor:
    """
    Extracts 3D bounding boxes enclosing tumor sub-regions from segmentation masks
    and expands them with clinical contextual margins to capture infiltrative edema.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the bounding box extractor.
        
        Args:
            config: Optional configuration dictionary.
        """
        roi_cfg = config.get("roi_extraction", {}) if config else {}
        seg_cfg = config.get("segmentation", {}) if config else {}

        self.default_margin = int(roi_cfg.get("context_margin_voxels", 10))
        self.min_margin = int(roi_cfg.get("min_margin_voxels", 6))
        self.min_tumor_voxels = int(roi_cfg.get("min_tumor_voxels", 10))
        self.target_labels = list(seg_cfg.get("target_labels", [1, 2, 4]))

    def extract_bounding_box(
        self,
        mask: np.ndarray,
        margin: Optional[int] = None,
        target_labels: Optional[List[int]] = None,
        spatial_shape: Optional[Tuple[int, int, int]] = None
    ) -> TumorBoundingBox:
        """
        Compute the 3D bounding box for whole tumor or specified sub-region labels.
        
        Args:
            mask: 3D integer numpy array (D, H, W).
            margin: Context margin in voxels. If None, uses default_margin.
            target_labels: Sub-region labels to include. If None, uses Whole Tumor [1, 2, 4].
            spatial_shape: Reference spatial dimensions (D, H, W) for boundary clamping.
            
        Returns:
            TumorBoundingBox object containing raw and margin-expanded bounds.
        """
        if mask.ndim != 3:
            raise ValueError(f"Expected 3D mask with shape (D, H, W), got ndim={mask.ndim}")

        d, h, w = spatial_shape if spatial_shape is not None else mask.shape
        margin_val = int(margin if margin is not None else self.default_margin)
        labels = target_labels if target_labels is not None else self.target_labels

        # Create binary mask for target labels
        binary_mask = np.isin(mask, labels)
        tumor_voxel_count = int(np.sum(binary_mask))

        # Handle empty mask / no tumor detected
        if tumor_voxel_count < self.min_tumor_voxels:
            # Fallback: center-crop around central brain coordinates
            center_z, center_y, center_x = d // 2, h // 2, w // 2
            half_span = 32
            cz_min = max(0, center_z - half_span)
            cz_max = min(d - 1, center_z + half_span)
            cy_min = max(0, center_y - half_span)
            cy_max = min(h - 1, center_y + half_span)
            cx_min = max(0, center_x - half_span)
            cx_max = min(w - 1, center_x + half_span)

            return TumorBoundingBox(
                z_min=cz_min, z_max=cz_max,
                y_min=cy_min, y_max=cy_max,
                x_min=cx_min, x_max=cx_max,
                span_z=cz_max - cz_min + 1,
                span_y=cy_max - cy_min + 1,
                span_x=cx_max - cx_min + 1,
                bbox_volume_voxels=(cz_max - cz_min + 1) * (cy_max - cy_min + 1) * (cx_max - cx_min + 1),
                centroid_z=float(center_z),
                centroid_y=float(center_y),
                centroid_x=float(center_x),
                margin_applied=0,
                crop_z_min=cz_min, crop_z_max=cz_max,
                crop_y_min=cy_min, crop_y_max=cy_max,
                crop_x_min=cx_min, crop_x_max=cx_max,
                crop_span_z=cz_max - cz_min + 1,
                crop_span_y=cy_max - cy_min + 1,
                crop_span_x=cx_max - cx_min + 1,
                crop_volume_voxels=(cz_max - cz_min + 1) * (cy_max - cy_min + 1) * (cx_max - cx_min + 1),
                tumor_voxels=tumor_voxel_count,
                is_empty=True
            )

        # 1. Compute exact bounding indices along each axis
        z_indices = np.where(np.any(binary_mask, axis=(1, 2)))[0]
        y_indices = np.where(np.any(binary_mask, axis=(0, 2)))[0]
        x_indices = np.where(np.any(binary_mask, axis=(0, 1)))[0]

        z_min, z_max = int(z_indices[0]), int(z_indices[-1])
        y_min, y_max = int(y_indices[0]), int(y_indices[-1])
        x_min, x_max = int(x_indices[0]), int(x_indices[-1])

        span_z = z_max - z_min + 1
        span_y = y_max - y_min + 1
        span_x = x_max - x_min + 1
        bbox_volume = span_z * span_y * span_x

        # 2. Compute 3D morphological centroid (center of mass)
        coords_z, coords_y, coords_x = np.where(binary_mask)
        centroid_z = float(np.mean(coords_z))
        centroid_y = float(np.mean(coords_y))
        centroid_x = float(np.mean(coords_x))

        # 3. Apply contextual margin with boundary clamping
        crop_z_min = max(0, z_min - margin_val)
        crop_z_max = min(d - 1, z_max + margin_val)

        crop_y_min = max(0, y_min - margin_val)
        crop_y_max = min(h - 1, y_max + margin_val)

        crop_x_min = max(0, x_min - margin_val)
        crop_x_max = min(w - 1, x_max + margin_val)

        crop_span_z = crop_z_max - crop_z_min + 1
        crop_span_y = crop_y_max - crop_y_min + 1
        crop_span_x = crop_x_max - crop_x_min + 1
        crop_volume = crop_span_z * crop_span_y * crop_span_x

        return TumorBoundingBox(
            z_min=z_min, z_max=z_max,
            y_min=y_min, y_max=y_max,
            x_min=x_min, x_max=x_max,
            span_z=span_z, span_y=span_y, span_x=span_x,
            bbox_volume_voxels=bbox_volume,
            centroid_z=round(centroid_z, 2),
            centroid_y=round(centroid_y, 2),
            centroid_x=round(centroid_x, 2),
            margin_applied=margin_val,
            crop_z_min=crop_z_min, crop_z_max=crop_z_max,
            crop_y_min=crop_y_min, crop_y_max=crop_y_max,
            crop_x_min=crop_x_min, crop_x_max=crop_x_max,
            crop_span_z=crop_span_z,
            crop_span_y=crop_span_y,
            crop_span_x=crop_span_x,
            crop_volume_voxels=crop_volume,
            tumor_voxels=tumor_voxel_count,
            is_empty=False
        )
