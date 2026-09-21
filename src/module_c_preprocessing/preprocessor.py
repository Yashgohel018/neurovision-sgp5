"""
NeuroVision AI: Module C - End-to-End Multimodal MRI Preprocessor
Coordinates loading, non-zero intensity normalization, foreground cropping,
and standardized 4-channel 3D tensor generation for a single subject.
"""
import os
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import nibabel as nib

from src.module_c_preprocessing.normalizer import IntensityNormalizer
from src.module_c_preprocessing.cropper import SpatialCropper


class MRIPreprocessor:
    """
    Transforms raw BraTS multi-sequence NIfTI volumes into standardized 3D tensors.
    
    Standardized Tensor Spec:
      - Shape: (4, 128, 128, 128) -> [C, D, H, W]
      - Channel Order: [T1, T1ce, T2, FLAIR]
      - Voxel Data Type: float32
      - Normalization: Non-zero z-score with [P1, P99] clamping
      - Background: Zero-valued air voxels outside brain parenchyma
      - Associated Mask: (128, 128, 128) uint8 with labels {0: BG, 1: NCR, 2: ED, 4: ET}
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor with normalizer, cropper, and config parameters.
        
        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.normalizer = IntensityNormalizer(self.config)
        self.cropper = SpatialCropper(self.config)

        mm_cfg = self.config.get("multimodal", {})
        self.modalities: List[str] = mm_cfg.get("modalities", ["t1", "t1ce", "t2", "flair"])
        self.include_seg: bool = mm_cfg.get("include_seg", True)
        self.target_shape: Tuple[int, int, int] = tuple(
            self.config.get("spatial", {}).get("target_shape", (128, 128, 128))
        )
        self.crop_strategy: str = self.config.get("spatial", {}).get("crop_strategy", "foreground_bbox")

    def process_arrays(
        self,
        modality_arrays: Dict[str, np.ndarray],
        seg_array: Optional[np.ndarray] = None,
        subject_id: str = "synthetic"
    ) -> Dict[str, Any]:
        """
        Preprocess in-memory 3D numpy arrays directly (ideal for unit tests).
        
        Args:
            modality_arrays: Dictionary mapping modality names to 3D numpy arrays.
            seg_array: Optional 3D segmentation mask.
            subject_id: Unique subject identifier.
            
        Returns:
            Dictionary containing processed 4D image tensor, mask, and metadata.
        """
        # 1. Stack raw modalities into 4D tensor (C, D, H, W)
        channels = [modality_arrays[m] for m in self.modalities if m in modality_arrays]
        if len(channels) != len(self.modalities):
            raise ValueError(f"Missing required modalities. Expected {self.modalities}, got {list(modality_arrays.keys())}")

        raw_4d = np.stack(channels, axis=0)  # Shape: (4, D, H, W)

        # 2. Compute non-zero foreground bounding box across modalities
        bbox = self.cropper.find_foreground_bbox(raw_4d, threshold=0.0)

        # 3. Crop empty background air
        if self.crop_strategy == "foreground_bbox":
            cropped_4d = self.cropper.crop_to_bbox(raw_4d, bbox, margin=4)
            cropped_mask = self.cropper.crop_to_bbox(seg_array, bbox, margin=4) if seg_array is not None else None
        else:
            cropped_4d = raw_4d
            cropped_mask = seg_array

        # 4. Standardize spatial shape to target_shape (128, 128, 128)
        std_4d = self.cropper.crop_or_pad_3d(cropped_4d, target_shape=self.target_shape, pad_value=0.0)
        std_mask = self.cropper.crop_or_pad_3d(cropped_mask, target_shape=self.target_shape, pad_value=0) if cropped_mask is not None else None

        # 5. Apply robust non-zero intensity normalization per modality
        norm_4d = self.normalizer.normalize(std_4d)

        return {
            "subject_id": subject_id,
            "image": norm_4d.astype(np.float32),  # (4, 128, 128, 128)
            "mask": std_mask.astype(np.uint8) if std_mask is not None else None,
            "bbox": bbox,
            "modalities": self.modalities,
            "target_shape": self.target_shape
        }

    def process_subject_row(self, row: Any) -> Dict[str, Any]:
        """
        Load and preprocess a single subject from a metadata row or series.
        
        Args:
            row: Pandas Series or dict containing subject metadata and paths.
            
        Returns:
            Dictionary containing processed 4D image tensor, mask, and clinical labels.
        """
        subject_id = row["subject_id"]
        modality_arrays: Dict[str, np.ndarray] = {}

        for mod in self.modalities:
            path_col = f"{mod}_path"
            file_path = row[path_col]
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Missing MRI file for subject {subject_id}: {file_path}")

            nii = nib.load(file_path)
            # Reorient to canonical RAS/LPS if needed, convert to float32
            data = nii.get_fdata(dtype=np.float32)
            modality_arrays[mod] = data

        seg_array = None
        if self.include_seg and "seg_path" in row and pd_not_null(row["seg_path"]):
            seg_path = row["seg_path"]
            if os.path.isfile(seg_path):
                seg_nii = nib.load(seg_path)
                seg_array = np.round(seg_nii.get_fdata()).astype(np.uint8)

        processed = self.process_arrays(
            modality_arrays=modality_arrays,
            seg_array=seg_array,
            subject_id=subject_id
        )

        # Attach clinical labels if available
        if "grade" in row:
            processed["grade"] = row["grade"]
        if "label" in row:
            processed["label"] = int(row["label"])

        return processed


def pd_not_null(val: Any) -> bool:
    """Helper to check non-null without pandas overhead."""
    if val is None:
        return False
    if isinstance(val, str) and val.lower() in ("nan", "none", ""):
        return False
    return True
