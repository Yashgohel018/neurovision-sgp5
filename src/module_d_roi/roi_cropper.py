"""
NeuroVision AI: Module D - 3D Tumor ROI Cropper & Standardizer
Extracts focused 3D Region of Interest (ROI) tensors and standardizes them
into fixed-size uniform dimensions (e.g. 96 x 96 x 96) for deep learning batching.
"""
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
from scipy.ndimage import zoom

from src.module_d_roi.bbox_extractor import TumorBoundingBox


class TumorROICropper:
    """
    Crops multimodal MRI volumes and masks to tumor-centered ROI bounding boxes
    and standardizes spatial dimensions using aspect-preserving or padding strategies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ROI cropper with configuration options.
        
        Args:
            config: Optional configuration dictionary.
        """
        roi_cfg = config.get("roi_extraction", {}) if config else {}
        self.target_shape = tuple(roi_cfg.get("target_roi_shape", (96, 96, 96)))
        self.standardize_method = roi_cfg.get("standardize_method", "pad_or_crop")
        self.pad_value = float(roi_cfg.get("pad_value", 0.0))
        self.mask_pad_value = int(roi_cfg.get("mask_pad_value", 0))

    def crop_to_bbox(
        self,
        volume: np.ndarray,
        bbox: TumorBoundingBox
    ) -> np.ndarray:
        """
        Crop a 3D or 4D array to the margin-expanded coordinates of the bounding box.
        
        Args:
            volume: 3D array (D, H, W) or 4D array (C, D, H, W).
            bbox: TumorBoundingBox instance.
            
        Returns:
            Cropped numpy array.
        """
        slice_z, slice_y, slice_x = bbox.to_slice_tuple()
        if volume.ndim == 4:
            return volume[:, slice_z, slice_y, slice_x]
        elif volume.ndim == 3:
            return volume[slice_z, slice_y, slice_x]
        else:
            raise ValueError(f"Expected 3D or 4D volume, got shape {volume.shape}")

    def pad_or_crop_3d(
        self,
        volume: np.ndarray,
        target_shape: Optional[Tuple[int, int, int]] = None,
        pad_value: Optional[float] = None
    ) -> np.ndarray:
        """
        Symmetrically pad or center-crop a volume to match exact target dimensions.
        
        Args:
            volume: 3D array (D, H, W) or 4D array (C, D, H, W).
            target_shape: Desired 3D spatial shape (D_tgt, H_tgt, W_tgt).
            pad_value: Constant value used for padding.
            
        Returns:
            Numpy array with exact spatial dimensions target_shape.
        """
        tgt = target_shape if target_shape is not None else self.target_shape
        pval = pad_value if pad_value is not None else self.pad_value

        is_4d = (volume.ndim == 4)
        spatial_shape = volume.shape[1:] if is_4d else volume.shape
        cur_d, cur_h, cur_w = spatial_shape
        tgt_d, tgt_h, tgt_w = tgt

        # 1. Handle Depth (Z)
        if cur_d > tgt_d:
            start = (cur_d - tgt_d) // 2
            volume = volume[:, start:start + tgt_d] if is_4d else volume[start:start + tgt_d]
        elif cur_d < tgt_d:
            pad_before = (tgt_d - cur_d) // 2
            pad_after = tgt_d - cur_d - pad_before
            pad_width = [(0, 0), (pad_before, pad_after), (0, 0), (0, 0)] if is_4d else [(pad_before, pad_after), (0, 0), (0, 0)]
            volume = np.pad(volume, pad_width, mode="constant", constant_values=pval)

        # 2. Handle Height (Y)
        cur_h = volume.shape[2] if is_4d else volume.shape[1]
        if cur_h > tgt_h:
            start = (cur_h - tgt_h) // 2
            volume = volume[:, :, start:start + tgt_h] if is_4d else volume[:, start:start + tgt_h]
        elif cur_h < tgt_h:
            pad_before = (tgt_h - cur_h) // 2
            pad_after = tgt_h - cur_h - pad_before
            pad_width = [(0, 0), (0, 0), (pad_before, pad_after), (0, 0)] if is_4d else [(0, 0), (pad_before, pad_after), (0, 0)]
            volume = np.pad(volume, pad_width, mode="constant", constant_values=pval)

        # 3. Handle Width (X)
        cur_w = volume.shape[3] if is_4d else volume.shape[2]
        if cur_w > tgt_w:
            start = (cur_w - tgt_w) // 2
            volume = volume[:, :, :, start:start + tgt_w] if is_4d else volume[:, :, start:start + tgt_w]
        elif cur_w < tgt_w:
            pad_before = (tgt_w - cur_w) // 2
            pad_after = tgt_w - cur_w - pad_before
            pad_width = [(0, 0), (0, 0), (0, 0), (pad_before, pad_after)] if is_4d else [(0, 0), (0, 0), (pad_before, pad_after)]
            volume = np.pad(volume, pad_width, mode="constant", constant_values=pval)

        return volume

    def aspect_preserving_fit(
        self,
        volume: np.ndarray,
        target_shape: Optional[Tuple[int, int, int]] = None,
        is_mask: bool = False
    ) -> np.ndarray:
        """
        Isotropically scale the volume so the longest axis fits inside target_shape,
        then symmetrically pad the remaining axes to target_shape.
        Preserves 1:1:1 geometric aspect ratio without stretching.
        
        Args:
            volume: 3D array (D, H, W) or 4D array (C, D, H, W).
            target_shape: Desired 3D spatial shape.
            is_mask: If True, uses nearest-neighbor interpolation and integer padding.
            
        Returns:
            Standardized volume.
        """
        tgt = target_shape if target_shape is not None else self.target_shape
        is_4d = (volume.ndim == 4)
        spatial_shape = volume.shape[1:] if is_4d else volume.shape

        scale_factor = min(tgt[0] / spatial_shape[0], tgt[1] / spatial_shape[1], tgt[2] / spatial_shape[2])
        # If already fits or scale_factor >= 1.0, we can directly pad
        if scale_factor >= 1.0:
            pval = self.mask_pad_value if is_mask else self.pad_value
            return self.pad_or_crop_3d(volume, target_shape=tgt, pad_value=pval)

        # Rescale isotropically
        order = 0 if is_mask else 1
        if is_4d:
            scaled_channels = [
                zoom(volume[c], zoom=scale_factor, order=order)
                for c in range(volume.shape[0])
            ]
            scaled_vol = np.stack(scaled_channels, axis=0)
        else:
            scaled_vol = zoom(volume, zoom=scale_factor, order=order)

        pval = self.mask_pad_value if is_mask else self.pad_value
        return self.pad_or_crop_3d(scaled_vol, target_shape=tgt, pad_value=pval)

    def resample_3d(
        self,
        volume: np.ndarray,
        target_shape: Optional[Tuple[int, int, int]] = None,
        is_mask: bool = False
    ) -> np.ndarray:
        """
        Resample volume directly to target dimensions using 3D spline/nearest interpolation.
        
        Args:
            volume: 3D array (D, H, W) or 4D array (C, D, H, W).
            target_shape: Desired 3D spatial shape.
            is_mask: If True, uses nearest-neighbor (order 0) interpolation.
            
        Returns:
            Resampled numpy array.
        """
        tgt = target_shape if target_shape is not None else self.target_shape
        is_4d = (volume.ndim == 4)
        spatial_shape = volume.shape[1:] if is_4d else volume.shape

        zoom_factors = [t / s for t, s in zip(tgt, spatial_shape)]
        order = 0 if is_mask else 1

        if is_4d:
            resampled_channels = [
                zoom(volume[c], zoom=zoom_factors, order=order)
                for c in range(volume.shape[0])
            ]
            resampled = np.stack(resampled_channels, axis=0)
        else:
            resampled = zoom(volume, zoom=zoom_factors, order=order)

        if is_mask:
            resampled = np.round(resampled).astype(volume.dtype)
        return resampled

    def standardize(
        self,
        volume: np.ndarray,
        target_shape: Optional[Tuple[int, int, int]] = None,
        method: Optional[str] = None,
        is_mask: bool = False
    ) -> np.ndarray:
        """
        Standardize cropped volume using the configured or specified method.
        
        Args:
            volume: 3D or 4D cropped array.
            target_shape: Desired 3D spatial shape.
            method: 'pad_or_crop', 'aspect_preserving_fit', or 'resample'.
            is_mask: If True, treats array as categorical mask.
            
        Returns:
            Standardized numpy array.
        """
        m = method if method is not None else self.standardize_method
        tgt = target_shape if target_shape is not None else self.target_shape

        if m == "pad_or_crop":
            pval = self.mask_pad_value if is_mask else self.pad_value
            return self.pad_or_crop_3d(volume, target_shape=tgt, pad_value=pval)
        elif m == "aspect_preserving_fit":
            return self.aspect_preserving_fit(volume, target_shape=tgt, is_mask=is_mask)
        elif m == "resample":
            return self.resample_3d(volume, target_shape=tgt, is_mask=is_mask)
        else:
            raise ValueError(f"Unknown standardize method: {m}")

    def extract_roi(
        self,
        image_4d: np.ndarray,
        bbox: TumorBoundingBox,
        mask_3d: Optional[np.ndarray] = None,
        target_shape: Optional[Tuple[int, int, int]] = None,
        method: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete extraction pipeline for a single subject:
        1. Crop multimodal MRI tensor and mask to bounding box.
        2. Standardize both image and mask to fixed target shape.
        
        Args:
            image_4d: Multimodal MRI tensor (4, D, H, W).
            bbox: TumorBoundingBox instance.
            mask_3d: Optional 3D ground-truth or predicted mask (D, H, W).
            target_shape: Optional target 3D shape (e.g. (96, 96, 96)).
            method: Standardization method.
            
        Returns:
            Dictionary containing 'roi_image', 'roi_mask', 'raw_crop_shape', 'target_shape'.
        """
        tgt = target_shape if target_shape is not None else self.target_shape
        m = method if method is not None else self.standardize_method

        # 1. Crop
        raw_crop_image = self.crop_to_bbox(image_4d, bbox)
        raw_crop_mask = self.crop_to_bbox(mask_3d, bbox) if mask_3d is not None else None

        raw_crop_shape = raw_crop_image.shape[1:]

        # 2. Standardize
        std_image = self.standardize(raw_crop_image, target_shape=tgt, method=m, is_mask=False).astype(np.float32)
        std_mask = None
        if raw_crop_mask is not None:
            std_mask = self.standardize(raw_crop_mask, target_shape=tgt, method=m, is_mask=True).astype(np.uint8)

        return {
            "roi_image": std_image,
            "roi_mask": std_mask,
            "raw_crop_shape": raw_crop_shape,
            "target_shape": tgt,
            "bbox": bbox
        }
