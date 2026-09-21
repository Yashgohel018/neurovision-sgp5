"""
NeuroVision AI: Module C - Spatial Cropping, Padding & Resampling
Extracts foreground brain tissue, removes background air, and standardizes
spatial dimensions to GPU-feasible 3D tensor shapes (e.g. 128 x 128 x 128).
"""
from typing import Tuple, Optional, Union, Dict, Any
import numpy as np
from scipy.ndimage import zoom


class SpatialCropper:
    """
    Manages 3D bounding box extraction, center cropping, padding, and resampling.
    
    BraTS MRI scans have native dimensions 240 x 240 x 155 (~8.9 million voxels).
    Over 65% of this volume is empty air space outside the skull.
    This class extracts the tight anatomical bounding box and standardizes
    volumes into fixed tensor dimensions for deep learning batching.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the spatial cropper with configuration parameters.
        
        Args:
            config: Optional configuration dictionary.
        """
        spatial_cfg = config.get("spatial", {}) if config else {}
        self.target_shape = tuple(spatial_cfg.get("target_shape", (128, 128, 128)))
        self.pad_value = float(spatial_cfg.get("pad_value", 0.0))
        self.crop_strategy = spatial_cfg.get("crop_strategy", "foreground_bbox")

    @staticmethod
    def find_foreground_bbox(
        volume: np.ndarray,
        threshold: float = 0.0
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Compute the 3D bounding box enclosing all voxels with intensity > threshold.
        
        Args:
            volume: 3D numpy array (D, H, W) or 4D array (C, D, H, W).
            threshold: Intensity cutoff for foreground detection.
            
        Returns:
            Tuple of ((z_min, z_max), (y_min, y_max), (x_min, x_max)) bounds (inclusive).
        """
        if volume.ndim == 4:
            # Union of foregrounds across all channels
            fg_mask = np.any(volume > threshold, axis=0)
        else:
            fg_mask = volume > threshold

        if not np.any(fg_mask):
            # Fallback to full volume if no foreground detected
            d, h, w = fg_mask.shape
            return ((0, d - 1), (0, h - 1), (0, w - 1))

        z_indices = np.where(np.any(fg_mask, axis=(1, 2)))[0]
        y_indices = np.where(np.any(fg_mask, axis=(0, 2)))[0]
        x_indices = np.where(np.any(fg_mask, axis=(0, 1)))[0]

        return (
            (int(z_indices[0]), int(z_indices[-1])),
            (int(y_indices[0]), int(y_indices[-1])),
            (int(x_indices[0]), int(x_indices[-1]))
        )

    @staticmethod
    def crop_to_bbox(
        volume: np.ndarray,
        bbox: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        margin: int = 0
    ) -> np.ndarray:
        """
        Crop a 3D volume or 4D tensor to the specified bounding box with margin.
        
        Args:
            volume: 3D array (D, H, W) or 4D array (C, D, H, W).
            bbox: Tuple of ((z_min, z_max), (y_min, y_max), (x_min, x_max)).
            margin: Additional voxel padding around the bounding box.
            
        Returns:
            Cropped numpy array.
        """
        is_4d = (volume.ndim == 4)
        spatial_shape = volume.shape[1:] if is_4d else volume.shape
        d, h, w = spatial_shape

        (z_min, z_max), (y_min, y_max), (x_min, x_max) = bbox

        z_start = max(0, z_min - margin)
        z_end = min(d, z_max + margin + 1)
        y_start = max(0, y_min - margin)
        y_end = min(h, y_max + margin + 1)
        x_start = max(0, x_min - margin)
        x_end = min(w, x_max + margin + 1)

        if is_4d:
            return volume[:, z_start:z_end, y_start:y_end, x_start:x_end]
        return volume[z_start:z_end, y_start:y_end, x_start:x_end]

    def crop_or_pad_3d(
        self,
        volume: np.ndarray,
        target_shape: Optional[Tuple[int, int, int]] = None,
        pad_value: Optional[float] = None
    ) -> np.ndarray:
        """
        Center-crop or symmetric-pad volume to exactly match target_shape.
        
        Args:
            volume: 3D array (D, H, W) or 4D array (C, D, H, W).
            target_shape: Desired 3D shape (D, H, W). Defaults to self.target_shape.
            pad_value: Padding constant. Defaults to self.pad_value.
            
        Returns:
            Numpy array of exact shape (*target_shape) or (C, *target_shape).
        """
        tgt = target_shape or self.target_shape
        pad_val = pad_value if pad_value is not None else self.pad_value

        is_4d = (volume.ndim == 4)
        spatial_shape = volume.shape[1:] if is_4d else volume.shape

        # Step 1: Crop dimensions that are larger than target
        crop_slices = []
        for curr_dim, tgt_dim in zip(spatial_shape, tgt):
            if curr_dim > tgt_dim:
                diff = curr_dim - tgt_dim
                start = diff // 2
                end = start + tgt_dim
                crop_slices.append(slice(start, end))
            else:
                crop_slices.append(slice(0, curr_dim))

        if is_4d:
            cropped = volume[:, crop_slices[0], crop_slices[1], crop_slices[2]]
        else:
            cropped = volume[crop_slices[0], crop_slices[1], crop_slices[2]]

        # Step 2: Pad dimensions that are smaller than target
        curr_spatial = cropped.shape[1:] if is_4d else cropped.shape
        pad_widths = []
        for curr_dim, tgt_dim in zip(curr_spatial, tgt):
            if curr_dim < tgt_dim:
                diff = tgt_dim - curr_dim
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_widths.append((pad_before, pad_after))
            else:
                pad_widths.append((0, 0))

        if is_4d:
            full_pad = [(0, 0)] + pad_widths
        else:
            full_pad = pad_widths

        if any(p[0] > 0 or p[1] > 0 for p in pad_widths):
            padded = np.pad(cropped, full_pad, mode="constant", constant_values=pad_val)
        else:
            padded = cropped

        return padded

    @staticmethod
    def resample_3d(
        volume: np.ndarray,
        target_shape: Tuple[int, int, int],
        is_mask: bool = False
    ) -> np.ndarray:
        """
        Resample a 3D volume or 4D tensor using 3D interpolation.
        
        Args:
            volume: 3D array (D, H, W) or 4D array (C, D, H, W).
            target_shape: Desired 3D shape (D, H, W).
            is_mask: If True, uses nearest-neighbor interpolation (order 0)
                     to preserve discrete segmentation class labels (0, 1, 2, 4).
                     If False, uses trilinear interpolation (order 1) for intensities.
                     
        Returns:
            Resampled numpy array matching target_shape.
        """
        is_4d = (volume.ndim == 4)
        spatial_shape = volume.shape[1:] if is_4d else volume.shape
        zoom_factors = [t / c for t, c in zip(target_shape, spatial_shape)]
        order = 0 if is_mask else 1

        if is_4d:
            resampled_channels = []
            for c in range(volume.shape[0]):
                res_c = zoom(volume[c], zoom_factors, order=order, prefilter=not is_mask)
                resampled_channels.append(res_c)
            result = np.stack(resampled_channels, axis=0)
        else:
            result = zoom(volume, zoom_factors, order=order, prefilter=not is_mask)

        if is_mask:
            # Preserve discrete segmentation integer type
            return np.round(result).astype(np.uint8)

        return result.astype(np.float32)
