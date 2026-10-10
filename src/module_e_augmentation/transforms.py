"""
NeuroVision AI: Module E - Multimodal 3D Augmentation Transforms
Synchronized spatial and intensity transformations for multi-sequence brain MRI.
"""
import random
from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy import ndimage


class Multimodal3DAugmentation:
    """
    Synchronized 3D data augmentation pipeline for multimodal brain MRI tensors.
    Guarantees that spatial operations (rotation, flips, scaling) are strictly identical
    across all MRI sequences (T1, T1ce, T2, FLAIR) and ground-truth segmentation masks.
    """

    def __init__(
        self,
        p_flip: float = 0.5,
        p_rotate: float = 0.6,
        max_angle: float = 10.0,
        p_noise: float = 0.5,
        noise_std: float = 0.02,
        p_scale: float = 0.5,
        scale_range: Tuple[float, float] = (0.92, 1.08),
        p_contrast: float = 0.4,
        gamma_range: Tuple[float, float] = (0.85, 1.15),
        seed: Optional[int] = None
    ):
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.max_angle = max_angle
        self.p_noise = p_noise
        self.noise_std = noise_std
        self.p_scale = p_scale
        self.scale_range = scale_range
        self.p_contrast = p_contrast
        self.gamma_range = gamma_range
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        multimodal_volume: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Args:
            multimodal_volume: 4D array of shape [C, D, H, W] (e.g. 4 channels for T1, T1ce, T2, FLAIR)
            mask: Optional 3D array of shape [D, H, W] (integer segmentation labels)
        Returns:
            augmented_volume: 4D array [C, D, H, W]
            augmented_mask: Optional 3D array [D, H, W]
        """
        vol = multimodal_volume.copy()
        m = mask.copy() if mask is not None else None

        # 1. Synchronized Sagittal Flip (Left-Right symmetry)
        if self.rng.random() < self.p_flip:
            # Flip width axis (axis 3 in 4D, axis 2 in 3D)
            vol = np.flip(vol, axis=3)
            if m is not None:
                m = np.flip(m, axis=2)

        # 2. Synchronized Small 3D Rotation (Axial plane rotation: around Z axis)
        if self.rng.random() < self.p_rotate:
            angle = float(self.rng.uniform(-self.max_angle, self.max_angle))
            # Rotate in H-W plane (axes 2 and 3 in vol, 1 and 2 in m)
            for c in range(vol.shape[0]):
                vol[c] = ndimage.rotate(
                    vol[c],
                    angle=angle,
                    axes=(1, 2),
                    reshape=False,
                    order=1,
                    mode="constant",
                    cval=0.0
                )
            if m is not None:
                m = ndimage.rotate(
                    m,
                    angle=angle,
                    axes=(1, 2),
                    reshape=False,
                    order=0,  # Nearest-neighbor for integer segmentation labels
                    mode="constant",
                    cval=0
                )

        # 3. Intensity Transformations (Applied to image channels only, never masks)
        for c in range(vol.shape[0]):
            ch = vol[c]
            nonzero_mask = ch != 0

            # 3a. Random Intensity Scaling
            if self.rng.random() < self.p_scale:
                scale_factor = float(self.rng.uniform(self.scale_range[0], self.scale_range[1]))
                ch[nonzero_mask] = ch[nonzero_mask] * scale_factor

            # 3b. Additive Gaussian Noise
            if self.rng.random() < self.p_noise:
                noise = self.rng.normal(0.0, self.noise_std, size=ch.shape).astype(np.float32)
                ch[nonzero_mask] = ch[nonzero_mask] + noise[nonzero_mask]

            # 3c. Random Contrast Adjustment (Gamma transform)
            if self.rng.random() < self.p_contrast:
                gamma = float(self.rng.uniform(self.gamma_range[0], self.gamma_range[1]))
                vals = ch[nonzero_mask]
                min_val = vals.min()
                max_val = vals.max()
                if max_val > min_val:
                    norm = (vals - min_val) / (max_val - min_val)
                    adjusted = np.power(norm, gamma) * (max_val - min_val) + min_val
                    ch[nonzero_mask] = adjusted

            vol[c] = ch

        return vol.astype(np.float32), m
