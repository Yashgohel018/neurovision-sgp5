"""
NeuroVision AI: Module C - MRI Intensity Normalization
Standardizes multi-sequence MRI volumes using non-zero z-score normalization
and robust percentile clipping to suppress extreme scanner artifacts.
"""
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np


class IntensityNormalizer:
    """
    Performs robust MRI intensity normalization tailored for brain imaging.
    
    In skull-stripped brain MRI, air background voxels have an intensity of 0.
    Standard whole-image normalization severely distorts intensity distributions
    because background voxels dominate the field of view (> 60%).
    
    This class enforces non-zero z-score standardization:
      1. Isolates foreground brain voxels (I(x) > 0 or user mask).
      2. Clips extreme scanner spikes to [P_low, P_high] percentiles.
      3. Normalizes foreground to mean = 0, std = 1.
      4. Preserves background air voxels as exact 0.0.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the normalizer with configuration settings.
        
        Args:
            config: Optional dictionary containing normalization configuration.
        """
        norm_cfg = config.get("normalization", {}) if config else {}
        self.method = norm_cfg.get("method", "z_score")
        self.nonzero_only = norm_cfg.get("nonzero_only", True)
        self.clip_percentiles = tuple(norm_cfg.get("clip_percentiles", [1.0, 99.0]))
        self.epsilon = float(norm_cfg.get("epsilon", 1e-7))

    def normalize(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None,
        method: Optional[str] = None
    ) -> np.ndarray:
        """
        Normalize a 3D MRI volume or 4D multi-channel tensor.
        
        Args:
            volume: 3D array (D, H, W) or 4D array (C, D, H, W).
            mask: Optional boolean foreground mask. If None and nonzero_only is True,
                  foreground is determined by volume > 0.
            method: Override normalization method ('z_score', 'min_max', 'none').
            
        Returns:
            Normalized float32 numpy array with identical spatial shape.
        """
        selected_method = method or self.method
        if selected_method == "none":
            return volume.astype(np.float32)

        # Handle 4D multi-channel tensor (C, D, H, W)
        if volume.ndim == 4:
            normalized_channels = []
            for c in range(volume.shape[0]):
                ch_mask = mask[c] if (mask is not None and mask.ndim == 4) else mask
                norm_ch = self._normalize_single_channel(volume[c], ch_mask, selected_method)
                normalized_channels.append(norm_ch)
            return np.stack(normalized_channels, axis=0)

        # Single channel 3D volume
        return self._normalize_single_channel(volume, mask, selected_method)

    def _normalize_single_channel(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray],
        method: str
    ) -> np.ndarray:
        """Helper to normalize a single 3D channel."""
        vol = volume.astype(np.float32)

        # Determine foreground mask
        if mask is not None:
            fg_mask = mask.astype(bool)
        elif self.nonzero_only:
            fg_mask = vol > 0
        else:
            fg_mask = np.ones_like(vol, dtype=bool)

        # Edge case: empty volume
        if not np.any(fg_mask):
            return np.zeros_like(vol, dtype=np.float32)

        fg_values = vol[fg_mask]

        # Percentile clipping
        if self.clip_percentiles:
            p_low, p_high = np.percentile(fg_values, self.clip_percentiles)
            vol_clipped = np.clip(vol, p_low, p_high)
            fg_values_clipped = vol_clipped[fg_mask]
        else:
            vol_clipped = vol
            fg_values_clipped = fg_values

        out = np.zeros_like(vol, dtype=np.float32)

        if method == "z_score":
            mean = float(np.mean(fg_values_clipped))
            std = float(np.std(fg_values_clipped))
            if std < self.epsilon:
                std = 1.0
            out[fg_mask] = (vol_clipped[fg_mask] - mean) / (std + self.epsilon)

        elif method == "min_max":
            v_min = float(np.min(fg_values_clipped))
            v_max = float(np.max(fg_values_clipped))
            denom = v_max - v_min
            if denom < self.epsilon:
                denom = 1.0
            out[fg_mask] = (vol_clipped[fg_mask] - v_min) / (denom + self.epsilon)

        else:
            raise ValueError(f"Unsupported normalization method: '{method}'. Choose 'z_score' or 'min_max'.")

        return out

    def get_normalization_stats(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate statistical parameters of foreground intensities prior to normalization.
        
        Args:
            volume: 3D MRI volume.
            mask: Optional foreground mask.
            
        Returns:
            Dictionary containing mean, std, median, min, max, p1, p99.
        """
        if mask is not None:
            fg = volume[mask.astype(bool)]
        elif self.nonzero_only:
            fg = volume[volume > 0]
        else:
            fg = volume.ravel()

        if len(fg) == 0:
            return {
                "mean": 0.0, "std": 0.0, "median": 0.0,
                "min": 0.0, "max": 0.0, "p1": 0.0, "p99": 0.0
            }

        p1, p99 = np.percentile(fg, [1.0, 99.0])
        return {
            "mean": float(np.mean(fg)),
            "std": float(np.std(fg)),
            "median": float(np.median(fg)),
            "min": float(np.min(fg)),
            "max": float(np.max(fg)),
            "p1": float(p1),
            "p99": float(p99)
        }
