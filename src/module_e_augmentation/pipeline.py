"""
NeuroVision AI: Module E - Augmentation Pipeline
Loads training sample volumes, applies synchronized 3D augmentation, and produces QA artifacts.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logger import setup_logger
from .transforms import Multimodal3DAugmentation

logger = setup_logger("AugmentationPipeline")


class AugmentationPipeline:
    def __init__(self, config: Dict):
        self.config = config
        trans_cfg = config.get("transforms", {})
        spatial = trans_cfg.get("spatial", {})
        intensity = trans_cfg.get("intensity", {})

        p_flip = spatial.get("random_flip_sagittal", {}).get("probability", 0.5)
        p_rotate = spatial.get("random_rotation_3d", {}).get("probability", 0.6)
        max_angle = spatial.get("random_rotation_3d", {}).get("max_angle_degrees", 10.0)

        p_noise = intensity.get("gaussian_noise", {}).get("probability", 0.5)
        noise_std = intensity.get("gaussian_noise", {}).get("std", 0.02)
        p_scale = intensity.get("intensity_scaling", {}).get("probability", 0.5)
        scale_range = tuple(intensity.get("intensity_scaling", {}).get("factor_range", [0.92, 1.08]))
        p_contrast = intensity.get("random_contrast", {}).get("probability", 0.4)
        gamma_range = tuple(intensity.get("random_contrast", {}).get("gamma_range", [0.85, 1.15]))

        self.augmentor = Multimodal3DAugmentation(
            p_flip=p_flip,
            p_rotate=p_rotate,
            max_angle=max_angle,
            p_noise=p_noise,
            noise_std=noise_std,
            p_scale=p_scale,
            scale_range=scale_range,
            p_contrast=p_contrast,
            gamma_range=gamma_range,
            seed=42
        )

    def generate_qa_figure(
        self,
        original_vol: np.ndarray,
        augmented_vol: np.ndarray,
        modality_names: List[str],
        output_path: str,
        slice_idx: Optional[int] = None
    ) -> str:
        """
        Creates a side-by-side QA figure comparing original vs augmented 3D MRI across all modalities.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        num_mods = min(len(modality_names), original_vol.shape[0])

        if slice_idx is None:
            slice_idx = original_vol.shape[1] // 2

        fig, axes = plt.subplots(2, num_mods, figsize=(3.5 * num_mods, 7), facecolor="#0B0F19")
        fig.suptitle(
            "NeuroVision AI - Module E: Synchronized Multimodal 3D Augmentation QA",
            color="#E2E8F0",
            fontsize=14,
            fontweight="bold",
            y=0.98
        )

        for col, mod in enumerate(modality_names[:num_mods]):
            # Original slice
            orig_slice = original_vol[col, slice_idx, :, :]
            axes[0, col].imshow(orig_slice, cmap="magma" if "flair" in mod.lower() else "bone")
            axes[0, col].set_title(f"Original {mod.upper()}", color="#93C5FD", fontsize=11, fontweight="semibold")
            axes[0, col].axis("off")

            # Augmented slice
            aug_slice = augmented_vol[col, slice_idx, :, :]
            axes[1, col].imshow(aug_slice, cmap="magma" if "flair" in mod.lower() else "bone")
            axes[1, col].set_title(f"Augmented {mod.upper()}", color="#34D399", fontsize=11, fontweight="semibold")
            axes[1, col].axis("off")

        axes[0, 0].set_ylabel("Original", color="#93C5FD", fontsize=12, labelpad=10)
        axes[1, 0].set_ylabel("Augmented", color="#34D399", fontsize=12, labelpad=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()
        logger.info(f"Generated QA comparison figure: {output_path}")
        return output_path
