"""
NeuroVision AI: Module D - 3D ROI Extraction Visualizer
Generates publication-quality diagnostic figures illustrating
bounding box extraction, peritumoral context envelopes, and ROI comparisons.
"""
from typing import Dict, Any, List, Optional
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("ROIVisualizer")


class ROIVisualizer:
    """
    Renders visual QA figures for Module D:
    1. Comparison between Full Brain, Bounding Envelope, and Extracted ROI.
    2. Cohort-wide spatial distribution of tumor centroids and spans.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, base_dir: str = "."):
        vis_cfg = config.get("visualization", {}) if config else {}
        self.figures_dir = os.path.join(base_dir, vis_cfg.get("figures_dir", "reports/figures"))
        os.makedirs(self.figures_dir, exist_ok=True)

    def plot_roi_extraction_case(
        self,
        full_image: np.ndarray,      # (4, D, H, W)
        full_mask: np.ndarray,       # (D, H, W)
        roi_image: np.ndarray,       # (4, D_roi, H_roi, W_roi)
        roi_mask: np.ndarray,        # (D_roi, H_roi, W_roi)
        bbox: Any,
        subject_id: str,
        grade: str = "HGG",
        save_path: Optional[str] = None
    ) -> str:
        """
        Render a 4-panel comparison of full brain vs margin-expanded bounding envelope vs cropped ROI.
        """
        if save_path is None:
            save_path = os.path.join(self.figures_dir, f"roi_extraction_{subject_id}.png")

        # Pick representative axial slice at tumor centroid
        cz = int(bbox.centroid_z)
        cz = max(0, min(full_image.shape[1] - 1, cz))

        # Full brain slices (T1ce: ch 1, FLAIR: ch 3)
        full_flair_slice = full_image[3, cz, :, :]
        full_t1ce_slice = full_image[1, cz, :, :]
        full_mask_slice = full_mask[cz, :, :] if full_mask is not None else None

        # ROI slice (mid-slice of extracted ROI)
        roi_mid_z = roi_image.shape[1] // 2
        roi_flair_slice = roi_image[3, roi_mid_z, :, :]
        roi_mask_slice = roi_mask[roi_mid_z, :, :] if roi_mask is not None else None

        fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=300)
        fig.patch.set_facecolor("#0f172a")

        # Panel 1: Full Brain FLAIR with Tumor Mask
        axes[0].set_facecolor("#0f172a")
        axes[0].imshow(full_flair_slice, cmap="gray", origin="lower")
        if full_mask_slice is not None and np.any(full_mask_slice > 0):
            masked = np.ma.masked_where(full_mask_slice == 0, full_mask_slice)
            axes[0].imshow(masked, cmap="autumn", alpha=0.6, origin="lower")
        axes[0].set_title(f"Full Brain FLAIR + Mask\n(Patient: {subject_id} - {grade})", color="white", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Full Brain T1ce with Bounding Box and Context Margin
        axes[1].set_facecolor("#0f172a")
        axes[1].imshow(full_t1ce_slice, cmap="bone", origin="lower")
        # Draw raw bounding box (red dashed)
        # Note: in (Y, X) coordinates for matplotlib Rectangle: (x_min, y_min), width, height
        raw_rect = Rectangle(
            (bbox.x_min, bbox.y_min),
            bbox.span_x, bbox.span_y,
            linewidth=1.8, edgecolor="#ef4444", facecolor="none", linestyle="--", label="Raw Tumor BBox"
        )
        axes[1].add_patch(raw_rect)
        # Draw context margin envelope (green solid)
        margin_rect = Rectangle(
            (bbox.crop_x_min, bbox.crop_y_min),
            bbox.crop_span_x, bbox.crop_span_y,
            linewidth=2.0, edgecolor="#10b981", facecolor="none", label=f"Context Envelope (+{bbox.margin_applied} vox)"
        )
        axes[1].add_patch(margin_rect)
        axes[1].legend(loc="upper right", fontsize=8, facecolor="#1e293b", labelcolor="white")
        axes[1].set_title(f"T1ce + Bounding Envelopes\nRaw [{bbox.span_y}x{bbox.span_x}] -> Context [{bbox.crop_span_y}x{bbox.crop_span_x}]", color="white", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        # Panel 3: Extracted Standardized 3D ROI (FLAIR)
        axes[2].set_facecolor("#0f172a")
        axes[2].imshow(roi_flair_slice, cmap="magma", origin="lower")
        axes[2].set_title(f"Standardized 3D ROI (FLAIR)\nTarget Shape: {roi_image.shape[1:]}", color="white", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        # Panel 4: Extracted 3D ROI Segmentation Mask (Multiclass)
        axes[3].set_facecolor("#0f172a")
        axes[3].imshow(roi_flair_slice, cmap="gray", origin="lower", alpha=0.5)
        if roi_mask_slice is not None and np.any(roi_mask_slice > 0):
            # Create discrete colormap for labels: 1 (Necrotic - Red), 2 (Edema - Green), 4 (Enhancing - Yellow)
            cmap_custom = ListedColormap(["#ef4444", "#10b981", "#eab308"])
            remapped = np.zeros_like(roi_mask_slice)
            remapped[roi_mask_slice == 1] = 1
            remapped[roi_mask_slice == 2] = 2
            remapped[roi_mask_slice == 4] = 3
            masked_roi = np.ma.masked_where(remapped == 0, remapped)
            axes[3].imshow(masked_roi, cmap=cmap_custom, origin="lower", alpha=0.85)

        axes[3].set_title(f"Standardized Tumor Mask\nNCR (Red) | ED (Grn) | ET (Yel)", color="white", fontsize=11, fontweight="bold")
        axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Generated ROI extraction case figure: {save_path}")
        return save_path

    def plot_cohort_roi_distribution(
        self,
        roi_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate cohort-wide morphometric distribution figures:
        1. Bounding box spans (Depth, Height, Width) by tumor grade.
        2. Tumor-to-brain volume fraction distribution.
        3. Spatial centroid distribution.
        """
        if save_path is None:
            save_path = os.path.join(self.figures_dir, "roi_spatial_distribution.png")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
        fig.patch.set_facecolor("#0f172a")

        # Subplot 1: Bounding Box Spans Boxplot
        axes[0].set_facecolor("#1e293b")
        span_data = [roi_df["bbox_span_z"], roi_df["bbox_span_y"], roi_df["bbox_span_x"]]
        bplot = axes[0].boxplot(
            span_data,
            patch_artist=True,
            tick_labels=["Depth (Z)", "Height (Y)", "Width (X)"],
            medianprops=dict(color="#f59e0b", linewidth=2)
        )
        colors = ["#38bdf8", "#818cf8", "#a78bfa"]
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        axes[0].set_title("Tumor Bounding Box Spans (Voxels)", color="white", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Voxel Span", color="white")
        axes[0].tick_params(colors="white")
        axes[0].grid(True, linestyle="--", alpha=0.3, color="#64748b")

        # Subplot 2: Tumor to Brain Ratio (HGG vs LGG)
        axes[1].set_facecolor("#1e293b")
        hgg_ratio = roi_df[roi_df["grade"] == "HGG"]["tumor_to_brain_ratio"] * 100
        lgg_ratio = roi_df[roi_df["grade"] == "LGG"]["tumor_to_brain_ratio"] * 100

        axes[1].hist(hgg_ratio, bins=20, alpha=0.7, color="#ef4444", label="HGG (High-Grade)")
        axes[1].hist(lgg_ratio, bins=20, alpha=0.7, color="#06b6d4", label="LGG (Low-Grade)")
        axes[1].set_title("Tumor-to-Brain Volume Fraction (%)", color="white", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Tumor Volume / Total Brain Volume (%)", color="white")
        axes[1].set_ylabel("Patient Count", color="white")
        axes[1].tick_params(colors="white")
        axes[1].legend(facecolor="#0f172a", labelcolor="white")
        axes[1].grid(True, linestyle="--", alpha=0.3, color="#64748b")

        # Subplot 3: 2D Spatial Centroid Scatter (Axial Plane: Y vs X)
        axes[2].set_facecolor("#1e293b")
        hgg_pts = roi_df[roi_df["grade"] == "HGG"]
        lgg_pts = roi_df[roi_df["grade"] == "LGG"]

        axes[2].scatter(hgg_pts["centroid_x"], hgg_pts["centroid_y"], c="#ef4444", alpha=0.6, s=25, label="HGG Centroids")
        axes[2].scatter(lgg_pts["centroid_x"], lgg_pts["centroid_y"], c="#38bdf8", alpha=0.8, s=35, marker="^", label="LGG Centroids")
        axes[2].set_title("Tumor Spatial Centroids (Axial Projection)", color="white", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("X Centroid Coordinate", color="white")
        axes[2].set_ylabel("Y Centroid Coordinate", color="white")
        axes[2].tick_params(colors="white")
        axes[2].legend(facecolor="#0f172a", labelcolor="white")
        axes[2].grid(True, linestyle="--", alpha=0.3, color="#64748b")

        plt.tight_layout()
        plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Generated cohort ROI distribution figure: {save_path}")
        return save_path
