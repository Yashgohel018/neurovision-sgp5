"""
NeuroVision AI: Module B - Diagnostic Visualizer
Generates publication-quality clinical figures and multimodal orthogonal slice overlays.
Uses headless Agg backend for robust automated reporting.
"""
from typing import Dict, Any, List, Optional
import os
import numpy as np
import pandas as pd
import nibabel as nib

# Headless backend to prevent display errors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns

from src.utils.logger import setup_logger

logger = setup_logger("MultimodalVisualizer")


class MultimodalVisualizer:
    """
    Renders publication-grade diagnostic plots and orthogonal MRI panels for NeuroVision AI.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        vis_cfg = self.config.get("visualization", {})
        self.dpi = vis_cfg.get("dpi", 300)
        self.palette = vis_cfg.get("palette", {
            "HGG": "#e63946",
            "LGG": "#1d3557",
            "WT": "#457b9d",
            "TC": "#e76f51",
            "ET": "#f4a261",
            "ED": "#2a9d8f",
            "NCR": "#e63946",
        })

    def plot_class_distribution(self, df: pd.DataFrame, save_path: str) -> str:
        """
        Renders a dual-panel figure (bar chart + donut chart) illustrating HGG vs LGG distribution.
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        counts = df["grade"].value_counts()
        total = len(df)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
        fig.suptitle(
            f"NeuroVision AI: Histopathological Grade Distribution (N={total})",
            fontsize=15, fontweight="bold", y=1.02
        )

        colors = [self.palette.get(g, "#333333") for g in counts.index]

        # 1. Bar Chart with exact count and percentage
        bars = ax1.bar(counts.index, counts.values, color=colors, edgecolor="black", width=0.5, alpha=0.9)
        ax1.set_ylabel("Patient Count", fontsize=12, fontweight="bold")
        ax1.set_xlabel("WHO Glioma Grade Category", fontsize=12, fontweight="bold")
        ax1.set_ylim(0, max(counts.values) * 1.15)
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            pct = (height / total) * 100
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0, height + (max(counts.values) * 0.02),
                f"{int(height)} ({pct:.1f}%)",
                ha="center", va="bottom", fontsize=11, fontweight="bold"
            )

        # 2. Donut Chart
        wedges, texts, autotexts = ax2.pie(
            counts.values,
            labels=[f"{g} (WHO Grade {'III/IV' if g=='HGG' else 'I/II'})" for g in counts.index],
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            pctdistance=0.75,
            wedgeprops=dict(width=0.45, edgecolor="black", linewidth=1.5),
        )
        for t in autotexts:
            t.set_fontsize(11)
            t.set_fontweight("bold")
            t.set_color("white")

        hgg_count = counts.get("HGG", 0)
        lgg_count = counts.get("LGG", 1)
        ratio = hgg_count / lgg_count if lgg_count > 0 else 0
        ax2.set_title(f"Class Imbalance Ratio: {ratio:.2f} : 1 (HGG : LGG)", fontsize=11, fontstyle="italic")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved class distribution figure to: {save_path}")
        return save_path

    def plot_tumor_volume_distributions(self, df: pd.DataFrame, save_path: str) -> str:
        """
        Plots box/violin distributions comparing tumor sub-region volumes (WT, TC, ET, ED, NCR)
        across HGG and LGG cohorts on a logarithmic scale.
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        metrics = [
            ("wt_volume_cm3", "Whole Tumor (WT)"),
            ("tc_volume_cm3", "Tumor Core (TC)"),
            ("et_volume_cm3", "Enhancing Tumor (ET)"),
            ("ed_volume_cm3", "Peritumoral Edema (ED)"),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=self.dpi)
        fig.suptitle(
            "Tumor Sub-Region Volumetric Burden Comparison: HGG vs. LGG (BraTS 2020)",
            fontsize=15, fontweight="bold", y=1.03
        )

        palette_grades = {"HGG": self.palette.get("HGG", "#e63946"), "LGG": self.palette.get("LGG", "#1d3557")}

        for idx, (col, label) in enumerate(metrics):
            ax = axes[idx]
            if col not in df.columns:
                continue

            # Plot with seaborn boxplot + stripplot for clinical transparency
            sns.boxplot(
                data=df, x="grade", y=col, hue="grade", legend=False,
                palette=palette_grades, ax=ax,
                width=0.45, boxprops=dict(alpha=0.7), showmeans=True,
                meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white")
            )
            sns.stripplot(
                data=df, x="grade", y=col, hue="grade", legend=False,
                palette=palette_grades, ax=ax,
                size=3.5, jitter=0.2, alpha=0.4, dodge=False
            )

            ax.set_title(label, fontsize=12, fontweight="bold")
            ax.set_xlabel("Glioma Grade", fontsize=11)
            ax.set_ylabel("Physical Volume (cm³)" if idx == 0 else "", fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            # Add log-scale if maximum volume is large
            if df[col].max() > 20:
                ax.set_yscale("log")
                ax.set_ylabel("Volume (cm³, log scale)" if idx == 0 else "")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved tumor volume distribution figure to: {save_path}")
        return save_path

    def plot_modality_intensity_distributions(
        self,
        intensity_samples: Dict[str, np.ndarray],
        save_path: str
    ) -> str:
        """
        Plots KDE density curves of non-zero tissue intensities across T1, T1ce, T2, and FLAIR.
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 5), dpi=self.dpi)
        colors = {"t1": "#457b9d", "t1ce": "#e63946", "t2": "#2a9d8f", "flair": "#e76f51"}

        for mod, intensities in intensity_samples.items():
            if len(intensities) > 0:
                # Subsample up to 25,000 voxels for smooth KDE rendering
                sample = intensities if len(intensities) <= 25000 else np.random.choice(intensities, 25000, replace=False)
                sns.kdeplot(
                    sample, ax=ax, label=mod.upper(),
                    color=colors.get(mod, "#333333"), linewidth=2.2, alpha=0.8
                )

        ax.set_title("Multi-Sequence Non-Zero MRI Intensity Profiles (BraTS 2020)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Voxel Intensity Value", fontsize=11, fontweight="bold")
        ax.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
        ax.legend(title="MRI Sequence", fontsize=10, title_fontsize=11)
        ax.grid(linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved modality intensity distribution figure to: {save_path}")
        return save_path

    def plot_multimodal_slices(
        self,
        subject_id: str,
        t1_path: str,
        t1ce_path: str,
        t2_path: str,
        flair_path: str,
        seg_path: str,
        save_path: str,
        grade: str = "HGG"
    ) -> str:
        """
        Renders a 5-column orthogonal slice panel (T1, T1ce, T2, FLAIR, and T1ce + Sub-region Mask Overlay)
        centered at the tumor's maximum axial extent.
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        # Load volumes
        t1 = nib.load(t1_path).get_fdata()
        t1ce = nib.load(t1ce_path).get_fdata()
        t2 = nib.load(t2_path).get_fdata()
        flair = nib.load(flair_path).get_fdata()
        seg = nib.load(seg_path).get_fdata().astype(np.int16)

        # Determine slice with maximum tumor burden
        tumor_slices = np.sum(np.isin(seg, [1, 2, 4]), axis=(0, 1))
        best_z = int(np.argmax(tumor_slices)) if np.max(tumor_slices) > 0 else seg.shape[2] // 2

        # Extract 2D axial slices (rotate 90 degrees for standard radiologic view)
        s_t1 = np.rot90(t1[:, :, best_z])
        s_t1ce = np.rot90(t1ce[:, :, best_z])
        s_t2 = np.rot90(t2[:, :, best_z])
        s_flair = np.rot90(flair[:, :, best_z])
        s_seg = np.rot90(seg[:, :, best_z])

        # Normalize 2D slices to [0, 1] for visual display
        def norm_slice(arr: np.ndarray) -> np.ndarray:
            p1, p99 = np.percentile(arr[arr > 0], (1, 99)) if np.any(arr > 0) else (0, 1)
            clipped = np.clip(arr, p1, p99)
            denom = p99 - p1 if p99 > p1 else 1.0
            return (clipped - p1) / denom

        disp_t1 = norm_slice(s_t1)
        disp_t1ce = norm_slice(s_t1ce)
        disp_t2 = norm_slice(s_t2)
        disp_flair = norm_slice(s_flair)

        # Build RGB overlay for segmentation
        # NCR (1) = Red, ED (2) = Green, ET (4) = Yellow
        rgb_overlay = np.repeat(disp_t1ce[:, :, np.newaxis], 3, axis=2)
        mask_ncr = s_seg == 1
        mask_ed = s_seg == 2
        mask_et = s_seg == 4

        alpha = 0.55
        rgb_overlay[mask_ncr] = (1 - alpha) * rgb_overlay[mask_ncr] + alpha * np.array([1.0, 0.1, 0.1])
        rgb_overlay[mask_ed] = (1 - alpha) * rgb_overlay[mask_ed] + alpha * np.array([0.1, 0.85, 0.3])
        rgb_overlay[mask_et] = (1 - alpha) * rgb_overlay[mask_et] + alpha * np.array([1.0, 0.85, 0.1])

        fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), dpi=self.dpi)
        fig.suptitle(
            f"NeuroVision AI Multimodal MRI Ingestion Panel — Patient: {subject_id} ({grade}) [Axial Slice Z={best_z}]",
            fontsize=14, fontweight="bold", y=1.02
        )

        titles = [
            "T1-Native\n(Anatomy)",
            "T1-Contrast (T1ce)\n(Vascular Enhancement)",
            "T2-Weighted\n(Hyperintense Edema)",
            "T2-FLAIR\n(CSF Nulled Infiltration)",
            "Multimodal Overlay\n(Red:NCR, Green:ED, Yellow:ET)",
        ]
        images = [disp_t1, disp_t1ce, disp_t2, disp_flair, rgb_overlay]

        for i, ax in enumerate(axes):
            if i < 4:
                ax.imshow(images[i], cmap="gray")
            else:
                ax.imshow(images[i])
            ax.set_title(titles[i], fontsize=11, fontweight="bold")
            ax.axis("off")

        # Add custom legend for mask sub-regions
        patches = [
            mpatches.Patch(color=[1.0, 0.1, 0.1], label="Necrotic Core (NCR - L1)"),
            mpatches.Patch(color=[0.1, 0.85, 0.3], label="Peritumoral Edema (ED - L2)"),
            mpatches.Patch(color=[1.0, 0.85, 0.1], label="Enhancing Rim (ET - L4)"),
        ]
        fig.legend(handles=patches, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), fontsize=11)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved multimodal slice panel to: {save_path}")
        return save_path

    def plot_correlation_matrix(self, df: pd.DataFrame, save_path: str) -> str:
        """
        Renders correlation heatmap between clinical features (Age, Survival) and tumor volumes.
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        features = ["age", "wt_volume_cm3", "tc_volume_cm3", "et_volume_cm3", "ed_volume_cm3", "et_to_wt_ratio"]
        labels = ["Age", "WT Vol", "TC Vol", "ET Vol", "ED Vol", "ET/WT Ratio"]

        # Coerce and clean
        sub_df = df[[f for f in features if f in df.columns]].copy()
        if "survival_days" in df.columns:
            surv_num = pd.to_numeric(df["survival_days"], errors="coerce")
            if surv_num.notna().sum() > 20:
                sub_df["Survival"] = surv_num
                labels.append("Survival (Days)")

        corr = sub_df.corr(method="spearman")

        fig, ax = plt.subplots(figsize=(8, 6.5), dpi=self.dpi)
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm", center=0.0,
            square=True, linewidths=1.0, cbar_kws={"shrink": 0.8},
            xticklabels=labels[:len(corr)], yticklabels=labels[:len(corr)], ax=ax
        )
        ax.set_title("Clinical Covariate & Tumor Volume Correlation Matrix (Spearman Rank)", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved correlation matrix heatmap to: {save_path}")
        return save_path
