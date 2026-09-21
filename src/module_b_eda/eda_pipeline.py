"""
NeuroVision AI: Module B - Exploratory Data Analysis & Volumetric QC Pipeline
Orchestrates volumetric quantification, statistical hypothesis testing,
intensity profiling, and diagnostic visualization across the BraTS 2020 cohort.
"""
from typing import Dict, Any, Optional, List
import os
import time
import numpy as np
import pandas as pd
import nibabel as nib

from src.utils.logger import setup_logger
from src.module_b_eda.volumetrics import TumorVolumetricAnalyzer
from src.module_b_eda.intensity_profiler import IntensityProfiler
from src.module_b_eda.statistical_analyzer import StatisticalAnalyzer
from src.module_b_eda.visualizer import MultimodalVisualizer

logger = setup_logger("EDAPipeline")


class EDAPipeline:
    """
    End-to-end pipeline orchestrator for Module B.
    Transforms raw metadata into rich volumetric features, cohort statistics, and diagnostic reports.
    """

    def __init__(self, config: Dict[str, Any], base_dir: str = "."):
        self.config = config
        self.base_dir = base_dir

        self.vol_analyzer = TumorVolumetricAnalyzer(config)
        self.intensity_profiler = IntensityProfiler(config)
        self.stat_analyzer = StatisticalAnalyzer(config)
        self.visualizer = MultimodalVisualizer(config)

        # Output locations
        out_cfg = config.get("outputs", {})
        self.tumor_stats_path = os.path.join(base_dir, out_cfg.get("tumor_statistics_csv", "data/tumor_statistics.csv"))
        self.dataset_summary_path = os.path.join(base_dir, out_cfg.get("dataset_summary_csv", "data/dataset_summary.csv"))
        self.figures_dir = os.path.join(base_dir, out_cfg.get("figures_dir", "reports/figures"))

        os.makedirs(os.path.dirname(self.tumor_stats_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.dataset_summary_path), exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def run(self, limit: Optional[int] = None, skip_figures: bool = False) -> Dict[str, Any]:
        """
        Executes the complete Module B workflow.
        """
        start_time = time.time()
        meta_rel = self.config.get("inputs", {}).get("metadata_csv", "data/dataset_metadata.csv")
        meta_path = os.path.join(self.base_dir, meta_rel)

        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Input metadata index not found at: {meta_path}. Run Module A first.")

        df_meta = pd.read_csv(meta_path)
        logger.info(f"Loaded master metadata index: {meta_path} ({len(df_meta)} subjects)")

        if limit is not None and limit > 0:
            logger.info(f"Applying limit: processing first {limit} subjects for rapid execution")
            df_meta = df_meta.head(limit).copy()

        # -------------------------------------------------------------
        # 1. 3D Tumor Volumetric Quantification
        # -------------------------------------------------------------
        logger.info("Phase 1: Starting 3D tumor sub-region volumetric quantification...")
        vol_records: List[Dict[str, Any]] = []

        for idx, row in df_meta.iterrows():
            subject_id = row["subject_id"]
            seg_path = row["seg_path"]
            spacing = (float(row.get("spacing_x", 1.0)), float(row.get("spacing_y", 1.0)), float(row.get("spacing_z", 1.0)))

            try:
                metrics = self.vol_analyzer.analyze_nifti_file(seg_path, spacing=spacing)
                metrics["subject_id"] = subject_id
                vol_records.append(metrics)
            except Exception as e:
                logger.error(f"Error quantifying tumor for {subject_id}: {e}")
                # Fallback zero record
                fallback = {k: 0 for k in ["wt_voxels", "wt_volume_cm3", "tc_voxels", "tc_volume_cm3", "et_voxels", "et_volume_cm3", "ed_voxels", "ed_volume_cm3", "ncr_voxels", "ncr_volume_cm3"]}
                fallback["subject_id"] = subject_id
                vol_records.append(fallback)

        df_vols = pd.DataFrame(vol_records)
        df_merged = pd.merge(df_meta, df_vols, on="subject_id", how="left")

        # Save tumor_statistics.csv
        df_merged.to_csv(self.tumor_stats_path, index=False)
        logger.info(f"Saved master tumor volumetric registry to: {self.tumor_stats_path}")

        # -------------------------------------------------------------
        # 2. Clinical Statistical Analysis & Cohort Summary
        # -------------------------------------------------------------
        logger.info("Phase 2: Performing clinical hypothesis testing and cohort summary...")
        grade_comp = self.stat_analyzer.compare_grades(df_merged)
        correlations = self.stat_analyzer.correlate_clinical_covariates(df_merged)
        df_summary = self.stat_analyzer.build_dataset_summary(df_merged)

        df_summary.to_csv(self.dataset_summary_path, index=False)
        logger.info(f"Saved dataset summary table to: {self.dataset_summary_path}")

        # -------------------------------------------------------------
        # 3. Multi-Sequence Intensity Profiling
        # -------------------------------------------------------------
        logger.info("Phase 3: Sampling non-zero multi-sequence intensity distributions...")
        intensity_samples: Dict[str, List[float]] = {"t1": [], "t1ce": [], "t2": [], "flair": []}
        profile_subjects = df_merged.head(min(30, len(df_merged)))  # Sample representative cohort for density curve

        for _, row in profile_subjects.iterrows():
            for mod in ["t1", "t1ce", "t2", "flair"]:
                path = row.get(f"{mod}_path")
                if path and os.path.isfile(path):
                    try:
                        arr = nib.load(path).get_fdata(dtype=np.float32)
                        nz = arr[arr > 0]
                        if len(nz) > 0:
                            # Subsample for lightweight memory
                            sub = nz[::10]
                            intensity_samples[mod].extend(sub.tolist())
                    except Exception as e:
                        logger.warning(f"Could not extract intensity sample for {row['subject_id']} ({mod}): {e}")

        # Convert to numpy arrays
        intensity_arrays = {m: np.array(v, dtype=np.float32) for m, v in intensity_samples.items()}

        # -------------------------------------------------------------
        # 4. Diagnostic Figures & Orthogonal Slices
        # -------------------------------------------------------------
        saved_figures: Dict[str, str] = {}
        if not skip_figures:
            logger.info("Phase 4: Generating publication-grade diagnostic figures...")
            fig_class = os.path.join(self.figures_dir, "class_distribution.png")
            fig_vols = os.path.join(self.figures_dir, "tumor_volume_distributions.png")
            fig_int = os.path.join(self.figures_dir, "modality_intensity_distributions.png")
            fig_slices = os.path.join(self.figures_dir, "multimodal_mri_slices.png")
            fig_corr = os.path.join(self.figures_dir, "correlation_matrix.png")

            saved_figures["class_distribution"] = self.visualizer.plot_class_distribution(df_merged, fig_class)
            saved_figures["tumor_volume_distributions"] = self.visualizer.plot_tumor_volume_distributions(df_merged, fig_vols)
            saved_figures["modality_intensity_distributions"] = self.visualizer.plot_modality_intensity_distributions(intensity_arrays, fig_int)
            saved_figures["correlation_matrix"] = self.visualizer.plot_correlation_matrix(df_merged, fig_corr)

            # Pick a representative HGG patient with visible enhancing rim
            hgg_samples = df_merged[df_merged["grade"] == "HGG"]
            rep_patient = hgg_samples.iloc[0] if len(hgg_samples) > 0 else df_merged.iloc[0]
            saved_figures["multimodal_mri_slices"] = self.visualizer.plot_multimodal_slices(
                subject_id=rep_patient["subject_id"],
                t1_path=rep_patient["t1_path"],
                t1ce_path=rep_patient["t1ce_path"],
                t2_path=rep_patient["t2_path"],
                flair_path=rep_patient["flair_path"],
                seg_path=rep_patient["seg_path"],
                save_path=fig_slices,
                grade=rep_patient.get("grade", "HGG")
            )

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"Module B execution completed in {elapsed} seconds.")

        # Outlier counts
        micro_count = int(df_merged["is_micro_tumor"].sum()) if "is_micro_tumor" in df_merged.columns else 0
        massive_count = int(df_merged["is_massive_tumor"].sum()) if "is_massive_tumor" in df_merged.columns else 0
        non_enhancing_count = int(df_merged["is_non_enhancing"].sum()) if "is_non_enhancing" in df_merged.columns else 0

        return {
            "total_subjects": len(df_merged),
            "hgg_count": int((df_merged["grade"] == "HGG").sum()),
            "lgg_count": int((df_merged["grade"] == "LGG").sum()),
            "grade_comparison": grade_comp,
            "correlations": correlations,
            "outlier_summary": {
                "micro_tumors": micro_count,
                "massive_tumors": massive_count,
                "non_enhancing_tumors": non_enhancing_count,
            },
            "saved_artifacts": {
                "tumor_statistics_csv": os.path.normpath(self.tumor_stats_path),
                "dataset_summary_csv": os.path.normpath(self.dataset_summary_path),
                "figures": {k: os.path.normpath(v) for k, v in saved_figures.items()},
            },
            "elapsed_seconds": elapsed,
        }
