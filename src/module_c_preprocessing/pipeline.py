"""
NeuroVision AI: Module C - End-to-End Preprocessing & Dataset Splitting Pipeline
Orchestrates patient-level splitting, zero-leakage enforcement, and sample tensor caching.
"""
import os
import time
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from src.utils.logger import setup_logger
from src.module_c_preprocessing.splitter import PatientDataSplitter
from src.module_c_preprocessing.preprocessor import MRIPreprocessor

logger = setup_logger("PreprocessingPipeline")


class PreprocessingPipeline:
    """
    High-level orchestrator for Module C.
    
    Responsibilities:
      1. Load master metadata from Module A (dataset_metadata.csv).
      2. Partition all subjects into zero-leakage train/val/test splits.
      3. Export stratified split CSVs and summary JSON.
      4. Optionally preprocess and cache tensor volumes for verification or fast training.
      5. Generate an execution audit report.
    """

    def __init__(self, config: Dict[str, Any], base_dir: str = "."):
        """
        Initialize the pipeline.
        
        Args:
            config: Full configuration dictionary.
            base_dir: Project root directory.
        """
        self.config = config
        self.base_dir = base_dir
        self.splitter = PatientDataSplitter(config)
        self.preprocessor = MRIPreprocessor(config)

        # Paths
        inputs_cfg = config.get("inputs", {})
        self.metadata_csv = os.path.join(base_dir, inputs_cfg.get("metadata_csv", "data/dataset_metadata.csv"))

        caching_cfg = config.get("caching", {})
        self.cache_dir = os.path.join(base_dir, caching_cfg.get("output_dir", "data/processed"))
        self.save_format = caching_cfg.get("save_format", "npz")

    def run(
        self,
        cache_samples: int = 0,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute the full Module C pipeline.
        
        Args:
            cache_samples: Number of subjects to preprocess and cache to disk (0 = split only).
            limit: Optional limit on total metadata rows for testing.
            
        Returns:
            Dictionary containing split statistics, cached paths, and execution metrics.
        """
        start_time = time.time()
        logger.info("Initializing Module C: MRI Preprocessing & Patient-Level Splitting...")

        if not os.path.isfile(self.metadata_csv):
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_csv}")

        df = pd.read_csv(self.metadata_csv)
        logger.info(f"Loaded master metadata with {len(df)} patient records.")

        if limit is not None and limit > 0:
            logger.info(f"Applying limit: restricting dataset to first {limit} records.")
            df = df.head(limit).copy()

        # Step 1: Stratified Patient-Level Splitting
        logger.info("Partitioning subjects into Train (70%), Val (15%), and Test (15%) splits...")
        train_df, val_df, test_df = self.splitter.split_dataframe(df)

        split_summary = self.splitter.export_splits(
            train_df, val_df, test_df, base_dir=self.base_dir
        )
        logger.info(f"Exported stratified split tables to: {os.path.join(self.base_dir, self.splitter.output_dir)}")

        # Step 2: Optional Preprocessing & Sample Tensor Caching
        cached_files: List[str] = []
        sample_stats: List[Dict[str, Any]] = []

        if cache_samples > 0:
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Preprocessing and caching {cache_samples} sample volumes to: {self.cache_dir}")

            # Sample evenly across train, val, and test
            sample_subjs = []
            for split_name, s_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
                take = max(1, cache_samples // 3)
                sample_subjs.extend(s_df.head(take).to_dict("records"))

            sample_subjs = sample_subjs[:cache_samples]

            for s_row in sample_subjs:
                s_id = s_row["subject_id"]
                try:
                    logger.info(f"Processing subject: {s_id} (Grade: {s_row.get('grade')})...")
                    processed = self.preprocessor.process_subject_row(s_row)

                    cache_path = os.path.join(self.cache_dir, f"{s_id}.npz")
                    np.savez_compressed(
                        cache_path,
                        image=processed["image"],
                        mask=processed["mask"] if processed["mask"] is not None else np.array([]),
                        grade=str(processed.get("grade", "")),
                        label=int(processed.get("label", -1))
                    )
                    cached_files.append(cache_path)

                    # Compute non-zero stats of processed T1ce and FLAIR for audit
                    t1ce_fg = processed["image"][1][processed["image"][1] != 0]
                    flair_fg = processed["image"][3][processed["image"][3] != 0]

                    sample_stats.append({
                        "subject_id": s_id,
                        "tensor_shape": list(processed["image"].shape),
                        "t1ce_mean": float(np.mean(t1ce_fg)) if len(t1ce_fg) > 0 else 0.0,
                        "t1ce_std": float(np.std(t1ce_fg)) if len(t1ce_fg) > 0 else 0.0,
                        "flair_mean": float(np.mean(flair_fg)) if len(flair_fg) > 0 else 0.0,
                        "flair_std": float(np.std(flair_fg)) if len(flair_fg) > 0 else 0.0,
                        "cache_path": cache_path
                    })
                    logger.info(f"  -> Successfully cached {s_id} (Shape: {processed['image'].shape})")
                except Exception as e:
                    logger.error(f"Error processing subject {s_id}: {e}")

        elapsed_seconds = round(time.time() - start_time, 2)
        logger.info(f"Module C completed in {elapsed_seconds} seconds.")

        return {
            "total_subjects": len(df),
            "split_summary": split_summary,
            "cached_files": cached_files,
            "sample_stats": sample_stats,
            "elapsed_seconds": elapsed_seconds
        }
