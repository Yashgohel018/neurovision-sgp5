"""
NeuroVision AI: Module C - Patient-Level Stratified Dataset Splitter
Guarantees zero data leakage by strictly partitioning data at the patient level
with exact class-stratification across High-Grade Glioma (HGG) and Low-Grade Glioma (LGG).
"""
import os
import json
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np


class PatientDataSplitter:
    """
    Executes patient-level stratified train / validation / test partitioning.
    
    CRITICAL MEDICAL AI PRINCIPLE:
    In medical imaging, splitting 2D slices or patches randomly across train and test
    causes catastrophic data leakage (spurious generalization), as slices from the same
    patient share identical anatomy, scanner noise, and pathology.
    This splitter guarantees that every patient ID is uniquely assigned to a single split.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the splitter with configuration parameters.
        
        Args:
            config: Optional configuration dictionary.
        """
        split_cfg = config.get("splitting", {}) if config else {}
        self.seed = int(split_cfg.get("seed", 42))
        self.train_ratio = float(split_cfg.get("train_ratio", 0.70))
        self.val_ratio = float(split_cfg.get("val_ratio", 0.15))
        self.test_ratio = float(split_cfg.get("test_ratio", 0.15))
        self.stratify_column = split_cfg.get("stratify_column", "grade")
        self.output_dir = split_cfg.get("output_dir", "data/splits")

        # Validate ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-3):
            raise ValueError(f"Split ratios must sum to 1.0, got: {total_ratio}")

    def split_dataframe(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Partition dataframe into train, val, and test subsets with stratification.
        
        Args:
            df: Master metadata dataframe containing 'subject_id' and stratify_column.
            
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        rng = np.random.RandomState(self.seed)

        if self.stratify_column not in df.columns:
            raise ValueError(f"Stratification column '{self.stratify_column}' not found in dataframe.")

        train_indices: List[int] = []
        val_indices: List[int] = []
        test_indices: List[int] = []

        # Group by stratify class (e.g. HGG, LGG)
        for class_label, group in df.groupby(self.stratify_column):
            indices = group.index.values.copy()
            rng.shuffle(indices)

            n_total = len(indices)
            n_train = int(np.round(n_total * self.train_ratio))
            n_val = int(np.round(n_total * self.val_ratio))
            # Test gets remaining to ensure exact sum
            n_test = n_total - n_train - n_val

            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]

            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
            test_indices.extend(test_idx)

        train_df = df.loc[train_indices].copy().reset_index(drop=True)
        val_df = df.loc[val_indices].copy().reset_index(drop=True)
        test_df = df.loc[test_indices].copy().reset_index(drop=True)

        # Verify zero leakage
        self.verify_zero_leakage(train_df, val_df, test_df)

        return train_df, val_df, test_df

    @staticmethod
    def verify_zero_leakage(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> bool:
        """
        Enforces that subject sets are strictly disjoint.
        
        Raises:
            ValueError: If any patient ID exists in more than one partition.
        """
        train_ids = set(train_df["subject_id"])
        val_ids = set(val_df["subject_id"])
        test_ids = set(test_df["subject_id"])

        train_val_leak = train_ids.intersection(val_ids)
        train_test_leak = train_ids.intersection(test_ids)
        val_test_leak = val_ids.intersection(test_ids)

        if train_val_leak:
            raise ValueError(f"DATA LEAKAGE DETECTED between Train and Val: {len(train_val_leak)} subjects: {train_val_leak}")
        if train_test_leak:
            raise ValueError(f"DATA LEAKAGE DETECTED between Train and Test: {len(train_test_leak)} subjects: {train_test_leak}")
        if val_test_leak:
            raise ValueError(f"DATA LEAKAGE DETECTED between Val and Test: {len(val_test_leak)} subjects: {val_test_leak}")

        return True

    def export_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        base_dir: str = "."
    ) -> Dict[str, Any]:
        """
        Export train/val/test CSVs and structured JSON summary to output_dir.
        
        Args:
            train_df: Training set dataframe.
            val_df: Validation set dataframe.
            test_df: Test set dataframe.
            base_dir: Base directory path.
            
        Returns:
            Dictionary containing paths and distribution statistics.
        """
        out_path = os.path.join(base_dir, self.output_dir)
        os.makedirs(out_path, exist_ok=True)

        train_csv = os.path.join(out_path, "train_subjects.csv")
        val_csv = os.path.join(out_path, "val_subjects.csv")
        test_csv = os.path.join(out_path, "test_subjects.csv")
        summary_json = os.path.join(out_path, "split_summary.json")

        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)

        def get_class_dist(subset_df: pd.DataFrame) -> Dict[str, Any]:
            counts = subset_df[self.stratify_column].value_counts().to_dict()
            total = len(subset_df)
            proportions = {k: round(v / total * 100, 2) for k, v in counts.items()} if total > 0 else {}
            return {
                "total": total,
                "counts": counts,
                "percentages": proportions,
                "subjects": subset_df["subject_id"].tolist()
            }

        summary = {
            "random_seed": self.seed,
            "ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio
            },
            "stratify_column": self.stratify_column,
            "splits": {
                "train": get_class_dist(train_df),
                "val": get_class_dist(val_df),
                "test": get_class_dist(test_df)
            },
            "total_patients": len(train_df) + len(val_df) + len(test_df),
            "files": {
                "train_csv": train_csv,
                "val_csv": val_csv,
                "test_csv": test_csv,
                "summary_json": summary_json
            }
        }

        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
