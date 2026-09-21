"""
NeuroVision AI: Dataset Validation and Data Schema Generator (Module A)
Validates imaging completeness, checks clinical covariate integrity,
and generates the data dictionary structured for the Two-Agent CDSS.
"""
import os
import json
from typing import Dict, Any
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("NeuroVisionValidator")


class BraTSValidator:
    """
    Validates indexed BraTS dataset and generates formal data dictionary and summary reports
    for the NeuroVision AI Two-Agent Clinical Decision Support System.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_config = config.get("dataset", config)
        self.val_config = config.get("validation", {})
        self.modalities = self.dataset_config.get("modalities", ["t1", "t1ce", "t2", "flair"])

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute full validation checks on the indexed DataFrame.
        """
        total_subjects = len(df)
        if total_subjects == 0:
            raise ValueError("Dataset DataFrame is empty! No subjects indexed.")

        complete_subjects = int(df["is_complete"].sum())
        incomplete_subjects = total_subjects - complete_subjects
        retrieval_eligible_count = int(df["retrieval_eligible"].sum()) if "retrieval_eligible" in df.columns else complete_subjects

        # Check missing modalities across subjects
        missing_counts = {}
        for mod in self.modalities + [self.dataset_config.get("segmentation_modality", "seg")]:
            col = f"{mod}_path"
            if col in df.columns:
                missing_counts[mod] = int(df[col].isna().sum())

        # Check shapes and spacings
        unique_shapes = df["shape"].dropna().unique().tolist()
        unique_spacings = df["voxel_spacing_mm"].dropna().unique().tolist()
        unique_orientations = df["orientation"].dropna().unique().tolist()

        # Class distribution
        grade_counts = df["grade"].value_counts().to_dict()
        label_counts = df["label"].value_counts().to_dict()

        hgg_count = grade_counts.get("HGG", 0)
        lgg_count = grade_counts.get("LGG", 0)
        imbalance_ratio = round(hgg_count / lgg_count, 2) if lgg_count > 0 else None

        # Survival & demographic stats
        valid_ages = df["age"].dropna().astype(float)
        age_stats = {
            "mean": round(float(valid_ages.mean()), 2) if len(valid_ages) > 0 else None,
            "min": round(float(valid_ages.min()), 2) if len(valid_ages) > 0 else None,
            "max": round(float(valid_ages.max()), 2) if len(valid_ages) > 0 else None,
            "available_records": len(valid_ages),
        }

        report = {
            "total_subjects": total_subjects,
            "complete_subjects": complete_subjects,
            "incomplete_subjects": incomplete_subjects,
            "retrieval_eligible_subjects": retrieval_eligible_count,
            "missing_modality_counts": missing_counts,
            "unique_shapes": unique_shapes,
            "unique_spacings": unique_spacings,
            "unique_orientations": unique_orientations,
            "grade_distribution": grade_counts,
            "label_distribution": label_counts,
            "hgg_to_lgg_ratio": f"{imbalance_ratio}:1" if imbalance_ratio else "N/A",
            "age_statistics": age_stats,
            "all_passed": (incomplete_subjects == 0) and (df["header_valid"].all()),
        }
        return report

    def generate_data_dictionary(self) -> Dict[str, Any]:
        """
        Creates a structured data dictionary defining all fields in dataset_metadata.csv,
        explicitly categorizing them for the Medical Analysis Agent and Clinical Decision Agent.
        """
        data_dict = {
            "project": "NeuroVision AI – Autonomous MRI Clinical Decision Support System",
            "dataset_name": "BraTS 2020 Multimodal Brain Tumor Dataset",
            "clinical_decision_target": {
                "field": "label",
                "derived_from": "grade",
                "encoding": {
                    "0": "Low-Grade Glioma (LGG) - WHO Grade I/II",
                    "1": "High-Grade Glioma (HGG) - WHO Grade III/IV"
                }
            },
            "subsystem_architecture": {
                "agent_1_medical_analysis": "Responsible for 3D MRI feature extraction, tumor ROI segmentation, volumetric computation, Grad-CAM visual heatmaps, and multimodal vector embeddings.",
                "agent_2_clinical_decision": "Responsible for integrating MRI findings with patient demographics, querying medical literature via RAG, multi-agent clinical reasoning (LangGraph), and generating evidence-supported clinical reports."
            },
            "columns": {
                "system_identifiers": {
                    "subject_id": {"type": "string", "description": "Unique patient identifier in BraTS 2020 (e.g. BraTS20_Training_001)"},
                    "patient_dir": {"type": "string", "description": "Filesystem directory path containing the patient's MRI sequences"}
                },
                "agent_1_imaging_inputs": {
                    "t1_path": {"type": "string", "description": "Path to NIfTI T1-weighted native MRI (anatomical structure)"},
                    "t1ce_path": {"type": "string", "description": "Path to NIfTI T1ce MRI (Gadolinium contrast, vascular tumor core)"},
                    "t2_path": {"type": "string", "description": "Path to NIfTI T2-weighted MRI (fluid-sensitive, peritumoral edema)"},
                    "flair_path": {"type": "string", "description": "Path to NIfTI FLAIR MRI (CSF-attenuated edema delineation)"},
                    "seg_path": {"type": "string", "description": "Path to ground truth segmentation mask (labels: 0, 1, 2, 4)"},
                    "is_complete": {"type": "boolean", "description": "True if all 4 MRI sequences and segmentation mask exist and are valid"},
                    "shape": {"type": "string", "description": "3D matrix dimensions (X, Y, Z voxels), standard: (240, 240, 155)"},
                    "voxel_spacing_mm": {"type": "string", "description": "Physical voxel resolution in mm, standard: (1.0, 1.0, 1.0) isotropic"},
                    "orientation": {"type": "string", "description": "Anatomical coordinate orientation (standard: LPS - Left, Posterior, Superior)"},
                    "data_type": {"type": "string", "description": "Header voxel intensity storage type (e.g., int16)"}
                },
                "agent_2_clinical_inputs": {
                    "grade": {"type": "string", "description": "Histopathological diagnosis: 'HGG' (High-Grade Glioma) or 'LGG' (Low-Grade Glioma)"},
                    "label": {"type": "integer", "description": "Binary integer target: 1 for HGG, 0 for LGG"},
                    "age": {"type": "float", "description": "Patient age at diagnosis in years (where documented)"},
                    "survival_days": {"type": "string/float", "description": "Overall survival duration in days from diagnosis"},
                    "extent_of_resection": {"type": "string", "description": "Surgical resection status (e.g., GTR = Gross Total Resection, STR = Subtotal Resection)"},
                    "has_clinical_history": {"type": "boolean", "description": "True if patient has documented clinical survival/demographic covariates"}
                },
                "case_retrieval_registry": {
                    "retrieval_eligible": {"type": "boolean", "description": "True if patient has complete imaging and confirmed grade, qualifying for FAISS similarity indexing"},
                    "tcia_id": {"type": "string", "description": "Corresponding Subject ID in The Cancer Imaging Archive (TCIA/TCGA)"},
                    "brats_2019_id": {"type": "string", "description": "Subject identifier in BraTS 2019 challenge cohort"},
                    "brats_2018_id": {"type": "string", "description": "Subject identifier in BraTS 2018 challenge cohort"},
                    "brats_2017_id": {"type": "string", "description": "Subject identifier in BraTS 2017 challenge cohort"}
                }
            }
        }
        return data_dict

    def export_artifacts(self, df: pd.DataFrame, base_dir: str = ".") -> Dict[str, str]:
        """
        Export dataset_metadata.csv and data_dictionary.json.
        """
        output_dir = os.path.join(base_dir, self.dataset_config.get("output_dir", "data"))
        os.makedirs(output_dir, exist_ok=True)

        metadata_csv_path = os.path.join(
            base_dir, self.dataset_config.get("metadata_csv", "data/dataset_metadata.csv")
        )
        data_dict_json_path = os.path.join(
            base_dir, self.dataset_config.get("data_dictionary_json", "data/data_dictionary.json")
        )

        # Save metadata CSV
        df.to_csv(metadata_csv_path, index=False)
        logger.info(f"Saved dataset metadata ({len(df)} rows) to: {metadata_csv_path}")

        # Save data dictionary JSON
        data_dict = self.generate_data_dictionary()
        with open(data_dict_json_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=2)
        logger.info(f"Saved data dictionary to: {data_dict_json_path}")

        return {
            "metadata_csv": os.path.normpath(metadata_csv_path),
            "data_dictionary_json": os.path.normpath(data_dict_json_path),
        }
