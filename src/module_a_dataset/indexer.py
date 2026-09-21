"""
NeuroVision AI: Dataset Indexer (Module A)
Scans patient directories, indexes 4-channel MRI sequences for the Medical Analysis Agent,
links clinical/demographic ground truth for the Clinical Decision Agent, and prepares
structured case records for the Historical Case Retrieval Database.
"""
import os
import glob # pyright: ignore[reportMissingImports]
from typing import Dict, Any, List, Optional
# pyrefly: ignore [missing-import]
import pandas as pd
from tqdm import tqdm

from src.utils.logger import setup_logger
from src.module_a_dataset.nifti_utils import get_nifti_metadata

logger = setup_logger("NeuroVisionIndexer")


class BraTSIndexer:
    """
    Indexes BraTS 3D MRI dataset into a patient-level structured DataFrame
    tailored for NeuroVision AI's two-agent CDSS pipeline.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("dataset", config)
        self.raw_data_dir = self.config["raw_data_dir"]
        self.modalities = self.config.get("modalities", ["t1", "t1ce", "t2", "flair"])
        self.seg_modality = self.config.get("segmentation_modality", "seg")
        self.target_col = self.config.get("target_column", "Grade")
        self.target_classes = self.config.get("target_classes", {"HGG": 1, "LGG": 0})
        self.known_anomalies = self.config.get("known_anomalies", {})

        self.name_mapping_df = self._load_name_mapping()
        self.survival_df = self._load_survival_info()

    def _load_name_mapping(self) -> Optional[pd.DataFrame]:
        file_name = self.config.get("name_mapping_file", "name_mapping.csv")
        csv_path = os.path.join(self.raw_data_dir, file_name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded name_mapping from {csv_path} with {len(df)} records.")
            return df
        logger.warning(f"name_mapping file not found at {csv_path}")
        return None

    def _load_survival_info(self) -> Optional[pd.DataFrame]:
        file_name = self.config.get("survival_info_file", "survival_info.csv")
        csv_path = os.path.join(self.raw_data_dir, file_name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded survival_info from {csv_path} with {len(df)} records.")
            return df
        logger.warning(f"survival_info file not found at {csv_path}")
        return None

    def find_modality_file(self, patient_dir: str, subject_id: str, modality: str) -> Optional[str]:
        """
        Locate the NIfTI file for a given modality, handling .nii, .nii.gz,
        known dataset anomalies, and fuzzy fallback patterns.
        """
        # 1. Check known anomalies from configuration
        if subject_id in self.known_anomalies and modality in self.known_anomalies[subject_id]:
            anomaly_name = self.known_anomalies[subject_id][modality]
            path = os.path.join(patient_dir, anomaly_name)
            if os.path.exists(path):
                return os.path.normpath(path)

        # 2. Standard convention: <subject_id>_<modality>.nii / .nii.gz
        candidates = [
            os.path.join(patient_dir, f"{subject_id}_{modality}.nii"),
            os.path.join(patient_dir, f"{subject_id}_{modality}.nii.gz"),
            os.path.join(patient_dir, f"{subject_id}_{modality.upper()}.nii"),
            os.path.join(patient_dir, f"{subject_id}_{modality.upper()}.nii.gz"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.normpath(c)

        # 3. Fallback fuzzy search inside patient directory
        pattern_nii = glob.glob(os.path.join(patient_dir, f"*{modality}*.nii*"))
        if pattern_nii:
            return os.path.normpath(pattern_nii[0])

        return None

    def scan_dataset(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Scan all patient folders, link modalities and labels, and extract spatial metadata.
        Explicitly constructs records for both Agent 1 (Medical Analysis) and Agent 2 (Clinical Decision).
        """
        if not os.path.exists(self.raw_data_dir):
            raise FileNotFoundError(f"Raw data directory does not exist: {self.raw_data_dir}")

        all_entries = sorted(os.listdir(self.raw_data_dir))
        subject_dirs = [
            d for d in all_entries
            if os.path.isdir(os.path.join(self.raw_data_dir, d)) and not d.startswith(".")
        ]

        logger.info(f"Discovered {len(subject_dirs)} subject directories in {self.raw_data_dir}")
        if limit is not None and limit > 0:
            subject_dirs = subject_dirs[:limit]
            logger.info(f"Limiting execution to first {limit} subjects.")

        records = []
        for subject_id in tqdm(subject_dirs, desc="Indexing NeuroVision Patients"):
            patient_path = os.path.join(self.raw_data_dir, subject_id)
            
            # --- AGENT 1 (Medical Analysis Agent) Inputs: 4 MRI Modalities + Segmentation ---
            modality_paths = {}
            missing_modalities = []
            for mod in self.modalities:
                path = self.find_modality_file(patient_path, subject_id, mod)
                modality_paths[f"{mod}_path"] = path
                if path is None:
                    missing_modalities.append(mod)

            seg_path = self.find_modality_file(patient_path, subject_id, self.seg_modality)
            modality_paths["seg_path"] = seg_path
            if seg_path is None:
                missing_modalities.append(self.seg_modality)

            # Spatial metadata from reference volume (FLAIR or T1)
            ref_path = modality_paths.get("flair_path") or modality_paths.get("t1_path")
            spatial_meta = {}
            if ref_path and os.path.exists(ref_path):
                meta = get_nifti_metadata(ref_path)
                spatial_meta["shape"] = str(meta["shape"])
                spatial_meta["shape_x"] = meta["shape"][0] if meta["shape"] else None
                spatial_meta["shape_y"] = meta["shape"][1] if meta["shape"] else None
                spatial_meta["shape_z"] = meta["shape"][2] if meta["shape"] else None
                spatial_meta["voxel_spacing_mm"] = str(meta["voxel_spacing"])
                spatial_meta["spacing_x"] = meta["voxel_spacing"][0] if meta["voxel_spacing"] else None
                spatial_meta["spacing_y"] = meta["voxel_spacing"][1] if meta["voxel_spacing"] else None
                spatial_meta["spacing_z"] = meta["voxel_spacing"][2] if meta["voxel_spacing"] else None
                spatial_meta["orientation"] = meta["orientation"]
                spatial_meta["data_type"] = meta["data_type"]
                spatial_meta["header_valid"] = meta["is_valid"]
            else:
                spatial_meta["shape"] = None
                spatial_meta["shape_x"] = None
                spatial_meta["shape_y"] = None
                spatial_meta["shape_z"] = None
                spatial_meta["voxel_spacing_mm"] = None
                spatial_meta["spacing_x"] = None
                spatial_meta["spacing_y"] = None
                spatial_meta["spacing_z"] = None
                spatial_meta["orientation"] = None
                spatial_meta["data_type"] = None
                spatial_meta["header_valid"] = False

            # --- AGENT 2 (Clinical Decision Agent) Inputs: Ground Truth Grade & Clinical Covariates ---
            grade = None
            label = None
            tcia_id = None
            brats_2017_id = None
            brats_2018_id = None
            brats_2019_id = None

            if self.name_mapping_df is not None:
                match = self.name_mapping_df[
                    self.name_mapping_df["BraTS_2020_subject_ID"] == subject_id
                ]
                if not match.empty:
                    row = match.iloc[0]
                    grade = str(row.get("Grade", "UNKNOWN")).strip()
                    label = self.target_classes.get(grade, -1)
                    tcia_id = str(row.get("TCGA_TCIA_subject_ID", ""))
                    brats_2017_id = str(row.get("BraTS_2017_subject_ID", ""))
                    brats_2018_id = str(row.get("BraTS_2018_subject_ID", ""))
                    brats_2019_id = str(row.get("BraTS_2019_subject_ID", ""))

            # Demographics and outcomes
            age = None
            survival_days = None
            resection = None

            if self.survival_df is not None:
                s_match = self.survival_df[self.survival_df["Brats20ID"] == subject_id]
                if not s_match.empty:
                    s_row = s_match.iloc[0]
                    age = s_row.get("Age", None)
                    survival_days = s_row.get("Survival_days", None)
                    resection = s_row.get("Extent_of_Resection", None)

            is_imaging_complete = len(missing_modalities) == 0
            has_clinical_history = pd.notna(age) and pd.notna(survival_days)

            record = {
                # --- General System Identification ---
                "subject_id": subject_id,
                "patient_dir": os.path.normpath(patient_path),
                
                # --- Agent 1 (Medical Analysis Agent) Inputs ---
                **modality_paths,
                "missing_modalities": ",".join(missing_modalities) if missing_modalities else "None",
                "is_complete": is_imaging_complete,
                **spatial_meta,
                
                # --- Agent 2 (Clinical Decision Agent) Inputs ---
                "grade": grade,
                "label": label,
                "age": age,
                "survival_days": survival_days,
                "extent_of_resection": resection,
                "has_clinical_history": bool(has_clinical_history),
                
                # --- Case-Based Retrieval (FAISS / Knowledge Base Registry) ---
                "retrieval_eligible": bool(is_imaging_complete and (label is not None and label in [0, 1])),
                "tcia_id": tcia_id,
                "brats_2019_id": brats_2019_id,
                "brats_2018_id": brats_2018_id,
                "brats_2017_id": brats_2017_id,
            }
            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Indexing complete. Indexed {len(df)} subjects. Complete records: {df['is_complete'].sum()}/{len(df)}.")
        return df
