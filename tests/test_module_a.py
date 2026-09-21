"""
Unit tests for Module A components.
"""
import os
import sys
import unittest
import pandas as pd

# Add learnNeuro to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.module_a_dataset.nifti_utils import get_nifti_metadata, safe_load_header
from src.module_a_dataset.indexer import BraTSIndexer
from src.module_a_dataset.validator import BraTSValidator


class TestModuleA(unittest.TestCase):

    def setUp(self):
        self.config = {
            "dataset": {
                "raw_data_dir": "D:/sgp_dataset_NeuroVision/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
                "name_mapping_file": "name_mapping.csv",
                "survival_info_file": "survival_info.csv",
                "modalities": ["t1", "t1ce", "t2", "flair"],
                "segmentation_modality": "seg",
                "target_column": "Grade",
                "target_classes": {"HGG": 1, "LGG": 0},
                "known_anomalies": {
                    "BraTS20_Training_355": {
                        "seg": "W39_1998.09.19_Segm.nii"
                    }
                }
            }
        }

    def test_nifti_utils_nonexistent_file(self):
        meta = get_nifti_metadata("non_existent_file.nii")
        self.assertFalse(meta["is_valid"])
        self.assertFalse(meta["exists"])
        self.assertIn("File not found", meta["error"])

    def test_indexer_anomaly_resolution(self):
        indexer = BraTSIndexer(self.config)
        # Check that patient 355's seg mask is resolved properly
        patient_355_dir = os.path.join(self.config["dataset"]["raw_data_dir"], "BraTS20_Training_355")
        if os.path.exists(patient_355_dir):
            seg_path = indexer.find_modality_file(patient_355_dir, "BraTS20_Training_355", "seg")
            self.assertIsNotNone(seg_path)
            self.assertTrue(seg_path.endswith("W39_1998.09.19_Segm.nii"))

    def test_validator_with_synthetic_dataframe(self):
        df_mock = pd.DataFrame([
            {
                "subject_id": "BraTS20_Training_001",
                "is_complete": True,
                "t1_path": "path/t1.nii",
                "t1ce_path": "path/t1ce.nii",
                "t2_path": "path/t2.nii",
                "flair_path": "path/flair.nii",
                "seg_path": "path/seg.nii",
                "shape": "(240, 240, 155)",
                "voxel_spacing_mm": "(1.0, 1.0, 1.0)",
                "orientation": "LPS",
                "data_type": "int16",
                "header_valid": True,
                "retrieval_eligible": True,
                "grade": "HGG",
                "label": 1,
                "age": 60.5,
                "survival_days": "289",
            },
            {
                "subject_id": "BraTS20_Training_002",
                "is_complete": True,
                "t1_path": "path/t1.nii",
                "t1ce_path": "path/t1ce.nii",
                "t2_path": "path/t2.nii",
                "flair_path": "path/flair.nii",
                "seg_path": "path/seg.nii",
                "shape": "(240, 240, 155)",
                "voxel_spacing_mm": "(1.0, 1.0, 1.0)",
                "orientation": "LPS",
                "data_type": "int16",
                "header_valid": True,
                "grade": "LGG",
                "label": 0,
                "age": 45.0,
                "survival_days": "600",
            }
        ])
        validator = BraTSValidator(self.config)
        report = validator.validate(df_mock)
        self.assertEqual(report["total_subjects"], 2)
        self.assertEqual(report["complete_subjects"], 2)
        self.assertEqual(report["grade_distribution"]["HGG"], 1)
        self.assertEqual(report["grade_distribution"]["LGG"], 1)
        self.assertTrue(report["all_passed"])


if __name__ == "__main__":
    unittest.main()
