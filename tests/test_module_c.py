"""
NeuroVision AI: Automated Unit Tests for Module C
Tests MRI intensity normalization, foreground bounding box extraction,
3D cropping/padding, discrete mask resampling, and zero-leakage patient-level splitting.
"""
import unittest
import numpy as np
import pandas as pd

from src.module_c_preprocessing.normalizer import IntensityNormalizer
from src.module_c_preprocessing.cropper import SpatialCropper
from src.module_c_preprocessing.splitter import PatientDataSplitter
from src.module_c_preprocessing.preprocessor import MRIPreprocessor


class TestModuleC(unittest.TestCase):

    def setUp(self):
        self.config = {
            "splitting": {
                "seed": 42,
                "train_ratio": 0.70,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "stratify_column": "grade",
                "output_dir": "data/splits"
            },
            "normalization": {
                "method": "z_score",
                "nonzero_only": True,
                "clip_percentiles": [1.0, 99.0],
                "epsilon": 1e-7
            },
            "spatial": {
                "target_shape": [64, 64, 64],
                "crop_strategy": "foreground_bbox",
                "pad_value": 0.0
            },
            "multimodal": {
                "modalities": ["t1", "t1ce", "t2", "flair"],
                "include_seg": True
            }
        }
        self.normalizer = IntensityNormalizer(self.config)
        self.cropper = SpatialCropper(self.config)
        self.splitter = PatientDataSplitter(self.config)
        self.preprocessor = MRIPreprocessor(self.config)

    def test_non_zero_z_score_normalization(self):
        """
        Verify that non-zero voxels are standardized to mean=0, std=1,
        and background air voxels (0) remain strictly zero.
        """
        vol = np.zeros((30, 30, 30), dtype=np.float32)
        # Create synthetic brain foreground with random Gaussian values
        np.random.seed(42)
        fg_indices = (slice(5, 25), slice(5, 25), slice(5, 25))
        vol[fg_indices] = np.random.normal(loc=150.0, scale=30.0, size=(20, 20, 20))

        norm_vol = self.normalizer.normalize(vol, method="z_score")

        # Background check
        bg_mask = (vol == 0)
        self.assertTrue(np.all(norm_vol[bg_mask] == 0.0), "Background air voxels must remain 0.0")

        # Foreground check
        fg_mask = ~bg_mask
        fg_vals = norm_vol[fg_mask]
        self.assertAlmostEqual(float(np.mean(fg_vals)), 0.0, places=2)
        self.assertAlmostEqual(float(np.std(fg_vals)), 1.0, places=2)

    def test_percentile_clipping(self):
        """
        Verify that extreme scanner artifact spikes are clamped by percentile thresholds.
        """
        vol = np.zeros((20, 20, 20), dtype=np.float32)
        vol[5:15, 5:15, 5:15] = 100.0
        # Add extreme scanner spike
        vol[10, 10, 10] = 50000.0

        norm_vol = self.normalizer.normalize(vol, method="z_score")
        fg_vals = norm_vol[vol > 0]
        # Max value should not be astronomical because 50000 is clipped to ~P99
        self.assertLess(float(np.max(fg_vals)), 10.0, "Percentile clipping failed to suppress spike")

    def test_min_max_normalization(self):
        """
        Verify min-max normalization scales non-zero intensities to [0, 1].
        """
        vol = np.zeros((20, 20, 20), dtype=np.float32)
        vol[5:15, 5:15, 5:15] = np.linspace(10, 200, 1000).reshape(10, 10, 10)

        norm_vol = self.normalizer.normalize(vol, method="min_max")
        fg_vals = norm_vol[vol > 0]

        self.assertAlmostEqual(float(np.min(fg_vals)), 0.0, places=2)
        self.assertAlmostEqual(float(np.max(fg_vals)), 1.0, places=2)

    def test_foreground_bbox_extraction(self):
        """
        Verify that bounding box extractor accurately identifies spatial boundaries.
        """
        vol = np.zeros((50, 50, 50), dtype=np.float32)
        vol[10:30, 15:35, 20:45] = 100.0  # z: 10-29, y: 15-34, x: 20-44

        bbox = SpatialCropper.find_foreground_bbox(vol)
        self.assertEqual(bbox[0], (10, 29))
        self.assertEqual(bbox[1], (15, 34))
        self.assertEqual(bbox[2], (20, 44))

    def test_crop_or_pad_3d(self):
        """
        Verify that volumes can be padded or cropped to an exact target shape.
        """
        # Test padding smaller volume
        small_vol = np.ones((20, 30, 40), dtype=np.float32)
        padded = self.cropper.crop_or_pad_3d(small_vol, target_shape=(50, 50, 50))
        self.assertEqual(padded.shape, (50, 50, 50))

        # Test cropping larger volume
        large_vol = np.ones((80, 70, 90), dtype=np.float32)
        cropped = self.cropper.crop_or_pad_3d(large_vol, target_shape=(50, 50, 50))
        self.assertEqual(cropped.shape, (50, 50, 50))

        # Test 4D tensor
        vol_4d = np.ones((4, 30, 40, 50), dtype=np.float32)
        res_4d = self.cropper.crop_or_pad_3d(vol_4d, target_shape=(64, 64, 64))
        self.assertEqual(res_4d.shape, (4, 64, 64, 64))

    def test_mask_resampling_preserves_discrete_labels(self):
        """
        Verify nearest-neighbor interpolation preserves discrete class labels {0, 1, 2, 4}
        without generating spurious intermediate floats.
        """
        mask = np.zeros((30, 30, 30), dtype=np.uint8)
        mask[5:10, 5:10, 5:10] = 1
        mask[10:15, 10:15, 10:15] = 2
        mask[15:20, 15:20, 15:20] = 4

        resampled = SpatialCropper.resample_3d(mask, target_shape=(45, 45, 45), is_mask=True)
        unique_labels = set(np.unique(resampled))
        self.assertTrue(unique_labels.issubset({0, 1, 2, 4}), f"Unexpected interpolated labels: {unique_labels}")
        self.assertEqual(resampled.dtype, np.uint8)

    def test_patient_level_splitting_zero_leakage(self):
        """
        Verify that patient splitting produces disjoint sets and exact stratification.
        """
        # Create mock cohort of 100 patients
        subjects = [f"Patient_{i:03d}" for i in range(100)]
        # 70 HGG, 30 LGG
        grades = ["HGG"] * 70 + ["LGG"] * 30
        df = pd.DataFrame({"subject_id": subjects, "grade": grades})

        train_df, val_df, test_df = self.splitter.split_dataframe(df)

        # Ratio checks (70 / 15 / 15 with discrete strata rounding)
        self.assertEqual(len(train_df), 70)
        self.assertAlmostEqual(len(val_df), 15, delta=1)
        self.assertAlmostEqual(len(test_df), 15, delta=1)
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), 100)

        # Zero-leakage disjointness check
        train_ids = set(train_df["subject_id"])
        val_ids = set(val_df["subject_id"])
        test_ids = set(test_df["subject_id"])

        self.assertEqual(len(train_ids.intersection(val_ids)), 0)
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)
        self.assertEqual(len(val_ids.intersection(test_ids)), 0)

        # Stratification ratio check (approx 70% HGG in each)
        hgg_train_pct = (train_df["grade"] == "HGG").mean()
        hgg_val_pct = (val_df["grade"] == "HGG").mean()
        hgg_test_pct = (test_df["grade"] == "HGG").mean()

        self.assertAlmostEqual(hgg_train_pct, 0.70, delta=0.05)
        self.assertAlmostEqual(hgg_val_pct, 0.70, delta=0.05)
        self.assertAlmostEqual(hgg_test_pct, 0.70, delta=0.05)

    def test_preprocessor_synthetic_pipeline(self):
        """
        Verify end-to-end processing of synthetic 4-channel volume and mask.
        """
        d, h, w = (40, 40, 40)
        modalities = {
            "t1": np.random.uniform(10, 100, (d, h, w)).astype(np.float32),
            "t1ce": np.random.uniform(10, 150, (d, h, w)).astype(np.float32),
            "t2": np.random.uniform(10, 120, (d, h, w)).astype(np.float32),
            "flair": np.random.uniform(10, 110, (d, h, w)).astype(np.float32),
        }
        mask = np.zeros((d, h, w), dtype=np.uint8)
        mask[15:25, 15:25, 15:25] = 4

        result = self.preprocessor.process_arrays(modalities, mask, subject_id="synthetic_01")

        self.assertEqual(result["image"].shape, (4, 64, 64, 64))
        self.assertEqual(result["mask"].shape, (64, 64, 64))
        self.assertEqual(result["image"].dtype, np.float32)
        self.assertEqual(result["mask"].dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
