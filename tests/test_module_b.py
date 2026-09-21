"""
NeuroVision AI: Automated Unit Tests for Module B
Tests volumetric quantification, bounding box extraction, statistical comparison,
and intensity profiling using synthetic test fixtures.
"""
import unittest
import numpy as np
import pandas as pd

from src.module_b_eda.volumetrics import TumorVolumetricAnalyzer
from src.module_b_eda.statistical_analyzer import StatisticalAnalyzer
from src.module_b_eda.intensity_profiler import IntensityProfiler


class TestModuleB(unittest.TestCase):

    def setUp(self):
        self.config = {
            "outliers": {
                "micro_tumor_threshold_cm3": 5.0,
                "massive_tumor_threshold_cm3": 150.0,
                "zero_et_threshold_cm3": 0.0,
            },
            "intensity_profiling": {
                "modalities": ["t1", "t1ce", "t2", "flair"],
                "percentiles": [1, 25, 50, 75, 99],
            }
        }
        self.analyzer = TumorVolumetricAnalyzer(self.config)

    def test_synthetic_mask_volumetrics(self):
        """
        Verify that exact known voxel counts correctly convert to physical cm³ volumes.
        """
        # Create a 20x20x20 volume (8,000 voxels)
        mask = np.zeros((20, 20, 20), dtype=np.int16)

        # Assign known voxel regions:
        # 100 voxels of NCR (1)
        # 200 voxels of ED (2)
        # 300 voxels of ET (4)
        mask[2:4, 2:7, 2:12] = 1   # 2 * 5 * 10 = 100
        mask[5:7, 5:15, 5:15] = 2  # 2 * 10 * 10 = 200
        mask[8:11, 8:18, 8:18] = 4 # 3 * 10 * 10 = 300

        # Spacing (1.0, 1.0, 1.0) -> voxel volume = 0.001 cm³
        metrics = self.analyzer.analyze_mask_array(mask, spacing=(1.0, 1.0, 1.0))

        self.assertEqual(metrics["ncr_voxels"], 100)
        self.assertEqual(metrics["ed_voxels"], 200)
        self.assertEqual(metrics["et_voxels"], 300)
        self.assertEqual(metrics["tc_voxels"], 400) # NCR (100) + ET (300)
        self.assertEqual(metrics["wt_voxels"], 600) # TC (400) + ED (200)

        self.assertAlmostEqual(metrics["ncr_volume_cm3"], 0.1, places=3)
        self.assertAlmostEqual(metrics["ed_volume_cm3"], 0.2, places=3)
        self.assertAlmostEqual(metrics["et_volume_cm3"], 0.3, places=3)
        self.assertAlmostEqual(metrics["tc_volume_cm3"], 0.4, places=3)
        self.assertAlmostEqual(metrics["wt_volume_cm3"], 0.6, places=3)

        # Ratios
        self.assertAlmostEqual(metrics["et_to_wt_ratio"], 300 / 600, places=3)
        self.assertAlmostEqual(metrics["tc_to_wt_ratio"], 400 / 600, places=3)
        self.assertAlmostEqual(metrics["ed_to_wt_ratio"], 200 / 600, places=3)

        # Outlier flags: 0.6 cm³ is < 5.0 cm³ -> micro tumor
        self.assertTrue(metrics["is_micro_tumor"])
        self.assertFalse(metrics["is_massive_tumor"])
        self.assertFalse(metrics["is_non_enhancing"])

    def test_non_enhancing_tumor_flag(self):
        """
        Verify that a tumor without Enhancing Tumor (ET=0) is correctly flagged.
        """
        mask = np.zeros((10, 10, 10), dtype=np.int16)
        mask[1:5, 1:5, 1:5] = 2  # Edema only, no ET
        metrics = self.analyzer.analyze_mask_array(mask)

        self.assertEqual(metrics["et_voxels"], 0)
        self.assertEqual(metrics["et_volume_cm3"], 0.0)
        self.assertTrue(metrics["is_non_enhancing"])
        self.assertIn("non_enhancing", metrics["outlier_flags"])

    def test_bounding_box_extraction(self):
        """
        Verify 3D bounding box coordinates and centroid calculation.
        """
        mask = np.zeros((30, 30, 30), dtype=np.int16)
        mask[10:16, 5:15, 20:25] = 4 # ET

        metrics = self.analyzer.analyze_mask_array(mask)
        self.assertEqual(metrics["bbox_min_x"], 10)
        self.assertEqual(metrics["bbox_max_x"], 15)
        self.assertEqual(metrics["bbox_span_x"], 6)

        self.assertEqual(metrics["bbox_min_y"], 5)
        self.assertEqual(metrics["bbox_max_y"], 14)
        self.assertEqual(metrics["bbox_span_y"], 10)

        self.assertEqual(metrics["bbox_min_z"], 20)
        self.assertEqual(metrics["bbox_max_z"], 24)
        self.assertEqual(metrics["bbox_span_z"], 5)

        # Centroid within bounds
        self.assertTrue(10 <= metrics["centroid_x"] <= 15)
        self.assertTrue(5 <= metrics["centroid_y"] <= 14)
        self.assertTrue(20 <= metrics["centroid_z"] <= 24)

    def test_statistical_analyzer(self):
        """
        Verify Mann-Whitney U test and dataset summary generation.
        """
        df_mock = pd.DataFrame({
            "subject_id": [f"Patient_{i:03d}" for i in range(20)],
            "grade": ["HGG"] * 10 + ["LGG"] * 10,
            "wt_volume_cm3": [45.0 + i * 2 for i in range(10)] + [15.0 + i * 1.5 for i in range(10)],
            "tc_volume_cm3": [25.0 + i for i in range(10)] + [5.0 + i * 0.5 for i in range(10)],
            "et_volume_cm3": [18.0 + i for i in range(10)] + [1.0 + i * 0.2 for i in range(10)],
            "ed_volume_cm3": [20.0 + i for i in range(10)] + [10.0 + i for i in range(10)],
            "ncr_volume_cm3": [7.0 + i for i in range(10)] + [4.0 + i * 0.3 for i in range(10)],
            "et_to_wt_ratio": [0.4] * 10 + [0.1] * 10,
            "tc_to_wt_ratio": [0.55] * 10 + [0.35] * 10,
            "ed_to_wt_ratio": [0.45] * 10 + [0.65] * 10,
            "age": [62.0] * 10 + [45.0] * 10,
        })

        stat_analyzer = StatisticalAnalyzer(self.config)
        comparison = stat_analyzer.compare_grades(df_mock)

        self.assertIn("wt_volume_cm3", comparison)
        self.assertTrue(comparison["wt_volume_cm3"]["is_significant"])
        self.assertLess(comparison["wt_volume_cm3"]["p_value"], 0.05)
        self.assertGreater(comparison["wt_volume_cm3"]["hgg_median"], comparison["wt_volume_cm3"]["lgg_median"])

        summary_df = stat_analyzer.build_dataset_summary(df_mock)
        self.assertIsInstance(summary_df, pd.DataFrame)
        self.assertGreater(len(summary_df), 5)


if __name__ == "__main__":
    unittest.main()
