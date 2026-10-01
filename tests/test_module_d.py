"""
NeuroVision AI: Automated Unit Tests for Module D
Tests 3D tumor bounding box extraction, peritumoral context margin expansion,
multimodal ROI cropping & standardizing, morphological centroid calculation,
categorical mask preservation, and dual-mode segmentation adapters.
"""
import unittest
import numpy as np

from src.module_d_roi.bbox_extractor import TumorBoundingBoxExtractor, TumorBoundingBox
from src.module_d_roi.roi_cropper import TumorROICropper
from src.module_d_roi.segmentor import GroundTruthSegmentor, HeuristicFLAIRSegmentor, DeepTumorSegmentor
from src.module_d_roi.roi_metadata import ROIMetadataGenerator


class TestModuleD(unittest.TestCase):

    def setUp(self):
        self.config = {
            "roi_extraction": {
                "context_margin_voxels": 8,
                "min_margin_voxels": 4,
                "target_roi_shape": [32, 32, 32],
                "standardize_method": "pad_or_crop",
                "pad_value": 0.0,
                "mask_pad_value": 0,
                "min_tumor_voxels": 5
            },
            "segmentation": {
                "mode": "ground_truth",
                "target_labels": [1, 2, 4]
            }
        }
        self.bbox_extractor = TumorBoundingBoxExtractor(self.config)
        self.cropper = TumorROICropper(self.config)
        self.metadata_gen = ROIMetadataGenerator(self.config)

    def test_bounding_box_extraction(self):
        """
        Verify tight 3D bounding box coordinates and spans on a synthetic tumor.
        """
        mask = np.zeros((50, 50, 50), dtype=np.uint8)
        # Place synthetic tumor: Z: 10..25 (span 16), Y: 15..30 (span 16), X: 20..40 (span 21)
        mask[10:26, 15:31, 20:41] = 2  # Edema

        bbox = self.bbox_extractor.extract_bounding_box(mask, margin=0)

        self.assertEqual(bbox.z_min, 10)
        self.assertEqual(bbox.z_max, 25)
        self.assertEqual(bbox.span_z, 16)

        self.assertEqual(bbox.y_min, 15)
        self.assertEqual(bbox.y_max, 30)
        self.assertEqual(bbox.span_y, 16)

        self.assertEqual(bbox.x_min, 20)
        self.assertEqual(bbox.x_max, 40)
        self.assertEqual(bbox.span_x, 21)

        expected_voxels = 16 * 16 * 21
        self.assertEqual(bbox.tumor_voxels, expected_voxels)
        self.assertFalse(bbox.is_empty)

    def test_peritumoral_margin_and_clamping(self):
        """
        Verify that contextual margin expands bounds symmetrically and clamps strictly
        to volume boundaries without negative indices or out-of-bound errors.
        """
        mask = np.zeros((40, 40, 40), dtype=np.uint8)
        # Tumor touches boundary: Z starts at 2, Y ends at 38
        mask[2:10, 30:39, 15:25] = 4  # Enhancing tumor

        bbox = self.bbox_extractor.extract_bounding_box(mask, margin=8)

        # Z: 2 - 8 = -6 -> clamped to 0
        self.assertEqual(bbox.crop_z_min, 0)
        self.assertEqual(bbox.crop_z_max, 9 + 8)

        # Y: 38 + 8 = 46 -> clamped to 39
        self.assertEqual(bbox.crop_y_min, 30 - 8)
        self.assertEqual(bbox.crop_y_max, 39)

        # X: within boundaries
        self.assertEqual(bbox.crop_x_min, 15 - 8)
        self.assertEqual(bbox.crop_x_max, 24 + 8)

    def test_morphological_centroid(self):
        """
        Verify that 3D morphological centroid matches exact center of mass.
        """
        mask = np.zeros((30, 30, 30), dtype=np.uint8)
        mask[10:15, 10:15, 10:15] = 1  # 5x5x5 cube from index 10 to 14

        bbox = self.bbox_extractor.extract_bounding_box(mask)

        # Center of mass of indices 10, 11, 12, 13, 14 is exactly 12.0
        self.assertAlmostEqual(bbox.centroid_z, 12.0, places=1)
        self.assertAlmostEqual(bbox.centroid_y, 12.0, places=1)
        self.assertAlmostEqual(bbox.centroid_x, 12.0, places=1)

    def test_multimodal_roi_cropping_and_shape(self):
        """
        Verify that 4-channel image and 3D mask are cropped and standardized
        to the exact configured target ROI dimensions (e.g. 4, 32, 32, 32).
        """
        image_4d = np.random.randn(4, 50, 50, 50).astype(np.float32)
        mask_3d = np.zeros((50, 50, 50), dtype=np.uint8)
        mask_3d[15:25, 15:25, 15:25] = 2

        bbox = self.bbox_extractor.extract_bounding_box(mask_3d, margin=4)
        roi_result = self.cropper.extract_roi(
            image_4d=image_4d,
            bbox=bbox,
            mask_3d=mask_3d,
            target_shape=(32, 32, 32)
        )

        roi_image = roi_result["roi_image"]
        roi_mask = roi_result["roi_mask"]

        self.assertEqual(roi_image.shape, (4, 32, 32, 32))
        self.assertEqual(roi_mask.shape, (32, 32, 32))
        self.assertEqual(roi_image.dtype, np.float32)
        self.assertEqual(roi_mask.dtype, np.uint8)

    def test_categorical_mask_integrity(self):
        """
        Verify that discrete labels {0, 1, 2, 4} are preserved without fractional artifacts.
        """
        mask_3d = np.zeros((40, 40, 40), dtype=np.uint8)
        mask_3d[10:15, 10:15, 10:15] = 1
        mask_3d[15:20, 15:20, 15:20] = 2
        mask_3d[20:25, 20:25, 20:25] = 4

        bbox = self.bbox_extractor.extract_bounding_box(mask_3d, margin=2)
        image_4d = np.ones((4, 40, 40, 40), dtype=np.float32)

        roi_result = self.cropper.extract_roi(
            image_4d=image_4d,
            bbox=bbox,
            mask_3d=mask_3d,
            target_shape=(32, 32, 32)
        )

        unique_labels = np.unique(roi_result["roi_mask"])
        for lbl in unique_labels:
            self.assertIn(lbl, [0, 1, 2, 4], f"Corrupt fractional label detected: {lbl}")

    def test_empty_mask_fallback(self):
        """
        Verify that an empty mask triggers the central brain fallback rather than crashing.
        """
        empty_mask = np.zeros((40, 40, 40), dtype=np.uint8)
        bbox = self.bbox_extractor.extract_bounding_box(empty_mask)

        self.assertTrue(bbox.is_empty)
        self.assertEqual(bbox.tumor_voxels, 0)
        self.assertGreater(bbox.crop_span_z, 0)

    def test_segmentor_adapters(self):
        """
        Verify GroundTruthSegmentor and HeuristicFLAIRSegmentor functionality.
        """
        gt_seg = GroundTruthSegmentor()
        heur_seg = HeuristicFLAIRSegmentor()
        deep_seg = DeepTumorSegmentor()

        # GT test
        gt_mask = np.ones((20, 20, 20), dtype=np.uint8)
        dummy_img = np.zeros((4, 20, 20, 20), dtype=np.float32)
        res = gt_seg.predict_mask(dummy_img, ground_truth_mask=gt_mask)
        self.assertEqual(res.shape, (20, 20, 20))

        # GT requires mask
        with self.assertRaises(ValueError):
            gt_seg.predict_mask(dummy_img, ground_truth_mask=None)

        # Heuristic test
        # Inject hyperintensity in FLAIR (ch 3)
        dummy_img[3, 5:15, 5:15, 5:15] = 100.0
        heur_mask = heur_seg.predict_mask(dummy_img)
        self.assertEqual(heur_mask.shape, (20, 20, 20))
        self.assertGreater(np.sum(heur_mask > 0), 0)

    def test_roi_metadata_generation(self):
        """
        Verify that compute_subject_record generates accurate quantitative metrics.
        """
        mask = np.zeros((30, 30, 30), dtype=np.uint8)
        mask[5:15, 5:15, 5:15] = 4  # Enhancing tumor (1000 voxels)
        image = np.ones((4, 30, 30, 30), dtype=np.float32)

        bbox = self.bbox_extractor.extract_bounding_box(mask, margin=2)
        record = self.metadata_gen.compute_subject_record(
            subject_id="BraTS20_Test_001",
            bbox=bbox,
            mask=mask,
            image_4d=image,
            grade="HGG",
            label=1,
            split="train"
        )

        self.assertEqual(record["subject_id"], "BraTS20_Test_001")
        self.assertEqual(record["grade"], "HGG")
        self.assertEqual(record["wt_voxels"], 1000)
        self.assertEqual(record["et_voxels"], 1000)
        self.assertEqual(record["et_to_wt_ratio"], 1.0)
        self.assertGreater(record["brain_voxels"], 0)
        self.assertIn("crop_span_z", record)


if __name__ == "__main__":
    unittest.main()
