"""
NeuroVision AI: Module D - End-to-End 3D Tumor ROI Extraction Pipeline
Orchestrates tumor bounding box extraction, peritumoral margin expansion,
standardized ROI cropping, morphometric metadata export, and visual QA.
"""
from typing import Dict, Any, List, Optional
import os
import time
import pandas as pd
import numpy as np
import nibabel as nib

from src.utils.logger import setup_logger
from src.module_c_preprocessing.preprocessor import MRIPreprocessor
from src.module_d_roi.bbox_extractor import TumorBoundingBoxExtractor, TumorBoundingBox
from src.module_d_roi.roi_cropper import TumorROICropper
from src.module_d_roi.segmentor import get_segmentor, BaseTumorSegmentor
from src.module_d_roi.roi_metadata import ROIMetadataGenerator
from src.module_d_roi.visualizer import ROIVisualizer

logger = setup_logger("ROIExtractionPipeline")


class ROIExtractionPipeline:
    """
    High-level execution engine for Module D.
    """

    def __init__(self, config: Dict[str, Any], base_dir: str = "."):
        self.config = config
        self.base_dir = base_dir

        self.bbox_extractor = TumorBoundingBoxExtractor(config)
        self.cropper = TumorROICropper(config)
        self.segmentor = get_segmentor(config)
        self.metadata_gen = ROIMetadataGenerator(config)
        self.visualizer = ROIVisualizer(config, base_dir=base_dir)

        # Paths
        inputs_cfg = config.get("inputs", {})
        self.metadata_csv = os.path.join(base_dir, inputs_cfg.get("metadata_csv", "data/dataset_metadata.csv"))
        self.processed_dir = os.path.join(base_dir, inputs_cfg.get("processed_dir", "data/processed"))
        self.splits_dir = os.path.join(base_dir, inputs_cfg.get("splits_dir", "data/splits"))
        self.prefer_preprocessed = inputs_cfg.get("prefer_preprocessed", True)

        caching_cfg = config.get("caching", {})
        self.output_dir = os.path.join(base_dir, caching_cfg.get("output_dir", "data/roi_tensors"))
        self.metadata_output = os.path.join(base_dir, caching_cfg.get("metadata_output", "data/roi_metadata.csv"))

        # Preprocessor fallback if loading raw NIfTIs
        self.preprocessor = MRIPreprocessor(config)

    def _load_subject_data(self, row: pd.Series) -> Dict[str, Any]:
        """
        Load 4D multimodal MRI image and 3D mask for a subject.
        Checks preprocessed cache first for maximum speed.
        """
        subject_id = row["subject_id"]
        cached_npz = os.path.join(self.processed_dir, f"{subject_id}.npz")

        if self.prefer_preprocessed and os.path.isfile(cached_npz):
            data = np.load(cached_npz)
            return {
                "image": data["image"].astype(np.float32),
                "mask": data["mask"].astype(np.uint8) if "mask" in data else None,
                "grade": str(data["grade"]) if "grade" in data else row.get("grade", "HGG"),
                "label": int(data["label"]) if "label" in data else row.get("label", 1)
            }

        # Otherwise, process from raw NIfTIs
        processed = self.preprocessor.process_subject_row(row)
        return {
            "image": processed["image"],
            "mask": processed["mask"],
            "grade": row.get("grade", "HGG"),
            "label": row.get("label", 1)
        }

    def _load_split_mapping(self) -> Dict[str, str]:
        """
        Map each subject_id to train, val, or test split.
        """
        mapping = {}
        for s_name in ["train", "val", "test"]:
            s_path = os.path.join(self.splits_dir, f"{s_name}_subjects.csv")
            if os.path.isfile(s_path):
                df_split = pd.read_csv(s_path)
                for sid in df_split["subject_id"]:
                    mapping[sid] = s_name
        return mapping

    def run(
        self,
        cache_samples: int = 6,
        save_all: bool = False,
        limit: Optional[int] = None,
        generate_figures: bool = True
    ) -> Dict[str, Any]:
        """
        Execute Module D pipeline.
        
        Args:
            cache_samples: Number of ROI tensors to save to disk.
            save_all: If True, saves ROI tensors for all processed subjects.
            limit: Optional limit on total cohort rows for fast runs.
            generate_figures: Whether to render visual QA figures.
            
        Returns:
            Dictionary with execution statistics and output paths.
        """
        start_time = time.time()
        logger.info("Starting Module D: 3D Tumor Segmentation & ROI Extraction Pipeline...")

        if not os.path.isfile(self.metadata_csv):
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_csv}")

        df = pd.read_csv(self.metadata_csv)
        total_available = len(df)
        if limit is not None and limit > 0:
            logger.info(f"Applying limit: processing first {limit} of {total_available} subjects.")
            df = df.head(limit).copy()

        split_map = self._load_split_mapping()
        os.makedirs(self.output_dir, exist_ok=True)

        records: List[Dict[str, Any]] = []
        cached_count = 0
        figures_generated: List[str] = []

        # Track representative cases for figure generation
        vis_sample_candidates = []

        tumor_stats_file = os.path.join(self.base_dir, self.config.get("inputs", {}).get("tumor_statistics_csv", "data/tumor_statistics.csv"))
        if os.path.isfile(tumor_stats_file):
            t_df = pd.read_csv(tumor_stats_file)
            merge_cols = [
                'subject_id', 'bbox_min_x', 'bbox_max_x', 'bbox_span_x',
                'bbox_min_y', 'bbox_max_y', 'bbox_span_y',
                'bbox_min_z', 'bbox_max_z', 'bbox_span_z',
                'centroid_x', 'centroid_y', 'centroid_z',
                'wt_voxels', 'tc_voxels', 'et_voxels', 'ed_voxels', 'ncr_voxels',
                'et_to_wt_ratio', 'tc_to_wt_ratio', 'ed_to_wt_ratio'
            ]
            cols_to_use = [c for c in merge_cols if c in t_df.columns]
            df = pd.merge(df, t_df[cols_to_use], on='subject_id', how='left', suffixes=('', '_stats'))

        total_to_process = len(df)

        # Determine which subjects to load and crop 3D volumes for
        subjects_to_cache = set()
        if cache_samples > 0:
            for s_name in ["train", "val", "test"]:
                subjs_in_split = [sid for sid, sp in split_map.items() if sp == s_name]
                take = max(1, cache_samples // 3)
                subjects_to_cache.update(subjs_in_split[:take])
            if len(subjects_to_cache) < cache_samples:
                subjects_to_cache.update(df["subject_id"].head(cache_samples).tolist())

        for idx, (_, row) in enumerate(df.iterrows()):
            subject_id = row["subject_id"]
            split_name = split_map.get(subject_id, "train")
            grade = row.get("grade", "HGG")
            label = row.get("label", 1)

            should_extract_3d = save_all or (subject_id in subjects_to_cache) or (len(vis_sample_candidates) < self.config.get("visualization", {}).get("sample_cases", 3))

            if should_extract_3d:
                # 1. Load data
                subj_data = self._load_subject_data(row)
                image_4d = subj_data["image"]
                gt_mask = subj_data["mask"]
                grade = subj_data["grade"]
                label = subj_data["label"]

                # 2. Predict/Extract mask
                pred_mask = self.segmentor.predict_mask(image_4d, ground_truth_mask=gt_mask)

                # 3. Compute 3D bounding box with context margin
                bbox = self.bbox_extractor.extract_bounding_box(pred_mask)

                # 4. Crop and standardize ROI
                roi_result = self.cropper.extract_roi(
                    image_4d=image_4d,
                    bbox=bbox,
                    mask_3d=pred_mask
                )
                roi_image = roi_result["roi_image"]
                roi_mask = roi_result["roi_mask"]

                # 5. Cache tensor if within budget
                should_cache = save_all or (cached_count < cache_samples and subject_id in subjects_to_cache)
                cached_path = None
                if should_cache:
                    out_file = os.path.join(self.output_dir, f"{subject_id}_roi.npz")
                    np.savez_compressed(
                        out_file,
                        roi_image=roi_image.astype(np.float32),
                        roi_mask=roi_mask.astype(np.uint8) if roi_mask is not None else np.zeros((1,), dtype=np.uint8),
                        grade=grade,
                        label=label,
                        subject_id=subject_id,
                        target_shape=np.array(self.cropper.target_shape)
                    )
                    cached_path = os.path.relpath(out_file, self.base_dir)
                    cached_count += 1

                # 6. Compute morphometric metadata
                record = self.metadata_gen.compute_subject_record(
                    subject_id=subject_id,
                    bbox=bbox,
                    mask=pred_mask,
                    image_4d=image_4d,
                    grade=grade,
                    label=label,
                    split=split_name,
                    cached_file=cached_path,
                    target_roi_shape=self.cropper.target_shape
                )
                records.append(record)

                # Retain sample cases for visualization
                if len(vis_sample_candidates) < self.config.get("visualization", {}).get("sample_cases", 3):
                    vis_sample_candidates.append({
                        "full_image": image_4d,
                        "full_mask": pred_mask,
                        "roi_image": roi_image,
                        "roi_mask": roi_mask,
                        "bbox": bbox,
                        "subject_id": subject_id,
                        "grade": grade
                    })
            else:
                # Fast path using pre-calculated bounding metrics from Module B
                margin = self.bbox_extractor.default_margin
                # Read from row
                bx_min = int(row.get("bbox_min_x", 0))
                bx_max = int(row.get("bbox_max_x", 0))
                by_min = int(row.get("bbox_min_y", 0))
                by_max = int(row.get("bbox_max_y", 0))
                bz_min = int(row.get("bbox_min_z", 0))
                bz_max = int(row.get("bbox_max_z", 0))
                sh_x = int(row.get("shape_x", 240))
                sh_y = int(row.get("shape_y", 240))
                sh_z = int(row.get("shape_z", 155))

                cz_min = max(0, bz_min - margin)
                cz_max = min(sh_z - 1, bz_max + margin)
                cy_min = max(0, by_min - margin)
                cy_max = min(sh_y - 1, by_max + margin)
                cx_min = max(0, bx_min - margin)
                cx_max = min(sh_x - 1, bx_max + margin)

                wt_v = int(row.get("wt_voxels", 0))
                tc_v = int(row.get("tc_voxels", 0))
                et_v = int(row.get("et_voxels", 0))
                ed_v = int(row.get("ed_voxels", 0))
                ncr_v = int(row.get("ncr_voxels", 0))

                span_z = bz_max - bz_min + 1
                span_y = by_max - by_min + 1
                span_x = bx_max - bx_min + 1

                c_span_z = cz_max - cz_min + 1
                c_span_y = cy_max - cy_min + 1
                c_span_x = cx_max - cx_min + 1

                # Approximate brain volume: ~1.2M voxels average
                brain_v = int(sh_x * sh_y * sh_z * 0.25)
                tumor_ratio = round(wt_v / brain_v, 4) if brain_v > 0 else 0.0

                records.append({
                    "subject_id": subject_id,
                    "grade": grade,
                    "label": label,
                    "split": split_name,
                    "bbox_min_z": bz_min,
                    "bbox_max_z": bz_max,
                    "bbox_span_z": span_z,
                    "bbox_min_y": by_min,
                    "bbox_max_y": by_max,
                    "bbox_span_y": span_y,
                    "bbox_min_x": bx_min,
                    "bbox_max_x": bx_max,
                    "bbox_span_x": span_x,
                    "bbox_volume_voxels": span_z * span_y * span_x,
                    "centroid_z": float(row.get("centroid_z", 0.0)),
                    "centroid_y": float(row.get("centroid_y", 0.0)),
                    "centroid_x": float(row.get("centroid_x", 0.0)),
                    "margin_applied": margin,
                    "crop_min_z": cz_min,
                    "crop_max_z": cz_max,
                    "crop_span_z": c_span_z,
                    "crop_min_y": cy_min,
                    "crop_max_y": cy_max,
                    "crop_span_y": c_span_y,
                    "crop_min_x": cx_min,
                    "crop_max_x": cx_max,
                    "crop_span_x": c_span_x,
                    "crop_volume_voxels": c_span_z * c_span_y * c_span_x,
                    "wt_voxels": wt_v,
                    "tc_voxels": tc_v,
                    "et_voxels": et_v,
                    "ed_voxels": ed_v,
                    "ncr_voxels": ncr_v,
                    "et_to_wt_ratio": float(row.get("et_to_wt_ratio", 0.0)),
                    "tc_to_wt_ratio": float(row.get("tc_to_wt_ratio", 0.0)),
                    "ed_to_wt_ratio": float(row.get("ed_to_wt_ratio", 0.0)),
                    "brain_voxels": brain_v,
                    "tumor_to_brain_ratio": tumor_ratio,
                    "aspect_ratio_zy": round(span_z / max(1, span_y), 3),
                    "aspect_ratio_yx": round(span_y / max(1, span_x), 3),
                    "target_roi_shape": str(self.cropper.target_shape),
                    "is_empty": (wt_v < 10),
                    "cached_file": None
                })

            if (idx + 1) % 50 == 0 or (idx + 1) == total_to_process:
                logger.info(f"Processed {idx + 1}/{total_to_process} subjects. (Cached: {cached_count})")

        # Export metadata CSV
        roi_df = self.metadata_gen.export_metadata(
            records=records,
            output_csv=self.metadata_output,
            base_dir=self.base_dir
        )

        # Generate Visualizations
        if generate_figures and len(vis_sample_candidates) > 0:
            for cand in vis_sample_candidates:
                fig_path = self.visualizer.plot_roi_extraction_case(
                    full_image=cand["full_image"],
                    full_mask=cand["full_mask"],
                    roi_image=cand["roi_image"],
                    roi_mask=cand["roi_mask"],
                    bbox=cand["bbox"],
                    subject_id=cand["subject_id"],
                    grade=cand["grade"]
                )
                figures_generated.append(fig_path)

            cohort_fig = self.visualizer.plot_cohort_roi_distribution(roi_df)
            figures_generated.append(cohort_fig)

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"Module D execution complete in {elapsed}s. {len(records)} ROIs processed.")

        return {
            "total_processed": len(records),
            "cached_tensors": cached_count,
            "metadata_csv": self.metadata_output,
            "figures_generated": figures_generated,
            "elapsed_seconds": elapsed,
            "target_roi_shape": self.cropper.target_shape,
            "mean_tumor_span_z": round(float(roi_df["bbox_span_z"].mean()), 2),
            "mean_tumor_span_y": round(float(roi_df["bbox_span_y"].mean()), 2),
            "mean_tumor_span_x": round(float(roi_df["bbox_span_x"].mean()), 2),
            "mean_tumor_to_brain_ratio": round(float(roi_df["tumor_to_brain_ratio"].mean()), 4)
        }
