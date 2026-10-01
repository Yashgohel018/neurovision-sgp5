"""
NeuroVision AI: Module D Pipeline Runner
3D Tumor Segmentation, Context Margin Expansion & Standardized ROI Extraction.

Usage:
    python scripts/run_module_d.py
    python scripts/run_module_d.py --cache-samples 6
    python scripts/run_module_d.py --config configs/roi_config.yaml
    python scripts/run_module_d.py --save-all
"""
import os
import sys
import argparse
import yaml

# Ensure learnNeuro root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import setup_logger
from src.module_d_roi.pipeline import ROIExtractionPipeline

logger = setup_logger("RunModuleD")


def parse_args():
    parser = argparse.ArgumentParser(description="NeuroVision AI: Module D 3D Tumor ROI Extraction Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "roi_config.yaml"),
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--cache-samples",
        type=int,
        default=6,
        help="Number of subjects to cache as 3D ROI tensors (e.g. --cache-samples 6)"
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Save ROI tensors for all subjects in cohort"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on total dataset records for rapid testing"
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure rendering"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 78)
    logger.info("   NeuroVision AI: Autonomous MRI Clinical Decision Support System")
    logger.info("   Module D: 3D Tumor Segmentation & Focused ROI Extraction")
    logger.info("=" * 78)

    if not os.path.isfile(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from: {args.config}")

    pipeline = ROIExtractionPipeline(config, base_dir=PROJECT_ROOT)
    result = pipeline.run(
        cache_samples=args.cache_samples,
        save_all=args.save_all,
        limit=args.limit,
        generate_figures=not args.no_figures
    )

    # -------------------------------------------------------------
    # Executive Terminal Report
    # -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("            NEUROVISION AI - MODULE D EXECUTIVE REPORT")
    print("=" * 78)
    print(f"Total Subjects Processed        : {result['total_processed']} patients (BraTS 2020 cohort)")
    print(f"Cached 3D ROI Tensors           : {result['cached_tensors']} volumes saved to disk")
    print(f"Execution Duration              : {result['elapsed_seconds']} seconds")
    print(f"Segmentation Architecture Mode  : {config.get('segmentation', {}).get('mode', 'ground_truth')}")
    print("-" * 78)
    print("Cohort Morphometric & Spatial ROI Statistics:")
    print(f"  * Mean Tumor Bounding Span (Z) : {result['mean_tumor_span_z']} voxels (native axial slices)")
    print(f"  * Mean Tumor Bounding Span (Y) : {result['mean_tumor_span_y']} voxels")
    print(f"  * Mean Tumor Bounding Span (X) : {result['mean_tumor_span_x']} voxels")
    print(f"  * Mean Tumor / Brain Fraction  : {result['mean_tumor_to_brain_ratio'] * 100:.2f}% of intracranial parenchyma")
    print(f"  * Peritumoral Context Margin   : +{config.get('roi_extraction', {}).get('context_margin_voxels', 10)} voxels symmetrically")

    print("-" * 78)
    print("Standardized 3D ROI Tensor Specifications:")
    print("  - Modality Channels (4)       : [T1, T1ce, T2, FLAIR]")
    print(f"  - Target ROI Tensor Dimensions: (4, {result['target_roi_shape'][0]}, {result['target_roi_shape'][1]}, {result['target_roi_shape'][2]}) [C, D, H, W]")
    print(f"  - Spatial Standardizer Method : {config.get('roi_extraction', {}).get('standardize_method', 'pad_or_crop')}")
    print(f"  - Morphological Labels        : 1 (Necrotic Core), 2 (Edema), 4 (Enhancing Tumor)")

    print("-" * 78)
    print("Exported Deliverables & Artifacts:")
    print(f"  [PASS] ROI Metadata Table        : {result['metadata_csv']}")
    if result.get("figures_generated"):
        for fig_path in result["figures_generated"]:
            rel = os.path.relpath(fig_path, PROJECT_ROOT)
            print(f"  [PASS] Diagnostic Figure         : {rel}")

    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
