"""
NeuroVision AI: Module B Pipeline Runner
Exploratory Data Analysis, Tumor Volumetric Quantification & Clinical QC.

Usage:
    python scripts/run_module_b.py
    python scripts/run_module_b.py --limit 10
    python scripts/run_module_b.py --skip-figures
    python scripts/run_module_b.py --config configs/eda_config.yaml
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
from src.module_b_eda.eda_pipeline import EDAPipeline

logger = setup_logger("RunModuleB")


def parse_args():
    parser = argparse.ArgumentParser(description="NeuroVision AI: Module B EDA & Volumetric QC")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "eda_config.yaml"),
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional subject limit for rapid testing (e.g. --limit 10)"
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure rendering to speed up testing"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 78)
    logger.info("   NeuroVision AI: Autonomous MRI Clinical Decision Support System")
    logger.info("   Module B: Exploratory Data Analysis & Volumetric Quality Control")
    logger.info("=" * 78)

    if not os.path.isfile(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from: {args.config}")

    # Run EDA Pipeline
    pipeline = EDAPipeline(config, base_dir=PROJECT_ROOT)
    result = pipeline.run(limit=args.limit, skip_figures=args.skip_figures)

    # -------------------------------------------------------------
    # Executive Terminal Report
    # -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("            NEUROVISION AI - MODULE B EXECUTIVE REPORT")
    print("=" * 78)
    print(f"Total Subjects Evaluated        : {result['total_subjects']}")
    print(f"Class Breakdown                 : {result['hgg_count']} HGG vs {result['lgg_count']} LGG")
    print(f"Execution Duration              : {result['elapsed_seconds']} seconds")
    print("-" * 78)
    print("Tumor Sub-Region Volumetric Comparison (HGG vs LGG):")

    comp = result.get("grade_comparison", {})
    for metric, label in [
        ("wt_volume_cm3", "Whole Tumor (WT, cm³)"),
        ("tc_volume_cm3", "Tumor Core (TC, cm³)"),
        ("et_volume_cm3", "Enhancing Tumor (ET, cm³)"),
        ("ed_volume_cm3", "Peritumoral Edema (ED, cm³)"),
        ("et_to_wt_ratio", "Enhancing Fraction (ET/WT)"),
    ]:
        if metric in comp:
            c = comp[metric]
            p_val = c["p_value"]
            p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
            sig = "***" if p_val < 0.001 else ("*" if p_val < 0.05 else "ns")
            print(f"  - {label:<28}: HGG median = {c['hgg_median']:>6.2f} | LGG median = {c['lgg_median']:>6.2f} (p = {p_str} {sig})")

    print("-" * 78)
    print("Clinical Outlier & Anomaly Detection:")
    outliers = result["outlier_summary"]
    print(f"  - Micro-Tumors (< 5 cm³)        : {outliers['micro_tumors']} subjects")
    print(f"  - Massive Tumors (> 150 cm³)    : {outliers['massive_tumors']} subjects")
    print(f"  - Non-Enhancing Tumors (ET = 0) : {outliers['non_enhancing_tumors']} subjects (typical in select LGG)")

    print("-" * 78)
    print("Generated System Deliverables & Artifacts:")
    artifacts = result["saved_artifacts"]
    print(f"  - Patient Volumetric Index      : {artifacts['tumor_statistics_csv']}")
    print(f"  - Cohort Summary Table          : {artifacts['dataset_summary_csv']}")
    if artifacts.get("figures"):
        print("  - Diagnostic Figures:")
        for name, path in artifacts["figures"].items():
            print(f"      • {name:<30}: {path}")
    print("=" * 78)

    logger.info("Module B execution completed successfully! 100% volumetric profiling established.")


if __name__ == "__main__":
    main()
