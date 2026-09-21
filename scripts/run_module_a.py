"""
NeuroVision AI: Module A Pipeline Runner
Ingestion, Indexing, and Quality Validation for the Two-Agent Clinical Decision Support System.

Usage:
    python scripts/run_module_a.py
    python scripts/run_module_a.py --limit 20
    python scripts/run_module_a.py --config configs/dataset_config.yaml
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
from src.module_a_dataset.indexer import BraTSIndexer
from src.module_a_dataset.validator import BraTSValidator

logger = setup_logger("RunModuleA")


def parse_args():
    parser = argparse.ArgumentParser(description="NeuroVision AI: Module A Ingestion & Validation")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "dataset_config.yaml"),
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional subject limit for rapid testing (e.g. --limit 10)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 75)
    logger.info("   NeuroVision AI: Autonomous MRI Clinical Decision Support System")
    logger.info("   Module A: Multimodal Ingestion, Agent Indexing & Data Validation")
    logger.info("=" * 75)

    if not os.path.isfile(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from: {args.config}")
    raw_dir = config["dataset"]["raw_data_dir"]
    logger.info(f"Raw Dataset Directory: {raw_dir}")

    # 1. Initialize and run Indexer
    indexer = BraTSIndexer(config)
    df = indexer.scan_dataset(limit=args.limit)

    # 2. Initialize and run Validator
    validator = BraTSValidator(config)
    report = validator.validate(df)

    # 3. Export Artifacts
    paths = validator.export_artifacts(df, base_dir=PROJECT_ROOT)

    # 4. Print Executive Validation Report tailored to NeuroVision AI
    print("\n" + "=" * 75)
    print("            NEUROVISION AI - MODULE A VALIDATION REPORT")
    print("=" * 75)
    print(f"Total Subjects Ingested           : {report['total_subjects']}")
    print(f"Agent 1 (Medical Analysis) Inputs : {report['complete_subjects']} / {report['total_subjects']} complete (4 MRI + SEG)")
    print(f"Agent 2 (Clinical Decision) Target: {report['total_subjects']} / {report['total_subjects']} mapped (HGG vs LGG)")
    print(f"Historical Case Retrieval Registry: {report['retrieval_eligible_subjects']} / {report['total_subjects']} eligible")
    print("-" * 75)
    print("Modality Breakdown (Agent 1 Inputs):")
    for mod, count in report["missing_modality_counts"].items():
        status = "[OK] 0 missing" if count == 0 else f"[!] {count} missing"
        print(f"  - {mod.upper():<8}: {status}")
    print("-" * 75)
    print("Clinical Classification Target Distribution (Agent 2):")
    for grade, count in report["grade_distribution"].items():
        pct = (count / report['total_subjects']) * 100
        print(f"  - {grade:<8}: {count:>4} patients ({pct:.1f}%)")
    print(f"  - Imbalance Ratio (HGG : LGG) : {report['hgg_to_lgg_ratio']}")
    print("-" * 75)
    print("Spatial Uniformity (Agent 1 3D Convolutions):")
    print(f"  - Matrix Dimensions (voxels)  : {report['unique_shapes']}")
    print(f"  - Voxel Resolution (isotropic): {report['unique_spacings']} mm")
    print(f"  - Anatomical Orientation      : {report['unique_orientations']}")
    print("-" * 75)
    if report["age_statistics"]["available_records"] > 0:
        print("Clinical Covariates & Demographics (Agent 2 RAG & Reasoning):")
        print(f"  - Documented Records : {report['age_statistics']['available_records']} patients")
        print(f"  - Patient Age Range  : {report['age_statistics']['min']} to {report['age_statistics']['max']} years (Mean: {report['age_statistics']['mean']})")
    print("-" * 75)
    print("Generated System Artifacts:")
    print(f"  - Master Metadata Index: {paths['metadata_csv']}")
    print(f"  - System Data Schema   : {paths['data_dictionary_json']}")
    print("=" * 75)

    if report["all_passed"]:
        logger.info("Module A execution SUCCESSFUL: 100% data integrity established for NeuroVision AI.")
    else:
        logger.warning("Module A execution completed with warnings. Review report above.")


if __name__ == "__main__":
    main()
