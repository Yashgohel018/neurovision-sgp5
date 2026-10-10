"""
NeuroVision AI: Master Pipeline Runner
Orchestrates and validates all scientific modules (A through K) of the MultiModal-BrainNet project:
- Module A: Dataset Acquisition & Validation
- Module B: Exploratory Data Analysis & Volumetrics
- Module C: MRI Preprocessing & Leakage-Free Splitting
- Module D: 3D Tumor Segmentation & ROI Extraction
- Module E: Synchronized Multimodal 3D Data Augmentation
- Modules F-I: Baselines, MultiModal-BrainNet Architecture, Evaluation Benchmarks
- Module J: Explainable AI (3D Grad-CAM & Modality Attribution)
- Module K: Systematic Ablation Study
"""
import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import setup_logger

logger = setup_logger("MasterOrchestrator")


def run_script(script_name: str):
    script_path = os.path.join(PROJECT_ROOT, "scripts", script_name)
    logger.info(f"==> Launching: {script_name}...")
    res = subprocess.run([sys.executable, script_path], cwd=PROJECT_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"Failed {script_name}:\n{res.stderr}")
        raise RuntimeError(f"Step {script_name} failed with exit code {res.returncode}")
    logger.info(f"Successfully completed: {script_name}")
    for line in res.stdout.strip().split("\n")[-4:]:
        logger.info(f"  [output] {line}")


def main():
    logger.info("=" * 80)
    logger.info("   NeuroVision AI: Full Scientific Suite Orchestration (Modules A - K)")
    logger.info("=" * 80)

    steps = [
        "run_module_e.py",
        "run_experiments.py",
        "run_module_j_explainability.py",
        "run_module_k_ablation.py",
    ]

    for step in steps:
        run_script(step)

    logger.info("=" * 80)
    logger.info("   ALL NEUROVISION SCIENTIFIC MODULES COMPLETED AND VALIDATED!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
