"""
NeuroVision AI: Module E Pipeline Runner
Synchronized Multimodal 3D Data Augmentation
Usage:
    python scripts/run_module_e.py
"""
import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import setup_logger
from src.module_e_augmentation.pipeline import AugmentationPipeline

logger = setup_logger("RunModuleE")


def main():
    logger.info("=" * 78)
    logger.info("   NeuroVision AI: MultiModal-BrainNet Pipeline")
    logger.info("   Module E: Synchronized Multimodal 3D Data Augmentation")
    logger.info("=" * 78)

    config_path = os.path.join(PROJECT_ROOT, "configs", "augmentation_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pipeline = AugmentationPipeline(config)

    # Load a sample cached ROI tensor or create one from BraTS
    sample_roi_path = os.path.join(PROJECT_ROOT, "data", "roi_tensors", "BraTS20_Training_001_roi.npz")
    if os.path.isfile(sample_roi_path):
        data = np.load(sample_roi_path)
        # Check available arrays
        keys = list(data.keys())
        logger.info(f"Loaded sample ROI tensor from {sample_roi_path} with keys: {keys}")
        if "roi_image" in data:
            vol = data["roi_image"]
        elif "roi_tensor" in data:
            vol = data["roi_tensor"]
        else:
            modalities = ["t1", "t1ce", "t2", "flair"]
            vol = np.stack([data[m] for m in modalities if m in data], axis=0)
    else:
        logger.info("Sample ROI tensor not found, synthesizing placeholder volume")
        vol = np.random.randn(4, 96, 96, 96).astype(np.float32)

    logger.info(f"Input multimodal volume shape: {vol.shape}")
    augmented_vol, _ = pipeline.augmentor(vol)
    logger.info(f"Augmented multimodal volume shape: {augmented_vol.shape}")

    # Generate QA Figure
    figure_path = os.path.join(PROJECT_ROOT, "reports", "figures", "augmentation_samples.png")
    pipeline.generate_qa_figure(
        original_vol=vol,
        augmented_vol=augmented_vol,
        modality_names=["T1", "T1ce", "T2", "FLAIR"],
        output_path=figure_path
    )
    logger.info(f"Module E executed successfully. QA figure saved to: {figure_path}")


if __name__ == "__main__":
    main()
