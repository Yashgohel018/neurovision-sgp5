from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_DIR = BASE_DIR / "data"
CASES_DIR = DATA_DIR / "cases"
MODELS_DIR = BASE_DIR / "models"

# BraTS 2020 Medical Imaging Dataset Paths
BRATS_DATA_DIR = PROJECT_ROOT / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"

# learnNeuro Intelligence & Pipeline Paths
LEARN_NEURO_DIR = PROJECT_ROOT / "learnNeuro"
LEARN_NEURO_DATA_DIR = LEARN_NEURO_DIR / "data"
DATASET_METADATA_CSV = LEARN_NEURO_DATA_DIR / "dataset_metadata.csv"
TUMOR_STATS_CSV = LEARN_NEURO_DATA_DIR / "tumor_statistics.csv"
ROI_METADATA_CSV = LEARN_NEURO_DATA_DIR / "roi_metadata.csv"
SPLITS_DIR = LEARN_NEURO_DATA_DIR / "splits"

# Scientific Reports & Deliverables
LEARN_NEURO_REPORTS_DIR = LEARN_NEURO_DIR / "reports"
LEARN_NEURO_FIGURES_DIR = LEARN_NEURO_REPORTS_DIR / "figures"
MODEL_COMPARISON_CSV = LEARN_NEURO_REPORTS_DIR / "model_comparison_table.csv"
BASELINE_RESULTS_CSV = LEARN_NEURO_REPORTS_DIR / "baseline_results.csv"
ABLATION_RESULTS_CSV = LEARN_NEURO_REPORTS_DIR / "ablation_results.csv"
ERROR_ANALYSIS_CSV = LEARN_NEURO_REPORTS_DIR / "error_analysis.csv"

# Trained Model Checkpoints
MORPHNET_MODEL_PATH = MODELS_DIR / "morphnet_baseline.joblib"
BRAINNET_CHECKPOINT_PATH = MODELS_DIR / "multimodal_brainnet_p2.pt"

# Ensure backend local directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CASES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "*",
]
