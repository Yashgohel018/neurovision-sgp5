# NeuroVision AI (MultiModal-BrainNet) 🧠🔬
> **Autonomous MRI Clinical Decision Support System (CDSS) for Brain Tumor Classification & Analysis**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: PyTorch & MONAI Ready](https://img.shields.io/badge/Framework-PyTorch%20%26%20MONAI-ee4c2c.svg)](https://monai.io/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)

---

## 📌 Executive Summary

**NeuroVision AI** is an enterprise-grade Clinical Decision Support System designed to assist neuro-oncologists and radiologists in diagnosing, grading, and analyzing brain tumors from multimodal 3D Magnetic Resonance Imaging (MRI).

It bridges an **autonomous 2-agent architecture** with an intensive **12-Module Deep Learning Engine** operating over the BraTS (Brain Tumor Segmentation) dataset:

1. **Medical Analysis Agent (Perception & Biomarker Extraction)**: Multi-sequence 3D CNN / Transformer encoders (T1, T1ce, T2, FLAIR), cross-attention fusion, segmentation-assisted ROI pooling, and Grad-CAM explainability.
2. **Clinical Decision Agent (Clinical Reasoning & Synthesis)**: Multi-modal RAG, clinical guideline retrieval (WHO CNS5), and structured diagnostic report generation.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               NEUROVISION AI PLATFORM                                  │
│  [ Clinician Interface: Multi-modal MRI Upload + Patient Data + Radiology Reports ]   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        AGENT 1: MEDICAL ANALYSIS AGENT                                 │
│  (Powered by BraTS Modules A through J)                                                │
│                                                                                        │
│  ┌─────────────────────────┐   ┌──────────────────────────┐   ┌─────────────────────┐  │
│  │ MRI Preprocessing & ROI │   │ Multimodal 3D Encoders   │   │ Explainable AI      │  │
│  │ (Modules A, B, C, D)    │──▶│ & Attention Fusion       │──▶│ (Grad-CAM / XAI)    │  │
│  │ 240x240x155 -> Crop     │   │ (Modules F, G, H, I)     │   │ (Module J)          │  │
│  └─────────────────────────┘   └─────────────┬────────────┘   └─────────────────────┘  │
│                                              │                                         │
│                                              ▼                                         │
│                               ┌───────────────────────────┐                            │
│                               │ Latent Embeddings (512-d) │                            │
│                               │ & HGG/LGG Probabilities   │                            │
│                               └──────────────┬────────────┘                            │
│                                              │                                         │
│                                              ▼                                         │
│                               ┌───────────────────────────┐                            │
│                               │ Visual & Numeric Evidence │                            │
│                               └──────────────┬────────────┘                            │
└──────────────────────────────────────────────┼─────────────────────────────────────────┘
                                               │
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        AGENT 2: CLINICAL DECISION AGENT                                │
│  (Reasoning & Report Generation Engine)                                                │
│                                                                                        │
│  ┌─────────────────────────┐   ┌──────────────────────────┐   ┌─────────────────────┐  │
│  │ Evidence Assimilation   │   │ Medical Knowledge Graph  │   │ Structured Report   │  │
│  │ (Embeddings + Saliency) │──▶│ & Vector DB (RAG)        │──▶│ Generator (FHIR)    │  │
│  │ & EHR Lab Values        │   │ (PubMed / NCCN / WHO)    │   │ & Oncologist Review │  │
│  └─────────────────────────┘   └──────────────────────────┘   └─────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

For complete architectural details, see [NEUROVISION_SYSTEM_ARCHITECTURE.md](NEUROVISION_SYSTEM_ARCHITECTURE.md).

---

## 🚀 Implemented Modules

### 🔍 Module A: Dataset Indexing & Integrity Validation
- **Automated Directory Traversal**: Recursive discovery of patient folders across complex hierarchies.
- **Multimodal Verification**: Verifies presence, orientation, voxel spacing, and dimensions for all four sequences:
  - **T1**: T1-weighted native
  - **T1ce / T1Gd**: T1-weighted contrast-enhanced
  - **T2**: T2-weighted
  - **FLAIR**: Fluid-Attenuated Inversion Recovery
  - **Seg**: Segmentation ground truth masks (Necrotic core, Edema, Enhancing tumor)
- **Metadata Generation**: Outputs structured CSV (`data/dataset_metadata.csv`) and verification summaries (`data/dataset_summary.csv`).
- Comprehensive documentation: [MODULE_A_EXPLANATION.md](MODULE_A_EXPLANATION.md)

### 📊 Module B: Exploratory Data Analysis (EDA) & Volumetrics
- **Intensity Profiling**: Modality-specific intensity distribution estimation (min, max, mean, std, percentiles).
- **Tumor Sub-region Volumetrics**: Voxel counts and volume calculation ($mm^3$) for:
  - **ET**: Enhancing Tumor (Label 4)
  - **TC**: Tumor Core (Labels 1 + 4)
  - **WT**: Whole Tumor (Labels 1 + 2 + 4)
- **Quality Control & Anomaly Detection**: Identifies outlier volumes, empty masks, and zero-slice scans.
- Comprehensive documentation: [MODULE_B_EXPLANATION.md](MODULE_B_EXPLANATION.md)

### 🔬 Module C: MRI Preprocessing & Patient-Level Splitting
- **Zero-Data-Leakage Splitting**: Partitions cohort into strictly disjoint, patient-level splits (70% Train: 258, 15% Val: 55, 15% Test: 56) with verified class stratification ($p = 0.9689$).
- **Non-Zero Z-Score Normalization**: Implements $(I(x) - \mu_{nz}) / \sigma_{nz}$ with $[P_1, P_{99}]$ percentile clamping, safeguarding soft-tissue contrast while preserving background air at $0.0$.
- **Spatial Standardization**: Automated brain foreground bounding box extraction and symmetric center-crop/padding to $4 \times 128 \times 128 \times 128$ model-ready tensors.
- **Nearest-Neighbor Mask Resampling**: Preserves categorical ground truth labels $\{0, 1, 2, 4\}$ without spurious fractional artifacts.
- Comprehensive documentation: [MODULE_C_EXPLANATION.md](MODULE_C_EXPLANATION.md)

---

## 📈 Analysis & Visualizations

The pipeline automatically generates high-resolution figures in `reports/figures/`:

| Visualization | Description |
|---|---|
| `multimodal_mri_slices.png` | Axial slices across T1, T1ce, T2, FLAIR with overlaid tumor segmentation masks |
| `class_distribution.png` | Cohort distribution across High-Grade Glioma (HGG) and Low-Grade Glioma (LGG) |
| `modality_intensity_distributions.png` | Sequence-wise pixel intensity distributions comparing skull-stripped regions |
| `tumor_volume_distributions.png` | Log-scale volume distributions of ET, TC, and WT across tumor grades |
| `correlation_matrix.png` | Correlation heatmaps between tumor sub-region volumes and clinical attributes |

---

## 📂 Repository Structure

```
learnNeuro/
├── configs/                          # Pipeline configuration files (YAML)
│   ├── dataset_config.yaml           # Module A configuration (paths, sequences, validation rules)
│   ├── eda_config.yaml               # Module B configuration (metrics, figure settings, thresholds)
│   └── preprocessing_config.yaml     # Module C configuration (splits, normalization, shapes)
├── data/                             # Generated metadata, splits and data dictionaries
│   ├── data_dictionary.json          # Schema definitions and data types
│   ├── dataset_metadata.csv          # Per-case sequence paths, dimensions, voxel spacings
│   ├── dataset_summary.csv           # Aggregate dataset metrics and validation flags
│   ├── tumor_statistics.csv          # Sub-region volumetric and intensity statistics
│   └── splits/                       # Zero-leakage patient-level stratified partitions
│       ├── train_subjects.csv        # 258 training subjects (79.5% HGG, 20.5% LGG)
│       ├── val_subjects.csv          # 55 validation subjects (80.0% HGG, 20.0% LGG)
│       ├── test_subjects.csv         # 56 test subjects (78.6% HGG, 21.4% LGG)
│       └── split_summary.json        # Formal verification audit & distribution hashes
├── docs/                             # Project planning and roadmap documents
│   ├── BraTS_Brain_Tumor_Classification_2_Month_Project_Plan.docx
│   └── project_plan_extracted.txt    # 12-Module weekly milestone plan
├── reports/
│   └── figures/                      # Generated publication-quality figures
├── scripts/                          # Entry-point execution scripts
│   ├── run_module_a.py               # Run dataset indexing and validation pipeline
│   ├── run_module_b.py               # Run exploratory data analysis and QC pipeline
│   └── run_module_c.py               # Run MRI preprocessing & patient splitting pipeline
├── src/                              # Source package modules
│   ├── module_a_dataset/             # Indexing, NIfTI utilities, integrity validators
│   ├── module_b_eda/                 # Volumetrics, intensity profiler, statistical analyzer, visualizer
│   ├── module_c_preprocessing/       # Normalizer, spatial cropper, patient splitter, preprocessor
│   └── utils/                        # Logging and general helpers
├── tests/                            # Unit and integration test suites
│   ├── test_module_a.py
│   ├── test_module_b.py
│   └── test_module_c.py
├── MODULE_A_EXPLANATION.md           # Deep-dive theoretical and technical guide for Module A
├── MODULE_B_EXPLANATION.md           # Deep-dive theoretical and technical guide for Module B
├── MODULE_C_EXPLANATION.md           # Deep-dive theoretical and technical guide for Module C
├── NEUROVISION_SYSTEM_ARCHITECTURE.md# Complete system architecture specification
├── requirements.txt                  # Python dependencies
└── README.md                         # Project overview and quickstart
```

---

## ⚡ Quickstart & Usage

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/Yashgohel018/neurovision-sgp5.git
cd neurovision-sgp5

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Module A (Dataset Indexer & Validator)
```bash
python scripts/run_module_a.py --config configs/dataset_config.yaml
```

### 3. Run Module B (EDA & Volumetrics Pipeline)
```bash
python scripts/run_module_b.py --config configs/eda_config.yaml
```

### 4. Run Module C (MRI Preprocessing & Patient Splitting)
```bash
python scripts/run_module_c.py --config configs/preprocessing_config.yaml
```

### 5. Run Automated Test Suite
```bash
python -m unittest discover -s tests
```

---

## 🗺️ 12-Module Implementation Roadmap

- [x] **Module A**: Dataset Indexing, Path Discovery & Integrity Verification
- [x] **Module B**: Exploratory Data Analysis (EDA) & Volumetric Quality Control
- [x] **Module C**: Voxel Intensity Normalization & Zero-Leakage Patient Splitting
- [ ] **Module D**: 3D ROI Cropping & Foreground Mask Extraction
- [ ] **Module E**: Multimodal 3D Data Loader with Elastic/Spatial Augmentations
- [ ] **Module F**: 3D ResNet Baseline Classifier
- [ ] **Module G**: Multimodal Parallel-Branch Fusion Network
- [ ] **Module H**: Multi-Head Spatial & Channel Cross-Attention Fusion
- [ ] **Module I**: Segmentation-Guided Mask-Attention Mechanism
- [ ] **Module J**: Explainable AI (3D Grad-CAM & Integrated Gradients)
- [ ] **Module K**: Clinical Decision Agent & RAG Evidence Retrieval
- [ ] **Module L**: Streamlit / FastAPI Clinical Deployment & Diagnostic Interface

---

## 📄 License
This project is licensed under the MIT License.
