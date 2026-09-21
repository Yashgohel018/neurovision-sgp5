# NeuroVision AI: System Architecture & Implementation Bridge
## Harmonizing the Autonomous 2-Agent CDSS with the BraTS 12-Module Deep Learning Engine

> **Project:** NeuroVision AI – Autonomous MRI Clinical Decision Support System  
> **Scientific Core:** MultiModal-BrainNet (BraTS 2020 Multi-Sequence Deep Learning)  
> **Repository Root:** `learnNeuro/`

---

## 1. System Vision & Architecture Mapping

**NeuroVision AI** is designed as an enterprise-grade Clinical Decision Support System (CDSS) powered by **two collaborating AI agents**:
1. **The Medical Analysis Agent (Perception & Perception-Level Retrieval)**
2. **The Clinical Decision Agent (Reasoning, RAG & Clinical Synthesis)**

The 12-module roadmap provided by your guide/professor (`BraTS_Brain_Tumor_Classification_2_Month_Project_Plan.docx`) serves as the **rigorous deep learning, computer vision, and medical imaging engine** that powers the Medical Analysis Agent and feeds structured evidence into the Clinical Decision Agent.

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
│                               │ Vector Similarity Search  │                            │
│                               │ (FAISS / Historical Cases)│                            │
│                               └──────────────┬────────────┘                            │
└──────────────────────────────────────────────┼─────────────────────────────────────────┘
                                               │ Structured Image Findings + Embeddings
                                               │ + Patient Demographics (Age, Resection)
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        AGENT 2: CLINICAL DECISION AGENT                                │
│  (Reasoning & Decision Support)                                                        │
│                                                                                        │
│  ┌─────────────────────────┐   ┌──────────────────────────┐   ┌─────────────────────┐  │
│  │ Medical Literature RAG  │   │ Multi-Agent Clinical     │   │ Structured CDSS     │  │
│  │ (WHO CNS Guidelines,    │──▶│ Reasoning (LangGraph)    │──▶│ Report Generation   │  │
│  │ Neuro-oncology Evidence)│   │ Evidence Integration     │   │ & Follow-up Actions │  │
│  └─────────────────────────┘   └──────────────────────────┘   └─────────────────────┘  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│              CLINICIAN DECISION SUPPORT REPORT & INTERFACE (Module L)                  │
│  - Predicted Tumor Grade (HGG vs LGG) & Calibrated Confidence                          │
│  - 3D Tumor Sub-region Volumes (WT, TC, ET in cm³)                                     │
│  - 3D Grad-CAM Visual Heatmaps overlaid on MRI                                         │
│  - Top-3 Clinically Similar Historical Cases                                           │
│  - Evidence-Based Diagnostic Rationale & Literature Citations                          │
│  - Suggested Next Steps (Perfusion MRI, Spectroscopy, Molecular Markers IDH/1p19q)    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module-to-Agent Mapping Matrix

Here is how each module from your professor's plan directly fulfills the NeuroVision AI vision:

| Module | Title | Role in NeuroVision AI | Primary Output / Deliverable |
| :--- | :--- | :--- | :--- |
| **Module A** | Dataset Acquisition & Indexing | Establishes the foundational multimodal registry, links HGG/LGG labels and clinical covariates (Age, Survival), handles naming quirks. | `dataset_metadata.csv`, `data_dictionary.json` *(COMPLETED)* |
| **Module B** | EDA & Data Quality Validation | Quantifies intensity profiles, tumor volumes (WT, TC, ET in $\text{cm}^3$), checks outliers, and exports 2D orthogonal slices with mask overlays. | `dataset_summary.csv`, `tumor_statistics.csv`, QC figures |
| **Module C** | MRI Preprocessing & Splitting | Implements skull-stripping validation, non-zero z-score normalization, spatial resampling, and patient-level train/val/test splitting (zero leakage). | `preprocessing_config.yaml`, preprocessed volumes, split manifests |
| **Module D** | Tumor Segmentation & ROI Extraction | Extracts 3D tumor bounding boxes with anatomical margins. In the final CDSS, allows automated tumor localization without requiring manual ground truth masks. | 3D Tumor ROI tensors, tumor spatial coordinates |
| **Module E** | Synchronized 3D Augmentation | Augments 3D multimodal volumes (rotations, intensity scaling, affine) to ensure clinical model robustness across diverse scanner types. | `augmentation_config.yaml`, transformed sample QA |
| **Module F** | Baseline Classification Models | Establishes baseline diagnostic benchmarks (2D ResNet, 3D CNN, single-modality) to prove superiority of multimodal fusion. | `baseline_results.csv`, benchmark model weights |
| **Module G** | Proposed Multimodal Architecture | 4 parallel 3D ResNet encoders (T1, T1ce, T2, FLAIR) + multimodal cross-attention fusion. Extracts the dense 512-d feature vectors for **Agent 1's Similar Case Retrieval**. | Best model architecture, latent feature extractor |
| **Module H** | Training & Optimization Pipeline | Handles 3.86:1 class imbalance using Weighted BCE / Focal Loss, mixed precision (FP16), early stopping, and TensorBoard logging. | `best_model.pt`, training history logs |
| **Module I** | Clinical Model Evaluation | Computes clinical metrics: Sensitivity (Recall), Specificity, Precision, F1-Score, ROC-AUC, PR-AUC, and 95% Confidence Intervals. | `final_metrics.csv`, ROC/PR curves, confusion matrices |
| **Module J** | Explainable AI (Grad-CAM) | Generates 3D activation heatmaps showing which voxels triggered the classification. Powers **Agent 1's Visual Explanation**. | Grad-CAM heatmaps, MRI-overlay visual reports |
| **Module K** | Ablation Study | Rigorously proves which modalities (T1 vs T1ce vs T2 vs FLAIR) contribute most to diagnosis, defending system design in reviews and papers. | `ablation_results.csv`, modality importance rankings |
| **Module L** | Web-Based CDSS Demonstration | Integrates the trained deep learning engine with the web interface, demonstrating the full multi-agent decision support report for clinicians. | Interactive Web Application (Streamlit / FastAPI + React) |

---

## 3. The Two-Agent Workflow in Detail

### Agent 1: Medical Analysis Agent (Perception & Retrieval Specialist)
* **Inputs:** Raw 3D MRI scans (T1, T1ce, T2, FLAIR), patient demographics (Age, Gender, Symptoms).
* **Processing Steps:**
  1. **Spatial Validation & Normalization:** Ingests NIfTI files, verifies orientation (LPS) and dimensions ($240 \times 240 \times 155$), applies non-zero z-score normalization.
  2. **Tumor Localization & Volumetrics:** Computes bounding box around tumor, calculates physical volumes of Enhancing Tumor (ET), Tumor Core (TC), and Whole Tumor (WT) in $\text{cm}^3$.
  3. **Multimodal Feature Extraction:** Passes the 4 sequences through the 3D deep learning encoders to generate a dense, unified latent embedding vector $\mathbf{z} \in \mathbb{R}^{512}$.
  4. **Grad-CAM Saliency Map Generation:** Computes gradients of the predicted class score with respect to the final convolutional feature maps, producing 3D heatmaps of suspicious tissue.
  5. **Case-Based Retrieval (Vector Search):** Queries the FAISS vector index of previously diagnosed patients using embedding $\mathbf{z}$ to find the **Top-3 most clinically similar cases** with known histopathology and survival outcomes.
* **Outputs to Agent 2:**
  - Tumor Grade Prediction: High-Grade Glioma (Probability: 92.4%) vs. Low-Grade Glioma.
  - Saliency Heatmaps & Key Slices.
  - Volumetric Breakdown (WT: $42.1\text{ cm}^3$, ET: $18.4\text{ cm}^3$, Edema: $23.7\text{ cm}^3$).
  - Top-3 Similar Cases with similarity scores and patient outcomes.

---

### Agent 2: Clinical Decision Agent (Reasoning, RAG & Report Specialist)
* **Inputs:** Structured findings from Agent 1 + Patient history + Radiology report text (if available).
* **Processing Steps:**
  1. **Retrieval-Augmented Generation (RAG):**
     - Queries local vector database containing curated medical literature: WHO Classification of Tumors of the Central Nervous System (5th Edition), NCCN Clinical Practice Guidelines in Oncology, and AJNR neuroimaging protocols.
     - Retrieves evidence concerning the specific imaging features identified (e.g., ring enhancement, thick irregular margins, extensive vasogenic edema, mass effect).
  2. **Clinical Reasoning Synthesis:**
     - Correlates the MRI findings with patient age (e.g., 65-year-old presenting with large ring-enhancing lesion strongly points to Glioblastoma).
     - Cross-references the Top-3 similar historical cases to assess treatment paths and survival trajectories.
  3. **Differential Diagnosis & Uncertainty Quantification:**
     - Computes calibrated confidence intervals.
     - Identifies potential mimic conditions (e.g., brain abscess, solitary metastasis, tumefactive demyelination) and explains why glioma is more likely.
  4. **Actionable Recommendations:**
     - Recommends appropriate confirmatory investigations (e.g., Perfusion MRI to assess cerebral blood volume [rCBV], MR Spectroscopy for choline/creatine ratio, stereotactic biopsy, or molecular testing for IDH1 mutation and MGMT promoter methylation).
  5. **Report Generation:**
     - Formats the entire synthesis into an interpretable, clinician-ready Decision Support Report.

---

## 4. Immediate Development Roadmap

With **Module A completed**, the unified pipeline proceeds as follows:

```
[Module A] Dataset Ingestion & Metadata Registry (COMPLETED)
     │
     ▼
[Module B] Exploratory Data Analysis & Volumetric QC (NEXT STEP)
     │  ├── Tumor volume distribution (WT, TC, ET in cm³)
     │  ├── Multi-sequence intensity profiling & signal-to-noise check
     │  └── Multi-modal 2D orthogonal slice generator with mask overlays
     │
     ▼
[Module C] Preprocessing Pipeline & Patient-Stratified Splits
     │  ├── Leakage-free train/val/test splits (70 / 15 / 15)
     │  ├── Non-zero z-score normalization
     │  └── 3D volume cropping & caching
     │
     ▼
[Modules D - J] Deep Learning Engine (Agent 1 Core)
     │  ├── 3D Multimodal Attention Network
     │  ├── Model Training & Imbalance Handling (Weighted BCE / Focal Loss)
     │  └── 3D Grad-CAM Explainability Engine
     │
     ▼
[Vector Retrieval & Agent 2 RAG Integration]
     │  ├── FAISS Vector Store of patient embeddings
     │  └── LangGraph-driven Clinical Decision Agent
     │
     ▼
[Module L] Full NeuroVision AI Web Application
```
