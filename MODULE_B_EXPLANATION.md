# NeuroVision AI — Module B: Exploratory Data Analysis & Volumetric Quality Control
## Quantitative Tumor Profiling, Clinical Hypothesis Testing & Multimodal Quality Control

> **Project:** NeuroVision AI – Autonomous MRI Clinical Decision Support System (CDSS)  
> **Scientific Core:** MultiModal-BrainNet (BraTS 2020 Deep Learning Engine)  
> **System Architecture:** Two-Agent Autonomous CDSS (Medical Analysis Agent + Clinical Decision Agent)  
> **Workspace Directory:** `learnNeuro/`  
> **Module Status:** COMPLETED & EMPIRICALLY VALIDATED (369/369 subjects profiled)

---

## Table of Contents
1. [Executive Overview: NeuroVision AI & The Role of Module B](#1-executive-overview-neurovision-ai--the-role-of-module-b)
2. [The Two-Agent Architecture: How Module B Feeds Both Agents](#2-the-two-agent-architecture-how-module-b-feeds-both-agents)
3. [Physical Tumor Sub-Regions: Biology, Labels & Mathematical Formulation](#3-physical-tumor-sub-regions-biology-labels--mathematical-formulation)
4. [Empirical Discovery: The Enhancing Fraction Biomarker ($p < 0.001$)](#4-empirical-discovery-the-enhancing-fraction-biomarker-p--0001)
5. [Outlier & Clinical Edge Case Detection: Non-Enhancing & Giant Lesions](#5-outlier--clinical-edge-case-detection-non-enhancing--giant-lesions)
6. [Multi-Sequence Intensity Profiling & Parameters for Module C](#6-multi-sequence-intensity-profiling--parameters-for-module-c)
7. [Detailed Code Walkthrough & Engine Architecture](#7-detailed-code-walkthrough--engine-architecture)
8. [Cohort Summary & Empirical Findings Table](#8-cohort-summary--empirical-findings-table)
9. [Master Viva & Project Evaluation Q&A (10 High-Yield Questions)](#9-master-viva--project-evaluation-qa-10-high-yield-questions)
10. [Seamless Transition: How Module B Connects to Module C](#10-seamless-transition-how-module-b-connects-to-module-c)

---

## 1. Executive Overview: NeuroVision AI & The Role of Module B

### The Clinical Challenge
In clinical neuro-oncology, a glioma cannot be characterized solely by its total volume. Two tumors of identical $80\text{ cm}^3$ volume may have radically different biological behaviors: one may be an aggressive **Glioblastoma (WHO Grade IV)** with a thick hypervascularized enhancing rim and central liquefactive necrosis, while the other may be an infiltrative **Oligodendroglioma (WHO Grade II)** dominated by non-enhancing edema. 

Raw pixel grids and unverified images cannot be fed blindly into deep learning models. Before training 3D convolutional neural networks or Vision Transformers, clinicians and medical AI engineers must:
1. Quantify the physical volumes ($\text{cm}^3$) of active tumor sub-regions.
2. Characterize multi-sequence MRI intensity distributions to design artifact-resistant normalization.
3. Statistically validate that the imaging features correlate with true histopathological diagnosis.
4. Detect extreme anomalies (e.g. microscopic tumors, giant lesions, non-enhancing cases).

### The Solution: Module B
**Module B** transforms the master file registry produced in Module A (`dataset_metadata.csv`) into a rich **quantitative volumetric and statistical database** (`tumor_statistics.csv`, `dataset_summary.csv`, and 5 high-resolution diagnostic figures).

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                NEUROVISION AI PLATFORM                                 │
│                                                                                        │
│   ┌────────────────────────────────────────────────────────────────────────────────┐   │
│   │                        MODULE A: MULTIMODAL REGISTRY                           │   │
│   │   Ingested BraTS 2020 (369 subjects) ──▶ data/dataset_metadata.csv [COMPLETED] │   │
│   └───────────────────────────────────────┬────────────────────────────────────────┘   │
│                                           │                                            │
│                                           ▼                                            │
│   ┌────────────────────────────────────────────────────────────────────────────────┐   │
│   │                        MODULE B: EDA & VOLUMETRIC QC                           │   │
│   │   - 3D Tumor Sub-region Volumetrics (WT, TC, ET, ED, NCR in cm³ & voxels)      │   │
│   │   - Multi-Sequence Intensity Profiling (T1, T1ce, T2, FLAIR percentiles)       │   │
│   │   - Statistical Profiling: HGG vs LGG (Mann-Whitney U & Volumetric Biomarkers) │   │
│   │   - Multimodal Orthogonal Slice Visualization with Sub-region Mask Overlays    │   │
│   │   - Outlier & Anomaly Detection (Micro-tumors, massive lesions, non-enhancing) │   │
│   │   ──▶ Exports: tumor_statistics.csv, dataset_summary.csv, QC Figures, Report   │   │
│   └───────────────────────┬────────────────────────────────┬───────────────────────┘   │
│                           │                                │                           │
│        [Physical Features]│                                │  [Statistical Evidence]   │
│        Bounding boxes &   │                                │  Volumetric differences & │
│        volumetrics (cm³)  │                                │  clinical correlation     │
│                           ▼                                ▼                           │
│   ┌─────────────────────────────────┐            ┌─────────────────────────────────┐   │
│   │  AGENT 1: MEDICAL ANALYSIS      │            │  AGENT 2: CLINICAL DECISION     │   │
│   │  - Feeds Module C Preprocessing │            │  - Evidence RAG synthesis       │   │
│   │  - Configures z-score clipping  │            │  - Tumor burden prognosticating │   │
│   │  - Informs 3D ROI crops (Mod D) │            │  - Differential diagnosis logic │   │
│   └─────────────────────────────────┘            └─────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Two-Agent Architecture: How Module B Feeds Both Agents

### 1. How Module B Powers Agent 1 (Medical Analysis Agent)
* **Spatial Bounding Box Geometry:** Module B calculates the exact 3D spatial extents $[x_{\min}, x_{\max}]$, $[y_{\min}, y_{\max}]$, $[z_{\min}, z_{\max}]$ and spans $(\Delta x, \Delta y, \Delta z)$ for every patient. In **Module D**, Agent 1 uses these coordinates to crop 3D tumor ROIs with anatomical safety margins (e.g., $128 \times 128 \times 128$), reducing background air voxels by over 80%.
* **Intensity Normalization Thresholds:** By computing the 1st and 99th non-zero percentiles ($P_1, P_{99}$) across T1, T1ce, T2, and FLAIR, Module B establishes the robust clipping boundaries for **Module C**'s z-score pipeline.
* **Tumor Centroids $(\bar{x}, \bar{y}, \bar{z})$:** Identifies the physical center of mass of the tumor, ensuring that key 2D visualization panels and 3D Grad-CAM saliency heatmaps (Module J) center precisely on the lesion.

### 2. How Module B Powers Agent 2 (Clinical Decision Agent)
* **Evidence-Based Volumetric Biomarkers:** Module B provides the empirical proof that Enhancing Tumor volume ($p < 0.001$) and Enhancing Fraction ($\text{ET}/\text{WT}$, $p < 0.001$) decisively differentiate HGG from LGG. Agent 2 integrates this statistical evidence into its LangGraph multi-agent reasoning chain.
* **Prognostic Covariate Correlation:** Agent 2 combines volumetric metrics with patient age (mean: 61.2 years) and survival days. In glioblastoma, large non-enhancing necrotic cores alongside high age correlate with poorer overall survival, forming the clinical rationale for recommended adjuvant chemotherapy.
* **Edge-Case Safety Guardrails:** Module B flags 27 non-enhancing cases ($\text{ET} = 0$). Agent 2 uses this flag to prevent diagnostic hallucinations and to alert the clinician that a non-enhancing tumor may represent an IDH-mutant Low-Grade Astrocytoma or Oligodendroglioma.

---

## 3. Physical Tumor Sub-Regions: Biology, Labels & Mathematical Formulation

### BraTS 2020 Ground Truth Voxel Labels
Each ground-truth mask (`seg.nii`) contains discrete integer labels:
* **Label 0:** Background (healthy brain tissue / skull-stripped air)
* **Label 1:** Necrotic and Non-Enhancing Tumor Core (NCR/NET)
* **Label 2:** Peritumoral Vasogenic Edema (ED)
* **Label 4:** GD-Enhancing Tumor (ET) *(Note: Label 3 was eliminated in BraTS 2017+ cohorts)*

### Clinical Sub-Region Composites
The clinical evaluation protocols define three nested evaluation regions:
1. **Whole Tumor (WT):** Labels $\{1, 2, 4\}$ — Encompasses the entire pathological lesion (active core, necrosis, and infiltrative peritumoral edema).
2. **Tumor Core (TC):** Labels $\{1, 4\}$ — Represents the surgically resectable gross tumor bulk (enhancing rim + necrotic core), excluding non-tumor edema.
3. **Enhancing Tumor (ET):** Label $\{4\}$ — The active, viable, neo-vascularized tumor rim characterized by disrupted blood-brain barrier permeability.

### Mathematical Formulations

#### 1. Physical Voxel Volume
Given voxel spacings $(s_x, s_y, s_z)$ in millimeters (in BraTS 2020: $1.0 \times 1.0 \times 1.0\text{ mm}^3$):
$$V_{\text{voxel}} = \frac{s_x \times s_y \times s_z}{1000.0} = \frac{1.0 \times 1.0 \times 1.0}{1000.0} = 0.001\text{ cm}^3$$

#### 2. Physical Sub-Region Volume ($\text{cm}^3$)
For any sub-region $R \in \{\text{WT}, \text{TC}, \text{ET}, \text{ED}, \text{NCR}\}$:
$$V(R) = N_R \times V_{\text{voxel}} = \frac{N_R}{1000.0}\text{ cm}^3$$
where $N_R$ is the total voxel count belonging to region $R$.

#### 3. Enhancing Fraction ($\text{ET} / \text{WT}$)
$$\text{Ratio}_{\text{ET/WT}} = \frac{V(\text{ET})}{V(\text{WT})}$$
This dimensionless ratio measures the proportion of the lesion undergoing aggressive neo-angiogenesis.

#### 4. 3D Tumor Centroid
The center of mass $(\bar{x}, \bar{y}, \bar{z})$ of Whole Tumor voxels:
$$\bar{x} = \frac{1}{N_{\text{WT}}} \sum_{i=1}^{N_{\text{WT}}} x_i, \quad \bar{y} = \frac{1}{N_{\text{WT}}} \sum_{i=1}^{N_{\text{WT}}} y_i, \quad \bar{z} = \frac{1}{N_{\text{WT}}} \sum_{i=1}^{N_{\text{WT}}} z_i$$

---

## 4. Empirical Discovery: The Enhancing Fraction Biomarker ($p < 0.001$)

The empirical findings from our execution across all 369 patients revealed fundamental biological differences between glioma grades:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                HGG vs. LGG EMPIRICAL VOLUMETRIC COMPARISON (N=369)                     │
├──────────────────────────────┬──────────────────┬──────────────────┬───────────────────┤
│ Metric                       │ HGG Median (IQR) │ LGG Median (IQR) │ Significance      │
├──────────────────────────────┼──────────────────┼──────────────────┼───────────────────┤
│ Enhancing Tumor Volume (ET)  │ 19.46 cm³ (23.38)│  0.82 cm³ (3.93) │ p < 0.001 (***)   │
│ Enhancing Fraction (ET / WT) │ 0.224     (0.19) │  0.007     (0.05)│ p < 0.001 (***)   │
│ Peritumoral Edema (ED)       │ 51.35 cm³ (53.86)│ 34.20 cm³ (53.07)│ p = 0.028 (*)     │
│ Tumor Core Volume (TC)       │ 32.54 cm³ (42.31)│ 44.59 cm³ (76.16)│ p = 0.008 (*)     │
│ Whole Tumor Volume (WT)      │ 91.13 cm³ (85.57)│ 86.32 cm³ (123.2)│ p = 0.347 (ns)    │
└──────────────────────────────┴──────────────────┴──────────────────┴───────────────────┘
```

### Key Biological Takeaways
1. **Total Tumor Volume (WT) Does NOT Distinguish Grade ($p = 0.347$, not significant):**  
   Low-Grade Gliomas can grow to enormous sizes (median: $86.3\text{ cm}^3$) prior to diagnosis because they lack rapid mass effect and severe edema. Therefore, a deep learning model trained only on total tumor volume cannot separate HGG from LGG.
2. **Enhancing Tumor (ET) is the Gold-Standard Discriminator ($p < 0.001$):**  
   HGG exhibits a median enhancing volume of $19.46\text{ cm}^3$ compared to only $0.82\text{ cm}^3$ for LGG.
3. **Enhancing Fraction ($\text{ET} / \text{WT}$) is a Decisive Diagnostic Feature:**  
   In HGG, active enhancing tissue accounts for $22.4\%$ of the total lesion. In LGG, enhancing tissue accounts for less than $0.7\%$.

---

## 5. Outlier & Clinical Edge Case Detection: Non-Enhancing & Giant Lesions

Module B automated outlier detection across all 369 patients, identifying three crucial edge cases:

1. **Non-Enhancing Tumors ($\text{ET} = 0\text{ cm}^3$): 27 Subjects Identified**
   * *Clinical Significance:* Certain Low-Grade Gliomas (specifically WHO Grade II Diffuse Astrocytomas and Oligodendrogliomas) have an intact blood-brain barrier and do not take up Gadolinium contrast.
   * *System Guardrail:* Deep learning models that rely solely on T1ce would fail on these cases. Multimodal fusion across T2 and FLAIR is mandatory.
2. **Massive Tumors ($\text{WT} > 150\text{ cm}^3$): 79 Subjects Identified**
   * *Computational Significance:* Tumors exceeding $150\text{ cm}^3$ create substantial midline shift and ventricle compression. In Module D, ROI bounding boxes for these cases require adaptive cropping so anatomical context is preserved.
3. **Microscopic Tumors ($\text{WT} < 5\text{ cm}^3$): 0 Subjects**
   * *Verification:* All 369 tumors are well-developed macro-lesions (minimum volume: $5.8\text{ cm}^3$), eliminating concerns of unresolvable sub-voxel artifacts.

---

## 6. Multi-Sequence Intensity Profiling & Parameters for Module C

Multi-sequence MRI intensities do not possess standardized physical units (unlike CT Hounsfield Units). Scanner gain, RF coil sensitivities, and patient geometry introduce wide intensity variations.

Module B extracted non-zero voxel distributions across the cohort to determine normalization parameters for Module C:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               MULTI-SEQUENCE NON-ZERO INTENSITY PROFILES (BraTS 2020)                  │
├───────────┬──────────────┬──────────────┬──────────────────┬───────────────────────────┤
│ Modality  │ 1st Pct (P1) │ 99th Pct(P99)│ Dynamic Range    │ Clinical Diagnostic Role  │
├───────────┼──────────────┼──────────────┼──────────────────┼───────────────────────────┤
│ T1        │ 12.0         │ 685.0        │ 0 – 1,100        │ Anatomical Reference      │
│ T1ce      │ 15.0         │ 890.0        │ 0 – 1,650        │ Vascular Rim Enhancement  │
│ T2        │ 24.0         │ 940.0        │ 0 – 1,800        │ Vasogenic Edema Signal    │
│ FLAIR     │ 18.0         │ 820.0        │ 0 – 1,450        │ CSF-Suppressed Edema      │
└───────────┴──────────────┴──────────────┴──────────────────┴───────────────────────────┘
```

### Direct Configuration for Module C Preprocessing:
1. **Foreground Masking:** Ignore all voxels with $I(x) = 0$.
2. **Robust Percentile Clipping:** Clip intensities to $[P_1, P_{99}]$ to eliminate extreme scanner spikes and RF hot spots.
3. **Non-Zero Z-Score Normalization:**
   $$\hat{I}(x) = \frac{I(x) - \mu_{\text{non-zero}}}{\sigma_{\text{non-zero}}}$$
   Ensuring all 4 modalities share zero mean and unit variance for 3D tensor processing.

---

## 7. Detailed Code Walkthrough & Engine Architecture

All Module B components are organized in `learnNeuro/`:

```
learnNeuro/
├── configs/
│   ├── dataset_config.yaml         # Module A configuration
│   └── eda_config.yaml             # Module B configuration (thresholds, palettes, DPI)
├── data/
│   ├── dataset_metadata.csv        # Master patient index (Module A)
│   ├── data_dictionary.json        # Formal data schema (Module A)
│   ├── tumor_statistics.csv        # Patient-level volumetric metrics (Module B)
│   └── dataset_summary.csv         # Cohort comparative summary table (Module B)
├── reports/
│   └── figures/                    # High-resolution publication-quality PNGs (300 DPI)
│       ├── class_distribution.png
│       ├── tumor_volume_distributions.png
│       ├── modality_intensity_distributions.png
│       ├── correlation_matrix.png
│       └── multimodal_mri_slices.png
├── src/
│   ├── module_a_dataset/           # Module A Ingestion Engine
│   └── module_b_eda/               # Module B EDA & Volumetric Engine
│       ├── __init__.py
│       ├── volumetrics.py          # Vectorized sub-region quantification & 3D bboxes
│       ├── intensity_profiler.py   # Multi-sequence non-zero intensity percentiles
│       ├── statistical_analyzer.py # Mann-Whitney U testing & clinical correlations
│       ├── visualizer.py           # Publication-grade plotting & orthogonal slice montages
│       └── eda_pipeline.py         # End-to-end pipeline orchestrator
├── scripts/
│   ├── run_module_a.py             # Module A runner
│   └── run_module_b.py             # Module B runner (CLI with full executive report)
├── tests/
│   ├── test_module_a.py            # Module A test suite (3/3 passing)
│   └── test_module_b.py            # Module B test suite (4/4 passing)
├── MODULE_A_EXPLANATION.md         # Reference guide for Module A
└── MODULE_B_EXPLANATION.md         # This comprehensive reference guide
```

---

## 8. Cohort Summary & Empirical Findings Table

The formal output saved to [`learnNeuro/data/dataset_summary.csv`](file:///c:/Users/yashg/Documents/sgp_5/learnNeuro/data/dataset_summary.csv):

| Parameter | Cohort Total ($N=369$) | HGG ($N=293, 79.4\%$) | LGG ($N=76, 20.6\%$) | $P$-Value | Significance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cohort Patient Count** | 369 (100.0%) | 293 (79.4%) | 76 (20.6%) | N/A | N/A |
| **Whole Tumor Volume (WT)** [Median (IQR)] | 90.74 cm³ (94.39) | 91.13 cm³ (85.57) | 86.32 cm³ (123.21) | 0.3468 | ns (not significant) |
| └─ Mean ± Std | 99.55 ± 59.43 | 96.44 ± 53.78 | 111.54 ± 76.72 | - | - |
| **Tumor Core Volume (TC)** [Median (IQR)] | 33.81 cm³ (45.79) | 32.54 cm³ (42.31) | 44.59 cm³ (76.16) | 0.0084 | * ($p < 0.05$) |
| └─ Mean ± Std | 41.82 ± 35.34 | 37.66 ± 29.26 | 57.85 ± 49.66 | - | - |
| **Enhancing Tumor Volume (ET)** [Median (IQR)] | 15.09 cm³ (25.63) | 19.46 cm³ (23.38) | 0.82 cm³ (3.93) | < 0.001 | *** ($p < 0.001$) |
| └─ Mean ± Std | 19.70 ± 19.02 | 23.39 ± 18.50 | 5.46 ± 13.53 | - | - |
| **Peritumoral Edema Volume (ED)** [Median (IQR)] | 49.55 cm³ (54.76) | 51.35 cm³ (53.86) | 34.20 cm³ (53.07) | 0.0285 | * ($p < 0.05$) |
| └─ Mean ± Std | 57.73 ± 40.12 | 58.78 ± 37.60 | 53.70 ± 48.72 | - | - |
| **Necrotic Core Volume (NCR)** [Median (IQR)] | 10.56 cm³ (21.52) | 9.16 cm³ (16.63) | 42.84 cm³ (74.38) | < 0.001 | *** ($p < 0.001$) |
| └─ Mean ± Std | 22.12 ± 29.86 | 14.27 ± 15.80 | 52.38 ± 47.27 | - | - |
| **Enhancing Fraction (ET / WT)** [Median (IQR)] | 0.200 (0.212) | 0.224 (0.192) | 0.007 (0.046) | < 0.001 | *** ($p < 0.001$) |
| └─ Mean ± Std | 0.200 ± 0.15 | 0.242 ± 0.13 | 0.053 ± 0.11 | - | - |
| **Patient Age (years)** [Mean ± Std] | 61.2 ± 11.9 | 61.2 ± 11.9 | N/A | N/A | N/A |

---

## 9. Master Viva & Project Evaluation Q&A (10 High-Yield Questions)

### Q1: What is the primary purpose of Module B in the NeuroVision AI architecture?
> **Answer:** "Module B performs Exploratory Data Analysis and volumetric quality control. It extracts physical 3D sub-region tumor volumes ($\text{cm}^3$) from ground-truth masks, profiles non-zero multi-sequence MRI intensities, statistically compares HGG vs. LGG cohorts using Mann-Whitney U hypothesis testing, detects clinical outliers (non-enhancing and giant tumors), and renders publication-quality orthogonal slice montages."

### Q2: What are the three nested tumor sub-regions evaluated in BraTS, and what label combinations define them?
> **Answer:** "The three sub-regions are:
> 1. **Whole Tumor (WT):** Labels 1 (NCR), 2 (ED), and 4 (ET).
> 2. **Tumor Core (TC):** Labels 1 (NCR) and 4 (ET), representing the gross resectable tumor core.
> 3. **Enhancing Tumor (ET):** Label 4, representing active, vascularized tissue with blood-brain barrier disruption."

### Q3: Why does Whole Tumor (WT) volume fail to statistically differentiate HGG from LGG ($p = 0.3468$)?
> **Answer:** "Low-Grade Gliomas are slow-growing, indolent neoplasms that can expand over months or years without creating acute neurological symptoms, allowing them to reach very large volumes before clinical discovery (LGG median volume: $86.32\text{ cm}^3$ vs HGG median volume: $91.13\text{ cm}^3$). Therefore, gross tumor volume alone is not a reliable biomarker of histological malignancy."

### Q4: Which volumetric feature is the strongest predictor of High-Grade Glioma malignancy?
> **Answer:** "Enhancing Tumor volume (ET) and the Enhancing Fraction ($\text{ET} / \text{WT}$) are the strongest discriminators, both achieving $p < 0.001$. HGG has a median enhancing volume of $19.46\text{ cm}^3$ and an enhancing fraction of $22.4\%$, whereas LGG has a median enhancing volume of only $0.82\text{ cm}^3$ and an enhancing fraction under $1\%$."

### Q5: How many non-enhancing cases ($\text{ET} = 0$) did you discover, and what is their clinical significance?
> **Answer:** "We discovered exactly 27 non-enhancing cases ($\text{ET} = 0\text{ cm}^3$). These cases correspond primarily to Low-Grade Astrocytomas and Oligodendrogliomas where the blood-brain barrier is intact. Identifying them is vital because single-channel models evaluating only T1ce would predict no tumor; our system requires multimodal fusion across T2 and FLAIR to capture non-enhancing disease."

### Q6: Why must MRI intensity profiling be restricted strictly to non-zero voxels?
> **Answer:** "In skull-stripped brain MRI, over $60\%$ of the $240 \times 240 \times 155$ matrix is background air voxels with an intensity of exactly 0. Including these zero voxels would heavily skew the mean, variance, and percentiles toward zero. Restricting profiling to $I(x) > 0$ extracts the true physiological distribution of brain parenchyma."

### Q7: How do the findings of Module B directly configure Module C (MRI Preprocessing)?
> **Answer:** "Module B establishes the 1st and 99th non-zero percentiles across all four modalities ($P_1 \approx 12$–$24$, $P_{99} \approx 685$–$940$). In Module C, we use these empirical percentiles to clip extreme scanner artifacts before applying non-zero z-score normalization, guaranteeing model stability."

### Q8: How does Module B support 3D ROI extraction in Module D?
> **Answer:** "Module B calculates the 3D bounding box coordinates $[x_{\min}, x_{\max}]$, $[y_{\min}, y_{\max}]$, $[z_{\min}, z_{\max}]$ and the Whole Tumor centroid $(\bar{x}, \bar{y}, \bar{z})$ for every subject. Module D uses these coordinates to extract compact, centered 3D patches (e.g. $128^3$), discarding over $80\%$ of uninformative background and dramatically accelerating 3D neural network training."

### Q9: What statistical test was used to compare HGG vs. LGG volumes, and why?
> **Answer:** "We used the non-parametric Mann-Whitney U test (Wilcoxon rank-sum test). Medical tumor volumes exhibit heavy right-skewed distributions with large variances that violate the normality assumptions required by Student's t-test. The Mann-Whitney U test evaluates rank distributions robustly without requiring normal distribution assumptions."

### Q10: How does Module B feed into Agent 2 (Clinical Decision Agent) in the final platform?
> **Answer:** "Agent 2 uses Module B's volumetric metrics ($\text{WT}$, $\text{TC}$, $\text{ET}$, $\text{ET}/\text{WT}$) as quantitative clinical evidence during LangGraph multi-agent reasoning. By comparing a new patient's volumetric profile against the cohort distributions documented in `dataset_summary.csv`, Agent 2 formulates evidence-supported clinical reports with calibrated confidence."

---

## 10. Seamless Transition: How Module B Connects to Module C

With **Module B completed**, validated, and documented, we transition directly into **Module C: MRI Preprocessing & Patient-Level Splitting**:
1. **Intensity Normalization:** Apply non-zero z-score normalization using the $[P_1, P_{99}]$ clipping thresholds established in Module B.
2. **Zero-Leakage Splitting:** Partition all 369 patients into stratified training ($70\%$), validation ($15\%$), and test ($15\%$) sets strictly at the patient level (preserving HGG : LGG class ratio across all folds).
3. **Spatial Cropping & Tensor Caching:** Implement bounding box cropping and convert NIfTI volumes into optimized PyTorch tensors for rapid GPU ingestion in Modules F, G, and H.
