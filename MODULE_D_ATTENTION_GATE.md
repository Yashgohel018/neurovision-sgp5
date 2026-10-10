# Module D Extension: Attention-Gated UNETR for Precise 3D Brain Tumor Segmentation & ROI Refinement

## 1. Scientific Foundation & Literature Alignment

**Referenced Publication**:
> *Attention-Gated UNETR Model for Precise Brain Tumor Segmentation in 3D Medical Imaging*  
> Ronak R Patel, Dhruvil Joshi, Keval Shah, Prachi Desai, Smit Gandhi, Miral Patel  
> CSPIT, Charotar University of Science and Technology (CHARUSAT) & G H Patel College of Engineering and Technology (GCET), Anand, Gujarat, India.

### Core Problem Solved
Standard convolutional segmentation networks (UNet, UNet++) frequently suffer from false-positive background responses and inaccurate delineation along fuzzy, infiltrating neoplastic boundaries in multisequence MRI (T1, T1ce, T2, FLAIR). The **Attention-Gated UNETR (AG-UNETR)** combines a 3D Vision Transformer encoder with skip-connection Attention Gates (AG) in the decoder to selectively suppress background brain parenchyma while enhancing high-gradient tumor margins.

---

## 2. Mathematical Architecture

```
4D MRI Input [4, 96, 96, 96]
  │
  ▼
Patch Embedding (16×16×16) ──► 216 Tokens [D=768] + Positional Embeddings
  │
  ├─► Transformer Blocks 1..4   ──► z3 Feature Map [64, 12, 12, 12]
  ├─► Transformer Blocks 5..8   ──► z6 Feature Map [128, 6, 6, 6]
  └─► Transformer Blocks 9..12  ──► z12 Feature Map [256, 6, 6, 6]
                                      │
                                  [Upsample ×2]
                                      │
          z6 ──► [3D Attention Gate 1] ◄── Gating Signal g1
                        │
                  [Concatenate]
                        │
                  [Decoder Stage 1] ──► Gating Signal g2
                                              │
                  z3 ──► [3D Attention Gate 2] ◄──
                                │
                          [Concatenate]
                                │
                          [Final 1×1×1 Conv]
                                │
                                ▼
                  3 Tumor Channels (ET, WT, TC) [3, 96, 96, 96]
```

### 3D Attention Gate (AG) Formulation
For skip feature map $x \in \mathbb{R}^{C_x \times D \times H \times W}$ and decoder gating signal $g \in \mathbb{R}^{C_g \times D_g \times H_g \times W_g}$:

$$\theta_x = W_x^T x, \quad \phi_g = W_g^T g$$
$$\alpha = \sigma\left(\psi^T \left(\text{ReLU}(\theta_x + \phi_g + b_g)\right) + b_\psi\right)$$
$$\hat{x} = x \cdot \alpha$$

Where:
- $W_x, W_g, \psi$ are $1 \times 1 \times 1$ 3D convolutions with 3D Batch Normalization.
- $\alpha \in [0, 1]$ is the dense spatial-channel attention coefficient.
- $\hat{x}$ suppresses irrelevant non-tumoral voxels while preserving microscopic tumor infiltrations.

---

## 3. Comparative Benchmark Results (BraTS 2020)

| Architecture | Enhancing Tumor (ET) Dice | Whole Tumor (WT) Dice | Tumor Core (TC) Dice | ET HD95 (mm) | WT HD95 (mm) | TC HD95 (mm) |
|---|---|---|---|---|---|---|
| Deep Random Forest [Shaikh & Phophalia] | 0.20 | 0.38 | 0.27 | 99.02 | 60.06 | 66.24 |
| Multi-Threshold Attention UNet (MTAU) [Awasthi] | 0.59 | 0.72 | 0.57 | 38.53 | 22.73 | 41.82 |
| U-Attention Net [Xu et al.] | 0.67 | 0.86 | 0.70 | 40.60 | 7.94 | 15.75 |
| Lightweight UNet [Tarasiewicz et al.] | 0.70 | 0.89 | 0.75 | 40.13 | 4.55 | 10.67 |
| **Attention-Gated UNETR (Proposed)** | **0.7727** | **0.8619** | **0.7629** | **7.47** | **9.81** | **9.81** |

---

## 4. Integration with NeuroVision AI

1. **Sub-Region Mask Composition**:
   - `Channel 0 (ET)`: Active Enhancing Tumor $\to$ BraTS Label 4
   - `Channel 1 (WT)`: Whole Infiltrating Lesion $\to$ BraTS Labels 1 + 2 + 4
   - `Channel 2 (TC)`: Tumor Core $\to$ BraTS Labels 1 + 4
   - Distinct Necrotic Core (NCR, Label 1) and Peritumoral Vasogenic Edema (ED, Label 2) are derived via:
     $$\text{NCR} = \text{TC} \setminus \text{ET}, \quad \text{ED} = \text{WT} \setminus \text{TC}$$

2. **Module D ROI Bounding Box Refinement**:
   - `AttentionGatedROISegmentor` provides high-confidence spatial masks for `TumorBoundingBoxExtractor`.
   - Incorporates a calibrated $+10$ voxel contextual margin around the attention-gated envelope to preserve the peritumoral infiltration zone.
