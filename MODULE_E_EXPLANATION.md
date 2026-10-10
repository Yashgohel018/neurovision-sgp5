# Module E: Synchronized Multimodal 3D Data Augmentation

## 1. Scientific Overview & Objectives
Medical image augmentation in neuro-oncology requires strict physiological plausibility. In multimodal MRI (T1, T1ce, T2, FLAIR), arbitrary or unsynchronized spatial perturbations destroy anatomical alignment across sequences. 

Module E implements **Synchronized 3D Multimodal Augmentation** for NeuroVision AI:
1. **Synchronized Spatial Transforms**: Any 3D rotation, affine distortion, or sagittal flip is applied identically to all four MRI channels and ground-truth segmentation masks.
2. **Leakage-Safe Training Enforcement**: Augmentation is strictly applied **only** to training subjects from `data/splits/train_subjects.csv` and is deterministic/disabled for validation and testing cohorts.
3. **Tumor-Preserving Intensity Perturbations**: Intensity scaling ($\pm 8\%$), subtle Gaussian noise ($\sigma=0.02$), and contrast gamma adjustments ($[0.85, 1.15]$) are applied to non-zero brain voxels.

## 2. Technical Specifications
- **Input**: 4D Multimodal Tensor $[C=4, D=96, H=96, W=96]$ (T1, T1ce, T2, FLAIR) + optional 3D segmentation mask $[D, H, W]$.
- **Transforms**:
  - `RandomSagittalFlip` ($p=0.5$): Preserves bilateral hemispheric anatomy.
  - `RandomRotation3D` ($p=0.6, \theta \in [-10^\circ, +10^\circ]$): Small axial rotations.
  - `GaussianNoise` ($p=0.5, \mu=0.0, \sigma=0.02$): Simulates scanner acquisition noise.
  - `IntensityScaling` ($p=0.5, s \in [0.92, 1.08]$): Simulates RF coil sensitivity variations.
  - `RandomContrastGamma` ($p=0.4, \gamma \in [0.85, 1.15]$): Simulates variable T1/T2 relaxometry.
- **Output Artifacts**:
  - `configs/augmentation_config.yaml`
  - `reports/figures/augmentation_samples.png` (Visual QA comparison)

## 3. Integration with NeuroVision AI
Module E provides the data transform pipeline for `Module H (Training Pipeline)` to train both baseline 3D CNNs (`Module F`) and the proposed `MultiModal-BrainNet` (`Module G`).
