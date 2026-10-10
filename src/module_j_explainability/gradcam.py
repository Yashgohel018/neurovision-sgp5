"""
NeuroVision AI: Module J - Explainable AI (XAI)
Implements 3D Grad-CAM and Multimodal Attention Attribution for MultiModal-BrainNet.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GradCAM3D:
    """
    3D Grad-CAM for convolutional feature maps in MultiModal-BrainNet encoders.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h1, h2])

    def generate_cam(
        self,
        t1: torch.Tensor,
        t1ce: torch.Tensor,
        t2: torch.Tensor,
        flair: torch.Tensor,
        target_shape: Tuple[int, int, int] = (96, 96, 96)
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Computes 3D Grad-CAM map.
        Returns:
            cam_3d: [D, H, W] normalized between 0 and 1
            attn_weights: [4] numpy array of modality weights
            pred_prob: float
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward
        logits, attn = self.model(t1, t1ce, t2, flair)
        prob = float(torch.sigmoid(logits).item())

        # Backward on positive logit
        logits.backward(retain_graph=True)

        if self.gradients is not None and self.activations is not None:
            # Global average pooling of gradients: [B, C, 1, 1, 1]
            weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
            # Weighted sum of feature maps
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam)
            # Upsample to target 3D shape
            cam = F.interpolate(cam, size=target_shape, mode="trilinear", align_corners=False)
            cam_np = cam.squeeze().cpu().numpy()
            # Normalize
            if cam_np.max() > cam_np.min():
                cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
            else:
                cam_np = np.zeros(target_shape, dtype=np.float32)
        else:
            cam_np = np.zeros(target_shape, dtype=np.float32)

        attn_np = attn.squeeze().detach().cpu().numpy() if attn is not None else np.array([0.25, 0.25, 0.25, 0.25])
        return cam_np, attn_np, prob

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []


def render_xai_overlay_figure(
    mri_slice: np.ndarray,
    cam_slice: np.ndarray,
    modality_weights: np.ndarray,
    subject_id: str,
    grade: str,
    prob: float,
    output_path: str
):
    """
    Renders multi-panel explanation figure showing MRI anatomy, Grad-CAM heatmap,
    alpha-blended overlay, and bar chart of sequence attention weights.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), facecolor="#0B0F19")
    for ax in axes:
        ax.set_facecolor("#111827")

    fig.suptitle(
        f"NeuroVision AI XAI Report: {subject_id} | True Grade: {grade} | MultiModal-BrainNet P(HGG)={prob:.1%}",
        color="#E2E8F0",
        fontsize=13,
        fontweight="bold",
        y=0.98
    )

    # 1. Structural MRI
    axes[0].imshow(mri_slice, cmap="bone")
    axes[0].set_title("Structural MRI (FLAIR)", color="#93C5FD", fontsize=11, fontweight="semibold")
    axes[0].axis("off")

    # 2. Grad-CAM Heatmap
    im2 = axes[1].imshow(cam_slice, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("3D Grad-CAM Neural Attention", color="#FBBF24", fontsize=11, fontweight="semibold")
    axes[1].axis("off")

    # 3. Alpha Blend Overlay
    axes[2].imshow(mri_slice, cmap="bone")
    axes[2].imshow(cam_slice, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("Decision Attribution Overlay", color="#34D399", fontsize=11, fontweight="semibold")
    axes[2].axis("off")

    # 4. Modality Attention Weights Bar Chart
    modalities = ["T1", "T1ce", "T2", "FLAIR"]
    colors = ["#60A5FA", "#EF4444", "#F59E0B", "#10B981"]
    bars = axes[3].bar(modalities, modality_weights * 100, color=colors, edgecolor="#374151")
    axes[3].set_title("Modality Attention Fusion (%)", color="#E2E8F0", fontsize=11, fontweight="semibold")
    axes[3].set_ylabel("Weight (%)", color="#9CA3AF")
    axes[3].set_ylim(0, 100)
    axes[3].tick_params(colors="#9CA3AF")
    axes[3].grid(True, linestyle="--", alpha=0.2, color="#374151", axis="y")

    for bar, val in zip(bars, modality_weights):
        axes[3].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val * 100:.1f}%",
            ha="center",
            color="#E2E8F0",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
