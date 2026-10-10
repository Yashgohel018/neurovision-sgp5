"""
NeuroVision AI: Module G - MultiModal-BrainNet
Segmentation-Assisted Multimodal Deep Learning for Brain Tumor Classification from BraTS MRI.
Implements:
- 4 Parallel Modality-Specific 3D Encoders (T1, T1ce, T2, FLAIR)
- P1: Late-Fusion via Concatenation
- P2: Multimodal Cross-Attention Fusion with Interpretability Weights
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder3D(nn.Module):
    """
    Dedicated 3D feature encoder for an individual MRI modality sequence.
    Input: [B, 1, D, H, W] -> Output: [B, embedding_dim]
    """
    def __init__(self, in_channels: int = 1, base_filters: int = 16, embedding_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(base_filters, base_filters * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_filters * 8),
            nn.ReLU(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Linear(base_filters * 8, embedding_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns (embedding, feature_map_before_gap for gradcam)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        gap = self.gap(c4)
        flat = torch.flatten(gap, 1)
        emb = self.proj(flat)
        return emb, c4


class MultimodalAttentionFusion(nn.Module):
    """
    Computes learnable self-attention / cross-modal weights across the 4 MRI sequence embeddings.
    Allows the model to dynamically weight T1ce for active vascularity vs FLAIR for edema.
    """
    def __init__(self, embedding_dim: int = 64, num_modalities: int = 4):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, embeddings: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: List of 4 tensors, each [B, embedding_dim]
        Returns:
            fused: [B, embedding_dim]
            weights: [B, 4] normalized attention weights
        """
        # Stack into [B, 4, embedding_dim]
        stacked = torch.stack(embeddings, dim=1)
        # Compute attention scores for each modality: [B, 4, 1]
        scores = self.attn_mlp(stacked)
        weights = F.softmax(scores, dim=1) # [B, 4, 1]
        # Weighted sum: [B, embedding_dim]
        fused = torch.sum(stacked * weights, dim=1)
        return fused, weights.squeeze(-1)


class MultiModalBrainNet(nn.Module):
    """
    MultiModal-BrainNet: Segmentation-Assisted Multimodal Deep Learning for Brain Tumor Classification.
    Accepts 4 MRI sequences (T1, T1ce, T2, FLAIR) and fuses features via either:
    - fusion_type='concat' (P1)
    - fusion_type='attention' (P2 - Recommended proposed model)
    """
    MODALITIES = ["t1", "t1ce", "t2", "flair"]

    def __init__(
        self,
        num_classes: int = 1,
        embedding_dim: int = 64,
        base_filters: int = 16,
        fusion_type: str = "attention",
        dropout: float = 0.3
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.embedding_dim = embedding_dim

        # 4 modality-specific encoders
        self.t1_encoder = ModalityEncoder3D(1, base_filters, embedding_dim)
        self.t1ce_encoder = ModalityEncoder3D(1, base_filters, embedding_dim)
        self.t2_encoder = ModalityEncoder3D(1, base_filters, embedding_dim)
        self.flair_encoder = ModalityEncoder3D(1, base_filters, embedding_dim)

        if fusion_type == "attention":
            self.fusion = MultimodalAttentionFusion(embedding_dim, num_modalities=4)
            classifier_in = embedding_dim
        elif fusion_type == "concat":
            classifier_in = embedding_dim * 4
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_in, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2.0),
            nn.Linear(32, num_classes)
        )

    def forward(
        self,
        t1: torch.Tensor,
        t1ce: torch.Tensor,
        t2: torch.Tensor,
        flair: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Args:
            Each input is [B, 1, D, H, W]
        Returns:
            logits: [B, num_classes]
            attention_weights: [B, 4] (if attention fusion) or None
        """
        emb_t1, _ = self.t1_encoder(t1)
        emb_t1ce, _ = self.t1ce_encoder(t1ce)
        emb_t2, _ = self.t2_encoder(t2)
        emb_flair, _ = self.flair_encoder(flair)

        embeddings = [emb_t1, emb_t1ce, emb_t2, emb_flair]

        if self.fusion_type == "attention":
            fused, attn_weights = self.fusion(embeddings)
        else:
            fused = torch.cat(embeddings, dim=1)
            attn_weights = None

        logits = self.classifier(fused)
        return logits, attn_weights
