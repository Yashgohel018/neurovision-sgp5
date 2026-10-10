"""
NeuroVision AI: Module F - Baseline Classification Models
Implements reference baselines specified in the BraTS classification project plan:
- B1: 2D ResNet on representative tumor slice
- B2: 3D CNN on 3D tumor ROI
- B3: Single-modality 3D model (FLAIR)
- B4: 4-channel early-fusion multimodal 3D model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Baseline2D_ResNet(nn.Module):
    """
    B1: 2D ResNet on representative tumor slice (4-channel input: T1, T1ce, T2, FLAIR slice).
    """
    def __init__(self, in_channels: int = 4, num_classes: int = 1, pretrained: bool = False):
        super().__init__()
        resnet = models.resnet18(weights=None)
        # Adapt first conv for in_channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class Baseline3D_CNN(nn.Module):
    """
    B2 & B3 & B4: Modular 3D CNN for tumor ROI.
    - in_channels=1: Single-modality 3D baseline (B3)
    - in_channels=4: Early-fusion 4-channel 3D baseline (B4)
    """
    def __init__(self, in_channels: int = 4, num_classes: int = 1, base_filters: int = 16):
        super().__init__()
        self.block1 = ConvBlock3D(in_channels, base_filters)
        self.block2 = ConvBlock3D(base_filters, base_filters * 2)
        self.block3 = ConvBlock3D(base_filters * 2, base_filters * 4)
        self.block4 = ConvBlock3D(base_filters * 4, base_filters * 8)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(base_filters * 8, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W]
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)
