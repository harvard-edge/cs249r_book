"""
MLPerf EDU: ResNet-18 — Pure PyTorch Implementation
=====================================================
Provenance: He et al. 2016, "Deep Residual Learning for Image Recognition"
Maps to: MLPerf Inference ResNet-50 (scaled down for education)

This is a COMPLETE, self-contained implementation with no external model
dependencies. Every layer is visible for pedagogical inspection.
Students can trace the full forward pass, count parameters, and modify
the residual connections.

Architecture:
    Input → Conv3x3(64) → BN → ReLU
    → Layer1: [BasicBlock(64)]  × 2
    → Layer2: [BasicBlock(128)] × 2  (stride=2)
    → Layer3: [BasicBlock(256)] × 2  (stride=2)
    → Layer4: [BasicBlock(512)] × 2  (stride=2)
    → AdaptiveAvgPool → FC(num_classes)

Total: 11.17M parameters for CIFAR-100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Standard ResNet basic block with two 3x3 convolutions and skip connection.
    
    Pedagogical insight: The skip connection allows gradients to flow directly
    to earlier layers, solving the degradation problem in deep networks.
    
    Parameters:
        Without shortcut: 2 × (C_in × C_out × 3 × 3) + 2 × C_out (BN) 
        With shortcut: adds C_in × C_out × 1 × 1 + C_out (BN)
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # First conv: may downsample spatially
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second conv: always same spatial size
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut: identity if dimensions match, 1x1 conv otherwise
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        # Element-wise addition: this IS the residual learning
        out += identity
        out = F.relu(out)
        return out


class ResNet18Local(nn.Module):
    """
    Complete ResNet-18 implementation — no torchvision dependency.
    
    Adapted for CIFAR-style 32×32 images:
    - Uses 3×3 initial conv (not 7×7) with stride=1 (not 2)
    - Removes initial max-pool layer
    - These are standard adaptations from He et al. for small images
    
    Args:
        num_classes: Number of output classes (default: 100 for CIFAR-100)
        in_channels: Input channels (default: 3 for RGB)
    """

    def __init__(self, num_classes=100, in_channels=3):
        super().__init__()
        self.in_planes = 64

        # Initial convolution — adapted for 32×32 CIFAR images
        # (ImageNet uses 7×7/stride=2 + maxpool, but CIFAR is already small)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Four residual stages
        self.layer1 = self._make_layer(64,  2, stride=1)  # 32×32
        self.layer2 = self._make_layer(128, 2, stride=2)  # 16×16
        self.layer3 = self._make_layer(256, 2, stride=2)  # 8×8
        self.layer4 = self._make_layer(512, 2, stride=2)  # 4×4

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # Weight initialization (Kaiming He et al.)
        self._init_weights()

    def _make_layer(self, planes, num_blocks, stride):
        """Build a residual stage with `num_blocks` BasicBlocks."""
        downsample = None
        if stride != 1 or self.in_planes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = [BasicBlock(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Kaiming initialization for all conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass:
            (B, 3, 32, 32) → conv/bn/relu → layer1-4 → pool → fc → (B, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))   # (B, 64, 32, 32)
        x = self.layer1(x)                     # (B, 64, 32, 32)
        x = self.layer2(x)                     # (B, 128, 16, 16)
        x = self.layer3(x)                     # (B, 256, 8, 8)
        x = self.layer4(x)                     # (B, 512, 4, 4)
        x = self.avgpool(x)                    # (B, 512, 1, 1)
        x = torch.flatten(x, 1)               # (B, 512)
        x = self.fc(x)                         # (B, num_classes)
        return x


# Convenience factory (drop-in replacement for torchvision.models.resnet18)
def resnet18(num_classes=100, **kwargs):
    """Build ResNet-18 for CIFAR. No torchvision dependency."""
    return ResNet18Local(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    model = ResNet18Local(num_classes=100)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet-18 (local): {n_params:,} parameters ({n_params/1e6:.1f}M)")
    
    # Quick forward pass test
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    assert out.shape == (4, 100), f"Expected (4, 100), got {out.shape}"
    print("✅ Forward pass verified")
