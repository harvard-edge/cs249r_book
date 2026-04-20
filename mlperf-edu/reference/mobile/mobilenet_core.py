"""
MLPerf EDU: MobileNetV2 Backbone — Pure PyTorch Implementation
================================================================
Provenance: Sandler et al. 2018, "MobileNetV2: Inverted Residuals
            and Linear Bottlenecks"
Maps to: MLPerf Inference MobileNet-EdgeTPU / SSD-MobileNet

This is a COMPLETE, self-contained implementation with no torchvision
model dependency. Implements the inverted residual block with linear
bottleneck as described in the original paper.

Key pedagogical concepts:
- Depthwise separable convolutions (parameter efficiency)
- Inverted residuals (expand → depthwise → project)
- Linear bottleneck (remove ReLU in narrow layers to preserve information)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    """Conv → BN → ReLU6 building block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """
    MobileNetV2 inverted residual block.
    
    Unlike standard residuals (wide→narrow→wide), this uses:
        narrow → EXPAND → depthwise → PROJECT → narrow
    
    The expansion increases channels for the depthwise conv to have
    more capacity, then the linear projection (no ReLU!) compresses
    back to avoid information loss in low-dimensional space.
    
    Args:
        inp: Input channels
        oup: Output channels  
        stride: Spatial stride (1 or 2)
        expand_ratio: Channel expansion factor (typically 6)
    """

    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion (1×1 conv)
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        
        layers.extend([
            # Depthwise convolution (3×3, groups=hidden_dim)
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # Linear projection — NOTE: no ReLU after this!
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2Local(nn.Module):
    """
    Complete MobileNetV2 — no torchvision dependency.
    
    Architecture follows Table 2 from the paper:
        t (expansion), c (channels), n (repeat), s (stride)
        
    Adapted for CIFAR-32 / small-image classification.
    
    Args:
        num_classes: Number of output classes
        width_mult: Channel width multiplier (default 1.0)
        in_channels: Input channels (default 3 for RGB)
    """

    def __init__(self, num_classes=100, width_mult=1.0, in_channels=3):
        super().__init__()

        # MobileNetV2 architecture specification from Table 2
        # [expansion_ratio, output_channels, num_blocks, stride]
        inverted_residual_setting = [
            [1, 16,  1, 1],
            [6, 24,  2, 1],  # stride=1 for CIFAR (was 2 for ImageNet)
            [6, 32,  3, 2],
            [6, 64,  4, 2],
            [6, 96,  3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # First layer
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        # Initial conv
        features = [ConvBNReLU(in_channels, input_channel, stride=1)]  # stride=1 for CIFAR

        # Inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel

        # Final conv
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Drop-in replacement for torchvision.models.mobilenet_v2
def mobilenet_v2(num_classes=100, **kwargs):
    return MobileNetV2Local(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    model = MobileNetV2Local(num_classes=100)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MobileNetV2 (local): {n_params:,} parameters ({n_params/1e6:.1f}M)")
    
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    assert out.shape == (4, 100), f"Expected (4, 100), got {out.shape}"
    print("✅ Forward pass verified")
