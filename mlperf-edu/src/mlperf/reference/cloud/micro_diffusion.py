"""
MLPerf EDU: Micro-Diffusion U-Net (Cloud Division)

A simplified U-Net denoising autoencoder for image generation,
mapping the MLPerf Training Stable Diffusion benchmark to laptop scale.

Architecture:
    Input image → Encoder (conv → downsample) → Bottleneck
    → Decoder (upsample + skip connections) → Reconstructed image

For training, the model learns to reconstruct clean images from
noisy inputs (denoising autoencoder objective). The time embedding
is a placeholder for the diffusion timestep conditioning that would
be used in a full DDPM pipeline.

Quality Target: MSE < 0.001 on CIFAR-10 reconstruction

Provenance: Ho et al. 2020, "Denoising Diffusion Probabilistic Models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownConv(nn.Module):
    """Encoder block: downsample 2x then double conv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    """Decoder block: upsample 2x + skip connection then double conv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad to match skip connection dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class MicroDiffusionUNet(nn.Module):
    """
    Micro-scale U-Net for denoising/generation (2.0M parameters).

    Encoder-decoder with skip connections. Currently used as a denoising
    autoencoder (reconstruct clean images). For full DDPM, add:
    - Sinusoidal time embeddings
    - Time-conditioned residual blocks
    - Iterative sampling loop
    """

    def __init__(self, n_channels=3, n_classes=3):
        super().__init__()

        # Time embedding (placeholder for full diffusion)
        self.time_embed = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # Encoder
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)

        # Decoder
        self.up1 = UpConv(256, 128)
        self.up2 = UpConv(128, 64)

        # Output projection
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x, t=None):
        """
        Args:
            x: (B, 3, H, W) input image
            t: (B,) optional timestep (unused in denoising AE mode)

        Returns:
            (B, 3, H, W) reconstructed image
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.outc(x)
