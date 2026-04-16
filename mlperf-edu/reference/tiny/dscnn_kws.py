"""
MLPerf EDU: DS-CNN Keyword Spotting (Tiny Division)

A depthwise-separable CNN for 12-class keyword spotting using the
Google Speech Commands v2 dataset.

Architecture:
    Waveform → Mel Spectrogram → DS-CNN (depthwise + pointwise conv blocks)
    → Global Average Pool → FC → 12 classes

The 12 classes follow the MLPerf Tiny standard:
    "yes", "no", "up", "down", "left", "right", "on", "off",
    "stop", "go", "unknown", "silence"

Systems Focus:
    - Model size constraint (<100KB for microcontroller deployment)
    - Depthwise-separable convolution efficiency vs standard convolution
    - Students measure parameter count, MACs, and latency

Quality Target:
    - Top-1 accuracy >= 90% on Speech Commands v2 test set

Dataset:
    Google Speech Commands v2 (Warden 2018)
    35 keyword classes → mapped to 12 MLPerf Tiny classes
    ~105K utterances, 1 second each, 16kHz mono WAV

Provenance: Zhang et al. 2017, "Hello Edge: Keyword Spotting on Microcontrollers"
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ---------------------------------------------------------------------------
# DS-CNN Architecture
# ---------------------------------------------------------------------------

class DSCNNBlock(nn.Module):
    """Depthwise-Separable Convolution Block.

    Splits the standard convolution into:
    1. Depthwise: one spatial filter per input channel (groups=in_channels)
    2. Pointwise: 1x1 convolution to mix channels

    Students measure: standard conv has C_in * C_out * K * K params,
    depthwise-separable has C_in * K * K + C_in * C_out params.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x


class DSCNN(nn.Module):
    """
    DS-CNN for keyword spotting (MLPerf Tiny reference architecture).

    Input: Mel spectrogram of shape (B, 1, n_mels, time_steps)
    Output: (B, num_classes) logits

    The model is deliberately small (~60K parameters) to fit on a
    microcontroller with <256KB SRAM. Students can quantize it to INT8
    and measure the compression ratio.
    """

    def __init__(self, num_classes=12, n_mels=40):
        super().__init__()

        # Initial convolution: maps 1-channel spectrogram to 64 filters
        self.conv_init = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(10, 4), stride=(2, 2), padding=(4, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 4 DS-CNN blocks (same channel dimension for simplicity)
        self.ds_blocks = nn.Sequential(
            DSCNNBlock(64, 64),
            DSCNNBlock(64, 64),
            DSCNNBlock(64, 48),
            DSCNNBlock(48, 48),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x, targets=None):
        """
        Forward pass compatible with the auto_trainer interface.

        Args:
            x: (B, 1, n_mels, time_steps) mel spectrogram
            targets: (B,) class labels for loss computation

        Returns:
            logits: (B, num_classes)
            loss: scalar if targets provided
        """
        x = self.conv_init(x)
        x = self.ds_blocks(x)
        x = self.pool(x).view(x.size(0), -1)
        logits = self.fc(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss


# ---------------------------------------------------------------------------
# Speech Commands v2 Dataset
# ---------------------------------------------------------------------------

# The 12 MLPerf Tiny keyword classes
MLPERF_KEYWORDS = [
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go",
]
# All other words map to "unknown", silence maps to "silence"


class SpeechCommandsMelDataset(torch.utils.data.Dataset):
    """
    Wraps torchaudio.datasets.SPEECHCOMMANDS with mel spectrogram transform.

    Maps the 35-class Speech Commands v2 to the 12-class MLPerf Tiny schema:
    - 10 target keywords + "unknown" + "silence"
    """

    def __init__(self, root="./data", subset="training", n_mels=40, target_sr=16000):
        self.n_mels = n_mels
        self.target_sr = target_sr
        self.target_length = target_sr  # 1 second

        # Build label map
        self.label_to_idx = {kw: i for i, kw in enumerate(MLPERF_KEYWORDS)}
        self.label_to_idx["unknown"] = 10
        self.label_to_idx["silence"] = 11

        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=480,
            hop_length=160,
            n_mels=n_mels,
        )

        # Load dataset
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=root, download=True, subset=subset
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, label, _, _ = self.dataset[idx]

        # Resample if needed
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sr)
            waveform = resampler(waveform)

        # Pad or trim to exactly 1 second
        if waveform.size(1) < self.target_length:
            pad = self.target_length - waveform.size(1)
            waveform = F.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.target_length]

        # Convert to mel spectrogram
        mel = self.mel_transform(waveform)  # (1, n_mels, time_steps)

        # Log mel (add small epsilon for numerical stability)
        mel = torch.log(mel + 1e-9)

        # Map label to MLPerf Tiny 12-class schema
        if label in self.label_to_idx:
            target = self.label_to_idx[label]
        else:
            target = self.label_to_idx["unknown"]

        return mel, target


def get_speech_commands_dataloaders(batch_size=64, data_dir="./data", num_workers=0):
    """
    Returns (train_loader, val_loader) for Speech Commands v2.

    Used by: DS-CNN keyword spotting.
    """
    train_ds = SpeechCommandsMelDataset(root=data_dir, subset="training")
    val_ds = SpeechCommandsMelDataset(root=data_dir, subset="validation")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    print("🎤 DS-CNN Keyword Spotting — Architecture Demo")

    model = DSCNN(num_classes=12)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Parameters: {total_params:,} ({total_params/1e3:.1f}K)")

    # Model size in bytes (FP32)
    model_size_bytes = total_params * 4
    print(f"💾 Model size: {model_size_bytes/1024:.1f} KB (FP32)")
    print(f"💾 Model size: {total_params/1024:.1f} KB (INT8, after quantization)")

    # Dummy forward pass
    dummy_mel = torch.randn(4, 1, 40, 101)  # (B, 1, n_mels, time_steps)
    dummy_labels = torch.randint(0, 12, (4,))
    logits, loss = model(dummy_mel, targets=dummy_labels)
    print(f"✅ Forward: logits={logits.shape}, loss={loss.item():.4f}")
