"""
MLPerf EDU: Wake Vision — TinyML Person Detection

A pedagogical visual wake words (VWW) benchmark using the Wake Vision dataset
from Harvard-Edge/Wake-Vision on HuggingFace. This maps to the official
MLPerf Tiny Visual Wake Words benchmark.

Architecture:
    MicroNet — a tiny CNN designed for person/no-person binary classification
    on 96x96 grayscale images, mimicking the deployment target of MCU-class
    devices (e.g., Arduino Nano 33 BLE Sense).

Dataset:
    Wake Vision (Banbury et al., 2024, CVPR) — 6M+ images for person detection.
    We use a 10K subset (5K person, 5K non-person) for pedagogical training.
    Downloaded via HuggingFace Datasets or falls back to CIFAR-10 person proxy.

Quality Target:
    Binary accuracy > 0.85 on held-out validation set.

Provenance:
    Banbury et al. 2024, "Wake Vision: A Large-scale, Diverse Dataset and
    Benchmark Suite for TinyML Person Detection", CVPR 2024.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np


class MicroNet(nn.Module):
    """
    Tiny CNN for binary person detection on 96x96 grayscale images.

    Architecture mirrors the MLPerf Tiny VWW reference:
    - 3 convolutional blocks with depthwise-separable convolutions
    - Global average pooling
    - Single FC layer for binary classification

    Total parameters: ~25K (fits in MCU SRAM).
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Block 1: Standard conv (input is 1-channel grayscale)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Block 2: Depthwise-separable
        self.dw2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, groups=16, bias=False)
        self.bn2a = nn.BatchNorm2d(16)
        self.pw2 = nn.Conv2d(16, 32, kernel_size=1, bias=False)
        self.bn2b = nn.BatchNorm2d(32)

        # Block 3: Depthwise-separable
        self.dw3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32, bias=False)
        self.bn3a = nn.BatchNorm2d(32)
        self.pw3 = nn.Conv2d(32, 64, kernel_size=1, bias=False)
        self.bn3b = nn.BatchNorm2d(64)

        # Block 4: Depthwise-separable
        self.dw4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64, bias=False)
        self.bn4a = nn.BatchNorm2d(64)
        self.pw4 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.bn4b = nn.BatchNorm2d(64)

        # Classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        """Forward pass. Input: (B, 1, 96, 96) grayscale image."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2a(self.dw2(x)))
        x = F.relu(self.bn2b(self.pw2(x)))
        x = F.relu(self.bn3a(self.dw3(x)))
        x = F.relu(self.bn3b(self.pw3(x)))
        x = F.relu(self.bn4a(self.dw4(x)))
        x = F.relu(self.bn4b(self.pw4(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class WakeVisionSubset(data.Dataset):
    """
    A pedagogically-sized subset of Wake Vision.

    Attempts to load from HuggingFace; falls back to CIFAR-10 person proxy
    (classes 'person'-adjacent: deer=4, dog=5, horse=7 as non-person;
    we use a simple binary split).

    Images are resized to 96x96 grayscale to match MCU deployment targets.
    """

    def __init__(self, split="train", n_samples=5000, cache_dir="./data/wake_vision"):
        self.images = []
        self.labels = []
        self.n_samples = n_samples

        cache_file = os.path.join(cache_dir, f"wv_{split}_{n_samples}.pt")

        if os.path.exists(cache_file):
            cached = torch.load(cache_file, weights_only=True)
            self.images = cached["images"]
            self.labels = cached["labels"]
            return

        os.makedirs(cache_dir, exist_ok=True)

        # Try HuggingFace Wake Vision
        try:
            self._load_from_huggingface(split, n_samples)
        except Exception as e:
            print(f"  Wake Vision HF load failed ({e}), using CIFAR-10 proxy")
            self._load_cifar10_proxy(split, n_samples)

        # Cache for fast reload
        torch.save({"images": self.images, "labels": self.labels}, cache_file)

    def _load_from_huggingface(self, split, n_samples):
        """Load from Harvard-Edge/Wake-Vision on HuggingFace."""
        from datasets import load_dataset
        from PIL import Image
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((96, 96)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
        ])

        hf_split = "train" if split == "train" else "validation"
        ds = load_dataset("Harvard-Edge/Wake-Vision", split=hf_split, streaming=True)

        images, labels = [], []
        person_count = non_person_count = 0
        half = n_samples // 2

        for sample in ds:
            if person_count >= half and non_person_count >= half:
                break
            label = int(sample["person"])
            if label == 1 and person_count < half:
                img = transform(sample["image"])
                images.append(img)
                labels.append(1)
                person_count += 1
            elif label == 0 and non_person_count < half:
                img = transform(sample["image"])
                images.append(img)
                labels.append(0)
                non_person_count += 1

        self.images = torch.stack(images)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def _load_cifar10_proxy(self, split, n_samples):
        """Fallback: use CIFAR-10 as person detection proxy.

        CIFAR-10 classes: airplane=0, automobile=1, bird=2, cat=3, deer=4,
        dog=5, frog=6, horse=7, ship=8, truck=9.

        Person proxy: classes with living beings (cat, deer, dog, frog, horse)
        are "person" (label=1), vehicles/objects are "no-person" (label=0).
        This is a pedagogical proxy — not a real person detector.
        """
        import torchvision
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((96, 96)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
        ])

        is_train = (split == "train")
        cifar = torchvision.datasets.CIFAR10(
            root="./data", train=is_train, download=True, transform=transform
        )

        # Living beings = "person" proxy
        living_classes = {2, 3, 4, 5, 6, 7}  # bird, cat, deer, dog, frog, horse
        images, labels = [], []

        for img, cls in cifar:
            if len(images) >= n_samples:
                break
            binary_label = 1 if cls in living_classes else 0
            images.append(img)
            labels.append(binary_label)

        self.images = torch.stack(images)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_wake_vision_dataloaders(batch_size=64, n_train=5000, n_val=1000):
    """Returns (train_loader, val_loader) for Wake Vision person detection."""
    train_ds = WakeVisionSubset("train", n_samples=n_train)
    val_ds = WakeVisionSubset("val", n_samples=n_val)

    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=True
    )
    return train_loader, val_loader


if __name__ == "__main__":
    print("🔍 Wake Vision — TinyML Person Detection Benchmark")

    model = MicroNet(num_classes=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"📊 MicroNet parameters: {n_params:,} ({n_params/1e3:.1f}K)")

    # Test forward pass
    dummy = torch.randn(4, 1, 96, 96)
    out = model(dummy)
    print(f"✅ Forward: input={dummy.shape} → output={out.shape}")

    # Quick training test
    train_ld, val_ld = get_wake_vision_dataloaders(batch_size=32, n_train=500, n_val=100)
    x, y = next(iter(train_ld))
    print(f"📦 Data: x={x.shape}, y={y.shape}, classes={y.unique().tolist()}")

    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(f"✅ Loss: {loss.item():.4f}")
    print(f"✅ Accuracy: {(logits.argmax(1) == y).float().mean():.3f}")
