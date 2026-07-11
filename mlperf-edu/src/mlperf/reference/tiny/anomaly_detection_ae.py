"""
MLPerf EDU: Anomaly Detection Autoencoder (Tiny Division)

A fully-connected autoencoder for machine sound anomaly detection,
matching the MLPerf Tiny anomaly detection benchmark.

Architecture:
    Audio → Mel Spectrogram → Flatten → Encoder (FC layers)
    → Bottleneck (8 dims) → Decoder (FC layers) → Reconstruct
    → Anomaly score = reconstruction error (MSE)

The MLPerf Tiny AD benchmark uses the ToyADMOS/DCASE2020 Task 2 dataset.
For pedagogical portability, we also support MNIST as a simpler alternative
(detect out-of-distribution digits), while the full pipeline uses the
same architecture on audio mel spectrograms.

Systems Focus:
    - Compression ratio: input_dim / bottleneck_dim
    - Model size constraint (<32KB for microcontroller)
    - Students measure reconstruction quality vs. bottleneck size

Quality Target:
    - Macro AUROC >= 0.93 on the versioned MNIST hard-curve protocol
    - Worst-class AUROC >= 0.90
    - Per-class AUROC improvement over no-training controls >= 0.20

Dataset:
    Primary: ToyADMOS (Koizumi et al. 2019) / DCASE 2020 Task 2
    Fallback: MNIST hard-curve-v1 (train on digit 5; detect 3, 8, and 9)

Provenance: MLPerf Tiny Benchmark Suite, Banbury et al. 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


MNIST_HARD_ANOMALY_PROTOCOL = "mnist-hard-curve-v1"
MNIST_HARD_NORMAL_CLASS = 5
MNIST_HARD_ANOMALY_CLASSES = (3, 8, 9)


class AnomalyDetectionAE(nn.Module):
    """
    Fully-connected autoencoder for anomaly detection.

    The model learns to reconstruct "normal" inputs. At inference,
    high reconstruction error indicates an anomaly.

    Architecture matches MLPerf Tiny reference:
    - Input: 640-dim (5 concatenated 128-dim mel frames)
    - Encoder: 640 → 128 → 128 → 128 → 128
    - Bottleneck: 128 → 8
    - Decoder: 8 → 128 → 128 → 128 → 128 → 640
    """

    def __init__(self, input_dim=640, bottleneck_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, bottleneck_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x, targets=None):
        """
        Forward pass. For training, targets=None and loss is reconstruction MSE.

        Args:
            x: (B, input_dim) flattened mel spectrogram frames
            targets: unused (reconstruction target is the input itself)

        Returns:
            reconstruction: (B, input_dim)
            loss: scalar MSE reconstruction loss
        """
        # Flatten if needed (e.g., from image input)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        loss = F.mse_loss(decoded, x)
        return decoded, loss

    def anomaly_score(self, x):
        """
        Compute per-sample anomaly scores (reconstruction error).

        Higher score = more anomalous.
        """
        self.eval()
        with torch.no_grad():
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            decoded = self.decoder(self.encoder(x))
            scores = ((decoded - x) ** 2).mean(dim=1)
        return scores


# ---------------------------------------------------------------------------
# MNIST Anomaly Detection Dataset
# ---------------------------------------------------------------------------


class MNISTAnomalyDataset(data.Dataset):
    """
    MNIST-based anomaly detection dataset.

    Training contains only digit 5. Evaluation pairs held-out fives with the
    visually related curved digits 3, 8, and 9. Returning the original digit
    labels allows the runner to report a separate AUROC for every anomaly
    class instead of allowing easy classes to hide a weak one.

    This is a pedagogical stand-in for ToyADMOS audio data, using the
    same autoencoder architecture. The principle is identical:
    train on normal → detect anomaly via reconstruction error.
    """

    def __init__(
        self,
        root="./data",
        train=True,
        normal_class=MNIST_HARD_NORMAL_CLASS,
        anomaly_classes=MNIST_HARD_ANOMALY_CLASSES,
    ):
        import torchvision
        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        full_dataset = torchvision.datasets.MNIST(
            root=root, train=train, download=True, transform=transform
        )

        anomaly_classes = tuple(int(value) for value in anomaly_classes)
        if normal_class in anomaly_classes:
            raise ValueError("normal_class cannot also be an anomaly class")
        if not anomaly_classes:
            raise ValueError("at least one anomaly class is required")

        if train:
            # Training: only normal class
            indices = [
                i for i, (_, label) in enumerate(full_dataset) if label == normal_class
            ]
            self.data = torch.stack([full_dataset[i][0] for i in indices])
            self.labels = torch.full(
                (len(indices),), int(normal_class), dtype=torch.long
            )
        else:
            # Evaluation: normal class plus the versioned hard anomaly set.
            selected = {int(normal_class), *anomaly_classes}
            indices = [
                i for i, (_, label) in enumerate(full_dataset) if label in selected
            ]
            self.data = torch.stack([full_dataset[i][0] for i in indices])
            self.labels = torch.tensor(
                [full_dataset[i][1] for i in indices], dtype=torch.long
            )

        self.normal_class = int(normal_class)
        self.anomaly_classes = anomaly_classes
        self.protocol = MNIST_HARD_ANOMALY_PROTOCOL

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].view(-1)  # Flatten 28x28 → 784
        label = self.labels[idx]
        return x, label


def get_mnist_anomaly_dataloaders(
    batch_size=64,
    data_dir="./data",
    normal_class=MNIST_HARD_NORMAL_CLASS,
    anomaly_classes=MNIST_HARD_ANOMALY_CLASSES,
    num_workers=0,
    seed=0,
):
    """
    Returns (train_loader, val_loader) for MNIST anomaly detection.

    Training contains only ``normal_class``. Evaluation contains the normal
    class and every class in ``anomaly_classes`` while preserving original
    digit labels for classwise scoring.
    """
    train_ds = MNISTAnomalyDataset(
        root=data_dir,
        train=True,
        normal_class=normal_class,
        anomaly_classes=anomaly_classes,
    )
    val_ds = MNISTAnomalyDataset(
        root=data_dir,
        train=False,
        normal_class=normal_class,
        anomaly_classes=anomaly_classes,
    )
    generator = torch.Generator().manual_seed(int(seed))

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        generator=generator,
    )
    val_loader = data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    print("🔍 Anomaly Detection Autoencoder — Architecture Demo")

    # MNIST version (input_dim = 784 = 28*28)
    model = AnomalyDetectionAE(input_dim=784, bottleneck_dim=8)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Parameters: {total_params:,} ({total_params / 1e3:.1f}K)")
    print(f"💾 Model size: {total_params * 4 / 1024:.1f} KB (FP32)")
    print(f"🔬 Compression ratio: 784 / 8 = {784 / 8:.0f}x")

    # Dummy forward
    dummy = torch.randn(4, 784)
    recon, loss = model(dummy)
    print(f"✅ Forward: recon={recon.shape}, loss={loss.item():.4f}")

    # Anomaly scores
    scores = model.anomaly_score(dummy)
    print(f"✅ Anomaly scores: {scores.tolist()}")
