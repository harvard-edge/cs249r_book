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
    - AUC >= 0.85 on held-out anomaly detection

Dataset:
    Primary: ToyADMOS (Koizumi et al. 2019) / DCASE 2020 Task 2
    Fallback: MNIST (train on one digit class, detect others as anomalies)

Provenance: MLPerf Tiny Benchmark Suite, Banbury et al. 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


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

    Training: only "normal" digits (e.g., digit 0)
    Validation: mix of normal + anomalous digits
    The model learns to reconstruct digit 0; other digits have higher error.

    This is a pedagogical stand-in for ToyADMOS audio data, using the
    same autoencoder architecture. The principle is identical:
    train on normal → detect anomaly via reconstruction error.
    """

    def __init__(self, root="./data", train=True, normal_class=0):
        import torchvision
        import torchvision.transforms as transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        full_dataset = torchvision.datasets.MNIST(
            root=root, train=train, download=True, transform=transform
        )

        if train:
            # Training: only normal class
            indices = [i for i, (_, label) in enumerate(full_dataset) if label == normal_class]
            self.data = torch.stack([full_dataset[i][0] for i in indices])
            self.labels = torch.zeros(len(indices), dtype=torch.long)  # all normal
        else:
            # Validation: all classes, with labels (0=normal, 1=anomaly)
            self.data = torch.stack([full_dataset[i][0] for i in range(len(full_dataset))])
            original_labels = torch.tensor([full_dataset[i][1] for i in range(len(full_dataset))])
            self.labels = (original_labels != normal_class).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].view(-1)  # Flatten 28x28 → 784
        label = self.labels[idx]
        return x, label


def get_mnist_anomaly_dataloaders(batch_size=64, data_dir="./data", normal_class=0, num_workers=0):
    """
    Returns (train_loader, val_loader) for MNIST anomaly detection.

    Training set: only normal_class digits
    Validation set: all digits (normal_class → label 0, others → label 1)
    """
    train_ds = MNISTAnomalyDataset(root=data_dir, train=True, normal_class=normal_class)
    val_ds = MNISTAnomalyDataset(root=data_dir, train=False, normal_class=normal_class)

    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    val_loader = data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    print("🔍 Anomaly Detection Autoencoder — Architecture Demo")

    # MNIST version (input_dim = 784 = 28*28)
    model = AnomalyDetectionAE(input_dim=784, bottleneck_dim=8)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Parameters: {total_params:,} ({total_params/1e3:.1f}K)")
    print(f"💾 Model size: {total_params * 4 / 1024:.1f} KB (FP32)")
    print(f"🔬 Compression ratio: 784 / 8 = {784/8:.0f}x")

    # Dummy forward
    dummy = torch.randn(4, 784)
    recon, loss = model(dummy)
    print(f"✅ Forward: recon={recon.shape}, loss={loss.item():.4f}")

    # Anomaly scores
    scores = model.anomaly_score(dummy)
    print(f"✅ Anomaly scores: {scores.tolist()}")
