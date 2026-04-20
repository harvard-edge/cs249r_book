"""
MLPerf EDU: Micro-LSTM — Time-Series Forecasting Workload
===========================================================
Provenance: Hochreiter & Schmidhuber 1997, "Long Short-Term Memory"
Maps to: MLPerf Training LSTM / RNN workloads

This implements a small LSTM for multivariate time-series
forecasting on the **ETTh1** dataset (Electricity Transformer
Temperature, hourly frequency).

Dataset: ETTh1 (Zhou et al., AAAI 2021 — Informer paper)
    - 17,420 hourly observations (Jul 2016 — Jul 2018)
    - 7 features: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT (oil temperature)
    - Target: predict OT (oil temperature) for next pred_len timesteps
    - Ships locally in data/etth1/ETTh1.csv (850 KB)
    - Provenance: Zhou et al., "Informer: Beyond Efficient Transformer
      for Long Sequence Time-Series Forecasting", AAAI 2021

Pedagogical concepts:
- Recurrent architecture: hidden state carries temporal context
- Gating mechanism: forget/input/output gates control information flow
- Vanishing gradients: why LSTMs outperform vanilla RNNs
- Multivariate input: using multiple correlated features
- Sequence-to-sequence vs sequence-to-one prediction

Architecture:
    Input(7) → LSTM(hidden=64, layers=2) → Linear(64, pred_len)

Total: ~51K parameters
Target: MSE < 0.05 on normalized OT prediction
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ETTH1_PATH = os.path.join(REPO_ROOT, "data", "etth1", "ETTh1.csv")


# ============================================================================
# ETTh1 Dataset — real electricity transformer temperature data
# ============================================================================

class ETTh1Dataset(Dataset):
    """
    ETTh1 hourly electricity transformer temperature dataset.

    Loads real sensor data from an electrical transformer station.
    The model learns to predict oil temperature (OT) using all 7
    features as input — a genuine multivariate forecasting task.

    Split: 12 months train / 4 months val / 4 months test
    (standard Informer convention: 12/4/4 month split)

    Args:
        split: "train", "val", or "test"
        seq_len: Input sequence length (lookback)
        pred_len: Prediction horizon
    """

    def __init__(self, split="train", seq_len=96, pred_len=24):
        super().__init__()

        if not os.path.exists(ETTH1_PATH):
            raise FileNotFoundError(
                f"ETTh1 dataset not found at {ETTH1_PATH}. "
                "Download: cd data/etth1 && curl -sL "
                "'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/"
                "main/ETT-small/ETTh1.csv' -o ETTh1.csv"
            )

        # Load CSV (skip date column)
        import csv
        data = []
        with open(ETTH1_PATH, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
            for row in reader:
                data.append([float(x) for x in row[1:]])  # Skip date

        data = np.array(data, dtype=np.float32)  # (17420, 7)

        # Standard Informer split: 12/4/4 months of hourly data
        # 12 months ≈ 8760 hours, 4 months ≈ 2880 hours
        train_end = 12 * 30 * 24      # ~8640
        val_end = train_end + 4 * 30 * 24  # ~11520

        if split == "train":
            raw = data[:train_end]
        elif split == "val":
            raw = data[train_end:val_end]
        else:
            raw = data[val_end:]

        # Normalize using training statistics (prevent data leakage)
        train_data = data[:train_end]
        self.mean = train_data.mean(axis=0)
        self.std = train_data.std(axis=0) + 1e-8
        normalized = (raw - self.mean) / self.std

        # Create sliding windows
        self.inputs = []
        self.targets = []
        n = len(normalized)

        for i in range(n - seq_len - pred_len + 1):
            x = normalized[i : i + seq_len]       # All 7 features
            y = normalized[i + seq_len : i + seq_len + pred_len, -1]  # OT only
            self.inputs.append(x)
            self.targets.append(y)

        self.inputs = torch.tensor(np.array(self.inputs))    # (N, seq_len, 7)
        self.targets = torch.tensor(np.array(self.targets))  # (N, pred_len)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class MicroLSTM(nn.Module):
    """
    Small LSTM for multivariate time-series forecasting.

    Predicts the next pred_len timesteps of oil temperature (OT)
    given seq_len history of all 7 features.

    Architecture:
        LSTM(input=7, hidden=64, layers=2, bidirectional=False)
        → Linear(64, pred_len)

    Students can explore:
    - Effect of hidden_dim on capacity
    - num_layers and depth vs width tradeoffs
    - Bidirectional vs unidirectional
    - Teacher forcing during training

    Args:
        input_dim: Feature dimension per timestep (7 for ETTh1)
        hidden_dim: LSTM hidden state size
        num_layers: Number of stacked LSTM layers
        pred_len: Prediction horizon
    """

    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, pred_len=24):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        """
        Args:
            x: Input sequence (B, seq_len, input_dim)
        Returns:
            predictions: (B, pred_len)
        """
        # LSTM encodes the full sequence
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use the last hidden state for prediction
        last_hidden = lstm_out[:, -1, :]  # (B, hidden_dim)
        pred = self.fc(last_hidden)  # (B, pred_len)
        return pred


def get_timeseries_dataloaders(batch_size=64, seq_len=96, pred_len=24, seed=42):
    """
    Create train/val DataLoaders for ETTh1 time-series forecasting.
    Compatible with dataset_factory interface.
    """
    train_ds = ETTh1Dataset(split="train", seq_len=seq_len, pred_len=pred_len)
    val_ds = ETTh1Dataset(split="val", seq_len=seq_len, pred_len=pred_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# Training loop for standalone testing
def train_and_evaluate(epochs=30, batch_size=64, lr=0.001, seed=42):
    """Train MicroLSTM on ETTh1 and report convergence."""
    torch.manual_seed(seed)

    train_loader, val_loader = get_timeseries_dataloaders(
        batch_size=batch_size, seed=seed
    )
    model = MicroLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"  ETTh1: {len(train_loader.dataset)} train, "
          f"{len(val_loader.dataset)} val windows")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches

        model.eval()
        val_loss = 0
        n_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                val_loss += F.mse_loss(pred, y).item()
                n_batches += 1
        val_loss /= n_batches

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}: train_mse={train_loss:.6f}  "
                  f"val_mse={val_loss:.6f}")

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "n_params": n_params,
    }


if __name__ == "__main__":
    model = MicroLSTM()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MicroLSTM: {n_params:,} parameters")
    print()
    print("Training on ETTh1 (electricity transformer temperature)...")
    results = train_and_evaluate(epochs=30)
    print(f"\n✅ Results:")
    print(f"   Final train MSE: {results['final_train_loss']:.6f}")
    print(f"   Final val MSE: {results['final_val_loss']:.6f}")
    print(f"   Parameters: {results['n_params']:,}")
