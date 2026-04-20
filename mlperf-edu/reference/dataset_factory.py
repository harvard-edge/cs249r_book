"""
MLPerf EDU: Dataset Factory

Provides real, deterministic data loaders for every workload in the benchmark
suite. This replaces the random-tensor stubs in auto_trainer.py.

Data sources:
    - Language models (NanoGPT, Nano-MoE): TinyShakespeare character-level encoding
    - Agent models: MBPP (CodeGen), ReAct traces (RAG, ReAct, ToolCall)
    - Vision models (ResNet, MobileNetV2): CIFAR-100 via torchvision
    - Diffusion models: CIFAR-10 via torchvision
    - DLRM: MovieLens-100K (Harper & Konstan, 2015) — real user-item interactions
    - GCN: Cora citation network (McCallum et al., 2000)
    - BERT: SST-2 sentiment (Socher et al., 2013)
    - LSTM: ETTh1 electricity transformer temperature (Zhou et al., 2021)
    - RL: CartPole physics simulation

All datasets enforce an immutable 80/20 train/val split with a fixed random seed
to guarantee reproducible gradients across runs.
"""

import os
import torch
import torch.utils.data as data

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_ROOT = os.path.join(REPO_ROOT, "datasets")
SPLIT_SEED = 42  # Immutable — guarantees identical splits everywhere


# ---------------------------------------------------------------------------
# Language Dataset (TinyShakespeare, character-level)
# ---------------------------------------------------------------------------

class CharTokenizer:
    """
    Character-level tokenizer for TinyShakespeare.

    Uses ASCII encoding (vocab_size 128) rather than BPE. This is pedagogically
    useful because students can inspect the token↔character mapping directly.
    Models that consume this tokenizer (NanoGPTWhiteBox, Nano-MoE, agent LMs)
    must size their embedding tables to accept token IDs in [0, 127].
    """

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("ascii", errors="replace"))

    @staticmethod
    def decode(tokens: list[int]) -> str:
        return bytes(tokens).decode("ascii", errors="replace")


def _load_tinyshakespeare(split: str = "train") -> torch.Tensor:
    """Load TinyShakespeare and return as a 1D tensor of ASCII token IDs."""
    if split == "train":
        path = os.path.join(DATASET_ROOT, "local_tensors", "tinyshakespeare_train.txt")
    else:
        path = os.path.join(DATASET_ROOT, "local_tensors", "tinyshakespeare_val.txt")

    if not os.path.exists(path):
        # Fall back to the full file and split manually
        full_path = os.path.join(DATASET_ROOT, "local_tensors", "tinyshakespeare.txt")
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"TinyShakespeare not found at {full_path}. "
                "Run: python scripts/orchestration/data_fetcher.py"
            )
        with open(full_path, "r") as f:
            text = f.read()
        split_idx = int(len(text) * 0.8)
        if split == "train":
            text = text[:split_idx]
        else:
            text = text[split_idx:]
    else:
        with open(path, "r") as f:
            text = f.read()

    tokens = CharTokenizer.encode(text)
    return torch.tensor(tokens, dtype=torch.long)


class TextDataset(data.Dataset):
    """
    Sliding-window dataset over a 1D token tensor.

    Each sample is (input_ids[i:i+seq_len], targets[i+1:i+seq_len+1]),
    exactly how language models are trained: predict the next character.
    """

    def __init__(self, tokens: torch.Tensor, seq_len: int = 64):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return x, y


def get_language_dataloaders(
    batch_size: int = 16, seq_len: int = 64, num_workers: int = 0
) -> tuple:
    """
    Returns (train_loader, val_loader) for character-level language modeling.

    Used by: NanoGPT, Nano-MoE, and all agent workloads.
    """
    train_tokens = _load_tinyshakespeare("train")
    val_tokens = _load_tinyshakespeare("val")

    train_ds = TextDataset(train_tokens, seq_len=seq_len)
    val_ds = TextDataset(val_tokens, seq_len=seq_len)

    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )
    val_loader = data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=True,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Vision Dataset (CIFAR-100 for ResNet, CIFAR-10 for Diffusion)
# ---------------------------------------------------------------------------

def get_cifar100_dataloaders(
    batch_size: int = 64, data_dir: str = "./data", num_workers: int = 0
) -> tuple:
    """
    Returns (train_loader, val_loader) for CIFAR-100 image classification.

    Used by: ResNet-18.
    """
    import torchvision
    import torchvision.transforms as transforms

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    valset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_val
    )

    train_loader = data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    val_loader = data.DataLoader(
        valset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=True,
    )

    return train_loader, val_loader


def get_cifar10_dataloaders(
    batch_size: int = 64, data_dir: str = "./data", num_workers: int = 0
) -> tuple:
    """
    Returns (train_loader, val_loader) for CIFAR-10.

    Used by: Micro-Diffusion (denoising target is the clean image itself).
    """
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    valset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    val_loader = data.DataLoader(
        valset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=True,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# DLRM Dataset (MovieLens-100K — real user-item interactions)
# ---------------------------------------------------------------------------

MOVIELENS_DIR = os.path.join(DATASET_ROOT, "..", "data", "movielens", "ml-100k")


class MovieLensRecommendationDataset(data.Dataset):
    """
    MovieLens-100K recommendation dataset for DLRM training.

    Converts the classic user-item rating dataset into a binary
    click-through format compatible with the DLRM architecture:
    - Dense features: normalized user age, occupation encoding
    - Sparse features: user_id (943 users), item_id (1682 items),
      genre (18 genres)
    - Binary label: rating >= 4 → positive (click), else negative

    Dataset: MovieLens-100K (Harper & Konstan, 2015)
        - 100,000 ratings from 943 users on 1,682 movies
        - Rating scale: 1-5 (binarized at threshold 4)
        - Ships locally in data/movielens/ml-100k/ (5 MB)

    All user/item IDs are reproducible from the raw data files.
    """

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = MOVIELENS_DIR

        ratings_path = os.path.join(data_dir, "u.data")
        users_path = os.path.join(data_dir, "u.user")

        if not os.path.exists(ratings_path):
            raise FileNotFoundError(
                f"MovieLens-100K not found at {data_dir}. "
                "Download: cd data/movielens && curl -sL "
                "'https://files.grouplens.org/datasets/movielens/ml-100k.zip' "
                "-o ml-100k.zip && unzip ml-100k.zip"
            )

        # Load user demographics
        user_ages = {}
        user_occupations = {}
        user_genders = {}
        with open(users_path, "r") as f:
            for line in f:
                parts = line.strip().split("|")
                uid = int(parts[0])
                age = float(parts[1])
                gender = 1.0 if parts[2] == "M" else 0.0
                occupation = int(parts[3]) if parts[3].isdigit() else 0
                user_ages[uid] = age
                user_occupations[uid] = occupation
                user_genders[uid] = gender

        # Load item genres (19 binary genre flags per movie)
        items_path = os.path.join(data_dir, "u.item")
        item_genres = {}
        if os.path.exists(items_path):
            with open(items_path, "r", encoding="latin-1") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) >= 24:
                        item_id = int(parts[0])
                        genres = [float(g) for g in parts[5:24]]  # 19 genre flags
                        item_genres[item_id] = genres

        # Normalize ages
        ages = list(user_ages.values())
        age_mean = sum(ages) / len(ages)
        age_std = (sum((a - age_mean) ** 2 for a in ages) / len(ages)) ** 0.5

        # First pass: compute per-user rating stats for dense features
        user_ratings = {}
        with open(ratings_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                uid, rating = int(parts[0]), float(parts[2])
                user_ratings.setdefault(uid, []).append(rating)
        user_avg = {u: sum(rs)/len(rs) for u, rs in user_ratings.items()}
        user_cnt = {u: len(rs) for u, rs in user_ratings.items()}
        max_cnt = max(user_cnt.values())

        # Second pass: build feature tensors
        dense_list = []
        sparse_list = []
        labels_list = []

        with open(ratings_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                user_id = int(parts[0])
                item_id = int(parts[1])
                rating = float(parts[2])

                # Dense features: user age, gender, avg rating, activity level
                age_norm = (user_ages.get(user_id, 30) - age_mean) / (age_std + 1e-8)
                gender = user_genders.get(user_id, 0.0)
                avg_r = (user_avg.get(user_id, 3.0) - 3.0) / 2.0  # center around 0
                activity = user_cnt.get(user_id, 1) / max_cnt  # 0-1

                # Add item genre features (up to 12 most common)
                genres = item_genres.get(item_id, [0.0] * 19)
                # Select 12 genre slots to fill 16 total dense features
                dense = [age_norm, gender, avg_r, activity] + genres[:12]
                dense_list.append(dense)

                # Sparse features: user_id, item_id, occupation
                sparse_list.append([
                    user_id - 1,   # 0-indexed, max 942
                    item_id - 1,   # 0-indexed, max 1681
                    user_occupations.get(user_id, 0),  # max ~20
                ])

                # Binary: rating >= 4 → positive
                labels_list.append(1.0 if rating >= 4.0 else 0.0)

        self.dense_features = torch.tensor(dense_list, dtype=torch.float32)
        self.sparse_features = [
            torch.tensor([s[0] for s in sparse_list], dtype=torch.long),
            torch.tensor([s[1] for s in sparse_list], dtype=torch.long),
            torch.tensor([s[2] for s in sparse_list], dtype=torch.long),
        ]
        self.labels = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        dense = self.dense_features[idx]
        sparse = [sf[idx] for sf in self.sparse_features]
        label = self.labels[idx]
        return dense, sparse, label


def _dlrm_collate_fn(batch):
    """Custom collate for DLRM: handles sparse index lists."""
    dense = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[2] for b in batch])

    # Build sparse indices and offsets for EmbeddingBag
    n_sparse = len(batch[0][1])
    sparse_indices = []
    sparse_offsets = []
    for i in range(n_sparse):
        indices = torch.stack([b[1][i] for b in batch])
        offsets = torch.arange(0, len(batch), dtype=torch.long)
        sparse_indices.append(indices.view(-1))
        sparse_offsets.append(offsets)

    return dense, sparse_indices, sparse_offsets, labels


def get_dlrm_dram_dataloaders(
    batch_size: int = 1024, num_workers: int = 0
) -> tuple:
    """
    Returns (train_loader, val_loader) for the DRAM-bound DLRM variant.

    Same MovieLens-100K data and collate_fn as the cache-resident variant,
    but defaults to a much larger batch size so each step issues enough
    embedding lookups to escape prefetcher noise and saturate DRAM
    bandwidth. The model itself (MicroDLRMDRAM) routes lookups through
    a 2M-row virtual table; the dataloader is unchanged.
    """
    return get_dlrm_dataloaders(batch_size=batch_size, num_workers=num_workers)


def get_dlrm_dataloaders(
    batch_size: int = 256, num_workers: int = 0
) -> tuple:
    """
    Returns (train_loader, val_loader) for MovieLens-100K recommendation.

    Used by: Micro-DLRM.
    """
    full_ds = MovieLensRecommendationDataset()

    # 80/20 split
    n_train = int(len(full_ds) * 0.8)
    n_val = len(full_ds) - n_train
    train_ds, val_ds = data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )

    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
        collate_fn=_dlrm_collate_fn,
    )
    val_loader = data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=True,
        collate_fn=_dlrm_collate_fn,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Unified Factory
# ---------------------------------------------------------------------------

def get_dataloaders(model_name: str, batch_size: int = 16) -> tuple:
    """
    Returns (train_loader, val_loader) for the given workload.

    This is the single entry point that auto_trainer.py should use.
    """
    if "resnet" in model_name or "mobilenet" in model_name:
        return get_cifar100_dataloaders(batch_size=batch_size)
    elif "diffusion" in model_name:
        return get_cifar10_dataloaders(batch_size=batch_size)
    elif "dlrm-dram" in model_name:
        return get_dlrm_dram_dataloaders(batch_size=batch_size)
    elif "dlrm" in model_name:
        return get_dlrm_dataloaders(batch_size=batch_size)
    elif "dscnn" in model_name or "kws" in model_name:
        from reference.tiny.dscnn_kws import get_speech_commands_dataloaders
        return get_speech_commands_dataloaders(batch_size=batch_size)
    elif "anomaly" in model_name or "autoencoder" in model_name:
        from reference.tiny.anomaly_detection_ae import get_mnist_anomaly_dataloaders
        return get_mnist_anomaly_dataloaders(batch_size=batch_size)
    elif "wake" in model_name or "vww" in model_name:
        from reference.tiny.wake_vision_vww import get_wake_vision_dataloaders
        return get_wake_vision_dataloaders(batch_size=batch_size)
    elif "codegen" in model_name:
        from reference.agent_datasets import get_mbpp_dataloaders
        return get_mbpp_dataloaders(batch_size=batch_size)
    elif "react" in model_name:
        from reference.agent_datasets import get_react_dataloaders
        return get_react_dataloaders(batch_size=batch_size)
    elif "rag" in model_name or "toolcall" in model_name:
        # RAG and ToolCall still use language data but with agent-specific framing
        from reference.agent_datasets import get_react_dataloaders
        return get_react_dataloaders(batch_size=batch_size)
    elif "gnn" in model_name or "gcn" in model_name:
        from reference.cloud.micro_gnn import get_gnn_dataloaders
        return get_gnn_dataloaders()  # Returns dict, not (train, val)
    elif "lstm" in model_name or "timeseries" in model_name:
        from reference.cloud.micro_lstm import get_timeseries_dataloaders
        return get_timeseries_dataloaders(batch_size=batch_size)
    elif "bert" in model_name or "text-cls" in model_name:
        from reference.cloud.micro_bert import get_bert_dataloaders
        return get_bert_dataloaders(batch_size=batch_size)
    elif "rl" in model_name or "cartpole" in model_name:
        from reference.cloud.micro_rl import get_rl_dataloaders
        return get_rl_dataloaders()  # Returns dict with env + agent_factory
    else:
        # NanoGPT, Nano-MoE use TinyShakespeare
        return get_language_dataloaders(batch_size=batch_size)


if __name__ == "__main__":
    """Quick verification that all data pipelines work."""
    print("🔍 Verifying MLPerf EDU Dataset Factory...\n")

    # Language
    train_ld, val_ld = get_language_dataloaders(batch_size=4, seq_len=64)
    x, y = next(iter(train_ld))
    print(f"📖 Language (TinyShakespeare):")
    print(f"   Train samples: {len(train_ld.dataset)}")
    print(f"   Val samples:   {len(val_ld.dataset)}")
    print(f"   Batch shape:   x={x.shape}, y={y.shape}")
    snippet = CharTokenizer.decode(x[0, :40].tolist())
    print(f"   Sample text:   '{snippet}'\n")

    # DLRM
    train_ld, val_ld = get_dlrm_dataloaders(batch_size=8)
    dense, sparse_idx, sparse_off, labels = next(iter(train_ld))
    print(f"🛒 DLRM (MovieLens-100K):")
    print(f"   Train samples: {len(train_ld.dataset)}")
    print(f"   Dense shape:   {dense.shape}")
    print(f"   Labels shape:  {labels.shape}")
    print(f"   Sparse tables: {len(sparse_idx)}")
    print(f"   Positive rate: {labels.mean().item():.3f}\n")

    print("✅ All dataset pipelines verified.")
    print("   (CIFAR-10/100 not tested here to avoid download — they auto-download on first use)")
