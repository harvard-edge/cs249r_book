"""
MLPerf EDU: Micro-BERT — Text Classification Workload
======================================================
Provenance: Devlin et al. 2019, "BERT: Pre-training of Deep
            Bidirectional Transformers for Language Understanding"
Maps to: MLPerf Training BERT

This implements a micro-scale bidirectional transformer encoder
for binary sentiment classification on the **SST-2** dataset
(Stanford Sentiment Treebank, Socher et al. 2013).

Dataset: SST-2 (Stanford Sentiment Treebank — Binary)
    - 67,349 training sentences from movie reviews
    - 872 validation sentences
    - Binary labels: 0 = negative, 1 = positive
    - Ships locally in data/sst2/SST-2/ (3.8 MB)
    - Provenance: Socher et al., EMNLP 2013

Pedagogical concepts:
- Bidirectional self-attention (vs causal in GPT)
- [CLS] token pooling for classification
- Position embeddings for sequence ordering
- Fine-tuning paradigm (pre-train + task head)
- Real NLU challenge: sarcasm, negation, context

Architecture:
    CharEmbed + PosEmbed → TransformerEncoder(2 layers, 4 heads, d=128)
    → [CLS] pooling → Linear(128, 2)

Total: ~0.5M parameters
Target: Val accuracy > 0.78 (with character-level tokenization)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SST2_DIR = os.path.join(REPO_ROOT, "data", "sst2", "SST-2")

# Special token indices
PAD_IDX = 0
CLS_IDX = 1
SEP_IDX = 2
UNK_IDX = 3


def build_vocab(path, max_vocab=5000):
    """Build word-level vocabulary from SST-2 training data."""
    counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                words = parts[0].lower().split()
                counter.update(words)
    # Top max_vocab words + 4 special tokens
    vocab = {"[PAD]": PAD_IDX, "[CLS]": CLS_IDX, "[SEP]": SEP_IDX, "[UNK]": UNK_IDX}
    for word, _ in counter.most_common(max_vocab):
        vocab[word] = len(vocab)
    return vocab


def word_encode(text, vocab, max_len=64):
    """Encode text as word-level tokens with [CLS] and [SEP]."""
    tokens = [CLS_IDX]
    for word in text.lower().split()[:max_len - 2]:
        tokens.append(vocab.get(word, UNK_IDX))
    tokens.append(SEP_IDX)
    if len(tokens) < max_len:
        tokens += [PAD_IDX] * (max_len - len(tokens))
    return tokens


# ============================================================================
# SST-2 Dataset — real binary sentiment from Stanford NLP
# ============================================================================

class SST2Dataset(Dataset):
    """
    SST-2 binary sentiment classification dataset.

    Loads real movie review sentences from the Stanford Sentiment
    Treebank. Each sentence is a short movie review fragment with
    a binary label (0=negative, 1=positive).

    Uses word-level whitespace tokenization with a 5K vocabulary built
    from the training data — no external tokenizer dependency needed.

    Args:
        split: "train" or "dev"
        vocab: Word-to-index dictionary (built from training data)
        max_len: Maximum sequence length
        max_samples: Limit samples for faster training (None = all)
    """

    def __init__(self, split="train", vocab=None, max_len=64, max_samples=None):
        super().__init__()
        self.max_len = max_len

        if split == "train":
            path = os.path.join(SST2_DIR, "train.tsv")
        else:
            path = os.path.join(SST2_DIR, "dev.tsv")

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"SST-2 dataset not found at {path}. "
                "Download: cd data/sst2 && curl -sL "
                "'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip' "
                "-o SST-2.zip && unzip SST-2.zip"
            )

        # Build vocab from training data if not provided
        if vocab is None:
            train_path = os.path.join(SST2_DIR, "train.tsv")
            vocab = build_vocab(train_path)
        self.vocab = vocab

        self.texts = []
        self.labels = []

        with open(path, "r", encoding="utf-8") as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    sentence = parts[0]
                    label = int(parts[1])
                    self.texts.append(word_encode(sentence, vocab, max_len))
                    self.labels.append(label)

        if max_samples is not None:
            self.texts = self.texts[:max_samples]
            self.labels = self.labels[:max_samples]

        self.texts = torch.tensor(self.texts, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# ============================================================================
# Micro-BERT Model
# ============================================================================

class MicroBERT(nn.Module):
    """
    Micro-scale BERT encoder for binary sentiment classification.

    Key differences from GPT:
    - Bidirectional attention (no causal mask)
    - [CLS] token pooling for classification
    - Smaller scale: 2 layers, 4 heads, d=128

    Args:
        vocab_size: Vocabulary size (~5004 for word-level)
        d_model: Hidden dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        max_len: Maximum sequence length
        num_classes: Number of classification labels (2 for SST-2)
    """

    def __init__(self, vocab_size=5004, d_model=128, nhead=4,
                 num_layers=2, max_len=64, num_classes=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.embed_norm = nn.LayerNorm(d_model)

        # Transformer encoder (bidirectional — no causal mask!)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head: [CLS] → Linear → Softmax
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids: Token indices (B, seq_len)
            targets: Optional class labels (B,) for computing loss
        Returns:
            logits: (B, num_classes)
            loss: Cross-entropy loss (if targets provided)
        """
        B, T = input_ids.shape

        # Embed tokens + positions
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.embed_norm(self.embed_dropout(x))

        # Create padding mask
        pad_mask = (input_ids == PAD_IDX)  # True where padded

        # Bidirectional transformer encoding
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        # Pool: take the [CLS] token (position 0)
        cls_output = x[:, 0, :]  # (B, d_model)

        # Classify
        logits = self.classifier(cls_output)  # (B, num_classes)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss


def get_bert_dataloaders(batch_size=64, seed=42, max_train=10000):
    """
    Create train/val DataLoaders for SST-2 sentiment classification.

    Uses a 10K subset of training data by default for fast iteration
    (full 67K available by setting max_train=None).
    """
    # Build vocabulary from full training data
    vocab = build_vocab(os.path.join(SST2_DIR, "train.tsv"))
    train_ds = SST2Dataset(split="train", vocab=vocab, max_samples=max_train)
    val_ds = SST2Dataset(split="dev", vocab=vocab)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, len(vocab)


# Training loop for standalone testing
def train_and_evaluate(epochs=20, batch_size=64, lr=1e-3, seed=42):
    """Train MicroBERT on SST-2 and report convergence."""
    torch.manual_seed(seed)

    train_loader, val_loader, vocab_size = get_bert_dataloaders(
        batch_size=batch_size, seed=seed
    )
    model = MicroBERT(vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"  SST-2: {len(train_loader.dataset)} train, "
          f"{len(val_loader.dataset)} val sentences")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits, loss = model(x, targets=y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        train_loss /= len(train_loader)
        train_acc = correct / total

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                logits, loss = model(x, targets=y)
                val_loss += loss.item()
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total

        if (epoch + 1) % 4 == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f} "
                  f"train_acc={train_acc:.3f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "final_train_acc": train_acc,
        "final_val_acc": val_acc,
        "n_params": n_params,
    }


if __name__ == "__main__":
    vocab = build_vocab(os.path.join(SST2_DIR, "train.tsv"))
    model = MicroBERT(vocab_size=len(vocab))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MicroBERT: {n_params:,} parameters")
    print(f"Vocab: {len(vocab)} (word-level from SST-2)")
    print()
    print("Training on SST-2 sentiment classification...")
    results = train_and_evaluate(epochs=20)
    print(f"\n✅ Results:")
    print(f"   Final val accuracy: {results['final_val_acc']:.3f}")
    print(f"   Final val loss: {results['final_val_loss']:.4f}")
    print(f"   Parameters: {results['n_params']:,}")
