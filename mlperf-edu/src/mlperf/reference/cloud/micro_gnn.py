"""
MLPerf EDU: Micro-GCN — Graph Neural Network Workload
======================================================
Provenance: Kipf & Welling 2017, "Semi-Supervised Classification with
            Graph Convolutional Networks"
Maps to: MLPerf Training GNN (Graph Neural Networks division)

This implements a minimal Graph Convolutional Network (GCN) for node
classification on the **Cora citation network** — a real, widely-cited
benchmark dataset (2,708 nodes, 5,429 edges, 7 classes).

Dataset: Cora (McCallum et al., 2000; Kipf & Welling 2017)
    - 2,708 scientific papers (nodes)
    - 5,429 citation links (edges)
    - 1,433 binary bag-of-words features per paper
    - 7 subject classes: Case_Based, Genetic_Algorithms, Neural_Networks,
      Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory
    - Standard split: 140 train / 500 val / 1000 test (semi-supervised)
    - Ships locally in data/cora/ (168KB compressed)

Pedagogical concepts:
- Message passing: each node aggregates features from neighbors
- Graph convolution: spectral approximation via Chebyshev polynomials
- Over-smoothing: why deeper GNNs lose discriminative power
- Adjacency normalization: D^{-1/2} A D^{-1/2}
- Semi-supervised learning: only 20 labels per class (140 total)

Architecture:
    GCNConv(1433, 64) → ReLU → Dropout
    → GCNConv(64, 32) → ReLU → Dropout
    → Linear(32, 7)

Total: ~94K parameters
Target: Test accuracy > 0.78 (production GCN achieves ~81%)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORA_DIR = os.path.join(REPO_ROOT, "data", "cora", "cora")


# ============================================================================
# Graph Convolution Layer — Pure PyTorch, no PyG / DGL dependency
# ============================================================================

class GCNConv(nn.Module):
    """
    Graph Convolutional Layer (Kipf & Welling 2017).

    Computes: H' = σ(D̃^{-1/2} Ã D̃^{-1/2} H W)
    where Ã = A + I (self-loops), D̃ = degree matrix of Ã

    This is equivalent to a 1-hop neighborhood aggregation:
    each node's new features = learned linear combination of its
    neighbors' features (including itself).

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj_norm):
        """
        Args:
            x: Node features (N, in_features)
            adj_norm: Normalized adjacency matrix D̃^{-1/2} Ã D̃^{-1/2} (N, N)
        Returns:
            Updated node features (N, out_features)
        """
        # Step 1: Linear transform features
        support = torch.mm(x, self.weight)  # (N, out_features)
        # Step 2: Aggregate from neighbors via normalized adjacency
        output = torch.spmm(adj_norm, support)  # (N, out_features)
        return output + self.bias


class MicroGCN(nn.Module):
    """
    2-layer GCN for node classification on the Cora dataset.

    Architecture:
        Input features → GCNConv(hidden) → ReLU → Dropout
                       → GCNConv(hidden2) → ReLU → Dropout
                       → Linear(num_classes)

    Args:
        nfeat: Number of input features per node (1433 for Cora)
        nhid: Hidden dimension (default 64)
        nclass: Number of output classes (7 for Cora)
        dropout: Dropout rate

    Exercise: Over-smoothing Analysis
        Try adding more GCN layers (3, 4, 5) and measure test accuracy.
        You should observe accuracy degradation because GCNs suffer from
        "over-smoothing": after too many message-passing rounds, all node
        representations converge to the same vector (Laplacian smoothing).
        This is a fundamental limitation of spectral GNNs and motivates
        architectures like GAT, GraphSAGE, or techniques like residual
        connections and jumping knowledge.
    """

    def __init__(self, nfeat=1433, nhid=64, nclass=7, dropout=0.5):
        super().__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid // 2)
        self.classifier = nn.Linear(nhid // 2, nclass)
        self.dropout = dropout

    def forward(self, x, adj_norm):
        """
        Args:
            x: Node features (N, nfeat)
            adj_norm: Normalized adjacency (N, N) sparse or dense
        Returns:
            log_softmax over classes (N, nclass)
        """
        x = F.relu(self.gc1(x, adj_norm))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj_norm))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


# ============================================================================
# Cora Dataset Loader — real citation network, shipped locally
# ============================================================================

CLASS_NAMES = [
    "Case_Based", "Genetic_Algorithms", "Neural_Networks",
    "Probabilistic_Methods", "Reinforcement_Learning",
    "Rule_Learning", "Theory"
]


def load_cora(data_dir=None):
    """
    Load the Cora citation network from local files.

    Data format:
        cora.content: <paper_id> <1433 binary features> <class_label>
        cora.cites:   <cited_paper_id> <citing_paper_id>

    Returns:
        dict with keys: x, y, adj_norm, train_mask, val_mask, test_mask,
                        n_classes, n_features, class_names
    """
    if data_dir is None:
        data_dir = CORA_DIR

    content_path = os.path.join(data_dir, "cora.content")
    cites_path = os.path.join(data_dir, "cora.cites")

    if not os.path.exists(content_path):
        raise FileNotFoundError(
            f"Cora dataset not found at {data_dir}. "
            "Download: cd data/cora && curl -sL "
            "'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz' | tar xz"
        )

    # ── Parse nodes ──────────────────────────────────────────────────────
    # Each line: paper_id feat_1 feat_2 ... feat_1433 class_label
    paper_ids = []
    features_list = []
    labels_list = []

    label_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    with open(content_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            paper_id = int(parts[0])
            feats = list(map(float, parts[1:-1]))
            label_str = parts[-1]

            paper_ids.append(paper_id)
            features_list.append(feats)
            labels_list.append(label_to_idx[label_str])

    n_nodes = len(paper_ids)
    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}

    features = torch.tensor(features_list, dtype=torch.float32)  # (2708, 1433)
    labels = torch.tensor(labels_list, dtype=torch.long)          # (2708,)

    # ── Parse edges ──────────────────────────────────────────────────────
    edges_src, edges_dst = [], []
    with open(cites_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            cited = int(parts[0])
            citing = int(parts[1])
            if cited in id_to_idx and citing in id_to_idx:
                src = id_to_idx[cited]
                dst = id_to_idx[citing]
                # Undirected
                edges_src.extend([src, dst])
                edges_dst.extend([dst, src])

    # ── Build adjacency matrix with self-loops ───────────────────────────
    n = n_nodes
    indices = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    values = torch.ones(len(edges_src), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, size=(n, n))

    # Add self-loops
    self_loop = torch.sparse_coo_tensor(
        torch.arange(n).unsqueeze(0).repeat(2, 1),
        torch.ones(n), size=(n, n)
    )
    adj = (adj + self_loop).coalesce()

    # ── Normalize: D^{-1/2} A D^{-1/2} ──────────────────────────────────
    adj_dense = adj.to_dense()
    # Clamp to binary (remove duplicate edges)
    adj_dense = adj_dense.clamp(max=1.0)
    degree = adj_dense.sum(dim=1)
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D = torch.diag(d_inv_sqrt)
    adj_norm = D @ adj_dense @ D
    adj_norm_sparse = adj_norm.to_sparse()

    # ── Standard Cora split (Kipf & Welling convention) ──────────────────
    # 20 nodes per class for training = 140 total (semi-supervised)
    # 500 for validation, 1000 for test
    rng = np.random.RandomState(42)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    # Select 20 labeled nodes per class for training
    for c in range(len(CLASS_NAMES)):
        class_indices = (labels == c).nonzero(as_tuple=True)[0].numpy()
        rng.shuffle(class_indices)
        train_mask[class_indices[:20]] = True

    # Remaining nodes: 500 val, 1000 test
    remaining = (~train_mask).nonzero(as_tuple=True)[0].numpy()
    rng.shuffle(remaining)
    val_mask[remaining[:500]] = True
    test_mask[remaining[500:1500]] = True

    return {
        "x": features,
        "y": labels,
        "adj_norm": adj_norm_sparse,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "n_classes": len(CLASS_NAMES),
        "n_features": features.shape[1],
        "class_names": CLASS_NAMES,
        "n_nodes": n_nodes,
        "n_edges": len(edges_src) // 2,
    }


def get_gnn_dataloaders(seed=42, **kwargs):
    """
    Returns the Cora graph data dict (GNNs don't use standard DataLoaders).
    Compatible with the dataset_factory interface.
    """
    return load_cora()


# ============================================================================
# Training loop for standalone testing
# ============================================================================

def train_and_evaluate(epochs=200, lr=0.01, seed=42):
    """Training loop on Cora — targets ~78-81% test accuracy."""
    data = load_cora()

    print(f"  Cora dataset: {data['n_nodes']} nodes, {data['n_edges']} edges")
    print(f"  Features: {data['n_features']}, Classes: {data['n_classes']}")
    print(f"  Train: {data['train_mask'].sum()}, Val: {data['val_mask'].sum()}, "
          f"Test: {data['test_mask'].sum()}")

    model = MicroGCN(nfeat=data["n_features"], nclass=data["n_classes"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data["x"], data["adj_norm"])
        loss = F.nll_loss(out[data["train_mask"]], data["y"][data["train_mask"]])
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(data["x"], data["adj_norm"])
            val_pred = out[data["val_mask"]].argmax(dim=1)
            val_acc = (val_pred == data["y"][data["val_mask"]]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={loss.item():.4f}  val_acc={val_acc:.3f}")

    # Test
    model.eval()
    with torch.no_grad():
        out = model(data["x"], data["adj_norm"])
        test_pred = out[data["test_mask"]].argmax(dim=1)
        test_acc = (test_pred == data["y"][data["test_mask"]]).float().mean().item()

    return {
        "final_loss": loss.item(),
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "n_params": sum(p.numel() for p in model.parameters()),
    }


if __name__ == "__main__":
    model = MicroGCN(nfeat=1433, nclass=7)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MicroGCN: {n_params:,} parameters")
    print()
    print("Training on Cora citation network...")
    results = train_and_evaluate(epochs=200)
    print(f"\n✅ Results:")
    print(f"   Test accuracy: {results['test_acc']:.3f}")
    print(f"   Best val accuracy: {results['best_val_acc']:.3f}")
    print(f"   Final loss: {results['final_loss']:.4f}")
    print(f"   Parameters: {results['n_params']:,}")
