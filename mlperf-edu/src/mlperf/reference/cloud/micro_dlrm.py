"""
MLPerf EDU: Micro-DLRM (Cloud Division)

A scaled-down Deep Learning Recommendation Model for click-through rate
prediction, mapping the MLPerf Training DLRM benchmark to laptop scale.

Architecture:
    Dense features → Bottom MLP → dense embedding
    Sparse features → EmbeddingBag tables → sparse embeddings
    [dense_emb, sparse_embs] → concatenate → Top MLP → sigmoid → CTR

The model demonstrates the unique memory access pattern of recommendation:
- Sparse embeddings → memory bandwidth bound (random lookups)
- Dense MLP → compute bound (matrix multiplications)

Dataset: MovieLens-100K (Harper & Konstan, 2015)
    - 100,000 ratings from 943 users on 1,682 movies
    - Binarized at threshold 4: rating >= 4 → positive click
    - Ships locally in data/movielens/ml-100k/ (5 MB)

Quality Target: Acc > 0.70 on MovieLens binary click prediction (best val ~71%)

Provenance: Naumov et al. 2019, "Deep Learning Recommendation Model"
"""

import torch
import torch.nn as nn


class MicroDLRMWhiteBox(nn.Module):
    """
    Micro-scale DLRM for MovieLens-100K recommendation.

    Implements the core DLRM pattern: separate processing of dense
    (continuous) and sparse (categorical) features, followed by
    feature interaction and CTR prediction.

    Default embedding sizes match MovieLens-100K:
    - user_id: 943 users
    - item_id: 1682 items  
    - occupation: 21 categories
    """

    def __init__(self,
                 m_spa=8,
                 num_embeddings=[943, 1682, 21],
                 ln_bot=[16, 8, 8],
                 ln_top=[32, 16, 1]):
        super().__init__()

        # Sparse: embedding tables for categorical features
        self.emb_l = nn.ModuleList([
            nn.EmbeddingBag(n, m_spa, mode="sum", sparse=False)
            for n in num_embeddings
        ])

        # Dense: bottom MLP for continuous features
        layers = []
        for i in range(len(ln_bot) - 1):
            layers.append(nn.Linear(ln_bot[i], ln_bot[i + 1]))
            layers.append(nn.ReLU())
        self.bot_l = nn.Sequential(*layers)

        # Feature interaction: concat dense output + all sparse embeddings
        cross_dim = ln_bot[-1] + len(num_embeddings) * m_spa

        # Top MLP: CTR prediction
        top_layers = []
        in_dim = cross_dim
        for out_dim in ln_top[:-1]:
            top_layers.append(nn.Linear(in_dim, out_dim))
            top_layers.append(nn.ReLU())
            in_dim = out_dim
        top_layers.append(nn.Linear(in_dim, ln_top[-1]))
        top_layers.append(nn.Sigmoid())
        self.top_l = nn.Sequential(*top_layers)

    def forward(self, dense_x, sparse_indices, sparse_offsets):
        """
        Args:
            dense_x: (B, 16) continuous features
            sparse_indices: list of (B,) index tensors for each embedding table
            sparse_offsets: list of (B,) offset tensors for EmbeddingBag

        Returns:
            (B, 1) click-through probability
        """
        # Process dense features through bottom MLP
        x_dense = self.bot_l(dense_x)

        # Lookup sparse embeddings
        x_sparse = []
        for i, emb in enumerate(self.emb_l):
            z = emb(sparse_indices[i], sparse_offsets[i])
            x_sparse.append(z)

        # Feature interaction: concatenate dense + sparse
        # NOTE: The official DLRM uses dot-product interaction:
        #   T = stack([x_dense] + x_sparse)  # (B, n_features, embed_dim)
        #   Z = bmm(T, T.transpose(1,2))     # (B, n, n) pairwise interactions
        #   flat = Z[triu_indices]            # upper triangle features
        # We use concat for simplicity. Switching to dot-product interaction
        # is a pedagogical exercise that exposes feature crossing and the
        # compute vs. memory tradeoff in sparse-dense architectures.
        interaction = torch.cat([x_dense] + x_sparse, dim=1)

        # Predict CTR
        return self.top_l(interaction)
