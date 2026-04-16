"""
MLPerf EDU: Micro-DLRM-DRAM (Cloud Division)

DRAM-bound DLRM variant. Same MLP topology as MicroDLRMWhiteBox, but
sparse lookups go through a virtual hashed embedding table sized to
exceed M1's last-level cache (~12 MB) by ~20x.

Why this exists
---------------
The original Micro-DLRM (943 users x 32 dim + 1682 items x 32 dim ~ 336 KB)
fits entirely in L2 on any modern laptop. Students profiling it observe
*compute-bound* behavior dominated by the MLP GEMMs. That is the *opposite*
of the production DLRM lesson, where embedding lookups against
hundreds-of-GB tables saturate DRAM bandwidth and stall on cache misses.

This variant restores the production memory-access regime while keeping
the data, vocabulary, and task identical. We project the real (small)
ID space into a much larger virtual table via a deterministic vectorized
hash, mirroring the "hash trick" used at scale (Weinberger et al. 2009;
Meta's hashed embedding tables in production DLRM).

Pedagogical contract
--------------------
On Apple Silicon (M1, ~68 GB/s unified bandwidth, ~12 MB LLC):
  - micro-dlrm           : sustained BW < 5 GB/s, MLP > 80% of step time.
  - micro-dlrm-dram (us) : sustained BW 50-65 GB/s, embedding lookups
                           > 70% of step time, step time linear in row
                           width but insensitive to MLP width.

The pair is meant to be run together so students *measure the bottleneck
transition* — same data, same MLP, different memory access pattern.

Why m_spa=256 (vs 8 in the cache variant)
-----------------------------------------
PyTorch's CPU EmbeddingBag has ~50 us of fixed overhead per call. With
an 8-dim row (32 bytes), 8192 lookups transfer only 256 KB of bytes
total -- well under L1's bandwidth limit, so the bottleneck is
PyTorch dispatch overhead and the table size is irrelevant. With a
256-dim row (1024 bytes), 8192 random lookups against an 8M-row table
transfer 8 MB through DRAM and the bandwidth signal becomes
unambiguous. Production DLRM uses 64-512 dim embeddings precisely
because anything smaller is dispatch-bound, not bandwidth-bound.

Provenance
----------
Naumov et al. 2019 (DLRM) + Weinberger et al. 2009 (hashing trick).
"""

import torch
import torch.nn as nn


# Mix constants from David Stafford's variant of MurmurHash3 64-bit
# finalizer. Vectorizable in PyTorch with int64 arithmetic.
_HASH_C1 = 0xBF58476D1CE4E5B9
_HASH_C2 = 0x94D049BB133111EB


def _hash_mod(idx: torch.Tensor, seed: int, mod: int) -> torch.Tensor:
    """Vectorized integer hash with output in [0, mod).

    Branch-free, runs on CPU/CUDA/MPS. Produces a uniform distribution
    over `mod` values for any input distribution. Each (seed, idx) pair
    maps to a deterministic virtual row.
    """
    x = idx.to(torch.int64) ^ seed
    x = (x ^ (x >> 30)) * _HASH_C1
    x = (x ^ (x >> 27)) * _HASH_C2
    x = x ^ (x >> 31)
    return (x.abs() % mod).to(torch.long)


class MicroDLRMDRAM(nn.Module):
    """DRAM-bound DLRM: same MLP topology, hash-mapped virtual embedding table.

    Defaults are tuned for an Apple M1 (12 MB LLC, 68 GB/s unified memory):
        virtual_table_size=2_000_000, m_spa=32  ->  256 MB table (~21x LLC).
    Adjust virtual_table_size up if your machine has > 32 MB LLC.
    """

    def __init__(self,
                 m_spa: int = 256,
                 virtual_table_size: int = 2_000_000,
                 num_hash_seeds=(0xA5A5A5A5, 0x5A5A5A5A, 0xC3C3C3C3),
                 ln_bot=(16, 8, 8),
                 ln_top=(32, 16, 1),
                 sparse_grad: bool = True):
        super().__init__()
        self.m_spa = m_spa
        self.virtual_table_size = virtual_table_size
        self.register_buffer(
            "hash_seeds",
            torch.tensor(list(num_hash_seeds), dtype=torch.int64),
        )

        # ONE physical table, shared across logical features via distinct seeds.
        # sparse=True keeps Adam moment buffers small (only touched rows
        # accumulate optimizer state).
        self.virtual_emb = nn.EmbeddingBag(
            virtual_table_size, m_spa, mode="sum", sparse=sparse_grad,
        )

        # Bottom MLP for dense features (identical to MicroDLRMWhiteBox).
        layers = []
        for i in range(len(ln_bot) - 1):
            layers.append(nn.Linear(ln_bot[i], ln_bot[i + 1]))
            layers.append(nn.ReLU())
        self.bot_l = nn.Sequential(*layers)

        # Top MLP for CTR prediction.
        cross_dim = ln_bot[-1] + len(num_hash_seeds) * m_spa
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
        """Same calling convention as MicroDLRMWhiteBox.

        Args:
            dense_x: (B, ln_bot[0]) continuous features.
            sparse_indices: list of (B,) int tensors, one per logical feature.
            sparse_offsets: list of (B,) int tensors for EmbeddingBag offsets.
        """
        x_dense = self.bot_l(dense_x)
        x_sparse = []
        for i, raw_idx in enumerate(sparse_indices):
            seed = int(self.hash_seeds[i].item())
            hashed = _hash_mod(raw_idx, seed, self.virtual_table_size)
            x_sparse.append(self.virtual_emb(hashed, sparse_offsets[i]))
        interaction = torch.cat([x_dense] + x_sparse, dim=1)
        return self.top_l(interaction)

    def working_set_bytes(self) -> int:
        """Total physical bytes of the virtual embedding table (fp32)."""
        return self.virtual_table_size * self.m_spa * 4
