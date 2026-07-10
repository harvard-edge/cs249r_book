"""
MLPerf EDU: Nano-MoE (Cloud Division)

A sparse Mixture-of-Experts language model with 8 experts and top-2
routing, mapping the MLPerf Training Switch Transformer benchmark
to laptop scale.

Architecture:
    Token embedding + positional embedding
    → N layers of [Self-Attention + Sparse MoE FFN]
    → Language model head

The MoE layer routes each token to 2 of 8 experts, demonstrating:
- Sparse computation: only 25% of expert parameters activate per token
- Total vs. active parameter distinction (17.4M total, ~5M active)
- Routing overhead and load balancing challenges

Quality Target: Cross-entropy loss < 0.05 on TinyShakespeare

Provenance: Shazeer et al. 2017, "Outrageously Large Neural Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single expert: a standard 2-layer FFN with SiLU activation."""

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class SparseMoERouter(nn.Module):
    """
    Top-K sparse Mixture-of-Experts layer.

    Routes each token to top_k experts based on a learned gating function.
    Students can measure:
    - Load balance across experts (are all experts used equally?)
    - Routing overhead (gate computation + gather/scatter)
    - Memory: all experts are in memory, but only top_k compute per token
    """

    def __init__(self, d_model: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(d_model, d_model * 4) for _ in range(num_experts)
        ])

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)

        # Compute routing probabilities
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=1)

        # Select top-k experts per token
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Dispatch tokens to experts and gather results
        final_output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            idx, nth_expert = torch.where(selected_experts == i)
            if idx.shape[0] > 0:
                expert_out = expert(x_flat[idx])
                final_output[idx] += expert_out * routing_weights[idx, nth_expert, None]

        return final_output.view(B, T, D)


class NanoMoEWhiteBox(nn.Module):
    """
    Sparse MoE language model (17.4M parameters).

    Replaces the standard FFN in each transformer layer with a
    SparseMoERouter, demonstrating sparse conditional computation.
    """

    def __init__(self, vocab_size=50257, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(256, d_model)

        self.layers = nn.ModuleList([
            nn.ModuleDict(dict(
                ln_1=nn.LayerNorm(d_model),
                attn=nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                ln_2=nn.LayerNorm(d_model),
                moe=SparseMoERouter(d_model, num_experts=8, top_k=2),
            ))
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        x = self.token_emb(idx) + self.pos_emb(pos)

        for block in self.layers:
            # Self-attention with residual
            attn_out, _ = block['attn'](
                block['ln_1'](x), block['ln_1'](x), block['ln_1'](x),
                need_weights=False
            )
            x = x + attn_out
            # Sparse MoE FFN with residual
            x = x + block['moe'](block['ln_2'](x))

        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
