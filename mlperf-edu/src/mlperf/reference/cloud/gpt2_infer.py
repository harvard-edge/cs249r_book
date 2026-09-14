import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    """Masked multi-head attention with optional KV-cache for autoregressive decode.

    The KV-cache path enables NanoGPTDecode (iter-3) to demonstrate
    bandwidth-bound decode behavior: each step appends one token's
    K and V, and attention re-reads the entire cached K, V from DRAM.
    """
    def __init__(self, n_embd=768, n_head=12, max_seq_len=1024, dropout=0.0, bias=True):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len))
                 .view(1, 1, max_seq_len, max_seq_len),
        )

    def forward(self, x, use_kv_cache=False, past_key_value=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # KV-cache append: re-reads past_k, past_v from DRAM each step.
        # The naive torch.cat allocates a fresh tensor and copies; that
        # is the load-bearing memory traffic the decode workload measures.
        if use_kv_cache and past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        present_key_value = (k, v) if use_kv_cache else None

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Causal mask: when T==1 (decode step), attend to all cached keys.
        T_k = k.size(-2)
        att = att.masked_fill(self.bias[:, :, T_k - T:T_k, :T_k] == 0, float('-inf'))
        att = self.attn_dropout(F.softmax(att, dim=-1))

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y)), present_key_value

class GPTBlock(nn.Module):
    def __init__(self, n_embd=768, n_head=12, max_seq_len=1024, dropout=0.0, bias=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd,
            n_head,
            max_seq_len=max_seq_len,
            dropout=dropout,
            bias=bias,
        )
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x, use_kv_cache=False, past_key_value=None):
        attn_out, present_kv = self.attn(self.ln_1(x), use_kv_cache, past_key_value)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv
