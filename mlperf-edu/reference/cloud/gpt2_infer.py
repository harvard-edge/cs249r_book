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
    def __init__(self, n_embd=768, n_head=12, max_seq_len=1024):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
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
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), present_key_value

class GPTBlock(nn.Module):
    def __init__(self, n_embd=768, n_head=12, max_seq_len=1024):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, max_seq_len=max_seq_len)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x, use_kv_cache=False, past_key_value=None):
        attn_out, present_kv = self.attn(self.ln_1(x), use_kv_cache, past_key_value)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv

class GPT2WhiteBox(nn.Module):
    """Reference GPT-2 architecture (124M params at default config).

    Forward returns (logits, present_key_values) for KV-cache support.
    Training callers can ignore the second element.
    """
    def __init__(self, vocab_size=50257, n_embd=768, n_head=12, n_layer=12, max_seq_len=1024):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(max_seq_len, n_embd)
        self.blocks = nn.ModuleList([
            GPTBlock(n_embd, n_head, max_seq_len=max_seq_len) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

    def forward(self, idx, use_kv_cache=False, past_key_values=None):
        b, t = idx.size()
        past_length = past_key_values[0][0].size(-2) if past_key_values is not None else 0
        pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        present_kvs = [] if use_kv_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, use_kv_cache=use_kv_cache, past_key_value=past_kv)
            if use_kv_cache:
                present_kvs.append(present_kv)
        x = self.ln_f(x)
        return self.lm_head(x), present_kvs
