import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    """
    Pure Pedagogical PyTorch implementation of Masked Multi-Head Attention.
    Stripped of all external macros to teach explicit Attention matrices.
    """
    def __init__(self, n_embd=768, n_head=12, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd) # Q, K, V
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        self.n_head = n_head
        self.n_embd = n_embd
        # Native causal mask simulating autoregressive bounds
        self.register_buffer("bias", torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 1. Native Mathematical Attention (Flash Attention Target optimization!)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Target bounding for Auto-Regressive tracking
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class GPTBlock(nn.Module):
    def __init__(self, n_embd=768, n_head=12):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2WhiteBox(nn.Module):
    """
    Teacher Golden Baseline for MLPerf Cloud Generation boundaries organically parsing 
    pure PyTorch modules functionally simulating parameter tracking organically natively.
    """
    def __init__(self, vocab_size=50257, n_embd=768, n_head=12, n_layer=12):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(1024, n_embd)
        
        self.blocks = nn.Sequential(*[GPTBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Tie weights mathematically cleanly
        self.wte.weight = self.lm_head.weight

    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        
        # Feature embeddings organically merging Native parameters structurally
        x = self.wte(idx) + self.wpe(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        
        return self.lm_head(x)
