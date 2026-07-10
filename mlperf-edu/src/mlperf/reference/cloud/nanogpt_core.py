import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, bias: bool = True):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer("bias", torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024))

    def forward(self, x, use_kv_cache=False, past_key_value=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # PEDAGOGICAL HOOK: KV-Caching
        if use_kv_cache:
            if past_key_value is not None:
                past_k, past_v = past_key_value
                k = torch.cat((past_k, k), dim=-2)
                v = torch.cat((past_v, v), dim=-2)
            present_key_value = (k, v)
        else:
            present_key_value = None

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Causal mask
        T_k = k.size(-2)
        att = att.masked_fill(self.bias[:, :, :T, :T_k] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, present_key_value

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x, use_kv_cache=False, past_key_value=None):
        attn_out, present_kv = self.attn(self.ln_1(x), use_kv_cache, past_key_value)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv

class GPT(nn.Module):
    """Pedagogical NanoGPT (124M param default)"""
    def __init__(self, vocab_size=50257, n_layer=12, n_head=12, n_embd=768, max_seq_len=1024):
        super().__init__()
        self.config = {
            'vocab_size': vocab_size, 'n_layer': n_layer, 
            'n_head': n_head, 'n_embd': n_embd, 'max_seq_len': max_seq_len
        }
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(max_seq_len, n_embd),
            h = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # weight tying

    def forward(self, idx, use_kv_cache=False, past_key_values=None):
        device = idx.device
        b, t = idx.size()
        
        # Position embeddings offset if using KV cache
        past_length = past_key_values[0][0].size(-2) if past_key_values is not None else 0
        pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=device)
        
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        
        present_key_values = [] if use_kv_cache else None
        
        for i, block in enumerate(self.transformer.h):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, use_kv_cache=use_kv_cache, past_key_value=past_kv)
            if use_kv_cache:
                present_key_values.append(present_kv)
                
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        return logits, present_key_values
        
    @classmethod
    def from_pretrained(cls, model_type="gpt2"):
        """Downloads canonical weights explicitly bridging the provenance."""
        try:
            from transformers import GPT2LMHeadModel
        except ImportError:
            raise ImportError("[Cloud] You must install `transformers` to load Golden Weights (pip install transformers)")
            
        print(f"[Provenance] 📥 Downloading Golden '{model_type}' weights via HuggingFace...")
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        model = cls()
        sd = model.state_dict()
        
        # Transfer logic
        keys = [k for k in sd_hf if not k.endswith('.attn.masked_bias')] 
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            elif k in sd:
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
                    
        print(f"[Provenance] ✅ {model_type} architecture hydrated flawlessly!")
        return model
