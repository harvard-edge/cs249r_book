"""
NanoGPT model definition for MLPerf EDU.

The canonical training entry point is `scripts/verify_training.py`, which
loads `NanoGPTWhiteBox` and trains it with the dataset_factory's
character-level TinyShakespeare loader. This file exports only the model
class and its tokenizer contract; do not add a `run_benchmark` here.

Vocab note: dataset_factory uses an ASCII char tokenizer over
TinyShakespeare with 65 unique chars. We use vocab_size=128 to give a
small safety margin while keeping the embedding matrix small (the GPT-2
vocab of 50,257 is BPE and does not apply to char-level data).
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from .gpt2_infer import GPTBlock

CHAR_VOCAB_SIZE = 128  # ASCII range; TinyShakespeare uses 65 of these.

# iter-6 (Han): GPT-2-Small canonical geometry as the "real LLM" stand-in.
# 124M params at d_model=768, n_head=12, n_layer=12. Trains in ~30 min on
# M5 Max; weights ~248 MB at fp16. This is the configuration that fills
# the (dram_bound, bandwidth_bound, device_saturated) cell when run with
# batch_size=32 ctx=2048 per Han's iter-6 sizing math.
GPT2_SMALL_CONFIG = dict(n_embd=768, n_head=12, n_layer=12, max_seq_len=4096)


class NanoGPTWhiteBox(nn.Module):
    """Char-level decoder-only transformer used for the NanoGPT workload.

    Default config (~11M params) is the iter-1/iter-3 small variant.
    Pass NanoGPTWhiteBox(**GPT2_SMALL_CONFIG) for the iter-6 124M
    variant used by the LLM serving workloads.
    """

    def __init__(self, vocab_size=CHAR_VOCAB_SIZE, n_embd=384, n_head=6, n_layer=6, max_seq_len=2048):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size, "n_embd": n_embd,
            "n_head": n_head, "n_layer": n_layer, "max_seq_len": max_seq_len,
        }
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(max_seq_len, n_embd)
        self.blocks = nn.ModuleList([
            GPTBlock(n_embd, n_head, max_seq_len=max_seq_len) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None, use_kv_cache=False, past_key_values=None):
        """Unified forward path for training and inference (incl. KV-cache decode).

        Returns:
            (logits, loss)            if targets is given (training).
            (logits, present_kvs)     if use_kv_cache=True (decode).
            (logits, None)            otherwise (single-shot inference / prefill).
        """
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
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits, present_kvs
