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


class NanoGPTWhiteBox(nn.Module):
    """Char-level decoder-only transformer used for the NanoGPT workload.

    With defaults (vocab=128, n_embd=384, n_head=6, n_layer=6) this is
    ~11M parameters — small enough to converge on TinyShakespeare in
    ~90 s on Apple Silicon, large enough to exhibit the O(N^2) attention
    pattern that motivates KV-cache and FlashAttention discussions.
    """

    def __init__(self, vocab_size=CHAR_VOCAB_SIZE, n_embd=384, n_head=6, n_layer=6, max_seq_len=1024):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size, "n_embd": n_embd,
            "n_head": n_head, "n_layer": n_layer, "max_seq_len": max_seq_len,
        }
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(max_seq_len, n_embd)
        self.blocks = nn.Sequential(*[GPTBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
