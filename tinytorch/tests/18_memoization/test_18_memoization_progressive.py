"""Module 18 composes earlier tensor, attention, and transformer mechanisms.

These are instructor-solution checks: missing exports and broken calls fail.
Numerical equivalence across full GPT layers lives in
``test_cached_generation_equivalence.py``.
"""
import numpy as np

from tinytorch.core.tensor import Tensor
from tinytorch.core.attention import MultiHeadAttention
from tinytorch.core.transformers import create_causal_mask
from tinytorch.perf.memoization import KVCache, CachedAttention


def test_cached_attention_matches_full_causal_attention():
    rng = np.random.default_rng(7)
    attention = MultiHeadAttention(embed_dim=16, num_heads=2)
    inputs = Tensor(rng.standard_normal((1, 4, 16)))
    expected = attention(inputs, create_causal_mask(4)).data
    cache = KVCache(1, 4, 1, 2, 8)
    cached = CachedAttention(attention, cache, 0)
    for position in range(4):
        actual = cached(inputs[:, position:position + 1])
        np.testing.assert_allclose(actual.data, expected[:, position:position + 1], atol=1e-5)
        cache.advance()
    assert cache.seq_pos == 4
    cache.reset()
    assert cache.get(0)[0].shape == (1, 2, 0, 8)
