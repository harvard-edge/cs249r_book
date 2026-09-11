"""
Regression: KV-cached generation must reproduce the uncached model exactly.

Written 2026-09 after three defects were found in the cache wiring (every layer
used the last block's weights, the first token was never cached, and every
single-token forward used position 0). A cache that changes the numbers is not
an optimization, so this test feeds a prompt one token at a time through the
cached model and compares each step's logits with the full uncached forward.
"""
import io
import contextlib
import numpy as np
import pytest

from tinytorch.core.tensor import Tensor
from tinytorch.core.transformers import GPT
from tinytorch.perf.memoization import CachedAttention, enable_kv_cache, disable_kv_cache


@pytest.mark.parametrize("num_layers", [1, 3])
def test_cached_generation_matches_uncached(num_layers):
    np.random.seed(0)
    model = GPT(vocab_size=50, embed_dim=32, num_layers=num_layers, num_heads=4)
    tokens = [3, 7, 2, 9, 11]

    full = model.forward(Tensor(np.array([tokens]))).data[0]        # (T, vocab)

    with contextlib.redirect_stdout(io.StringIO()):
        cache = enable_kv_cache(model)
    assert all(isinstance(b.attention, CachedAttention) for b in model.blocks)

    for t, token in enumerate(tokens):
        step = model.forward(Tensor(np.array([[token]])), start_pos=cache.seq_pos).data[0, -1]
        cache.advance()
        assert np.allclose(step, full[t], atol=1e-5), f"position {t} differs from the uncached model"

    with contextlib.redirect_stdout(io.StringIO()):
        disable_kv_cache(model)
    assert not any(isinstance(b.attention, CachedAttention) for b in model.blocks)
    assert np.allclose(model.forward(Tensor(np.array([tokens]))).data[0], full), "disable must restore the model"


def test_generation_at_context_boundary_avoids_unused_final_forward():
    from tinytorch.perf.memoization import _cached_generate
    model = GPT(vocab_size=12, embed_dim=16, num_layers=1, num_heads=2, max_seq_len=3)
    expected = model.generate(Tensor([[1, 2]]), 1, temperature=0).data[0, -1]
    cache = enable_kv_cache(model)
    assert _cached_generate(model, [1, 2], 1, 0, cache) == [expected]
    assert cache.seq_pos == 2
    assert _cached_generate(model, [1], 0, 0, cache) == []
    assert cache.seq_pos == 0
    for prompt, count, temperature in [([], 1, 1), ([1], -1, 1), ([1], 1, -1), ([1, 2, 3], 2, 1)]:
        with pytest.raises(ValueError):
            _cached_generate(model, prompt, count, temperature, cache)
