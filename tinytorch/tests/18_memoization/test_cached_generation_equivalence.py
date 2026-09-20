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

    with cache.generation():
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


@pytest.mark.parametrize("mask_rank", [2, 3, 4])
def test_cached_attention_respects_explicit_prefix_mask(mask_rank):
    from tinytorch.core.attention import MultiHeadAttention
    from tinytorch.perf.memoization import KVCache

    attention = MultiHeadAttention(8, 2)
    inputs = Tensor(np.arange(16).reshape(1, 2, 8) / 10)
    expected = attention(inputs, Tensor(np.eye(2))).data[:, 1:2]
    cache = KVCache(1, 4, 1, 2, 4)
    cached = CachedAttention(attention, cache, 0)
    mask = np.array([0, 1]).reshape((1,) * (mask_rank - 1) + (2,))
    with cache.generation():
        cached(inputs[:, :1])
        cache.advance()
        actual = cached(inputs[:, 1:2], Tensor(mask))
    np.testing.assert_allclose(actual.data, expected, atol=1e-6)


@pytest.mark.parametrize("mask", [[[0, 0]], [[1, 0.5]], [[1, np.nan]], [[1, 1, 1]]])
def test_cached_attention_rejects_invalid_masks(mask):
    from tinytorch.core.attention import MultiHeadAttention
    from tinytorch.perf.memoization import KVCache

    cache = KVCache(1, 4, 1, 2, 4)
    cached = CachedAttention(MultiHeadAttention(8, 2), cache, 0)
    token = Tensor(np.ones((1, 1, 8)))
    with cache.generation():
        cached(token)
        cache.advance()
        with pytest.raises(ValueError):
            cached(token, Tensor(mask))


def test_cached_generation_does_not_change_ordinary_forward_or_generate():
    from tinytorch.perf.memoization import _cached_generate

    model = GPT(vocab_size=12, embed_dim=16, num_layers=2, num_heads=2, max_seq_len=8)
    token = Tensor([[3]])
    expected_forward = model(token).data.copy()
    expected_generation = model.generate(token, 3, temperature=0).data.copy()
    cache = enable_kv_cache(model)
    _cached_generate(model, [1, 2], 2, 0, cache)
    position = cache.seq_pos
    np.testing.assert_allclose(model(token).data, expected_forward, atol=1e-6)
    np.testing.assert_array_equal(model.generate(token, 3, temperature=0).data, expected_generation)
    assert cache.seq_pos == position


def test_generation_scope_restores_ordinary_forward_after_exception():
    from tinytorch.perf.memoization import _cached_generate

    model = GPT(vocab_size=12, embed_dim=16, num_layers=1, num_heads=2, max_seq_len=8)
    token = Tensor([[3]])
    expected = model(token).data.copy()
    cache = enable_kv_cache(model)
    # First token populates history; the second fails embedding validation.
    with pytest.raises((ValueError, IndexError)):
        _cached_generate(model, [1, 99], 1, 0, cache)
    np.testing.assert_allclose(model(token).data, expected, atol=1e-6)
    with cache.generation():
        with pytest.raises(RuntimeError):
            with cache.generation():
                raise RuntimeError("interrupted request")
        assert cache._generation_active
    assert not cache._generation_active
