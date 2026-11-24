"""Tests for the KV-cache decode exercise (see README.md)."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _impl import load  # noqa: E402

impl = load(Path(__file__).resolve().parent)

T, D_MODEL, D_HEAD = 64, 32, 16


class CountingProjector:
    """``x @ w`` that counts how many token rows it has projected."""

    def __init__(self):
        self.rows = 0

    def __call__(self, x, w):
        self.rows += 1 if x.ndim == 1 else x.shape[0]
        return x @ w


@pytest.fixture
def model():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((T, D_MODEL))
    w = [rng.standard_normal((D_MODEL, D_HEAD)) / np.sqrt(D_MODEL) for _ in range(3)]
    return x, w


def _decode_all(x, w, project=np.matmul):
    cache = impl.KVCache(max_len=T, d_head=D_HEAD)
    outs = np.stack([impl.decode_step(x[t], *w, cache, project=project) for t in range(T)])
    return outs, cache


def test_matches_full_recompute(model):
    x, w = model
    outs, _ = _decode_all(x, w)
    np.testing.assert_allclose(outs, impl.full_causal_attention(x, *w), rtol=1e-5, atol=1e-5)


def test_each_token_is_projected_once(model):
    # Recomputing K and V for the whole prefix each step projects
    # 3 * T * (T + 1) / 2 rows (quadratic); a KV cache projects 3 * T (linear).
    x, w = model
    counter = CountingProjector()
    _decode_all(x, w, project=counter)
    assert counter.rows == 3 * T


def test_step_reads_one_cached_row_per_earlier_token(model):
    x, w = model
    cache = impl.KVCache(max_len=T, d_head=D_HEAD)
    for t in range(T):
        impl.decode_step(x[t], *w, cache)
        keys, values = cache.view()
        assert keys.shape == values.shape == (t + 1, D_HEAD)
    assert cache.length == T


def test_cache_is_preallocated_and_bounded(model):
    x, w = model
    cache = impl.KVCache(max_len=2, d_head=D_HEAD)
    buffer = cache.keys
    impl.decode_step(x[0], *w, cache)
    impl.decode_step(x[1], *w, cache)
    assert cache.keys is buffer                     # written in place, never regrown
    with pytest.raises(ValueError):
        impl.decode_step(x[2], *w, cache)


def test_kv_cache_bytes_llama2_7b():
    # Llama-2-7B: 32 layers, 32 KV heads of size 128; FP16; 4,096-token context.
    assert impl.kv_cache_bytes(32, 32, 128, 4096, 2) == 2 * 1024**3          # 2 GiB
    # Grouped-query attention with 8 KV heads cuts it by 4x.
    assert impl.kv_cache_bytes(32, 8, 128, 4096, 2) == 512 * 1024**2
    assert impl.kv_cache_bytes(32, 32, 128, 4096, 2, batch=8) == 16 * 1024**3
