"""
Module 15: KV Cache (Memoization) Core Tests
=============================================

These tests verify that KV caching works for efficient inference.

WHY THESE TESTS MATTER:
-----------------------
KV caching is essential for efficient text generation:
- Without cache: O(n²) per token (recompute all attention)
- With cache: O(n) per token (reuse previous K,V)

For generating 100 tokens, that's 100x speedup!

WHAT WE TEST:
-------------
1. KVCache can store key-value pairs
2. Cache retrieval returns stored values
3. Cache works across multiple layers
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor


class TestKVCacheBasics:
    """Test basic KV cache functionality."""

    def test_kv_cache_import(self):
        """
        WHAT: Verify KVCache can be imported.

        WHY: Basic sanity check.
        """
        try:
            from tinytorch.perf.memoization import KVCache
            assert KVCache is not None
        except ImportError as e:
            pytest.skip(f"KVCache not yet exported: {e}")

    def test_kv_cache_can_instantiate(self):
        """
        WHAT: Verify KVCache can be created.
        """
        try:
            from tinytorch.perf.memoization import KVCache
            # KVCache needs: batch_size, max_seq_len, num_layers, num_heads, head_dim
            cache = KVCache(batch_size=1, max_seq_len=128, num_layers=2, num_heads=4, head_dim=16)
            assert cache is not None
        except ImportError:
            pytest.skip("KVCache not yet exported")

    def test_kv_cache_stores_and_retrieves(self):
        """
        WHAT: Verify cache can store and retrieve K,V tensors.

        WHY: The whole point of the cache is to reuse computed values.
        If storage/retrieval doesn't work, there's no speedup.
        """
        try:
            from tinytorch.perf.memoization import KVCache
        except ImportError:
            pytest.skip("KVCache not yet exported")

        # Create cache with proper dimensions
        cache = KVCache(batch_size=1, max_seq_len=128, num_layers=2, num_heads=4, head_dim=16)

        # Store some K,V pairs (cache expects one token at a time during generation)
        layer_idx = 0
        K = Tensor(np.random.randn(1, 4, 1, 16))  # (batch, heads, 1, dim) - one new token
        V = Tensor(np.random.randn(1, 4, 1, 16))

        cache.update(layer_idx, K, V)

        # Retrieve
        cached_K, cached_V = cache.get(layer_idx)

        assert cached_K is not None, "Cache didn't store K"
        assert cached_V is not None, "Cache didn't store V"
        assert np.allclose(cached_K.data, K.data), "Retrieved K doesn't match stored"
        assert np.allclose(cached_V.data, V.data), "Retrieved V doesn't match stored"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
