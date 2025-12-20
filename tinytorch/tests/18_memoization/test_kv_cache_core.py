"""
Module 18: KV Cache (Memoization) Core Tests
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
        cache.advance()  # Must advance after update to make values retrievable

        # Retrieve
        cached_K, cached_V = cache.get(layer_idx)

        assert cached_K is not None, "Cache didn't store K"
        assert cached_V is not None, "Cache didn't store V"
        assert cached_K.shape == K.shape, f"K shape mismatch: {cached_K.shape} vs {K.shape}"
        assert np.allclose(cached_K.data, K.data), "Retrieved K doesn't match stored"
        assert np.allclose(cached_V.data, V.data), "Retrieved V doesn't match stored"


class TestKVCacheAdvanced:
    """Advanced KV cache tests for multiple tokens and layers."""

    def test_kv_cache_multiple_tokens(self):
        """
        WHAT: Verify cache can accumulate multiple tokens.

        WHY: During generation, we add one token at a time. The cache must
        correctly accumulate all previous K,V pairs.
        """
        try:
            from tinytorch.perf.memoization import KVCache
        except ImportError:
            pytest.skip("KVCache not yet exported")

        cache = KVCache(batch_size=1, max_seq_len=10, num_layers=1, num_heads=2, head_dim=8)

        # Add 3 tokens
        for token_idx in range(3):
            K = Tensor(np.full((1, 2, 1, 8), token_idx, dtype=np.float32))
            V = Tensor(np.full((1, 2, 1, 8), token_idx + 10, dtype=np.float32))
            cache.update(layer_idx=0, key=K, value=V)
            cache.advance()

        # Retrieve should give all 3 tokens
        cached_K, cached_V = cache.get(layer_idx=0)

        assert cached_K.shape == (1, 2, 3, 8), f"Expected (1,2,3,8), got {cached_K.shape}"
        assert cached_V.shape == (1, 2, 3, 8), f"Expected (1,2,3,8), got {cached_V.shape}"

        # Verify values are in order
        assert cached_K.data[0, 0, 0, 0] == 0, "First token K wrong"
        assert cached_K.data[0, 0, 1, 0] == 1, "Second token K wrong"
        assert cached_K.data[0, 0, 2, 0] == 2, "Third token K wrong"

    def test_kv_cache_multiple_layers(self):
        """
        WHAT: Verify cache works correctly across multiple transformer layers.

        WHY: Real transformers have multiple layers, each with its own K,V cache.
        """
        try:
            from tinytorch.perf.memoization import KVCache
        except ImportError:
            pytest.skip("KVCache not yet exported")

        num_layers = 4
        cache = KVCache(batch_size=1, max_seq_len=10, num_layers=num_layers, num_heads=2, head_dim=8)

        # Update each layer with different values
        for layer_idx in range(num_layers):
            K = Tensor(np.full((1, 2, 1, 8), layer_idx * 10, dtype=np.float32))
            V = Tensor(np.full((1, 2, 1, 8), layer_idx * 10 + 1, dtype=np.float32))
            cache.update(layer_idx, K, V)

        cache.advance()

        # Verify each layer has correct values
        for layer_idx in range(num_layers):
            cached_K, cached_V = cache.get(layer_idx)
            expected_k_val = layer_idx * 10
            expected_v_val = layer_idx * 10 + 1

            assert cached_K.data[0, 0, 0, 0] == expected_k_val, (
                f"Layer {layer_idx} K wrong: expected {expected_k_val}, got {cached_K.data[0,0,0,0]}"
            )
            assert cached_V.data[0, 0, 0, 0] == expected_v_val, (
                f"Layer {layer_idx} V wrong: expected {expected_v_val}, got {cached_V.data[0,0,0,0]}"
            )

    def test_kv_cache_seq_pos_tracking(self):
        """
        WHAT: Verify seq_pos counter tracks correctly.

        WHY: seq_pos determines where in the cache to write next and how
        much valid data to return. Incorrect tracking breaks generation.
        """
        try:
            from tinytorch.perf.memoization import KVCache
        except ImportError:
            pytest.skip("KVCache not yet exported")

        cache = KVCache(batch_size=1, max_seq_len=100, num_layers=1, num_heads=2, head_dim=8)

        # Initially at 0
        assert cache.seq_pos == 0, "Initial seq_pos should be 0"

        # Add tokens and check seq_pos
        for expected_pos in range(1, 6):
            K = Tensor(np.zeros((1, 2, 1, 8)))
            V = Tensor(np.zeros((1, 2, 1, 8)))
            cache.update(0, K, V)
            cache.advance()
            assert cache.seq_pos == expected_pos, (
                f"seq_pos should be {expected_pos}, got {cache.seq_pos}"
            )

    def test_kv_cache_raises_on_invalid_layer(self):
        """
        WHAT: Verify cache raises error for invalid layer index.

        WHY: Trying to access a non-existent layer is a programming error
        that should be caught early.
        """
        try:
            from tinytorch.perf.memoization import KVCache
        except ImportError:
            pytest.skip("KVCache not yet exported")

        cache = KVCache(batch_size=1, max_seq_len=10, num_layers=2, num_heads=2, head_dim=8)

        K = Tensor(np.zeros((1, 2, 1, 8)))
        V = Tensor(np.zeros((1, 2, 1, 8)))

        # Valid layers are 0 and 1
        with pytest.raises(ValueError):
            cache.update(layer_idx=5, key=K, value=V)  # Invalid layer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
