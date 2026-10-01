"""
Module 12: Attention Values, Checked Against softmax(QK^T / sqrt(d_k)) V
=========================================================================

WHY THESE TESTS MATTER:
-----------------------
Shape checks and "weights sum to 1" both pass if the 1/sqrt(d_k) factor is
dropped, or if multi-head attention scales by the model width instead of the
head width. Those bugs sharpen the softmax and only surface later as slower
or unstable training. These tests compare the actual numbers with the
formula computed independently in float64.
"""

import numpy as np
import pytest

from tinytorch.core.tensor import Tensor
from tinytorch.core.attention import scaled_dot_product_attention, MultiHeadAttention

rng = np.random.default_rng(12)


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _reference_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -np.inf, scores)
    weights = _softmax(scores)
    return weights @ V, weights


class TestScaledDotProductAttentionValues:

    def test_matches_formula(self):
        Q = rng.standard_normal((2, 5, 16))
        K = rng.standard_normal((2, 5, 16))
        V = rng.standard_normal((2, 5, 16))
        out, weights = scaled_dot_product_attention(Tensor(Q), Tensor(K), Tensor(V))
        ref_out, ref_w = _reference_attention(Q, K, V)
        np.testing.assert_allclose(weights.data, ref_w, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(out.data, ref_out, rtol=1e-4, atol=1e-5)

    def test_scale_is_one_over_sqrt_dk(self):
        """Two keys whose raw scores differ by exactly d_k: the weight ratio is exp(sqrt(d_k))."""
        d_k = 16
        Q = np.zeros((1, 1, d_k)); Q[0, 0, :] = 1.0
        K = np.zeros((1, 2, d_k)); K[0, 0, :] = 1.0          # raw score d_k, second key scores 0
        V = np.eye(2, d_k)[None]
        _, weights = scaled_dot_product_attention(Tensor(Q), Tensor(K), Tensor(V))
        w = weights.data[0, 0]
        # scaled scores are sqrt(d_k) = 4 and 0, so w0 / w1 = e^4
        np.testing.assert_allclose(w[0] / w[1], np.exp(np.sqrt(d_k)), rtol=1e-4)

    def test_causal_mask_values(self):
        Q = rng.standard_normal((1, 4, 8))
        K = rng.standard_normal((1, 4, 8))
        V = rng.standard_normal((1, 4, 8))
        mask = np.tril(np.ones((1, 4, 4)))
        out, weights = scaled_dot_product_attention(Tensor(Q), Tensor(K), Tensor(V), mask=Tensor(mask))
        ref_out, ref_w = _reference_attention(Q, K, V, mask)
        np.testing.assert_allclose(weights.data, ref_w, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(out.data, ref_out, rtol=1e-4, atol=1e-5)


class TestMultiHeadAttentionValues:

    def test_matches_per_head_formula(self):
        """Each head scales by 1/sqrt(head_dim), not 1/sqrt(embed_dim)."""
        batch, seq, embed_dim, num_heads = 2, 4, 8, 2
        head_dim = embed_dim // num_heads
        mha = MultiHeadAttention(embed_dim, num_heads)
        x = rng.standard_normal((batch, seq, embed_dim))
        out = mha(Tensor(x))

        def proj(layer, h):
            return h @ layer.weight.data.astype(np.float64) + layer.bias.data.astype(np.float64)

        def split(h):
            return h.reshape(batch, seq, num_heads, head_dim).transpose(0, 2, 1, 3)

        Q = split(proj(mha.q_proj, x))
        K = split(proj(mha.k_proj, x))
        V = split(proj(mha.v_proj, x))
        attended, _ = _reference_attention(Q, K, V)
        merged = attended.transpose(0, 2, 1, 3).reshape(batch, seq, embed_dim)
        expected = proj(mha.out_proj, merged)
        np.testing.assert_allclose(out.data, expected, rtol=1e-4, atol=1e-5)
