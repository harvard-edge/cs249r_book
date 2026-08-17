"""
Boundary-value tests for attention with a single-token sequence.

seq_len=1 is a real, common case: the first token of autoregressive
generation. The causal mask degenerates to a 1x1 all-ones matrix (the
one token can only attend to itself), an edge case no existing test
exercised.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.attention import MultiHeadAttention, scaled_dot_product_attention
from tinytorch.core.autograd import enable_autograd

enable_autograd()


class TestAttentionSingleToken:
    def test_scaled_dot_product_attention_seq_len_one(self):
        Q = Tensor(np.random.randn(1, 1, 4).astype(np.float32), requires_grad=True)
        K = Tensor(np.random.randn(1, 1, 4).astype(np.float32), requires_grad=True)
        V = Tensor(np.random.randn(1, 1, 4).astype(np.float32), requires_grad=True)

        output, weights = scaled_dot_product_attention(Q, K, V)

        assert output.shape == (1, 1, 4)
        assert weights.shape == (1, 1, 1)
        # A single token attending only to itself must get weight 1.0.
        np.testing.assert_allclose(weights.data, np.ones((1, 1, 1)), atol=1e-5)
        np.testing.assert_allclose(output.data, V.data, atol=1e-5)

    def test_multihead_attention_seq_len_one_with_causal_mask(self):
        mha = MultiHeadAttention(embed_dim=8, num_heads=2)
        x = Tensor(np.random.randn(1, 1, 8).astype(np.float32), requires_grad=True)
        mask = Tensor(np.tril(np.ones((1, 1, 1))))

        out = mha(x, mask)

        assert out.shape == (1, 1, 8)
        assert np.all(np.isfinite(out.data))

        out.sum().backward()
        assert np.all(np.isfinite(x.grad))

    def test_multihead_attention_seq_len_one_batch_greater_than_one(self):
        """Multiple independent single-token sequences in the same batch
        (e.g. the first generation step for several prompts at once)."""
        mha = MultiHeadAttention(embed_dim=8, num_heads=2)
        x = Tensor(np.random.randn(4, 1, 8).astype(np.float32), requires_grad=True)
        mask = Tensor(np.tril(np.ones((1, 1, 1))))

        out = mha(x, mask)

        assert out.shape == (4, 1, 8)
        assert np.all(np.isfinite(out.data))
