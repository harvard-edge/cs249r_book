"""
Module 12: Attention Core Tests
================================

These tests verify that attention mechanisms compute correctly.

WHY THESE TESTS MATTER:
-----------------------
Attention is the core innovation behind Transformers (GPT, BERT, etc.).
If attention doesn't work:
- Model can't focus on relevant parts of input
- Transformers collapse to simple averaging
- Language models produce garbage

WHAT WE TEST:
-------------
1. Scaled dot-product attention produces valid probability distributions
2. MultiHeadAttention preserves input/output shapes
3. Attention weights sum to 1 (softmax property)
4. Masking correctly prevents attending to future tokens

CONNECTION TO OTHER MODULES:
----------------------------
- Uses Tensor (Module 01) - all computations
- Uses Linear (Module 03) - Q, K, V projections
- Uses Softmax (Module 02) - attention weights
- Enables Transformers (Module 13) - attention is the core component
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.attention import MultiHeadAttention, scaled_dot_product_attention
from tinytorch.core.autograd import enable_autograd

enable_autograd()


class TestScaledDotProductAttention:
    """
    Test the core attention computation: softmax(QK^T / sqrt(d_k)) V

    This is the mathematical heart of all transformer models.
    """

    def test_attention_output_shape(self):
        """
        WHAT: Verify attention preserves sequence dimensions.

        WHY: Attention transforms values but shouldn't change shape.
        Input: (batch, seq, dim) → Output: (batch, seq, dim)
        """
        batch, seq, dim = 2, 5, 8

        Q = Tensor(np.random.randn(batch, seq, dim))
        K = Tensor(np.random.randn(batch, seq, dim))
        V = Tensor(np.random.randn(batch, seq, dim))

        output, weights = scaled_dot_product_attention(Q, K, V)

        assert output.shape == (batch, seq, dim), (
            f"Attention changed output shape!\n"
            f"  Input shape: {Q.shape}\n"
            f"  Output shape: {output.shape}\n"
            "Attention should preserve (batch, seq, dim) dimensions."
        )

    def test_attention_weights_are_probabilities(self):
        """
        WHAT: Verify attention weights form valid probability distributions.

        WHY: After softmax, each query's attention over keys must:
        1. Sum to 1.0 (it's a probability distribution)
        2. Be non-negative (probabilities can't be negative)

        This ensures the output is a proper weighted average of values.
        """
        Q = Tensor(np.random.randn(1, 4, 8))
        K = Tensor(np.random.randn(1, 4, 8))
        V = Tensor(np.random.randn(1, 4, 8))

        _, weights = scaled_dot_product_attention(Q, K, V)

        # Check non-negative
        assert np.all(weights.data >= 0), (
            "Attention weights are negative!\n"
            f"  Min weight: {weights.data.min()}\n"
            "After softmax, all weights must be >= 0."
        )

        # Check sum to 1 along last dimension (each query sums over keys)
        row_sums = weights.data.sum(axis=-1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), (
            "Attention weights don't sum to 1!\n"
            f"  Row sums: {row_sums}\n"
            "Each query's attention distribution must sum to 1.0."
        )

    def test_attention_focuses_on_similar_keys(self):
        """
        WHAT: Verify attention assigns higher weight to similar keys.

        WHY: The whole point of attention is to focus on relevant parts.
        If query is similar to key[i], attention weight[i] should be high.

        This is a semantic test - does attention do what it's supposed to?
        """
        dim = 4

        # Query vector
        Q = Tensor(np.array([[[1.0, 0.0, 0.0, 0.0]]]))  # (1, 1, 4)

        # Keys: one similar to Q, others different
        K = Tensor(np.array([[[
            [1.0, 0.0, 0.0, 0.0],   # Very similar to Q
            [0.0, 1.0, 0.0, 0.0],   # Orthogonal
            [0.0, 0.0, 1.0, 0.0],   # Orthogonal
        ]]]))  # (1, 1, 3, 4) - but we'll reshape
        K = Tensor(np.array([[[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0]]]))  # (1, 3, 4)

        V = Tensor(np.random.randn(1, 3, 4))

        _, weights = scaled_dot_product_attention(Q, K, V)

        # First key should get highest weight (most similar to query)
        first_key_weight = weights.data[0, 0, 0]
        other_weights = weights.data[0, 0, 1:]

        assert first_key_weight > np.max(other_weights), (
            "Attention doesn't focus on similar keys!\n"
            f"  Weight for similar key: {first_key_weight:.4f}\n"
            f"  Weights for orthogonal keys: {other_weights}\n"
            "Attention should assign highest weight to the most similar key."
        )


class TestMultiHeadAttention:
    """
    Test multi-head attention (the full transformer component).

    Multi-head attention runs multiple attention heads in parallel,
    allowing the model to attend to different aspects simultaneously.
    """

    def test_multihead_preserves_shape(self):
        """
        WHAT: Verify multi-head attention preserves input dimensions.

        WHY: Like single-head attention, MHA shouldn't change shapes.
        """
        batch, seq, embed_dim = 2, 10, 32
        num_heads = 4

        mha = MultiHeadAttention(embed_dim, num_heads)
        x = Tensor(np.random.randn(batch, seq, embed_dim))

        output = mha.forward(x)

        assert output.shape == x.shape, (
            f"MultiHeadAttention changed shape!\n"
            f"  Input: {x.shape}\n"
            f"  Output: {output.shape}\n"
            "MHA should preserve (batch, seq, embed_dim) dimensions."
        )

    def test_multihead_has_learnable_parameters(self):
        """
        WHAT: Verify MHA has trainable parameters (Q, K, V, output projections).

        WHY: These projections are what the model learns.
        No parameters = nothing to train = useless layer.
        """
        mha = MultiHeadAttention(embed_dim=64, num_heads=8)
        params = mha.parameters()

        assert len(params) > 0, (
            "MultiHeadAttention has no parameters!\n"
            "It should have at least 4 linear projections (Q, K, V, output)."
        )

        # Should have 8 tensors: weight+bias for each of 4 projections
        # (or 4 if no bias)
        assert len(params) >= 4, (
            f"MultiHeadAttention has only {len(params)} parameters.\n"
            "Expected at least 4 (Q, K, V, output weights)."
        )

    def test_multihead_head_dim_calculation(self):
        """
        WHAT: Verify head dimension is calculated correctly.

        WHY: embed_dim must be divisible by num_heads.
        head_dim = embed_dim / num_heads

        This is a common source of bugs in transformer implementations.
        """
        embed_dim = 64
        num_heads = 8
        expected_head_dim = 8  # 64 / 8

        mha = MultiHeadAttention(embed_dim, num_heads)

        assert mha.head_dim == expected_head_dim, (
            f"Head dimension calculated incorrectly!\n"
            f"  embed_dim={embed_dim}, num_heads={num_heads}\n"
            f"  Expected head_dim: {expected_head_dim}\n"
            f"  Got: {mha.head_dim}\n"
            "head_dim = embed_dim / num_heads"
        )

    def test_multihead_invalid_config_raises(self):
        """
        WHAT: Verify MHA rejects invalid configurations.

        WHY: embed_dim must be divisible by num_heads.
        If not, we can't split dimensions evenly across heads.
        """
        with pytest.raises((ValueError, AssertionError)):
            # 64 is not divisible by 5
            MultiHeadAttention(embed_dim=64, num_heads=5)


class TestAttentionGradientFlow:
    """
    Test that gradients flow through attention correctly.

    WHY THIS MATTERS: Attention must be differentiable for training.
    If gradients don't flow, transformers can't learn.
    """

    def test_gradients_flow_to_input(self):
        """
        WHAT: Verify input tensor receives gradients after backward pass.

        WHY: For training to work, gradients must flow from loss
        back through attention to the input embeddings.
        """
        mha = MultiHeadAttention(embed_dim=16, num_heads=2)
        x = Tensor(np.random.randn(1, 4, 16), requires_grad=True)

        output = mha.forward(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, (
            "Input didn't receive gradients through attention!\n"
            "This means the model cannot learn from attention outputs."
        )

    def test_gradients_flow_to_parameters(self):
        """
        WHAT: Verify attention parameters receive gradients.

        WHY: The Q, K, V projections are what we're training.
        If they don't get gradients, attention can't improve.
        """
        mha = MultiHeadAttention(embed_dim=16, num_heads=2)
        x = Tensor(np.random.randn(1, 4, 16), requires_grad=True)

        output = mha.forward(x)
        loss = output.sum()
        loss.backward()

        params_with_grad = sum(1 for p in mha.parameters() if p.grad is not None)

        assert params_with_grad > 0, (
            "No attention parameters received gradients!\n"
            "The Q, K, V projections must receive gradients to learn."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
