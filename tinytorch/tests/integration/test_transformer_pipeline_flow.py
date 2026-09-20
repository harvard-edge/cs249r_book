"""Module 13 integration: gradients through transformer blocks and stacks."""
import numpy as np
from tinytorch.core.tensor import Tensor
import tinytorch.core.autograd
from tinytorch.core.transformers import TransformerBlock

rng = np.random.default_rng(7)


class TestTransformerGradientFlow:
    """
    Critical Test: Verify gradients flow through complete Transformer.

    Common bugs caught:
    - Residual connection gradients
    - Layer norm gradient issues
    - Deep network vanishing gradients
    """

    def test_transformer_block_gradient_flow(self):
        """Gradients must flow through a complete transformer block"""
        embed_dim = 32
        num_heads = 4
        ff_dim = 64

        block = TransformerBlock(embed_dim, num_heads, ff_dim=ff_dim)

        x = Tensor(
            rng.standard_normal((1, 8, embed_dim)),
            requires_grad=True
        )

        output = block.forward(x)
        # Use Tensor operation to preserve computation graph
        loss = output.sum()
        loss.backward()

        # Input must receive gradients (for stacking blocks)
        assert x.grad is not None, (
            "Transformer block input did not receive gradients!"
        )

        # Gradient should not be too small (vanishing)
        grad_norm = np.linalg.norm(x.grad)
        assert grad_norm > 1e-6, (
            f"Vanishing gradients in transformer block: {grad_norm}"
        )

    def test_stacked_transformer_blocks(self):
        """Gradients must flow through multiple stacked blocks"""
        embed_dim = 32
        num_heads = 4
        ff_dim = 64
        num_layers = 4

        blocks = [TransformerBlock(embed_dim, num_heads, ff_dim=ff_dim) for _ in range(num_layers)]

        x = Tensor(
            rng.standard_normal((1, 8, embed_dim)),
            requires_grad=True
        )

        # Forward through all blocks
        h = x
        for block in blocks:
            h = block.forward(h)

        # Use Tensor operation to preserve computation graph
        loss = h.sum()
        loss.backward()

        # Input must receive gradients through all layers
        assert x.grad is not None, (
            f"Gradients did not flow through {num_layers} transformer blocks!"
        )

        # Check gradient magnitude is reasonable
        grad_norm = np.linalg.norm(x.grad)
        assert grad_norm > 1e-8, (
            f"Severe vanishing gradients through {num_layers} blocks: {grad_norm}"
        )
