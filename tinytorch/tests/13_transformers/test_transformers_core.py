"""
Module 13: Transformers - Core Functionality Tests
===================================================

WHY TRANSFORMERS MATTER:
-----------------------
Transformers power modern AI:
- GPT, ChatGPT, Claude (language)
- BERT (understanding)
- Vision Transformers (images)
- Whisper (speech)

WHAT STUDENTS LEARN:
-------------------
1. Self-attention: every token attends to every other token
2. Multi-head: parallel attention for different relationships
3. Feed-forward: process each position independently
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.autograd import enable_autograd
from tinytorch.core.tensor import Tensor
from tinytorch.core.transformers import TransformerBlock


class TestTransformerBlock:
    """Test Transformer block functionality."""

    def test_transformer_block_shape(self):
        """
        WHAT: Verify TransformerBlock preserves shape.

        WHY: Transformers stack many blocks.
        Each must output same shape as input for stacking.

        STUDENT LEARNING: Transformer blocks are residual:
        output = x + attention(norm(x))
        output = output + ffn(norm(output))
        """
        block = TransformerBlock(embed_dim=256, num_heads=8)

        # Sequence of embeddings
        x = Tensor(np.random.randn(2, 20, 256))  # (batch, seq, embed)

        output = block(x)

        assert output.shape == x.shape, (
            f"TransformerBlock should preserve shape.\n"
            f"  Input: {x.shape}\n"
            f"  Output: {output.shape}"
        )

    def test_transformer_stack(self):
        """
        WHAT: Verify multiple transformer blocks can be stacked.

        WHY: GPT has 12-96 blocks. They must chain correctly.

        STUDENT LEARNING: Deeper = more complex patterns learned.
        But also harder to train (vanishing gradients).
        """
        # Stack of 4 blocks
        blocks = [TransformerBlock(embed_dim=128, num_heads=4) for _ in range(4)]

        x = Tensor(np.random.randn(2, 10, 128))

        for block in blocks:
            x = block(x)

        assert x.shape == (2, 10, 128), (
            "Shape should be preserved through all blocks"
        )


class TestTransformerGradients:
    """Test gradient flow through transformers."""

    def test_transformer_gradients(self):
        """
        WHAT: Verify gradients flow through TransformerBlock.

        WHY: Transformers are deep - gradients must flow through
        all attention and FFN layers for training.

        STUDENT LEARNING: Residual connections help gradients flow:
        output = x + f(x)
        d_output/d_x = 1 + df/dx (always ≥ 1!)
        """
        enable_autograd()

        block = TransformerBlock(embed_dim=64, num_heads=4)
        x = Tensor(np.random.randn(1, 5, 64), requires_grad=True)

        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, (
            "Input should receive gradients through Transformer.\n"
            "Check attention and FFN gradient implementations."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
