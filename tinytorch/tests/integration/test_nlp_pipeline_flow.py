"""
NLP Pipeline Flow Integration Tests
====================================

Tests that the NLP pipeline works end-to-end:
1. Tokenization produces valid token IDs
2. Embeddings convert tokens to vectors
3. Attention mechanisms process sequences
4. Transformers combine everything correctly
5. Gradients flow back through the entire pipeline

These tests catch issues at module boundaries in the NLP stack.

Modules tested: 10-13 (Tokenization → Embeddings → Attention → Transformers)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd

# Enable autograd
enable_autograd()


class TestEmbeddingGradientFlow:
    """
    Critical Test: Verify gradients flow through embeddings.

    Common bugs caught:
    - Embedding lookup not differentiable
    - Wrong gradient accumulation for repeated tokens
    - Shape mismatches between embedding and attention
    """

    @pytest.mark.skip(reason="Embedding gradient flow requires advanced autograd integration")
    def test_embedding_receives_gradients(self):
        """Embedding weights must receive gradients during training"""
        try:
            from tinytorch.core.embeddings import Embedding
        except ImportError:
            pytest.skip("Embedding module not yet implemented")

        vocab_size = 100
        embed_dim = 32
        embedding = Embedding(vocab_size, embed_dim)

        # Token IDs (as Tensor)
        token_ids = Tensor(np.array([1, 5, 3, 7, 2]))

        # Forward pass
        embedded = embedding.forward(token_ids)

        # Simple loss: sum of embeddings
        loss = Tensor(np.array([[embedded.data.sum()]]), requires_grad=True)
        loss.backward()

        # Embedding weights should have gradients
        assert embedding.weight.grad is not None, (
            "Embedding weights did not receive gradients!"
        )

        # Only used token embeddings should have non-zero gradients
        for token_id in token_ids:
            grad_row = embedding.weight.grad[token_id]
            assert np.any(grad_row != 0), (
                f"Token {token_id} embedding has zero gradient!"
            )

    @pytest.mark.skip(reason="Embedding gradient flow requires advanced autograd integration")
    def test_repeated_tokens_accumulate_gradients(self):
        """Same token appearing twice should have accumulated gradient"""
        try:
            from tinytorch.core.embeddings import Embedding
        except ImportError:
            pytest.skip("Embedding module not yet implemented")

        vocab_size = 10
        embed_dim = 4
        embedding = Embedding(vocab_size, embed_dim)

        # Token 5 appears twice (as Tensor)
        token_ids = Tensor(np.array([5, 2, 5, 3]))

        embedded = embedding.forward(token_ids)

        # Loss that weights all positions equally
        loss = Tensor(np.array([[embedded.data.sum()]]), requires_grad=True)
        loss.backward()

        # Token 5 should have ~2x the gradient of token 2 or 3
        grad_5 = np.linalg.norm(embedding.weight.grad[5])
        grad_2 = np.linalg.norm(embedding.weight.grad[2])

        # Allow some tolerance
        assert grad_5 > grad_2 * 1.5, (
            f"Repeated token gradient not accumulated!\n"
            f"  Token 5 (appears 2x) grad: {grad_5}\n"
            f"  Token 2 (appears 1x) grad: {grad_2}\n"
            f"  Expected ratio ~2, got {grad_5/grad_2:.2f}"
        )


class TestAttentionGradientFlow:
    """
    Critical Test: Verify gradients flow through attention mechanism.

    Common bugs caught:
    - Softmax gradient issues
    - Attention weights not differentiable
    - Query/Key/Value projection gradients
    """

    def test_attention_all_projections_receive_gradients(self):
        """Q, K, V projections must all receive gradients"""
        try:
            from tinytorch.core.attention import MultiHeadAttention
        except ImportError:
            pytest.skip("Attention module not yet implemented")

        embed_dim = 32
        num_heads = 4
        seq_len = 8
        batch_size = 2

        attention = MultiHeadAttention(embed_dim, num_heads)

        # Random input sequence
        x = Tensor(
            np.random.randn(batch_size, seq_len, embed_dim),
            requires_grad=True
        )

        # Forward pass (self-attention - single input for Q, K, V)
        output = attention.forward(x)

        # Simple loss - use tensor operation to maintain computation graph
        loss = output.sum()
        loss.backward()

        # All projection matrices should have gradients
        projections = ['W_q', 'W_k', 'W_v', 'W_o']
        for proj_name in projections:
            if hasattr(attention, proj_name):
                proj = getattr(attention, proj_name)
                if hasattr(proj, 'weight'):
                    assert proj.weight.grad is not None, (
                        f"{proj_name} did not receive gradients!"
                    )

    def test_attention_input_receives_gradients(self):
        """Input to attention must receive gradients for residual connections"""
        try:
            from tinytorch.core.attention import MultiHeadAttention
        except ImportError:
            pytest.skip("Attention module not yet implemented")

        embed_dim = 16
        num_heads = 2

        attention = MultiHeadAttention(embed_dim, num_heads)

        x = Tensor(
            np.random.randn(1, 4, embed_dim),
            requires_grad=True
        )

        output = attention.forward(x)
        # Use tensor operation to maintain computation graph
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, (
            "Input to attention did not receive gradients!\n"
            "This breaks residual connections in Transformers."
        )

        assert x.grad.shape == x.shape, (
            f"Input gradient shape mismatch: {x.grad.shape} vs {x.shape}"
        )


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
        try:
            from tinytorch.core.transformers import TransformerBlock
        except ImportError:
            pytest.skip("Transformer module not yet implemented")

        embed_dim = 32
        num_heads = 4
        ff_dim = 64

        block = TransformerBlock(embed_dim, num_heads, ff_dim)

        x = Tensor(
            np.random.randn(1, 8, embed_dim),
            requires_grad=True
        )

        output = block.forward(x)
        loss = Tensor(np.array([[output.data.sum()]]), requires_grad=True)
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
        try:
            from tinytorch.core.transformers import TransformerBlock
        except ImportError:
            pytest.skip("Transformer module not yet implemented")

        embed_dim = 32
        num_heads = 4
        ff_dim = 64
        num_layers = 4

        blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]

        x = Tensor(
            np.random.randn(1, 8, embed_dim),
            requires_grad=True
        )

        # Forward through all blocks
        h = x
        for block in blocks:
            h = block.forward(h)

        loss = Tensor(np.array([[h.data.sum()]]), requires_grad=True)
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


class TestNLPPipelineEndToEnd:
    """
    Integration Test: Full NLP pipeline from tokens to loss.

    This tests the complete flow:
    tokens → embedding → attention → linear → loss
    """

    @pytest.mark.skip(reason="NLP pipeline gradient flow requires advanced autograd integration")
    def test_complete_nlp_forward_backward(self):
        """Complete NLP pipeline must work end-to-end"""
        try:
            from tinytorch.core.embeddings import Embedding
            from tinytorch.core.attention import MultiHeadAttention
            from tinytorch.core.layers import Linear
            from tinytorch.core.losses import CrossEntropyLoss
        except ImportError:
            pytest.skip("NLP modules not yet implemented")

        vocab_size = 100
        embed_dim = 32
        num_heads = 4
        num_classes = 10
        seq_len = 8

        # Build pipeline
        embedding = Embedding(vocab_size, embed_dim)
        attention = MultiHeadAttention(embed_dim, num_heads)
        classifier = Linear(embed_dim, num_classes)
        loss_fn = CrossEntropyLoss()

        # Input: token IDs (as Tensor)
        token_ids = Tensor(np.random.randint(0, vocab_size, seq_len))
        target = Tensor(np.array([[3]]))  # Class 3

        # Forward pass
        embedded = embedding.forward(token_ids)  # [seq_len, embed_dim]
        # Reshape for attention: add batch dimension
        embedded_batched = Tensor(embedded.data.reshape(1, seq_len, embed_dim), requires_grad=True)
        attended = attention.forward(embedded_batched)  # [1, seq_len, embed_dim]

        # Mean pooling over sequence
        pooled = Tensor(attended.data.mean(axis=0, keepdims=True), requires_grad=True)

        logits = classifier.forward(pooled)  # [1, num_classes]
        loss = loss_fn.forward(logits, target)

        # Backward pass
        loss.backward()

        # Verify gradients flowed to embedding
        assert embedding.weight.grad is not None, (
            "Gradients did not flow back to embeddings!"
        )

        # Verify classifier received gradients
        assert classifier.weight.grad is not None, (
            "Classifier did not receive gradients!"
        )


# Quick smoke tests for CI
@pytest.mark.quick
class TestQuickNLPSmoke:
    """Fast tests for CI"""

    def test_embedding_forward_works(self):
        """Embedding forward should not crash"""
        try:
            from tinytorch.core.embeddings import Embedding
            from tinytorch.core.tensor import Tensor
        except ImportError:
            pytest.skip("Embedding module not yet implemented")

        embedding = Embedding(100, 32)
        indices = Tensor(np.array([1, 2, 3]))
        result = embedding.forward(indices)
        assert result.shape[0] == 3
        assert result.shape[1] == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
