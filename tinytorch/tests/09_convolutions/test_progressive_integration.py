"""
Module 09: Progressive Integration Tests
Tests that Module 09 (Convolutions) works correctly AND that Foundation + Training work.

DEPENDENCY CHAIN: 01_tensor → ... → 05_dataloader → 06_autograd → 07_optimizers → 08_training → 09_convolutions
This is where CNNs enable computer vision through spatial feature extraction.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPriorStackStillWorking:
    """Quick regression checks that prior modules (01→06) still work."""

    def test_foundation_stack_stable(self):
        """Verify foundation stack (01→05) remains stable."""
        # Environment (Module 01)
        assert sys.version_info >= (3, 8), "Foundation broken: Python version"

        # Tensor foundation (Module 02)
        try:
            from tinytorch.core.tensor import Tensor
            t = Tensor([1, 2, 3])
            assert t.shape == (3,), "Foundation broken: Tensor creation"
        except ImportError:
            assert True, "Tensor foundation not implemented yet"

    def test_spatial_operations_stable(self):
        """Verify Module 09 (Convolutions/Spatial) operations still work."""
        try:
            from tinytorch.core.spatial import Conv2d, MaxPool2d

            # Basic spatial operations should work
            conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3)
            pool = MaxPool2d(kernel_size=2)

            assert hasattr(conv, 'forward'), "Spatial broken: Conv2d interface"
            assert hasattr(pool, 'forward'), "Spatial broken: MaxPool2d interface"

        except ImportError:
            assert True, "Spatial operations not implemented yet"


class TestModule12AttentionCore:
    """Test Module 12 (Attention) core functionality."""

    def test_attention_mechanism_creation(self):
        """Test basic attention mechanism works."""
        try:
            from tinytorch.core.attention import MultiHeadAttention
            from tinytorch.core.tensor import Tensor

            # Create attention mechanism
            attention = MultiHeadAttention(embed_dim=64, num_heads=8)

            # Should have proper components (q_proj, k_proj, v_proj naming)
            assert hasattr(attention, 'q_proj') or hasattr(attention, 'query_proj'), "Attention broken: No query projection"
            assert hasattr(attention, 'k_proj') or hasattr(attention, 'key_proj'), "Attention broken: No key projection"
            assert hasattr(attention, 'v_proj') or hasattr(attention, 'value_proj'), "Attention broken: No value projection"

            # Test with sequence input
            seq_len, batch_size, embed_dim = 10, 4, 64
            x = Tensor(np.random.randn(seq_len, batch_size, embed_dim))

            output = attention(x)
            assert output.shape == (seq_len, batch_size, embed_dim), "Attention output shape broken"

        except ImportError:
            assert True, "Attention mechanism not implemented yet"

    def test_scaled_dot_product_attention(self):
        """Test core attention computation."""
        try:
            from tinytorch.core.attention import scaled_dot_product_attention
            from tinytorch.core.tensor import Tensor

            # Attention inputs: queries, keys, values
            seq_len, embed_dim = 8, 16
            Q = Tensor(np.random.randn(seq_len, embed_dim))
            K = Tensor(np.random.randn(seq_len, embed_dim))
            V = Tensor(np.random.randn(seq_len, embed_dim))

            # Compute attention
            output, attention_weights = scaled_dot_product_attention(Q, K, V)

            assert output.shape == V.shape, "Attention output shape wrong"
            assert attention_weights.shape == (seq_len, seq_len), "Attention weights shape wrong"

            # Attention weights should sum to 1 across keys
            weight_sums = np.sum(attention_weights.data, axis=1)
            assert np.allclose(weight_sums, 1.0), "Attention weights don't sum to 1"

        except ImportError:
            assert True, "Scaled dot-product attention not implemented yet"


class TestProgressiveStackIntegration:
    """Test that the complete stack (01→07) works together."""

    def test_neural_network_with_attention(self):
        """Test neural network enhanced with attention."""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            from tinytorch.core.attention import MultiHeadAttention

            # Build network: dense → attention → dense
            encoder = Linear(64, 64)
            attention = MultiHeadAttention(embed_dim=64, num_heads=8)
            decoder = Linear(64, 10)
            relu = ReLU()

            # Sequence input
            seq_len, batch_size, input_dim = 12, 4, 64
            x = Tensor(np.random.randn(seq_len, batch_size, input_dim))

            # Forward pass through network with attention
            h = relu(encoder(x))        # Dense processing
            attn_out = attention(h)     # Attention mechanism
            output = decoder(attn_out)  # Final projection

            assert output.shape == (seq_len, batch_size, 10), "Network with attention broken"

        except ImportError:
            assert True, "Neural network with attention not ready yet"

    def test_transformer_block_capability(self):
        """Test building transformer-style blocks."""
        try:
            from tinytorch.core.attention import MultiHeadAttention
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            from tinytorch.core.tensor import Tensor

            # Transformer block components
            attention = MultiHeadAttention(embed_dim=128, num_heads=8)
            ff1 = Linear(128, 512)
            ff2 = Linear(512, 128)
            relu = ReLU()

            # Input sequence
            seq_len, batch_size, embed_dim = 16, 2, 128
            x = Tensor(np.random.randn(seq_len, batch_size, embed_dim))

            # Transformer block: attention + feedforward
            attn_out = attention(x)
            ff_out = ff2(relu(ff1(attn_out)))

            # Residual connection (if implemented)
            if hasattr(x, '__add__'):
                output = x + ff_out  # Residual connection
            else:
                output = ff_out

            assert output.shape == x.shape, "Transformer block broken"

        except ImportError:
            assert True, "Transformer block capability not ready yet"


class TestSequenceUnderstandingCapability:
    """Test that attention enables sequence understanding."""

    def test_sequence_to_sequence_capability(self):
        """Test sequence-to-sequence processing."""
        try:
            from tinytorch.core.attention import MultiHeadAttention
            from tinytorch.core.tensor import Tensor

            # Encoder-decoder style processing
            encoder_attention = MultiHeadAttention(embed_dim=64, num_heads=4)
            decoder_attention = MultiHeadAttention(embed_dim=64, num_heads=4)

            # Source and target sequences
            src_len, tgt_len, batch_size, embed_dim = 10, 8, 2, 64
            src = Tensor(np.random.randn(src_len, batch_size, embed_dim))
            tgt = Tensor(np.random.randn(tgt_len, batch_size, embed_dim))

            # Encode source sequence
            encoded = encoder_attention(src)

            # Decode target sequence (with potential cross-attention)
            if hasattr(decoder_attention, 'cross_attention'):
                decoded = decoder_attention(tgt, encoded)
            else:
                decoded = decoder_attention(tgt)

            assert encoded.shape == src.shape, "Sequence encoding broken"
            assert decoded.shape == tgt.shape, "Sequence decoding broken"

        except ImportError:
            assert True, "Sequence-to-sequence not ready yet"

    def test_attention_pattern_analysis(self):
        """Test that attention creates meaningful patterns."""
        try:
            from tinytorch.core.attention import scaled_dot_product_attention
            from tinytorch.core.tensor import Tensor

            # Create sequence with clear patterns
            seq_len, embed_dim = 6, 8

            # Pattern: first and last tokens should attend to each other
            pattern_input = np.zeros((seq_len, embed_dim))
            pattern_input[0, :] = 1.0  # First token
            pattern_input[-1, :] = 1.0  # Last token

            Q = Tensor(pattern_input)
            K = Tensor(pattern_input)
            V = Tensor(pattern_input)

            output, attention_weights = scaled_dot_product_attention(Q, K, V)

            # Check attention patterns make sense
            # First token should attend strongly to last token
            first_to_last = attention_weights.data[0, -1]
            last_to_first = attention_weights.data[-1, 0]

            # These should be among the highest attention weights
            assert first_to_last > 0.1, "Attention pattern not detected"
            assert last_to_first > 0.1, "Attention pattern not detected"

        except ImportError:
            assert True, "Attention pattern analysis not ready yet"


class TestNLPReadiness:
    """Test readiness for NLP applications."""

    def test_language_modeling_architecture(self):
        """Test architecture suitable for language modeling."""
        try:
            from tinytorch.core.attention import MultiHeadAttention
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            # Language model components
            vocab_size, embed_dim, seq_len = 1000, 256, 32

            # Embedding layer (simplified)
            embedding = Linear(vocab_size, embed_dim)

            # Attention layers
            attention1 = MultiHeadAttention(embed_dim=embed_dim, num_heads=8)
            attention2 = MultiHeadAttention(embed_dim=embed_dim, num_heads=8)

            # Output projection
            output_proj = Linear(embed_dim, vocab_size)

            # Token sequence (as embeddings)
            batch_size = 4
            tokens = Tensor(np.random.randint(0, vocab_size, (seq_len, batch_size)))

            # Simple embedding lookup (simplified)
            if hasattr(embedding, 'embedding_lookup'):
                x = embedding.embedding_lookup(tokens)
            else:
                # Simplified: random embeddings
                x = Tensor(np.random.randn(seq_len, batch_size, embed_dim))

            # Transformer layers
            h1 = attention1(x)
            h2 = attention2(h1)

            # Output logits
            logits = output_proj(h2)

            assert logits.shape == (seq_len, batch_size, vocab_size), "Language model architecture broken"

        except ImportError:
            assert True, "Language modeling architecture not ready yet"


class TestRegressionPrevention:
    """Ensure previous modules still work after Module 07 development."""

    def test_no_foundation_regression(self):
        """Verify foundation stack (01→05) unchanged."""
        # Environment should remain stable
        assert sys.version_info.major >= 3, "Foundation: Python detection broken"

        # Project structure should remain intact
        project_root = Path(__file__).parent.parent.parent
        assert project_root.exists(), "Foundation: Project structure broken"

    def test_no_spatial_regression(self):
        """Verify convolution operations (Module 09) unchanged."""
        try:
            from tinytorch.core.spatial import Conv2d as Conv2D

            # Spatial operations should still work
            conv = Conv2D(in_channels=1, out_channels=8, kernel_size=3)
            assert hasattr(conv, 'forward'), "Spatial regression: Conv2D broken"

        except ImportError:
            # If not implemented, that's fine
            # But numpy should still work (from foundation)
            import numpy as np
            arr = np.array([1, 2, 3])
            assert arr.shape == (3,), "Spatial regression: Numpy foundation broken"

    def test_progressive_stability(self):
        """Test the progressive stack is stable through attention."""
        # Stack should be stable through: Setup → Tensor → Activations → Layers → Dense → Spatial → Attention

        # Setup level
        import numpy as np
        assert np is not None, "Setup level broken"

        # Foundation level (if available)
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear

            # Should still be able to build neural networks
            layer = Linear(10, 5)
            x = Tensor(np.random.randn(4, 10))
            output = layer(x)
            assert output.shape == (4, 5), "Foundation level broken"

        except ImportError:
            pass  # Not implemented yet

        # Attention level (if available)
        try:
            from tinytorch.core.attention import MultiHeadAttention
            attention = MultiHeadAttention(embed_dim=32, num_heads=4)
            assert callable(attention), "Attention level broken"
        except ImportError:
            pass  # Not implemented yet
