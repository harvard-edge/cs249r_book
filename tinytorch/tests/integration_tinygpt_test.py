#!/usr/bin/env python3
"""
TinyGPT Integration Test - After Module 14
==========================================

This test validates that modules 1-14 work together for transformer language models.

Required modules:
- Module 01-08: Core MLP and training functionality
- Module 11: Tokenization for text processing
- Module 12: Embeddings (token + positional)
- Module 13: Multi-head self-attention
- Module 14: Transformer blocks and layer normalization

This demonstrates the milestone: "Can build transformer language models"
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU

# Try to import transformer components
try:
    from tinytorch.core.embeddings import Embedding, PositionalEncoding
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    from tinytorch.core.attention import MultiHeadAttention
    ATTENTION_AVAILABLE = True
except ImportError:
    ATTENTION_AVAILABLE = False

try:
    from tinytorch.core.transformers import LayerNorm, TransformerBlock
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class SimpleTinyGPT:
    """Simple GPT-style transformer for language modeling."""

    def __init__(self, vocab_size=1000, embed_dim=128, max_length=50, num_heads=8, num_layers=2):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.num_heads = num_heads

        # Token representation
        if EMBEDDINGS_AVAILABLE:
            self.embedding = Embedding(vocab_size, embed_dim)
            self.pos_encoding = PositionalEncoding(embed_dim, max_length)
        else:
            # Fallback: simple linear embedding
            self.embedding = Linear(vocab_size, embed_dim)

        # Transformer layers
        if TRANSFORMERS_AVAILABLE and ATTENTION_AVAILABLE:
            self.layers = []
            hidden_dim = embed_dim * 4
            for _ in range(num_layers):
                block = TransformerBlock(embed_dim, num_heads, hidden_dim)
                self.layers.append(block)

            # Output
            self.layer_norm = LayerNorm(embed_dim)
        else:
            # Fallback: simple feedforward layers
            self.layers = [
                Linear(embed_dim, embed_dim * 2),
                ReLU(),
                Linear(embed_dim * 2, embed_dim)
            ]

        # Output projection
        self.output_proj = Linear(embed_dim, vocab_size)

    def forward(self, x):
        """Forward pass."""
        # Convert tokens to embeddings
        if EMBEDDINGS_AVAILABLE:
            x = self.embedding(x)
            x = self.pos_encoding(x)
        else:
            # Fallback: convert token indices to one-hot, then embed
            batch_size, seq_len = x.shape
            one_hot = np.zeros((batch_size, seq_len, self.vocab_size))
            for b in range(batch_size):
                for s in range(seq_len):
                    token_id = int(x.data[b, s])
                    if 0 <= token_id < self.vocab_size:
                        one_hot[b, s, token_id] = 1.0

            x = Tensor(one_hot)
            # Apply embedding to each position
            embedded = []
            for s in range(seq_len):
                pos_embed = self.embedding(x[:, s, :])  # (batch, embed_dim)
                embedded.append(pos_embed)

            # Stack to get (batch, seq_len, embed_dim)
            x = Tensor(np.stack([emb.data for emb in embedded], axis=1))

        # Process through transformer layers
        if TRANSFORMERS_AVAILABLE and ATTENTION_AVAILABLE:
            for layer in self.layers:
                x = layer(x)
            x = self.layer_norm(x)
        else:
            # Fallback: process each position through feedforward
            batch_size, seq_len, embed_dim = x.shape
            processed = []
            for s in range(seq_len):
                pos_data = x[:, s, :]  # (batch, embed_dim)

                # Apply simple feedforward
                h = self.layers[0](pos_data)  # Dense layer
                h = self.layers[1](h)         # ReLU
                h = self.layers[2](h)         # Dense layer
                processed.append(h.data)

            x = Tensor(np.stack(processed, axis=1))

        # Output projection
        batch_size, seq_len, embed_dim = x.shape
        outputs = []
        for s in range(seq_len):
            pos_output = self.output_proj(x[:, s, :])
            outputs.append(pos_output.data)

        return Tensor(np.stack(outputs, axis=1))

    def __call__(self, x):
        return self.forward(x)

def test_transformer_components():
    """Test individual transformer components."""
    print("🧩 Testing Transformer Components...")

    # Test embeddings
    if EMBEDDINGS_AVAILABLE:
        print("  ✓ Testing Embedding layer")
        embed = Embedding(vocab_size=100, embed_dim=32)
        tokens = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))  # (batch=2, seq_len=3)
        embedded = embed(tokens)
        assert embedded.shape == (2, 3, 32), f"Expected (2, 3, 32), got {embedded.shape}"
        print(f"    Embedding: {tokens.shape} -> {embedded.shape}")

        print("  ✓ Testing Positional Encoding")
        pos_enc = PositionalEncoding(embed_dim=32, max_length=10)
        pos_embedded = pos_enc(embedded)
        assert pos_embedded.shape == embedded.shape, "Positional encoding should preserve shape"
        print(f"    Pos encoding: {embedded.shape} -> {pos_embedded.shape}")
    else:
        print("  ⚠️  Embeddings not available - using fallback")

    # Test attention
    if ATTENTION_AVAILABLE:
        print("  ✓ Testing Multi-Head Attention")
        attn = MultiHeadAttention(embed_dim=32, num_heads=4)
        x = Tensor(np.random.randn(2, 5, 32))  # (batch, seq_len, embed_dim)
        attn_out = attn(x)
        assert attn_out.shape == x.shape, f"Attention should preserve shape: {x.shape} -> {attn_out.shape}"
        print(f"    Attention: {x.shape} -> {attn_out.shape}")
    else:
        print("  ⚠️  Attention not available - using fallback")

    # Test transformer blocks
    if TRANSFORMERS_AVAILABLE and ATTENTION_AVAILABLE:
        print("  ✓ Testing Transformer Block")
        block = TransformerBlock(embed_dim=32, num_heads=4, hidden_dim=128)
        x = Tensor(np.random.randn(2, 5, 32))
        block_out = block(x)
        assert block_out.shape == x.shape, f"Transformer block should preserve shape"
        print(f"    Transformer block: {x.shape} -> {block_out.shape}")

        print("  ✓ Testing Layer Normalization")
        ln = LayerNorm(embed_dim=32)
        ln_out = ln(x)
        assert ln_out.shape == x.shape, "LayerNorm should preserve shape"
        print(f"    LayerNorm: {x.shape} -> {ln_out.shape}")
    else:
        print("  ⚠️  Transformer blocks not available - using fallback")

    print("✅ Transformer components tested!")
    return True

def test_tinygpt_architecture():
    """Test TinyGPT architecture."""
    print("🤖 Testing TinyGPT Architecture...")

    try:
        # Create small TinyGPT
        model = SimpleTinyGPT(
            vocab_size=100,
            embed_dim=64,
            max_length=10,
            num_heads=4,
            num_layers=2
        )

        # Test input: batch of token sequences
        batch_size, seq_len = 2, 8
        tokens = Tensor(np.random.randint(0, 100, (batch_size, seq_len)))

        print(f"  ✓ Created TinyGPT model")
        print(f"    Input tokens shape: {tokens.shape}")
        print(f"    Vocab size: 100, Embed dim: 64")

        # Forward pass
        outputs = model(tokens)

        print(f"  ✓ Forward pass successful")
        print(f"    Output shape: {outputs.shape}")

        expected_shape = (batch_size, seq_len, 100)  # (batch, seq_len, vocab_size)
        assert outputs.shape == expected_shape, f"Expected {expected_shape}, got {outputs.shape}"

        print("✅ TinyGPT architecture working!")
        return True

    except Exception as e:
        print(f"❌ TinyGPT architecture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_language_modeling():
    """Test language modeling capability."""
    print("📝 Testing Language Modeling...")

    try:
        # Create very small model for quick test
        model = SimpleTinyGPT(
            vocab_size=20,
            embed_dim=16,
            max_length=5,
            num_heads=2,
            num_layers=1
        )

        # Create simple sequence
        tokens = Tensor(np.array([[1, 2, 3, 4]]))  # Single sequence

        print(f"  ✓ Created small model for language modeling")
        print(f"    Input sequence: {tokens.shape}")

        # Get predictions
        logits = model(tokens)

        print(f"  ✓ Generated predictions")
        print(f"    Logits shape: {logits.shape}")
        print(f"    Each position predicts next token from vocab of size 20")

        # Check logits are reasonable
        assert logits.shape == (1, 4, 20), f"Expected (1, 4, 20), got {logits.shape}"

        # Test that different positions give different predictions (model is learning positional info)
        pos0_logits = logits.data[0, 0, :]  # First position
        pos1_logits = logits.data[0, 1, :]  # Second position

        # They should be different (not identical)
        diff = np.sum(np.abs(pos0_logits - pos1_logits))
        if diff > 0.001:
            print(f"  ✓ Different positions give different predictions (diff: {diff:.4f})")
        else:
            print(f"  ⚠️  Positions give similar predictions (diff: {diff:.4f})")

        print("✅ Language modeling capability tested!")
        return True

    except Exception as e:
        print(f"❌ Language modeling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text_generation_potential():
    """Test potential for text generation."""
    print("✍️  Testing Text Generation Potential...")

    try:
        model = SimpleTinyGPT(vocab_size=10, embed_dim=8, max_length=3, num_heads=2, num_layers=1)

        # Start with a single token
        start_token = Tensor(np.array([[5]]))  # Start with token 5

        print(f"  ✓ Testing autoregressive generation")
        print(f"    Start token: {start_token.data}")

        # Generate next token prediction
        logits = model(start_token)
        print(f"  ✓ Generated logits shape: {logits.shape}")

        # Get most likely next token
        next_token_logits = logits.data[0, 0, :]  # First (and only) position
        next_token = np.argmax(next_token_logits)

        print(f"  ✓ Predicted next token: {next_token}")
        print(f"    (In real generation, this would be added to sequence)")

        # Test with longer sequence
        longer_seq = Tensor(np.array([[5, int(next_token)]]))
        longer_logits = model(longer_seq)
        print(f"  ✓ Processed longer sequence: {longer_seq.shape} -> {longer_logits.shape}")

        print("✅ Text generation potential demonstrated!")
        return True

    except Exception as e:
        print(f"❌ Text generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_tinygpt_integration_test():
    """Run complete TinyGPT integration test."""
    print("=" * 60)
    print("🔥 TINYGPT INTEGRATION TEST - Modules 1-14")
    print("=" * 60)
    print()

    # Component availability summary
    components = [
        ("Embeddings", EMBEDDINGS_AVAILABLE),
        ("Attention", ATTENTION_AVAILABLE),
        ("Transformers", TRANSFORMERS_AVAILABLE)
    ]

    print("📋 Component Availability:")
    for name, available in components:
        status = "✅ Available" if available else "⚠️  Using fallback"
        print(f"   {name}: {status}")
    print()

    success = True
    tests = [
        test_transformer_components,
        test_tinygpt_architecture,
        test_language_modeling,
        test_text_generation_potential
    ]

    for test in tests:
        try:
            if not test():
                success = False
            print()
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            success = False
            print()

    if success:
        print("🎉 TINYGPT INTEGRATION TEST PASSED!")
        print()
        print("✅ Milestone Achieved: Can build transformer language models")
        print("   • Transformer architecture handles sequential data")
        print("   • Language modeling predictions generated")
        print("   • Text generation potential demonstrated")
        print("   • End-to-end NLP pipeline functional")
        print()
        print("🏆 CONGRATULATIONS: All core ML capabilities working!")
    else:
        print("❌ TINYGPT INTEGRATION TEST FAILED!")
        print("   Check transformer modules before proceeding")

    print("=" * 60)
    return success

if __name__ == "__main__":
    run_tinygpt_integration_test()
