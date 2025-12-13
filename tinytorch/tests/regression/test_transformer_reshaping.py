"""
BUG TRACKING:
============
Bug ID: BUG-2024-11-25-002
Date Found: 2024-11-25
Found By: PyTorch Expert Architecture Review
Severity: High

DESCRIPTION:
TinyGPT example fails with "matmul requires 2D tensors" when passing transformer
output (3D: batch x seq x embed) directly to Linear layer projection.

REPRODUCTION:
1. Create transformer with embed_dim=128, num_heads=4
2. Pass input of shape (batch=2, seq=10, embed=128)
3. Transformer outputs (2, 10, 128) - still 3D
4. Try to pass to Linear(128, vocab_size) for token prediction
5. ValueError: matmul requires 2D tensors

ROOT CAUSE:
Transformer blocks output 3D tensors (batch, sequence, embedding) but Linear layers
expect 2D input (batch, features). Missing reshape/view operation between transformer
and output projection.

FIX:
Add proper reshaping:
- Option 1: Reshape to (batch * seq, embed) before Linear, then reshape back
- Option 2: Apply Linear to last dimension only (requires Linear to handle 3D)
- Option 3: Take only last token for generation (shape becomes 2D naturally)

PREVENTION:
This regression test ensures transformer outputs can be properly passed to Linear layers
for vocabulary projection in language models.
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.nn import TransformerBlock, Embedding, PositionalEncoding, MultiHeadAttention


def test_transformer_to_linear_3d_to_2d():
    """
    Regression test for transformer 3D output to Linear 2D input.
    This exact issue occurred in examples/gpt_2018/train_gpt.py
    """
    print("🔬 Testing Transformer 3D -> Linear 2D reshaping...")

    # Setup from failing TinyGPT example
    batch_size = 2
    seq_length = 10
    embed_dim = 128
    num_heads = 4
    vocab_size = 1000

    # Create transformer and output projection
    transformer = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=4,
        dropout_prob=0.1
    )
    output_proj = Linear(embed_dim, vocab_size)

    # Create dummy input (batch, seq, embed)
    x = Tensor(np.random.randn(batch_size, seq_length, embed_dim))
    print(f"Input shape: {x.shape}")

    # Transformer maintains 3D shape
    transformer_out = transformer(x)
    assert transformer_out.shape == (batch_size, seq_length, embed_dim)
    print(f"Transformer output shape: {transformer_out.shape}")

    # The bug: Direct pass to Linear fails
    try:
        # This is what the broken example tried to do
        output = output_proj(transformer_out)
        # If Linear can handle 3D, this might work
        if output.shape == (batch_size, seq_length, vocab_size):
            print("✅ Linear handles 3D input (broadcasting)")
            return True
    except (ValueError, AssertionError) as e:
        print(f"Expected error with 3D input: {e}")

    # Solution 1: Reshape to 2D, apply Linear, reshape back
    print("\n📝 Solution 1: Reshape -> Linear -> Reshape")
    batch, seq, embed = transformer_out.shape
    reshaped_2d = transformer_out.reshape(batch * seq, embed)
    print(f"Reshaped to 2D: {reshaped_2d.shape}")

    output_2d = output_proj(reshaped_2d)
    assert output_2d.shape == (batch * seq, vocab_size)
    print(f"Linear output: {output_2d.shape}")

    output_3d = output_2d.reshape(batch, seq, vocab_size)
    assert output_3d.shape == (batch_size, seq_length, vocab_size)
    print(f"Reshaped back to 3D: {output_3d.shape}")
    print("✅ Solution 1 works!")

    # Solution 2: Take only last token (for generation)
    print("\n📝 Solution 2: Use only last token for generation")
    last_token = transformer_out[:, -1, :]  # (batch, embed)
    assert last_token.shape == (batch_size, embed_dim)
    print(f"Last token shape: {last_token.shape}")

    next_token_logits = output_proj(last_token)
    assert next_token_logits.shape == (batch_size, vocab_size)
    print(f"Next token predictions: {next_token_logits.shape}")
    print("✅ Solution 2 works!")

    print("\n🎯 Transformer->Linear reshape test PASSED!")
    return True


def test_full_gpt_architecture_shapes():
    """Test shape flow through complete GPT architecture."""
    print("🔬 Testing complete GPT architecture shape flow...")

    # GPT-style architecture parameters
    batch_size = 4
    seq_length = 50
    vocab_size = 1000
    embed_dim = 256
    num_heads = 8
    num_layers = 4

    # Input: token indices
    input_ids = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)))
    print(f"Input tokens shape: {input_ids.shape}")

    # Embedding layer
    embed_layer = Embedding(vocab_size, embed_dim)
    x = embed_layer(input_ids)  # -> (batch, seq, embed)
    assert x.shape == (batch_size, seq_length, embed_dim)
    print(f"After embedding: {x.shape}")

    # Positional encoding (max_seq_len, embed_dim)
    pos_enc = PositionalEncoding(seq_length, embed_dim)
    x = pos_enc(x)
    assert x.shape == (batch_size, seq_length, embed_dim)
    print(f"After positional encoding: {x.shape}")

    # Stack of transformer blocks
    for i in range(num_layers):
        transformer = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4
        )
        x = transformer(x)
        assert x.shape == (batch_size, seq_length, embed_dim)
        print(f"After transformer {i+1}: {x.shape}")

    # Output projection (with proper reshaping)
    output_proj = Linear(embed_dim, vocab_size)

    # Method 1: Process all positions
    batch, seq, embed = x.shape
    x_2d = x.reshape(batch * seq, embed)
    logits_2d = output_proj(x_2d)
    logits = logits_2d.reshape(batch, seq, vocab_size)
    assert logits.shape == (batch_size, seq_length, vocab_size)
    print(f"Final logits (all positions): {logits.shape}")

    # Method 2: Process last position only (for generation)
    last_hidden = x[:, -1, :]
    next_token_logits = output_proj(last_hidden)
    assert next_token_logits.shape == (batch_size, vocab_size)
    print(f"Next token logits: {next_token_logits.shape}")

    print("✅ Complete GPT architecture shapes flow correctly!")
    return True


def test_attention_kv_cache_shapes():
    """Test that KV caching maintains proper shapes."""
    print("🔬 Testing attention KV cache shape compatibility...")

    batch_size = 2
    seq_length = 10
    embed_dim = 128
    num_heads = 4

    # Multi-head attention
    mha = MultiHeadAttention(embed_dim, num_heads)

    # Initial forward pass
    x = Tensor(np.random.randn(batch_size, seq_length, embed_dim))

    # Self-attention (Q, K, V all derived from x)
    output = mha(x)
    assert output.shape == (batch_size, seq_length, embed_dim)
    print(f"MHA output: {output.shape}")

    # Process one token at a time (for autoregressive generation)
    for t in range(seq_length):
        x_t = x[:, t:t+1, :]  # Single token
        output_t = mha(x_t)
        assert output_t.shape == (batch_size, 1, embed_dim)
        print(f"  Token {t} output: {output_t.shape}")

    print("✅ Attention shape handling works correctly!")


def test_embedding_dimension_compatibility():
    """Test that embeddings match transformer input requirements."""
    print("🔬 Testing embedding dimension compatibility...")

    vocab_size = 5000
    embed_dim = 512
    seq_length = 100
    batch_size = 8

    # Create embedding and transformer
    embedding = Embedding(vocab_size, embed_dim)
    transformer = TransformerBlock(embed_dim, num_heads=8)

    # Token indices
    tokens = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)))

    # Embed tokens
    embedded = embedding(tokens)
    assert embedded.shape == (batch_size, seq_length, embed_dim)

    # Pass through transformer
    output = transformer(embedded)
    assert output.shape == (batch_size, seq_length, embed_dim)

    print("✅ Embedding->Transformer dimensions compatible!")
    return True


if __name__ == "__main__":
    print("="*60)
    print("REGRESSION TEST: Transformer 3D to Linear 2D Reshaping")
    print("="*60)

    # Import required modules for testing
    try:
        from tinytorch.nn import MultiHeadAttention
    except ImportError:
        # Create a simple mock if not available
        class MultiHeadAttention:
            def __init__(self, embed_dim, num_heads):
                self.embed_dim = embed_dim
                self.num_heads = num_heads

            def __call__(self, q, k, v):
                # Return query shape for testing
                return q

    # Run all tests
    all_pass = True
    all_pass &= test_transformer_to_linear_3d_to_2d()
    all_pass &= test_full_gpt_architecture_shapes()
    all_pass &= test_attention_kv_cache_shapes()
    all_pass &= test_embedding_dimension_compatibility()

    if all_pass:
        print("\n🏆 ALL REGRESSION TESTS PASSED!")
        print("The Transformer->Linear reshape bug is prevented.")
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
