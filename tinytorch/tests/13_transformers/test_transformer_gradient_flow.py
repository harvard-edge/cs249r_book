"""
Test gradient flow through complete transformer architecture.

This test validates that all transformer components (embeddings, attention,
LayerNorm, MLP) properly propagate gradients during backpropagation.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.transformer import GPT, MultiHeadAttention, LayerNorm, MLP
from tinytorch.core.losses import CrossEntropyLoss


def test_multihead_attention_gradient_flow():
    """Test that all MultiHeadAttention parameters receive gradients."""
    print("Testing MultiHeadAttention gradient flow...")

    batch_size, seq_len, embed_dim = 2, 8, 16
    num_heads = 4

    # Create attention module
    mha = MultiHeadAttention(embed_dim, num_heads)

    # Forward pass
    x = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    output = mha.forward(x)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check all parameters have gradients
    params = mha.parameters()
    params_with_grad = 0
    params_without_grad = []

    for i, param in enumerate(params):
        if param.grad is not None and np.abs(param.grad).max() > 1e-10:
            params_with_grad += 1
        else:
            params_without_grad.append(i)

    assert params_with_grad == len(params), \
        f"All {len(params)} MHA parameters should have gradients, but only {params_with_grad} do. Missing: {params_without_grad}"

    print(f"✅ All {len(params)} MultiHeadAttention parameters receive gradients")


def test_layernorm_gradient_flow():
    """Test that LayerNorm parameters receive gradients."""
    print("Testing LayerNorm gradient flow...")

    batch_size, seq_len, embed_dim = 2, 8, 16

    # Create LayerNorm
    ln = LayerNorm(embed_dim)

    # Forward pass
    x = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    output = ln.forward(x)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check parameters have gradients
    params = ln.parameters()
    assert len(params) == 2, "LayerNorm should have 2 parameters (gamma, beta)"

    for i, param in enumerate(params):
        assert param.grad is not None, f"Parameter {i} should have gradient"
        assert np.abs(param.grad).max() > 1e-10, f"Parameter {i} gradient should be non-zero"

    print("✅ LayerNorm gradient flow works correctly")


def test_mlp_gradient_flow():
    """Test that MLP parameters receive gradients."""
    print("Testing MLP gradient flow...")

    batch_size, seq_len, embed_dim = 2, 8, 16

    # Create MLP
    mlp = MLP(embed_dim)

    # Forward pass
    x = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    output = mlp.forward(x)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check all parameters have gradients
    params = mlp.parameters()
    for i, param in enumerate(params):
        assert param.grad is not None, f"MLP parameter {i} should have gradient"
        assert np.abs(param.grad).max() > 1e-10, f"MLP parameter {i} gradient should be non-zero"

    print(f"✅ All {len(params)} MLP parameters receive gradients")


def test_full_gpt_gradient_flow():
    """Test that all GPT model parameters receive gradients end-to-end."""
    print("Testing full GPT gradient flow...")

    # Create small GPT model
    vocab_size = 20
    embed_dim = 16
    num_layers = 2
    num_heads = 2
    max_seq_len = 32

    model = GPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len
    )

    # Create input and targets
    batch_size = 2
    seq_len = 8
    tokens = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
    targets = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))

    # Forward pass
    logits = model.forward(tokens)

    # Compute loss
    logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
    targets_flat = targets.reshape(batch_size * seq_len)
    loss_fn = CrossEntropyLoss()
    loss = loss_fn.forward(logits_flat, targets_flat)

    print(f"   Loss: {loss.data:.3f}")

    # Backward pass
    loss.backward()

    # Check gradient flow to all parameters
    params = model.parameters()
    params_with_grad = 0
    params_without_grad = []

    for i, param in enumerate(params):
        if param.grad is not None and np.abs(param.grad).max() > 1e-10:
            params_with_grad += 1
        else:
            params_without_grad.append(i)

    # Report detailed results
    print(f"   Parameters with gradients: {params_with_grad}/{len(params)}")

    if params_without_grad:
        print(f"   ⚠️  Parameters WITHOUT gradients: {params_without_grad}")

        # Provide parameter mapping for debugging
        print("\n   Parameter breakdown:")
        param_idx = 0
        print(f"     {param_idx}: Token embedding weight")
        param_idx += 1
        print(f"     {param_idx}: Position embedding weight")
        param_idx += 1

        for block_idx in range(num_layers):
            print(f"     Block {block_idx}:")
            print(f"       {param_idx}-{param_idx+7}: Attention (Q/K/V/out + biases)")
            param_idx += 8
            print(f"       {param_idx}-{param_idx+1}: LayerNorm 1 (gamma, beta)")
            param_idx += 2
            print(f"       {param_idx}-{param_idx+1}: LayerNorm 2 (gamma, beta)")
            param_idx += 2
            print(f"       {param_idx}-{param_idx+3}: MLP (2 linears + biases)")
            param_idx += 4

        print(f"     {param_idx}-{param_idx+1}: Final LayerNorm (gamma, beta)")
        param_idx += 2
        print(f"     {param_idx}: LM head weight")

        raise AssertionError(f"Expected all {len(params)} parameters to have gradients, but {len(params_without_grad)} don't")

    print(f"✅ All {len(params)} GPT parameters receive gradients")


def test_attention_mask_gradient_flow():
    """Test that attention with masking preserves gradient flow."""
    print("Testing attention with causal mask gradient flow...")

    batch_size, seq_len, embed_dim = 2, 4, 16
    num_heads = 4

    # Create attention module
    mha = MultiHeadAttention(embed_dim, num_heads)

    # Create causal mask
    mask = Tensor(-1e9 * np.triu(np.ones((seq_len, seq_len)), k=1))

    # Forward pass
    x = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    output = mha.forward(x, mask)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check all parameters have gradients
    params = mha.parameters()
    params_with_grad = sum(1 for p in params if p.grad is not None and np.abs(p.grad).max() > 1e-10)

    assert params_with_grad == len(params), \
        f"Masking should not break gradient flow. Expected {len(params)} params with grads, got {params_with_grad}"

    print("✅ Attention with masking preserves gradient flow")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TRANSFORMER GRADIENT FLOW TEST SUITE")
    print("="*70 + "\n")

    test_multihead_attention_gradient_flow()
    test_layernorm_gradient_flow()
    test_mlp_gradient_flow()
    test_attention_mask_gradient_flow()
    test_full_gpt_gradient_flow()

    print("\n" + "="*70)
    print("✅ ALL TRANSFORMER GRADIENT FLOW TESTS PASSED")
    print("="*70 + "\n")
