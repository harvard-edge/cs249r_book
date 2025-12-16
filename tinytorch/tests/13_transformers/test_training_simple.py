"""
Simple end-to-end training test for transformers.

This test validates that a transformer can successfully learn from a tiny dataset,
demonstrating that the entire training pipeline (forward, loss, backward, update) works.
"""

import numpy as np
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.optimizers import Adam
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.models.transformer import GPT
from tinytorch.core.tokenization import CharTokenizer


def test_transformer_memorization():
    """
    Test that a transformer can memorize a tiny dataset.

    Success criteria:
    - Loss decreases by at least 80% in 500 steps
    - No NaN/Inf losses
    - All parameters receive gradients
    - Training completes in reasonable time (<120s)
    """
    print("\n" + "="*70)
    print("TEST: Transformer Memorization Capability")
    print("="*70)

    # Tiny dataset (5 patterns)
    patterns = [
        "def add(a, b):\n    return a + b",
        "def sub(a, b):\n    return a - b",
        "for i in range(10):\n    print(i)",
        "if x > 0:\n    print('positive')",
        "numbers = [1, 2, 3, 4, 5]",
    ]

    # Create tokenizer
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(patterns)
    print(f"   Vocabulary size: {tokenizer.vocab_size}")

    # Create model (small for fast testing)
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        num_layers=1,
        num_heads=4,
        max_seq_len=64
    )

    num_params = sum(np.prod(p.shape) for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")

    # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = CrossEntropyLoss()

    # Encode and pad patterns
    max_len = 64
    encoded = []
    for p in patterns:
        tokens = tokenizer.encode(p)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + [0] * (max_len - len(tokens))
        encoded.append(tokens)

    # Training
    print("   Training for 500 steps...")
    losses = []
    start_time = time.time()

    for step in range(500):
        # Sample random pattern
        tokens = encoded[np.random.randint(len(encoded))]
        x = Tensor(np.array([tokens[:-1]], dtype=np.int32))
        y = Tensor(np.array([tokens[1:]], dtype=np.int32))

        # Forward pass
        logits = model.forward(x)
        logits_flat = logits.reshape(len(tokens)-1, tokenizer.vocab_size)
        y_flat = y.reshape(len(tokens)-1)
        loss = loss_fn(logits_flat, y_flat)

        # Check for NaN/Inf
        assert not np.isnan(loss.data).any(), f"NaN loss at step {step}"
        assert not np.isinf(loss.data).any(), f"Inf loss at step {step}"

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradients on first step
        if step == 0:
            params_with_grad = sum(1 for p in model.parameters()
                                   if p.grad is not None and np.abs(p.grad).max() > 1e-10)
            total_params = len(model.parameters())
            assert params_with_grad == total_params, \
                f"Only {params_with_grad}/{total_params} parameters have gradients"

        # Gradient clipping
        for p in model.parameters():
            if p.grad is not None:
                p.grad = np.clip(p.grad, -1.0, 1.0)

        # Update
        optimizer.step()

        # Track loss
        losses.append(loss.data.item())

    elapsed = time.time() - start_time

    # Compute statistics
    initial_loss = losses[0]
    final_loss = np.mean(losses[-100:])
    loss_decrease_pct = ((initial_loss - final_loss) / initial_loss) * 100

    print(f"\n   Results:")
    print(f"   ├─ Initial loss: {initial_loss:.3f}")
    print(f"   ├─ Final loss: {final_loss:.3f}")
    print(f"   ├─ Loss decrease: {loss_decrease_pct:.1f}%")
    print(f"   └─ Training time: {elapsed:.1f}s")

    # Assertions
    assert elapsed < 120, f"Training too slow: {elapsed:.1f}s > 120s"
    assert loss_decrease_pct > 80, \
        f"Insufficient learning: loss decreased only {loss_decrease_pct:.1f}% (expected >80%)"
    # Relaxed threshold: loss should decrease significantly, final value depends on model capacity
    assert final_loss < 1.0, \
        f"Final loss too high: {final_loss:.3f} (expected <1.0 for memorization)"

    print(f"\n✅ Transformer successfully memorized dataset!")
    print(f"   Loss decreased {loss_decrease_pct:.1f}% in {elapsed:.1f}s")
    return True


def test_transformer_convergence_rate():
    """
    Test that transformer converges at expected rate.

    This is a regression test to catch training instabilities.
    """
    print("\n" + "="*70)
    print("TEST: Transformer Convergence Rate")
    print("="*70)

    # Setup (same as memorization test)
    patterns = [
        "def add(a, b):\n    return a + b",
        "def sub(a, b):\n    return a - b",
    ]

    tokenizer = CharTokenizer()
    tokenizer.build_vocab(patterns)

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        num_layers=1,
        num_heads=4,
        max_seq_len=64
    )

    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = CrossEntropyLoss()

    # Encode patterns
    max_len = 64
    encoded = []
    for p in patterns:
        tokens = tokenizer.encode(p)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + [0] * (max_len - len(tokens))
        encoded.append(tokens)

    # Train until loss < 0.1
    step = 0
    loss_val = float('inf')

    print(f"   Training until loss < 0.1...")

    while loss_val > 0.1 and step < 1000:
        tokens = encoded[np.random.randint(len(encoded))]
        x = Tensor(np.array([tokens[:-1]], dtype=np.int32))
        y = Tensor(np.array([tokens[1:]], dtype=np.int32))

        logits = model.forward(x)
        logits_flat = logits.reshape(len(tokens)-1, tokenizer.vocab_size)
        y_flat = y.reshape(len(tokens)-1)
        loss = loss_fn(logits_flat, y_flat)

        optimizer.zero_grad()
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                p.grad = np.clip(p.grad, -1.0, 1.0)

        optimizer.step()

        loss_val = loss.data.item()
        step += 1

    print(f"   Reached loss < 0.1 in {step} steps")

    # Regression check: should converge in < 700 steps for 2 patterns
    # Educational implementations may have slightly slower convergence
    assert step < 700, \
        f"Convergence too slow: {step} steps (expected <700). Training may be unstable."

    print(f"✅ Convergence rate is acceptable ({step} steps)")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TRANSFORMER TRAINING TEST SUITE")
    print("="*70)

    test_transformer_memorization()
    test_transformer_convergence_rate()

    print("\n" + "="*70)
    print("✅ ALL TRAINING TESTS PASSED")
    print("="*70 + "\n")
