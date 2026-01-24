"""
Training Flow Integration Tests
================================

Tests that the complete training pipeline works:
1. Forward pass produces valid outputs
2. Loss computes correctly
3. Backward pass populates gradients
4. Optimizer updates weights
5. Loss decreases over iterations

These tests catch issues that unit tests miss - where modules
work individually but fail when connected.

Modules tested: 01-08 (Tensor → Training)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU, Sigmoid
from tinytorch.core.losses import MSELoss, CrossEntropyLoss
from tinytorch.core.optimizers import SGD, Adam
from tinytorch.core.autograd import enable_autograd

# Enable autograd for all tests
enable_autograd()


class TestOptimzerActuallyUpdatesWeights:
    """
    Critical Test: Verify optimizer.step() actually changes weights.

    Common bugs caught:
    - Optimizer not connected to parameters
    - Gradients not flowing to weights
    - Learning rate is zero
    - step() not implemented correctly
    """

    def test_sgd_updates_weights(self):
        """SGD must modify weights after step()"""
        layer = Linear(2, 1)
        optimizer = SGD([layer.weight, layer.bias], lr=0.1)

        # Store initial weights
        initial_weight = layer.weight.data.copy()
        initial_bias = layer.bias.data.copy()

        # Forward + backward
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        target = Tensor([[5.0]])

        output = layer.forward(x)
        loss = MSELoss().forward(output, target)
        loss.backward()

        # Verify gradients exist
        assert layer.weight.grad is not None, "Weight gradient is None!"
        assert layer.bias.grad is not None, "Bias gradient is None!"

        # Step should update weights
        optimizer.step()

        # Weights MUST be different
        weight_changed = not np.allclose(initial_weight, layer.weight.data)
        bias_changed = not np.allclose(initial_bias, layer.bias.data)

        assert weight_changed, (
            f"SGD.step() did not change weights!\n"
            f"  Before: {initial_weight}\n"
            f"  After:  {layer.weight.data}\n"
            f"  Grad:   {layer.weight.grad}"
        )
        assert bias_changed, "SGD.step() did not change bias!"

    def test_adam_updates_weights(self):
        """Adam must modify weights after step()"""
        layer = Linear(2, 1)
        optimizer = Adam([layer.weight, layer.bias], lr=0.1)

        initial_weight = layer.weight.data.copy()

        x = Tensor([[1.0, 2.0]], requires_grad=True)
        target = Tensor([[5.0]])

        output = layer.forward(x)
        loss = MSELoss().forward(output, target)
        loss.backward()

        optimizer.step()

        assert not np.allclose(initial_weight, layer.weight.data), (
            "Adam.step() did not change weights!"
        )


class TestTrainingReducesLoss:
    """
    Critical Test: Verify that training actually reduces loss.

    Common bugs caught:
    - Gradients have wrong sign
    - Learning rate too high (divergence)
    - Optimizer not using gradients correctly
    - Loss function returning wrong values
    """

    def test_mlp_loss_decreases(self):
        """A simple MLP must learn XOR-like pattern"""
        # Simple 2-layer network
        layer1 = Linear(2, 4)
        relu = ReLU()
        layer2 = Linear(4, 1)
        sigmoid = Sigmoid()
        loss_fn = MSELoss()

        params = [layer1.weight, layer1.bias, layer2.weight, layer2.bias]
        optimizer = SGD(params, lr=0.5)

        # XOR-like data
        X = Tensor([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.]
        ], requires_grad=True)
        y = Tensor([[0.], [1.], [1.], [0.]])

        # Track loss over time
        losses = []

        for epoch in range(100):
            # Zero gradients
            for p in params:
                if p.grad is not None:
                    p.grad = np.zeros_like(p.grad)

            # Forward
            h = relu.forward(layer1.forward(X))
            out = sigmoid.forward(layer2.forward(h))
            loss = loss_fn.forward(out, y)

            losses.append(float(loss.data))

            # Backward
            loss.backward()

            # Update
            optimizer.step()

        # Loss MUST decrease
        initial_loss = losses[0]
        final_loss = losses[-1]

        assert final_loss < initial_loss, (
            f"Training did not reduce loss!\n"
            f"  Initial: {initial_loss:.4f}\n"
            f"  Final:   {final_loss:.4f}\n"
            f"  Loss history: {losses[:5]}...{losses[-5:]}"
        )

        # Loss should decrease (at least 5% - being lenient for test stability)
        improvement = (initial_loss - final_loss) / initial_loss
        assert improvement > 0.05, (
            f"Training improved loss by only {improvement*100:.1f}%\n"
            f"  Expected at least 5% improvement"
        )


class TestGradientChainNotBroken:
    """
    Critical Test: Verify gradient chain is not broken.

    Common bugs caught:
    - requires_grad not propagating
    - Operations not recording grad_fn
    - Intermediate tensors breaking the chain
    """

    def test_deep_network_gradient_chain(self):
        """Gradients must flow through 5 layers"""
        # Use fixed seed for reproducibility - prevents flaky test due to
        # random initialization that might kill all ReLUs
        np.random.seed(42)

        layers = [Linear(4, 4) for _ in range(5)]
        relu = ReLU()

        x = Tensor(np.random.randn(1, 4), requires_grad=True)
        target = Tensor(np.random.randn(1, 4))

        # Forward through all layers
        h = x
        for layer in layers:
            h = relu.forward(layer.forward(h))

        loss = MSELoss().forward(h, target)
        loss.backward()

        # ALL layers must have gradients
        for i, layer in enumerate(layers):
            assert layer.weight.grad is not None, (
                f"Layer {i} weight.grad is None - gradient chain broken!"
            )
            assert layer.bias.grad is not None, (
                f"Layer {i} bias.grad is None - gradient chain broken!"
            )

            # Gradients should be non-trivial
            grad_norm = np.linalg.norm(layer.weight.grad)
            assert grad_norm > 1e-10, (
                f"Layer {i} has vanishing gradients: {grad_norm}"
            )

    def test_input_receives_gradients(self):
        """Input tensor must receive gradients for visualization/debugging"""
        layer = Linear(3, 2)
        x = Tensor([[1., 2., 3.]], requires_grad=True)
        target = Tensor([[1., 0.]])

        output = layer.forward(x)
        loss = MSELoss().forward(output, target)
        loss.backward()

        assert x.grad is not None, "Input tensor did not receive gradients!"
        assert x.grad.shape == x.shape, (
            f"Input gradient shape mismatch: {x.grad.shape} vs {x.shape}"
        )


class TestZeroGradWorks:
    """
    Critical Test: Verify zero_grad clears gradients properly.

    Common bugs caught:
    - Gradients accumulating across batches
    - zero_grad not actually zeroing
    - Memory leaks from gradient accumulation
    """

    def test_gradients_dont_accumulate_after_zero_grad(self):
        """Gradients must not accumulate when zero_grad is called"""
        layer = Linear(2, 1)
        optimizer = SGD([layer.weight, layer.bias], lr=0.1)

        x = Tensor([[1., 2.]], requires_grad=True)
        target = Tensor([[1.]])

        # First forward/backward
        out1 = layer.forward(x)
        loss1 = MSELoss().forward(out1, target)
        loss1.backward()

        grad_after_first = layer.weight.grad.copy()

        # Zero gradients
        optimizer.zero_grad()

        # Verify zeroed
        assert layer.weight.grad is None or np.allclose(layer.weight.grad, 0), (
            "zero_grad() did not clear weight gradients!"
        )

        # Second forward/backward
        out2 = layer.forward(x)
        loss2 = MSELoss().forward(out2, target)
        loss2.backward()

        grad_after_second = layer.weight.grad.copy()

        # Gradients should be similar magnitude (not accumulated)
        ratio = np.linalg.norm(grad_after_second) / np.linalg.norm(grad_after_first)
        assert 0.5 < ratio < 2.0, (
            f"Gradients appear to be accumulating!\n"
            f"  First grad norm: {np.linalg.norm(grad_after_first)}\n"
            f"  Second grad norm: {np.linalg.norm(grad_after_second)}\n"
            f"  Ratio: {ratio} (should be ~1.0)"
        )


class TestBatchTraining:
    """
    Critical Test: Verify batch training works correctly.

    Common bugs caught:
    - Shape mismatches with batches
    - Mean vs sum reduction issues
    - Gradient scaling problems
    """

    def test_batch_gradients_are_averaged(self):
        """Gradients should be averaged over batch (not summed)"""
        layer = Linear(2, 1)

        # Single sample
        x1 = Tensor([[1., 2.]], requires_grad=True)
        target1 = Tensor([[3.]])

        out1 = layer.forward(x1)
        loss1 = MSELoss().forward(out1, target1)
        loss1.backward()

        single_grad = layer.weight.grad.copy()

        # Reset
        layer.weight.grad = None
        layer.bias.grad = None

        # Batch of same sample repeated 4 times
        x_batch = Tensor([[1., 2.]] * 4, requires_grad=True)
        target_batch = Tensor([[3.]] * 4)

        out_batch = layer.forward(x_batch)
        loss_batch = MSELoss().forward(out_batch, target_batch)
        loss_batch.backward()

        batch_grad = layer.weight.grad.copy()

        # Gradients should be similar (averaged, not 4x)
        ratio = np.linalg.norm(batch_grad) / np.linalg.norm(single_grad)
        assert 0.8 < ratio < 1.2, (
            f"Batch gradients not properly averaged!\n"
            f"  Single sample grad norm: {np.linalg.norm(single_grad)}\n"
            f"  Batch (4x same) grad norm: {np.linalg.norm(batch_grad)}\n"
            f"  Ratio: {ratio} (should be ~1.0, got {ratio:.2f})"
        )


# Quick smoke test for CI
@pytest.mark.quick
class TestQuickTrainingSmoke:
    """Fast tests for CI - just verify nothing crashes"""

    def test_simple_training_step(self):
        """One training step should not crash"""
        layer = Linear(2, 1)
        opt = SGD([layer.weight, layer.bias], lr=0.1)

        x = Tensor([[1., 2.]], requires_grad=True)
        y = Tensor([[1.]])

        out = layer.forward(x)
        loss = MSELoss().forward(out, y)
        loss.backward()
        opt.step()

        assert True  # If we got here, it works


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
