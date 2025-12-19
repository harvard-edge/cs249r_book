"""
Module 07: Optimizer Core Tests
================================

These tests verify that optimizers correctly update model parameters.

WHY THESE TESTS MATTER:
-----------------------
Optimizers are the "learning" part of machine learning. If they don't work:
- Weights never change → model never learns
- Weights explode → training diverges
- Weights update incorrectly → model learns wrong things

WHAT WE TEST:
-------------
1. SGD actually modifies weights after step()
2. Adam maintains momentum correctly
3. Learning rate affects update magnitude
4. zero_grad() properly clears gradients

CONNECTION TO OTHER MODULES:
----------------------------
- Uses Tensor (Module 01) - optimizers update tensor.data
- Uses autograd (Module 06) - optimizers read tensor.grad
- Enables Training (Module 08) - optimizers make learning possible
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.optimizers import SGD, Adam
from tinytorch.core.autograd import enable_autograd

enable_autograd()


class TestSGDBasics:
    """
    Test SGD (Stochastic Gradient Descent) optimizer.

    SGD is the simplest optimizer: weight = weight - lr * gradient

    If SGD doesn't work, nothing else will - it's the foundation.
    """

    def test_sgd_updates_weights(self):
        """
        WHAT: Verify SGD.step() actually changes parameter values.

        WHY: The most basic requirement - if step() doesn't change weights,
        the model can never learn anything.

        HOW: Create parameter, set gradient, call step(), check weight changed.
        """
        # Create a simple parameter
        param = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        initial_values = param.data.copy()

        # Set up optimizer
        optimizer = SGD([param], lr=0.1)

        # Simulate gradient (as if from backward pass)
        param.grad = np.array([1.0, 1.0, 1.0])

        # Update weights
        optimizer.step()

        # Weights MUST be different now
        assert not np.allclose(param.data, initial_values), (
            "SGD.step() did not change weights!\n"
            f"  Before: {initial_values}\n"
            f"  After:  {param.data}\n"
            f"  Gradient: {param.grad}\n"
            "This means the model cannot learn."
        )

    def test_sgd_update_direction(self):
        """
        WHAT: Verify SGD moves weights in the correct direction (opposite to gradient).

        WHY: Gradient descent DESCENDS - it moves AGAINST the gradient.
        If we move WITH the gradient, we'd maximize loss, not minimize it.

        MATH: weight_new = weight_old - lr * gradient
        So if gradient is positive, weight should DECREASE.
        """
        param = Tensor([10.0], requires_grad=True)
        optimizer = SGD([param], lr=1.0)  # lr=1 for easy math

        # Positive gradient means "increasing this weight increases loss"
        param.grad = np.array([2.0])

        optimizer.step()

        # Weight should DECREASE (10 - 1.0 * 2.0 = 8.0)
        expected = 8.0
        assert np.isclose(param.data[0], expected), (
            f"SGD moved in wrong direction!\n"
            f"  Initial: 10.0, Gradient: 2.0, LR: 1.0\n"
            f"  Expected: {expected} (10 - 1*2)\n"
            f"  Got: {param.data[0]}\n"
            "Gradient descent should move OPPOSITE to gradient."
        )

    def test_sgd_learning_rate_scales_update(self):
        """
        WHAT: Verify learning rate controls the size of weight updates.

        WHY: Learning rate is the most important hyperparameter.
        - Too high → training explodes
        - Too low → training takes forever
        - Just right → smooth convergence
        """
        # Same initial state, same gradient, different learning rates
        param_slow = Tensor([10.0], requires_grad=True)
        param_fast = Tensor([10.0], requires_grad=True)

        sgd_slow = SGD([param_slow], lr=0.01)
        sgd_fast = SGD([param_fast], lr=1.0)

        # Same gradient
        param_slow.grad = np.array([1.0])
        param_fast.grad = np.array([1.0])

        sgd_slow.step()
        sgd_fast.step()

        # Fast should move 100x more than slow
        slow_change = abs(10.0 - param_slow.data[0])
        fast_change = abs(10.0 - param_fast.data[0])

        assert fast_change > slow_change * 50, (
            "Learning rate doesn't properly scale updates!\n"
            f"  lr=0.01 moved by: {slow_change}\n"
            f"  lr=1.0 moved by: {fast_change}\n"
            "The fast optimizer should move ~100x more."
        )


class TestAdamBasics:
    """
    Test Adam optimizer (Adaptive Moment Estimation).

    Adam is smarter than SGD - it maintains running averages of gradients
    and adapts learning rate per-parameter. Most modern training uses Adam.
    """

    def test_adam_updates_weights(self):
        """
        WHAT: Verify Adam.step() changes parameter values.

        WHY: Same as SGD - no update = no learning.
        """
        param = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        initial_values = param.data.copy()

        optimizer = Adam([param], lr=0.1)
        param.grad = np.array([1.0, 1.0, 1.0])

        optimizer.step()

        assert not np.allclose(param.data, initial_values), (
            "Adam.step() did not change weights!"
        )

    def test_adam_momentum_accumulates(self):
        """
        WHAT: Verify Adam's momentum builds up over multiple steps.

        WHY: Adam maintains exponential moving averages of gradients.
        With consistent gradient direction, updates should accelerate.
        This is why Adam often converges faster than SGD.
        """
        param = Tensor([0.0], requires_grad=True)
        optimizer = Adam([param], lr=0.1)

        # Apply same gradient 5 times
        for i in range(5):
            param.grad = np.array([1.0])
            optimizer.step()

        position_after_5 = param.data[0]

        # Continue for 5 more
        for i in range(5):
            param.grad = np.array([1.0])
            optimizer.step()

        position_after_10 = param.data[0]

        # Momentum should cause acceleration - later steps move more
        first_5_distance = abs(position_after_5 - 0.0)
        second_5_distance = abs(position_after_10 - position_after_5)

        # Second batch should move at least as much (momentum building)
        assert second_5_distance >= first_5_distance * 0.8, (
            "Adam momentum doesn't appear to be working!\n"
            f"  First 5 steps moved: {first_5_distance}\n"
            f"  Second 5 steps moved: {second_5_distance}\n"
            "With consistent gradients, momentum should help later steps."
        )


class TestZeroGrad:
    """
    Test gradient clearing functionality.

    WHY THIS MATTERS: Gradients accumulate by default. Without zero_grad():
    - Batch 1 gradients + Batch 2 gradients = wrong update
    - Memory grows unbounded
    - Training produces garbage
    """

    def test_zero_grad_clears_gradients(self):
        """
        WHAT: Verify zero_grad() sets all gradients to zero/None.

        WHY: Each training iteration should start fresh.
        """
        param = Tensor([1.0, 2.0], requires_grad=True)
        optimizer = SGD([param], lr=0.1)

        # Simulate a backward pass
        param.grad = np.array([5.0, 10.0])

        # Clear gradients
        optimizer.zero_grad()

        # Gradients should be cleared
        assert param.grad is None or np.allclose(param.grad, 0), (
            "zero_grad() did not clear gradients!\n"
            f"  Gradient after zero_grad: {param.grad}\n"
            "This will cause gradient accumulation bugs in training."
        )


class TestMultipleParameters:
    """
    Test optimizers with multiple parameters (like real models).

    Real models have many parameters (weights, biases, etc.).
    Optimizer must update ALL of them correctly.
    """

    def test_optimizer_updates_all_parameters(self):
        """
        WHAT: Verify optimizer updates every parameter, not just the first.

        WHY: A bug that only updates some parameters would cause
        parts of the model to never learn.
        """
        # Simulate a 2-layer network's parameters
        weights1 = Tensor(np.random.randn(3, 2), requires_grad=True)
        bias1 = Tensor(np.zeros(2), requires_grad=True)
        weights2 = Tensor(np.random.randn(2, 1), requires_grad=True)
        bias2 = Tensor(np.zeros(1), requires_grad=True)

        params = [weights1, bias1, weights2, bias2]
        initial_values = [p.data.copy() for p in params]

        optimizer = SGD(params, lr=0.1)

        # Set gradients for all
        for p in params:
            p.grad = np.ones_like(p.data)

        optimizer.step()

        # ALL parameters must have changed
        for i, (param, initial) in enumerate(zip(params, initial_values)):
            assert not np.allclose(param.data, initial), (
                f"Parameter {i} was not updated!\n"
                f"  Before: {initial}\n"
                f"  After: {param.data}\n"
                "Optimizer must update ALL parameters."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
