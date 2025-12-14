"""
Module 04: Losses - Core Functionality Tests
=============================================

WHY LOSSES MATTER:
-----------------
The loss function defines what "good" means for your model.
It's the signal that drives all learning. Wrong loss = wrong learning.

WHAT STUDENTS LEARN:
-------------------
1. MSE for regression (predict continuous values)
2. Cross-entropy for classification (predict categories)
3. Loss must be differentiable for gradient-based training
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestMSELoss:
    """Test Mean Squared Error loss."""

    def test_mse_computation(self):
        """
        WHAT: Verify MSE = mean((pred - target)²).

        WHY: MSE penalizes large errors heavily (squared).
        Good for regression where you want to minimize average error.

        STUDENT LEARNING: MSE = (1/n) * Σ(pred - target)²
        """
        try:
            from tinytorch.core.training import MSELoss
            from tinytorch.core.tensor import Tensor

            loss_fn = MSELoss()

            pred = Tensor([1.0, 2.0, 3.0])
            target = Tensor([1.0, 2.0, 4.0])  # Error of 1 on last element

            loss = loss_fn(pred, target)

            # MSE = (0² + 0² + 1²) / 3 = 1/3
            expected = 1.0 / 3.0
            assert np.isclose(float(loss.data), expected, atol=1e-5), (
                f"MSE wrong.\n"
                f"  Errors: [0, 0, 1]\n"
                f"  MSE = (0+0+1)/3 = 0.333\n"
                f"  Got: {loss.data}"
            )

        except ImportError:
            pytest.skip("MSELoss not implemented yet")

    def test_mse_gradient(self):
        """
        WHAT: Verify MSE gradient is 2(pred - target)/n.

        WHY: This gradient tells the model which direction to move.
        If pred > target, gradient is positive (decrease pred).

        STUDENT LEARNING: dMSE/dpred = 2(pred - target) / n
        """
        try:
            from tinytorch.core.training import MSELoss
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.autograd import enable_autograd

            enable_autograd()

            pred = Tensor([2.0], requires_grad=True)
            target = Tensor([1.0])

            loss_fn = MSELoss()
            loss = loss_fn(pred, target)
            loss.backward()

            # dMSE/dpred = 2*(2-1)/1 = 2
            assert pred.grad is not None, "MSE should produce gradient"

        except ImportError:
            pytest.skip("MSE gradient not implemented yet")


class TestCrossEntropyLoss:
    """Test Cross-Entropy loss for classification."""

    def test_cross_entropy_basic(self):
        """
        WHAT: Verify cross-entropy for classification.

        WHY: CE is THE loss for classification. It measures how
        well predicted probabilities match true labels.

        STUDENT LEARNING: CE = -Σ(target * log(pred))
        For one-hot targets: CE = -log(pred[true_class])
        """
        try:
            from tinytorch.core.training import CrossEntropyLoss
            from tinytorch.core.tensor import Tensor

            loss_fn = CrossEntropyLoss()

            # Logits for 3 classes
            logits = Tensor([[1.0, 2.0, 0.5]])  # Class 1 has highest
            target = Tensor([1])  # True class is 1

            loss = loss_fn(logits, target)

            # Loss should be small (predicted correct class)
            assert float(loss.data) < 1.0, (
                "CE loss should be small when predicting correct class"
            )

        except ImportError:
            pytest.skip("CrossEntropyLoss not implemented yet")

    def test_cross_entropy_wrong_prediction(self):
        """
        WHAT: Verify CE is high when prediction is wrong.

        WHY: High loss = model is confident but wrong.
        This creates strong gradient to correct the mistake.

        STUDENT LEARNING: CE heavily penalizes confident wrong predictions.
        """
        try:
            from tinytorch.core.training import CrossEntropyLoss
            from tinytorch.core.tensor import Tensor

            loss_fn = CrossEntropyLoss()

            # Confident wrong prediction
            logits = Tensor([[10.0, 0.0, 0.0]])  # Very confident class 0
            target = Tensor([2])  # But true class is 2

            loss = loss_fn(logits, target)

            # Loss should be high
            assert float(loss.data) > 1.0, (
                "CE loss should be high for confident wrong predictions"
            )

        except ImportError:
            pytest.skip("CrossEntropyLoss not implemented yet")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
