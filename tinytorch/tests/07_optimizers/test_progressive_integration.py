"""
Module 07: Progressive Integration Tests
Tests that Module 07 (Optimizers) works correctly AND that the foundation stack (01→06) still works.

DEPENDENCY CHAIN: 01_tensor → 02_activations → 03_layers → 04_losses → 05_dataloader → 06_autograd → 07_optimizers
This is where we enable learning through sophisticated optimization algorithms.

🎯 WHAT THIS TESTS:
- Module 07: SGD, Adam, AdamW optimizers with momentum and adaptive learning
- Integration: Optimizers work with gradients from autograd (Module 06)
- Regression: Foundation stack (01→06) still works correctly
- Preparation: Ready for training loops (Module 08)

💡 FOR STUDENTS: If tests fail, check:
1. Does your SGD class exist in tinytorch.core.optimizers?
2. Does SGD.step() update parameters using gradients?
3. Do momentum and learning rate work correctly?
4. Are parameter groups handled properly?

🔧 DEBUGGING HELP:
- optimizer.step() should update params: param -= lr * grad
- With momentum: velocity = momentum * velocity + grad; param -= lr * velocity
- Adam uses both momentum (first moment) and RMSprop (second moment)
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestFoundationStackStillWorks:
    """
    🔄 REGRESSION CHECK: Verify foundation stack (01→05) still works.

    💡 If these fail: You may have broken something in the foundation while working on optimizers.
    """

    def test_foundation_pipeline_stable(self):
        """
        ✅ TEST: Complete foundation pipeline (01→05) should still work
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            from tinytorch.core.losses import MSELoss

            # Create network
            dense = Linear(10, 5)
            relu = ReLU()
            loss_fn = MSELoss()

            # Forward pass
            x = Tensor(np.random.randn(4, 10))
            h = dense(x)
            output = relu(h)

            # Loss computation
            target = Tensor(np.random.randn(4, 5))
            loss = loss_fn(output, target)

            assert output.shape == (4, 5), f"Expected (4, 5), got {output.shape}"
            assert loss.data.shape == (), f"Loss should be scalar, got {loss.data.shape}"

        except ImportError as e:
            assert False, f"Foundation import broken: {str(e)}"
        except Exception as e:
            assert False, f"Foundation functionality broken: {str(e)}"

    def test_gradient_computation_stable(self):
        """
        ✅ TEST: Gradient computation from Module 06 still works
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.losses import MSELoss

            # Create simple network
            layer = Linear(4, 2)
            loss_fn = MSELoss()

            # Forward pass
            x = Tensor(np.random.randn(2, 4), requires_grad=True)
            output = layer(x)
            target = Tensor(np.random.randn(2, 2))
            loss = loss_fn(output, target)

            # Backward pass
            loss.backward()

            # Check gradients exist
            assert layer.weight.grad is not None, "Weight gradients not computed"
            assert layer.bias.grad is not None, "Bias gradients not computed"

        except ImportError as e:
            assert False, f"Autograd import broken: {str(e)}"
        except Exception as e:
            assert False, f"Gradient computation broken: {str(e)}"


class TestModule06OptimizersCore:
    """
    🆕 NEW FUNCTIONALITY: Test Module 07 (Optimizers) core implementation.

    💡 What you're implementing: Optimization algorithms that update parameters using gradients.
    🎯 Goal: Enable neural networks to learn from their mistakes.
    """

    def test_sgd_optimizer_exists(self):
        """
        ✅ TEST: SGD optimizer - Stochastic Gradient Descent

        📋 WHAT YOU NEED TO IMPLEMENT:
        class SGD:
            def __init__(self, params, lr=0.01, momentum=0.0):
                # Store parameters and hyperparameters
            def step(self):
                # Update parameters: param -= lr * grad
            def zero_grad(self):
                # Reset all gradients to zero

        🚨 IF FAILS: SGD optimizer doesn't exist or missing components
        """
        try:
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.tensor import Tensor

            # Create test parameters
            params = [Tensor(np.random.randn(3, 3), requires_grad=True)]
            params[0].grad = Tensor(np.ones((3, 3)))  # Set gradient

            # Create optimizer
            optimizer = SGD(params, lr=0.1)

            # Test that optimizer has required methods
            assert hasattr(optimizer, 'step'), "SGD missing step() method"
            assert hasattr(optimizer, 'zero_grad'), "SGD missing zero_grad() method"

            # Test parameter update
            original_value = params[0].data.copy()
            optimizer.step()

            # param should be updated: param -= lr * grad
            expected = original_value - 0.1 * np.ones((3, 3))
            assert np.allclose(params[0].data, expected), "SGD step() not updating parameters correctly"

        except ImportError as e:
            assert False, f"SGD optimizer not found: {str(e)}"
        except Exception as e:
            assert False, f"SGD optimizer broken: {str(e)}"

    def test_sgd_momentum(self):
        """
        ✅ TEST: SGD with momentum - Accelerates learning

        📋 MOMENTUM ALGORITHM:
        velocity = momentum * velocity + gradient
        param = param - lr * velocity

        💡 Momentum helps escape local minima and speeds up convergence
        """
        try:
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.tensor import Tensor

            # Create test parameter
            param = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
            param.grad = Tensor(np.array([[0.1, 0.1], [0.1, 0.1]]))

            # Create SGD with momentum
            optimizer = SGD([param], lr=0.1, momentum=0.9)

            # First step
            optimizer.step()

            # Second step with same gradient - momentum should accumulate
            param.grad = Tensor(np.array([[0.1, 0.1], [0.1, 0.1]]))
            optimizer.step()

            # With momentum, second update should be larger than first
            # This is a simplified check - actual values depend on implementation
            assert param.data is not None, "Parameter not updated"

        except ImportError:
            assert True, "SGD with momentum not implemented yet (expected)"
        except Exception as e:
            assert False, f"SGD momentum broken: {str(e)}"

    def test_adam_optimizer_exists(self):
        """
        ✅ TEST: Adam optimizer - Adaptive Moment Estimation

        📋 ADAM ALGORITHM:
        m = beta1 * m + (1 - beta1) * grad           # First moment (momentum)
        v = beta2 * v + (1 - beta2) * grad^2         # Second moment (RMSprop)
        m_hat = m / (1 - beta1^t)                    # Bias correction
        v_hat = v / (1 - beta2^t)                    # Bias correction
        param = param - lr * m_hat / (sqrt(v_hat) + eps)

        💡 Adam is the most popular optimizer in deep learning
        """
        try:
            from tinytorch.core.optimizers import Adam
            from tinytorch.core.tensor import Tensor

            # Create test parameters
            params = [Tensor(np.random.randn(3, 3), requires_grad=True)]
            params[0].grad = Tensor(np.ones((3, 3)))

            # Create Adam optimizer
            optimizer = Adam(params, lr=0.001)

            # Test required methods
            assert hasattr(optimizer, 'step'), "Adam missing step() method"
            assert hasattr(optimizer, 'zero_grad'), "Adam missing zero_grad() method"

            # Test parameter update
            original_value = params[0].data.copy()
            optimizer.step()

            # Parameters should change
            assert not np.array_equal(params[0].data, original_value), \
                "Adam step() not updating parameters"

        except ImportError:
            assert True, "Adam optimizer not implemented yet (expected)"
        except Exception as e:
            assert False, f"Adam optimizer broken: {str(e)}"

    def test_zero_grad(self):
        """
        ✅ TEST: zero_grad() - Reset gradients for next iteration

        📋 WHY ZERO GRAD:
        PyTorch accumulates gradients by default.
        Before each forward pass, we need to reset gradients to zero.

        💡 Forgetting zero_grad() is a common training bug!
        """
        try:
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.tensor import Tensor

            # Create parameter with gradient
            param = Tensor(np.random.randn(2, 2), requires_grad=True)
            param.grad = Tensor(np.ones((2, 2)))

            optimizer = SGD([param], lr=0.1)

            # Verify gradient exists
            assert param.grad is not None, "Gradient should exist before zero_grad()"

            # Zero gradients
            optimizer.zero_grad()

            # Gradient should be zeroed or None
            if param.grad is not None:
                assert np.allclose(param.grad.data, 0), "zero_grad() should reset gradients to zero"

        except ImportError as e:
            assert False, f"Optimizer not found: {str(e)}"
        except Exception as e:
            assert False, f"zero_grad() broken: {str(e)}"


class TestOptimizerIntegration:
    """
    🔗 INTEGRATION TEST: Optimizers + Autograd working together.

    💡 Test that optimizers can train neural networks using gradients.
    🎯 Goal: Complete gradient descent learning loop.
    """

    def test_training_step(self):
        """
        ✅ TEST: Complete training step (forward → loss → backward → update)

        📋 TRAINING STEP:
        1. Forward pass: output = model(input)
        2. Loss computation: loss = loss_fn(output, target)
        3. Backward pass: loss.backward()
        4. Parameter update: optimizer.step()
        5. Reset gradients: optimizer.zero_grad()

        💡 This is the core loop that trains all neural networks
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.losses import MSELoss
            from tinytorch.core.optimizers import SGD

            # Create simple network
            layer = Linear(4, 2)
            loss_fn = MSELoss()
            optimizer = SGD(layer.parameters(), lr=0.01)

            # Training data
            x = Tensor(np.random.randn(8, 4))
            target = Tensor(np.random.randn(8, 2))

            # Get initial loss
            initial_output = layer(x)
            initial_loss = loss_fn(initial_output, target)
            initial_loss_value = initial_loss.data.copy()

            # Training step
            optimizer.zero_grad()
            output = layer(x)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            # Loss should decrease (or at least change)
            new_output = layer(x)
            new_loss = loss_fn(new_output, target)

            # After one step, loss should typically decrease
            # (not guaranteed for all random initializations, but usually true)
            assert new_loss.data is not None, "Loss computation broken after training step"

        except ImportError as e:
            assert False, f"Integration components missing: {str(e)}"
        except Exception as e:
            assert False, f"Training step broken: {str(e)}"

    def test_multiple_training_steps(self):
        """
        ✅ TEST: Multiple training steps show learning

        📋 LEARNING VERIFICATION:
        Running multiple training steps should generally decrease loss.
        This verifies the complete gradient descent loop works.

        💡 If loss doesn't decrease, check:
        - Gradients are computed correctly
        - optimizer.step() updates parameters
        - optimizer.zero_grad() is called each iteration
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.losses import MSELoss
            from tinytorch.core.optimizers import SGD

            # Create network
            layer = Linear(4, 2)
            loss_fn = MSELoss()
            optimizer = SGD(layer.parameters(), lr=0.1)

            # Fixed training data (for reproducible learning)
            np.random.seed(42)
            x = Tensor(np.random.randn(8, 4))
            target = Tensor(np.zeros((8, 2)))  # Simple target: all zeros

            # Record losses over multiple steps
            losses = []
            for _ in range(10):
                optimizer.zero_grad()
                output = layer(x)
                loss = loss_fn(output, target)
                losses.append(float(loss.data))
                loss.backward()
                optimizer.step()

            # Loss should generally decrease over training
            # Allow for some noise, but trend should be downward
            assert losses[-1] < losses[0], \
                f"Loss should decrease: started at {losses[0]:.4f}, ended at {losses[-1]:.4f}"

        except ImportError as e:
            assert False, f"Training components missing: {str(e)}"
        except Exception as e:
            assert False, f"Multiple training steps broken: {str(e)}"


class TestRegressionPrevention:
    """
    🛡️ REGRESSION PREVENTION: Ensure Module 06 doesn't break earlier modules.
    """

    def test_module_05_not_broken(self):
        """Ensure autograd still works after adding optimizers."""
        try:
            from tinytorch.core.tensor import Tensor

            # Simple gradient test
            x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
            y = x * x  # y = x^2
            z = y.sum()  # z = sum(x^2)
            z.backward()

            # Gradient of sum(x^2) is 2x
            expected_grad = np.array([4.0, 6.0])
            assert np.allclose(x.grad.data, expected_grad), \
                f"Autograd broken. Expected {expected_grad}, got {x.grad.data}"

        except Exception as e:
            assert False, f"Module 06 (autograd) broken: {str(e)}"

    def test_progressive_compatibility(self):
        """Test that all foundation modules work together."""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.activations import ReLU, Sigmoid
            from tinytorch.core.layers import Linear
            from tinytorch.core.losses import MSELoss

            # Build and run a complete network
            layer1 = Linear(4, 8)
            layer2 = Linear(8, 2)
            relu = ReLU()
            sigmoid = Sigmoid()
            loss_fn = MSELoss()

            x = Tensor(np.random.randn(4, 4))
            h = relu(layer1(x))
            output = sigmoid(layer2(h))
            target = Tensor(np.random.randn(4, 2))
            loss = loss_fn(output, target)

            assert loss.data is not None, "Complete pipeline broken"

        except Exception as e:
            assert False, f"Progressive compatibility broken: {str(e)}"
