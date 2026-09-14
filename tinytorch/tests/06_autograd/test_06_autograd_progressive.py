"""
Module 06: Progressive Integration Tests
Tests that Module 06 (Autograd) works correctly AND that prior modules (01→05) still work.

DEPENDENCY CHAIN: 01_tensor → 02_activations → 03_layers → 04_losses → 05_dataloader → 06_autograd

⚠️ IMPORTANT: This test ONLY uses modules 01-06.
   Future modules (07_optimizers, 09_convolutions, 12_attention, etc.) are NOT tested here.

🎯 WHAT THIS TESTS:
- Module 06: Automatic differentiation, gradient computation, backward pass
- Integration: Autograd works with all prior modules (01-05)
- Regression: All previous modules still work correctly

💡 FOR STUDENTS: If tests fail, check:
1. Does Tensor support requires_grad=True?
2. Does backward() compute gradients correctly?
3. Do gradients accumulate properly?
4. Are computation graphs built during forward pass?
"""

import numpy as np
rng = np.random.default_rng(7)
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestAutogradCore:
    """
    🆕 NEW FUNCTIONALITY: Test Module 06 (Autograd) core implementation.

    Tests automatic differentiation capabilities.
    """

    def test_requires_grad_attribute(self):
        """
        ✅ TEST: Tensor supports requires_grad flag
        """
        try:
            from tinytorch.core.tensor import Tensor

            # Test creating tensor with requires_grad
            x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

            assert hasattr(x, 'requires_grad'), "Tensor missing requires_grad attribute"
            assert x.requires_grad == True, "requires_grad not set correctly"

            # Test default (no gradient tracking)
            y = Tensor([1.0, 2.0, 3.0])
            # Default should be False or tensor doesn't track by default

        except TypeError:
            # Tensor doesn't support requires_grad yet - that's what we're implementing
            raise
        except ImportError as e:
            assert False, f"Tensor import failed: {e}"

    def test_grad_attribute(self):
        """
        ✅ TEST: Tensor has grad attribute for storing gradients
        """
        try:
            from tinytorch.core.tensor import Tensor

            x = Tensor([1.0, 2.0], requires_grad=True)

            assert hasattr(x, 'grad'), "Tensor missing grad attribute"
            # Gradient should start as None before backward pass
            assert x.grad is None, "grad should be None before backward()"

        except TypeError:
            raise
        except ImportError as e:
            assert False, f"Tensor import failed: {e}"

    def test_backward_method(self):
        """
        ✅ TEST: Tensor has backward() method for gradient computation
        """
        try:
            from tinytorch.core.tensor import Tensor

            x = Tensor([2.0], requires_grad=True)

            assert hasattr(x, 'backward'), "Tensor missing backward() method"

            # Try calling backward
            try:
                x.backward(Tensor([1.0]))
                # If successful, gradient should be set
                assert x.grad is not None, "Required autograd capability is missing"
                assert x.grad.shape == x.shape, "Gradient shape mismatch"
            except (TypeError, ValueError):
                # Some implementations don't support backward on leaf tensors
                raise

        except TypeError:
            raise
        except ImportError as e:
            assert False, f"Tensor import failed: {e}"

    def test_simple_gradient(self):
        """
        ✅ TEST: Simple gradient computation y = x * 2
        """
        try:
            from tinytorch.core.tensor import Tensor

            x = Tensor([3.0], requires_grad=True)
            y = x * 2  # dy/dx = 2

            # y should also track gradients
            assert hasattr(y, 'requires_grad') and y.requires_grad, "Required autograd capability is missing"
            y.backward(Tensor([1.0]))

            assert x.grad is not None, "Required autograd capability is missing"
            expected = np.array([2.0])
            assert np.allclose(x.grad.data, expected), \
                f"Gradient wrong. Expected {expected}, got {x.grad.data}"

        except (TypeError, AttributeError):
            raise
        except ImportError as e:
            assert False, f"Import failed: {e}"

    def test_scalar_left_gradient(self):
        """
        ✅ TEST: Scalar arithmetic works naturally from either side.
        """
        try:
            from tinytorch.core.tensor import Tensor

            def grad_data(tensor):
                return tensor.grad.data if hasattr(tensor.grad, "data") else tensor.grad

            x = Tensor([3.0], requires_grad=True)
            y = 2 * x  # dy/dx = 2
            y.backward(Tensor([1.0]))
            assert x.grad is not None, "Required autograd capability is missing"
            assert np.allclose(grad_data(x), [2.0]), "Gradient wrong for scalar * tensor"

            x = Tensor([4.0], requires_grad=True)
            y = 10 - x  # dy/dx = -1
            y.backward(Tensor([1.0]))
            assert x.grad is not None, "Required autograd capability is missing"
            assert np.allclose(grad_data(x), [-1.0]), "Gradient wrong for scalar - tensor"

            x = Tensor([4.0], requires_grad=True)
            y = 12 / x  # dy/dx = -12 / x^2
            y.backward(Tensor([1.0]))
            assert x.grad is not None, "Required autograd capability is missing"
            assert np.allclose(grad_data(x), [-0.75]), "Gradient wrong for scalar / tensor"

        except (TypeError, AttributeError):
            raise
        except ImportError as e:
            assert False, f"Import failed: {e}"

    def test_chain_rule(self):
        """
        ✅ TEST: Chain rule: z = (x + y) * 2
        """
        try:
            from tinytorch.core.tensor import Tensor

            x = Tensor([1.0], requires_grad=True)
            y = Tensor([2.0], requires_grad=True)

            # z = (x + y) * 2
            # dz/dx = 2, dz/dy = 2
            sum_xy = x + y
            z = sum_xy * 2

            assert hasattr(z, 'backward'), "Required autograd capability is missing"
            try:
                z.backward(Tensor([1.0]))

                assert x.grad is not None and y.grad is not None, "Required autograd capability is missing"
                assert np.allclose(x.grad.data, [2.0]), "x gradient wrong"
                assert np.allclose(y.grad.data, [2.0]), "y gradient wrong"
            except (TypeError, ValueError):
                raise

        except (TypeError, AttributeError):
            raise
        except ImportError as e:
            assert False, f"Import failed: {e}"


class TestAutogradWithLayers:
    """
    🔗 INTEGRATION: Autograd + Layers (Module 03)

    Tests that gradients flow through neural network layers.
    """

    def test_linear_layer_gradients(self):
        """
        ✅ TEST: Gradients flow through Linear layer
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear

            # Create layer
            layer = Linear(4, 2)
            # Module 06 opts parameters into tracking explicitly; optimizers come next.
            for parameter in layer.parameters():
                parameter.requires_grad = True

            # Input with gradient tracking
            x = Tensor(rng.standard_normal((2, 4)), requires_grad=True)

            # Forward pass
            output = layer(x)

            # Backward pass
            assert hasattr(output, 'backward'), "Required autograd capability is missing"
            try:
                output.backward(Tensor(np.ones(output.shape)))

                # Check input gradient
                assert x.grad is not None, "Required autograd capability is missing"
                assert x.grad.shape == x.shape, "Input gradient shape wrong"

                # Check layer parameter gradients
                assert hasattr(layer, 'weight') and layer.weight.grad is not None, "Required autograd capability is missing"
                assert layer.weight.grad.shape == layer.weight.shape, \
                    "Weight gradient shape wrong"
            except (TypeError, ValueError, AttributeError):
                raise

        except TypeError:
            raise
        except ImportError as e:
            assert False, f"Import failed: {e}"

    def test_activation_gradients(self):
        """
        ✅ TEST: Gradients flow through activation functions
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.activations import ReLU, Sigmoid

            # Test ReLU gradient
            x = Tensor(np.array([-1.0, 0.0, 1.0, 2.0]), requires_grad=True)
            relu = ReLU()

            y = relu(x)

            assert hasattr(y, 'backward'), "Required autograd capability is missing"
            try:
                y.backward(Tensor(np.ones(y.shape)))

                assert x.grad is not None, "Required autograd capability is missing"
                # ReLU gradient: 0 for x<0, 1 for x>0
                expected = np.array([0.0, 0.0, 1.0, 1.0])
                # Allow some flexibility in gradient at x=0
                assert x.grad.data[0] == 0.0, "ReLU grad wrong for negative"
                assert x.grad.data[3] == 1.0, "ReLU grad wrong for positive"
            except (TypeError, ValueError, AttributeError):
                raise

        except TypeError:
            raise
        except ImportError as e:
            assert False, f"Import failed: {e}"

    def test_tanh_activation_gradient(self):
        """
        ✅ TEST: Tanh propagates gradients (regression test for #1341)

        Tanh was previously missing from the set of operations with a backward,
        so tanh(x).backward() silently failed to populate x.grad.
        """
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.activations import Tanh

        # tanh'(0) = 1 - tanh(0)² = 1 - 0 = 1
        # tanh'(1) = 1 - tanh(1)² ≈ 1 - 0.7616² ≈ 0.4200
        x = Tensor(np.array([0.0, 1.0, -1.0]), requires_grad=True)
        tanh = Tanh()

        y = tanh(x).sum()
        y.backward(np.ones_like(y.data))

        assert x.grad is not None, "Tanh did not produce a gradient"
        np.testing.assert_allclose(x.grad, [1.0, 0.4199743, 0.4199743], rtol=1e-5)


class TestAutogradWithLosses:
    """
    🔗 INTEGRATION: Autograd + Losses (Module 04)

    Tests that gradients flow from loss functions.
    """

    def test_mse_loss_gradient(self):
        """
        ✅ TEST: Gradients from MSE loss
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.losses import MSELoss

            pred = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
            target = Tensor(np.array([1.5, 2.0, 2.5]))

            loss_fn = MSELoss()
            loss = loss_fn(pred, target)

            assert hasattr(loss, 'backward'), "Required autograd capability is missing"
            try:
                loss.backward()

                assert pred.grad is not None, "Required autograd capability is missing"
                # MSE gradient: 2*(pred - target)/n
                assert pred.grad.shape == pred.shape, "Loss gradient shape wrong"
            except (TypeError, ValueError, AttributeError):
                raise

        except TypeError:
            raise
        except ImportError as e:
            assert False, f"Import failed: {e}"


class TestAutogradWithDataLoader:
    """
    🔗 INTEGRATION: Autograd + DataLoader (Module 05)

    Tests that autograd works with data loading pipeline.
    """

    def test_batch_gradients(self):
        """
        ✅ TEST: Gradients work with batched data from DataLoader
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.dataloader import TensorDataset, DataLoader
            from tinytorch.core.layers import Linear
            from tinytorch.core.losses import MSELoss

            # Create dataset
            data = Tensor(rng.standard_normal((20, 4)))
            targets = Tensor(rng.standard_normal((20, 2)))
            dataset = TensorDataset(data, targets)
            dataloader = DataLoader(dataset, batch_size=4)

            # Create model
            layer = Linear(4, 2)
            # Module 06 opts parameters into tracking explicitly; optimizers come next.
            for parameter in layer.parameters():
                parameter.requires_grad = True
            loss_fn = MSELoss()

            # Test gradient computation with batches
            for batch_x, batch_y in dataloader:
                # Forward pass
                output = layer(batch_x)
                loss = loss_fn(output, batch_y)

                # Backward pass
                assert hasattr(loss, 'backward'), "Required autograd capability is missing"
                try:
                    loss.backward()

                    # Check layer has gradients
                    assert hasattr(layer, 'weight') and layer.weight.grad is not None, "Required autograd capability is missing"
                    assert layer.weight.grad.shape == layer.weight.shape, \
                        "Batch gradient shape wrong"
                except (TypeError, ValueError, AttributeError):
                    raise

                break  # Test one batch

        except TypeError:
            raise
        except ImportError as e:
            assert False, f"Import failed: {e}"


class TestRegressionPrevention:
    """
    🔄 REGRESSION: Verify all previous modules (01-05) still work correctly.
    """

    def test_tensor_still_works(self):
        """
        ✅ TEST: Module 01 (Tensor) still works
        """
        try:
            from tinytorch.core.tensor import Tensor

            a = Tensor([1.0, 2.0, 3.0])
            b = Tensor([4.0, 5.0, 6.0])

            c = a + b
            assert np.allclose(c.data, [5.0, 7.0, 9.0]), "Tensor addition broken"

            d = a * b
            assert np.allclose(d.data, [4.0, 10.0, 18.0]), "Tensor multiplication broken"

        except Exception as e:
            assert False, f"Module 01 regression: {e}"

    def test_activations_still_work(self):
        """
        ✅ TEST: Module 02 (Activations) still works
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.activations import ReLU, Sigmoid

            x = Tensor([-1.0, 0.0, 1.0])

            relu = ReLU()
            r = relu(x)
            assert r.data[0] == 0.0, "ReLU broken"

            sigmoid = Sigmoid()
            s = sigmoid(x)
            assert 0 < s.data[2] < 1, "Sigmoid broken"

        except Exception as e:
            assert False, f"Module 02 regression: {e}"

    def test_layers_still_work(self):
        """
        ✅ TEST: Module 03 (Layers) still works
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear

            layer = Linear(5, 3)
            # Module 06 opts parameters into tracking explicitly; optimizers come next.
            for parameter in layer.parameters():
                parameter.requires_grad = True
            x = Tensor(rng.standard_normal((2, 5)))
            output = layer(x)

            assert output.shape == (2, 3), f"Linear broken: {output.shape}"

        except Exception as e:
            assert False, f"Module 03 regression: {e}"

    def test_losses_still_work(self):
        """
        ✅ TEST: Module 04 (Losses) still works
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.losses import MSELoss

            pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
            target = Tensor([[1.5, 2.5], [3.5, 4.5]])

            mse = MSELoss()
            loss = mse(pred, target)

            assert loss.data.size == 1, "MSE loss broken"

        except Exception as e:
            assert False, f"Module 04 regression: {e}"

    def test_dataloader_still_works(self):
        """
        ✅ TEST: Module 05 (DataLoader) still works
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.dataloader import TensorDataset, DataLoader

            data = Tensor(rng.standard_normal((10, 4)))
            targets = Tensor(np.arange(10).astype(float))

            dataset = TensorDataset(data, targets)
            dataloader = DataLoader(dataset, batch_size=2)

            batch_count = sum(1 for _ in dataloader)
            assert batch_count == 5, "DataLoader broken"

        except Exception as e:
            assert False, f"Module 05 regression: {e}"


class TestModule06Completion:
    """
    ✅ COMPLETION CHECK: Module 06 ready for next module (Optimizers).
    """

    def test_autograd_foundation_complete(self):
        """
        ✅ FINAL TEST: Autograd foundation ready for optimizers

        🎯 SUCCESS = Ready for Module 07: Optimizers!
        """
        capabilities = {
            "requires_grad attribute": False,
            "grad attribute": False,
            "backward method": False,
            "gradient computation": False,
            "layer integration": False,
        }

        try:
            from tinytorch.core.tensor import Tensor

            # Test 1: requires_grad
            try:
                x = Tensor([1.0], requires_grad=True)
                assert hasattr(x, 'requires_grad') and x.requires_grad, "Required autograd capability is missing"
                capabilities["requires_grad attribute"] = True
            except TypeError:
                raise

            # Test 2: grad attribute
            try:
                x = Tensor([1.0], requires_grad=True)
                assert hasattr(x, 'grad'), "Required autograd capability is missing"
                capabilities["grad attribute"] = True
            except TypeError:
                raise

            # Test 3: backward method
            try:
                x = Tensor([1.0], requires_grad=True)
                assert hasattr(x, 'backward'), "Required autograd capability is missing"
                capabilities["backward method"] = True
            except TypeError:
                raise

            # Test 4: gradient computation
            try:
                x = Tensor([2.0], requires_grad=True)
                y = x * 3
                assert hasattr(y, 'backward'), "Required autograd capability is missing"
                y.backward(Tensor([1.0]))
                assert x.grad is not None, "Required autograd capability is missing"
                capabilities["gradient computation"] = True
            except (TypeError, AttributeError):
                raise

            # Test 5: layer integration
            try:
                from tinytorch.core.layers import Linear
                layer = Linear(2, 1)
                # Module 06 opts parameters into tracking explicitly; optimizers come next.
                for parameter in layer.parameters():
                    parameter.requires_grad = True
                x = Tensor(rng.standard_normal((1, 2)), requires_grad=True)
                out = layer(x)
                assert hasattr(out, 'backward'), "Required autograd capability is missing"
                out.backward(Tensor([[1.0]]))
                assert layer.weight.grad is not None, "Required autograd capability is missing"
                capabilities["layer integration"] = True
            except (TypeError, AttributeError, ImportError):
                raise

            # Report progress
            completed = sum(capabilities.values())
            total = len(capabilities)

            if completed < total:
                progress = "\n".join(
                    f"  {'✅' if v else '❌'} {k}"
                    for k, v in capabilities.items()
                )
                print(f"\nAutograd Progress ({completed}/{total}):\n{progress}")

            # For now, pass if at least basic structure exists
            assert capabilities["requires_grad attribute"] or completed >= 2, \
                f"Autograd not ready: {capabilities}"

        except ImportError as e:
            assert False, f"Module 06 import failed: {e}"
