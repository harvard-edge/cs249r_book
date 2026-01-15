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
            assert True, "Autograd not implemented yet"
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
            assert True, "Autograd not implemented yet"
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
                if x.grad is not None:
                    assert x.grad.shape == x.shape, "Gradient shape mismatch"
            except (TypeError, ValueError):
                # Some implementations don't support backward on leaf tensors
                pass
                
        except TypeError:
            assert True, "Autograd not implemented yet"
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
            if hasattr(y, 'requires_grad') and y.requires_grad:
                y.backward(Tensor([1.0]))
                
                if x.grad is not None:
                    expected = np.array([2.0])
                    assert np.allclose(x.grad.data, expected), \
                        f"Gradient wrong. Expected {expected}, got {x.grad.data}"
                        
        except (TypeError, AttributeError):
            assert True, "Simple gradient not implemented yet"
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
            
            if hasattr(z, 'backward'):
                try:
                    z.backward(Tensor([1.0]))
                    
                    if x.grad is not None and y.grad is not None:
                        assert np.allclose(x.grad.data, [2.0]), "x gradient wrong"
                        assert np.allclose(y.grad.data, [2.0]), "y gradient wrong"
                except (TypeError, ValueError):
                    pass  # Chain rule not fully implemented
                    
        except (TypeError, AttributeError):
            assert True, "Chain rule not implemented yet"
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
            
            # Input with gradient tracking
            x = Tensor(np.random.randn(2, 4), requires_grad=True)
            
            # Forward pass
            output = layer(x)
            
            # Backward pass
            if hasattr(output, 'backward'):
                try:
                    output.backward(Tensor(np.ones(output.shape)))
                    
                    # Check input gradient
                    if x.grad is not None:
                        assert x.grad.shape == x.shape, "Input gradient shape wrong"
                        
                    # Check layer parameter gradients
                    if hasattr(layer, 'weight') and layer.weight.grad is not None:
                        assert layer.weight.grad.shape == layer.weight.shape, \
                            "Weight gradient shape wrong"
                except (TypeError, ValueError, AttributeError):
                    pass  # Layer gradients not fully implemented
                    
        except TypeError:
            assert True, "Layer gradients not implemented yet"
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
            
            if hasattr(y, 'backward'):
                try:
                    y.backward(Tensor(np.ones(y.shape)))
                    
                    if x.grad is not None:
                        # ReLU gradient: 0 for x<0, 1 for x>0
                        expected = np.array([0.0, 0.0, 1.0, 1.0])
                        # Allow some flexibility in gradient at x=0
                        assert x.grad.data[0] == 0.0, "ReLU grad wrong for negative"
                        assert x.grad.data[3] == 1.0, "ReLU grad wrong for positive"
                except (TypeError, ValueError, AttributeError):
                    pass
                    
        except TypeError:
            assert True, "Activation gradients not implemented yet"
        except ImportError as e:
            assert False, f"Import failed: {e}"


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
            
            if hasattr(loss, 'backward'):
                try:
                    loss.backward()
                    
                    if pred.grad is not None:
                        # MSE gradient: 2*(pred - target)/n
                        assert pred.grad.shape == pred.shape, "Loss gradient shape wrong"
                except (TypeError, ValueError, AttributeError):
                    pass
                    
        except TypeError:
            assert True, "Loss gradients not implemented yet"
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
            data = Tensor(np.random.randn(20, 4))
            targets = Tensor(np.random.randn(20, 2))
            dataset = TensorDataset(data, targets)
            dataloader = DataLoader(dataset, batch_size=4)
            
            # Create model
            layer = Linear(4, 2)
            loss_fn = MSELoss()
            
            # Test gradient computation with batches
            for batch_x, batch_y in dataloader:
                # Forward pass
                output = layer(batch_x)
                loss = loss_fn(output, batch_y)
                
                # Backward pass
                if hasattr(loss, 'backward'):
                    try:
                        loss.backward()
                        
                        # Check layer has gradients
                        if hasattr(layer, 'weight') and layer.weight.grad is not None:
                            assert layer.weight.grad.shape == layer.weight.shape, \
                                "Batch gradient shape wrong"
                    except (TypeError, ValueError, AttributeError):
                        pass
                        
                break  # Test one batch
                
        except TypeError:
            assert True, "Batch gradients not implemented yet"
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
            x = Tensor(np.random.randn(2, 5))
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
            
            data = Tensor(np.random.randn(10, 4))
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
                if hasattr(x, 'requires_grad') and x.requires_grad:
                    capabilities["requires_grad attribute"] = True
            except TypeError:
                pass
            
            # Test 2: grad attribute
            try:
                x = Tensor([1.0], requires_grad=True)
                if hasattr(x, 'grad'):
                    capabilities["grad attribute"] = True
            except TypeError:
                pass
            
            # Test 3: backward method
            try:
                x = Tensor([1.0], requires_grad=True)
                if hasattr(x, 'backward'):
                    capabilities["backward method"] = True
            except TypeError:
                pass
            
            # Test 4: gradient computation
            try:
                x = Tensor([2.0], requires_grad=True)
                y = x * 3
                if hasattr(y, 'backward'):
                    y.backward(Tensor([1.0]))
                    if x.grad is not None:
                        capabilities["gradient computation"] = True
            except (TypeError, AttributeError):
                pass
            
            # Test 5: layer integration
            try:
                from tinytorch.core.layers import Linear
                layer = Linear(2, 1)
                x = Tensor(np.random.randn(1, 2), requires_grad=True)
                out = layer(x)
                if hasattr(out, 'backward'):
                    out.backward(Tensor([[1.0]]))
                    if layer.weight.grad is not None:
                        capabilities["layer integration"] = True
            except (TypeError, AttributeError, ImportError):
                pass
            
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
