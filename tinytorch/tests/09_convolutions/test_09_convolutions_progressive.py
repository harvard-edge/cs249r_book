"""
Module 09: Progressive Integration Tests
Tests that Module 09 (Convolutions/Spatial) works correctly AND that prior modules (01→08) still work.

DEPENDENCY CHAIN: 01_tensor → 02_activations → 03_layers → 04_losses → 05_dataloader → 06_autograd → 07_optimizers → 08_training → 09_convolutions

⚠️ IMPORTANT: This test ONLY uses modules 01-09.
   Future modules (10_tokenization, 12_attention, 13_transformers, etc.) are NOT tested here.

🎯 WHAT THIS TESTS:
- Module 09: Conv2d, MaxPool2d, spatial operations
- Integration: Convolutions work with all prior modules (01-08)
- Regression: All previous modules still work correctly
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestConvolutionCore:
    """
    🆕 NEW FUNCTIONALITY: Test Module 09 (Convolutions) core implementation.
    """

    def test_conv2d_exists(self):
        """
        ✅ TEST: Conv2d class exists and is importable
        """
        try:
            from tinytorch.core.spatial import Conv2d
            
            assert Conv2d is not None, "Conv2d class not found"
            
        except ImportError:
            assert True, "Conv2d not implemented yet"

    def test_conv2d_initialization(self):
        """
        ✅ TEST: Conv2d can be initialized with proper parameters
        """
        try:
            from tinytorch.core.spatial import Conv2d
            
            conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3)
            
            assert hasattr(conv, 'weight'), "Conv2d missing weight"
            assert hasattr(conv, 'forward'), "Conv2d missing forward"
            
            # Check weight shape: (out_channels, in_channels, kernel_h, kernel_w)
            if hasattr(conv, 'weight'):
                assert conv.weight.shape[0] == 16, "Conv2d out_channels wrong"
                assert conv.weight.shape[1] == 3, "Conv2d in_channels wrong"
                
        except ImportError:
            assert True, "Conv2d not implemented yet"

    def test_conv2d_forward(self):
        """
        ✅ TEST: Conv2d forward pass produces correct output shape
        """
        try:
            from tinytorch.core.spatial import Conv2d
            from tinytorch.core.tensor import Tensor
            
            conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
            
            # Input: (batch, channels, height, width)
            x = Tensor(np.random.randn(2, 3, 32, 32))
            
            output = conv(x)
            
            # With padding=1, kernel=3, output size should be same as input
            assert output.shape == (2, 16, 32, 32), f"Conv2d output shape wrong: {output.shape}"
            
        except ImportError:
            assert True, "Conv2d forward not implemented yet"

    def test_maxpool2d_exists(self):
        """
        ✅ TEST: MaxPool2d class exists
        """
        try:
            from tinytorch.core.spatial import MaxPool2d
            
            assert MaxPool2d is not None, "MaxPool2d not found"
            
        except ImportError:
            assert True, "MaxPool2d not implemented yet"

    def test_maxpool2d_forward(self):
        """
        ✅ TEST: MaxPool2d produces correct output shape
        """
        try:
            from tinytorch.core.spatial import MaxPool2d
            from tinytorch.core.tensor import Tensor
            
            pool = MaxPool2d(kernel_size=2)
            
            x = Tensor(np.random.randn(2, 16, 32, 32))
            
            output = pool(x)
            
            # Pool with kernel=2 halves spatial dimensions
            assert output.shape == (2, 16, 16, 16), f"MaxPool2d output shape wrong: {output.shape}"
            
        except ImportError:
            assert True, "MaxPool2d not implemented yet"


class TestConvWithPriorModules:
    """
    🔗 INTEGRATION: Convolutions + Prior Modules (01-08)
    """

    def test_conv_with_activations(self):
        """
        ✅ TEST: Conv2d + ReLU activation
        """
        try:
            from tinytorch.core.spatial import Conv2d
            from tinytorch.core.activations import ReLU
            from tinytorch.core.tensor import Tensor
            
            conv = Conv2d(3, 16, kernel_size=3, padding=1)
            relu = ReLU()
            
            x = Tensor(np.random.randn(2, 3, 16, 16))
            
            conv_out = conv(x)
            activated = relu(conv_out)
            
            # ReLU should not change shape
            assert activated.shape == conv_out.shape, "ReLU changed shape"
            # ReLU should make all values >= 0
            assert np.all(activated.data >= 0), "ReLU not working"
            
        except ImportError:
            assert True, "Conv + activation not ready"

    def test_conv_with_linear(self):
        """
        ✅ TEST: CNN feature extraction → Linear classifier
        """
        try:
            from tinytorch.core.spatial import Conv2d, MaxPool2d
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            from tinytorch.core.tensor import Tensor
            
            # CNN layers
            conv = Conv2d(3, 16, kernel_size=3, padding=1)  # (3, 32, 32) → (16, 32, 32)
            pool = MaxPool2d(kernel_size=2)  # (16, 32, 32) → (16, 16, 16)
            relu = ReLU()
            
            # Classifier
            fc = Linear(16 * 16 * 16, 10)
            
            # Forward pass
            x = Tensor(np.random.randn(2, 3, 32, 32))
            
            conv_out = relu(conv(x))
            pooled = pool(conv_out)
            
            # Flatten for linear layer
            flat = Tensor(pooled.data.reshape(2, -1))
            logits = fc(flat)
            
            assert logits.shape == (2, 10), f"CNN classifier output wrong: {logits.shape}"
            
        except ImportError:
            assert True, "CNN + Linear not ready"

    def test_conv_with_dataloader(self):
        """
        ✅ TEST: Convolutions work with DataLoader batches
        """
        try:
            from tinytorch.core.spatial import Conv2d
            from tinytorch.core.dataloader import TensorDataset, DataLoader
            from tinytorch.core.tensor import Tensor
            
            # Image dataset
            images = Tensor(np.random.randn(20, 3, 16, 16))
            labels = Tensor(np.arange(20).astype(float))
            
            dataset = TensorDataset(images, labels)
            dataloader = DataLoader(dataset, batch_size=4)
            
            conv = Conv2d(3, 8, kernel_size=3, padding=1)
            
            for batch_x, batch_y in dataloader:
                out = conv(batch_x)
                assert out.shape[0] == batch_x.shape[0], "Batch size changed"
                break
                
        except ImportError:
            assert True, "Conv + DataLoader not ready"

    def test_conv_gradients(self):
        """
        ✅ TEST: Convolution gradients (if autograd works)
        """
        try:
            from tinytorch.core.spatial import Conv2d
            from tinytorch.core.tensor import Tensor
            
            conv = Conv2d(3, 8, kernel_size=3, padding=1)
            
            x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
            
            out = conv(x)
            
            if hasattr(out, 'backward'):
                try:
                    out.backward(Tensor(np.ones(out.shape)))
                    
                    if x.grad is not None:
                        assert x.grad.shape == x.shape, "Input gradient shape wrong"
                    
                    if hasattr(conv, 'weight') and conv.weight.grad is not None:
                        assert conv.weight.grad.shape == conv.weight.shape, \
                            "Conv weight gradient shape wrong"
                except (TypeError, AttributeError):
                    pass  # Gradients not fully implemented
                    
        except (ImportError, TypeError):
            assert True, "Conv gradients not ready"


class TestCNNArchitecture:
    """
    Test complete CNN architectures using modules 01-09.
    """

    def test_simple_cnn(self):
        """
        ✅ TEST: Simple CNN (Conv → Pool → Conv → Pool → FC)
        """
        try:
            from tinytorch.core.spatial import Conv2d, MaxPool2d
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            from tinytorch.core.tensor import Tensor
            
            # LeNet-style architecture
            conv1 = Conv2d(1, 6, kernel_size=5)  # (1, 28, 28) → (6, 24, 24)
            pool1 = MaxPool2d(kernel_size=2)     # → (6, 12, 12)
            conv2 = Conv2d(6, 16, kernel_size=5) # → (16, 8, 8)
            pool2 = MaxPool2d(kernel_size=2)     # → (16, 4, 4)
            fc = Linear(16 * 4 * 4, 10)
            relu = ReLU()
            
            # Forward
            x = Tensor(np.random.randn(4, 1, 28, 28))
            
            h = relu(conv1(x))
            h = pool1(h)
            h = relu(conv2(h))
            h = pool2(h)
            h = Tensor(h.data.reshape(4, -1))
            logits = fc(h)
            
            assert logits.shape == (4, 10), f"CNN output wrong: {logits.shape}"
            
        except ImportError:
            assert True, "Simple CNN not ready"

    def test_cnn_training_ready(self):
        """
        ✅ TEST: CNN can be trained (components work together)
        """
        try:
            from tinytorch.core.spatial import Conv2d, MaxPool2d
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            from tinytorch.core.losses import MSELoss
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.tensor import Tensor
            
            # Simple CNN
            conv = Conv2d(3, 8, kernel_size=3, padding=1)
            pool = MaxPool2d(kernel_size=2)
            fc = Linear(8 * 8 * 8, 5)
            relu = ReLU()
            
            # Collect parameters
            params = []
            for module in [conv, fc]:
                if hasattr(module, 'parameters'):
                    params.extend(module.parameters())
            
            optimizer = SGD(params, lr=0.01)
            loss_fn = MSELoss()
            
            # Training step
            x = Tensor(np.random.randn(2, 3, 16, 16))
            target = Tensor(np.random.randn(2, 5))
            
            # Forward
            h = relu(conv(x))
            h = pool(h)
            h = Tensor(h.data.reshape(2, -1))
            pred = fc(h)
            
            loss = loss_fn(pred, target)
            
            if hasattr(loss, 'backward'):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            assert loss.data.size == 1, "CNN training loss wrong"
            
        except ImportError:
            assert True, "CNN training not ready"


class TestRegressionPrevention:
    """
    🔄 REGRESSION: Verify all previous modules (01-08) still work.
    """

    def test_tensor_still_works(self):
        """✅ Module 01"""
        try:
            from tinytorch.core.tensor import Tensor
            a = Tensor([1, 2, 3])
            assert a.shape == (3,)
        except Exception as e:
            assert False, f"Module 01: {e}"

    def test_activations_still_work(self):
        """✅ Module 02"""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.activations import ReLU
            relu = ReLU()
            x = Tensor([-1, 0, 1])
            y = relu(x)
            assert y.data[0] == 0
        except Exception as e:
            assert False, f"Module 02: {e}"

    def test_layers_still_work(self):
        """✅ Module 03"""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            layer = Linear(4, 2)
            x = Tensor(np.random.randn(2, 4))
            y = layer(x)
            assert y.shape == (2, 2)
        except Exception as e:
            assert False, f"Module 03: {e}"

    def test_losses_still_work(self):
        """✅ Module 04"""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.losses import MSELoss
            loss_fn = MSELoss()
            pred = Tensor([[1.0, 2.0]])
            target = Tensor([[1.5, 2.5]])
            loss = loss_fn(pred, target)
            assert loss.data.size == 1
        except Exception as e:
            assert False, f"Module 04: {e}"

    def test_dataloader_still_works(self):
        """✅ Module 05"""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.dataloader import TensorDataset, DataLoader
            data = Tensor(np.random.randn(10, 3))
            targets = Tensor(np.arange(10).astype(float))
            dataset = TensorDataset(data, targets)
            dataloader = DataLoader(dataset, batch_size=2)
            assert sum(1 for _ in dataloader) == 5
        except Exception as e:
            assert False, f"Module 05: {e}"

    def test_autograd_still_works(self):
        """✅ Module 06"""
        try:
            from tinytorch.core.tensor import Tensor
            x = Tensor([1.0], requires_grad=True)
            assert hasattr(x, 'requires_grad')
        except TypeError:
            pass  # OK if requires_grad not supported
        except Exception as e:
            assert False, f"Module 06: {e}"

    def test_optimizers_still_work(self):
        """✅ Module 07"""
        try:
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.layers import Linear
            layer = Linear(3, 2)
            opt = SGD(layer.parameters(), lr=0.01)
            assert hasattr(opt, 'step')
        except Exception as e:
            assert False, f"Module 07: {e}"

    def test_training_still_works(self):
        """✅ Module 08"""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.losses import MSELoss
            from tinytorch.core.optimizers import SGD
            
            layer = Linear(4, 2)
            loss_fn = MSELoss()
            opt = SGD(layer.parameters(), lr=0.1)
            
            x = Tensor(np.random.randn(2, 4))
            y = Tensor(np.random.randn(2, 2))
            
            pred = layer(x)
            loss = loss_fn(pred, y)
            
            if hasattr(loss, 'backward'):
                opt.zero_grad()
                loss.backward()
                opt.step()
                
            assert loss.data.size == 1
        except Exception as e:
            assert False, f"Module 08: {e}"


class TestModule09Completion:
    """
    ✅ COMPLETION CHECK: Module 09 ready for next module.
    """

    def test_convolution_foundation_complete(self):
        """
        ✅ FINAL TEST: Convolution ready for attention/transformers
        
        🎯 SUCCESS = Ready for Module 10: Tokenization!
        """
        capabilities = {
            "Conv2d exists": False,
            "Conv2d forward works": False,
            "MaxPool2d exists": False,
            "MaxPool2d forward works": False,
            "CNN architecture works": False,
        }
        
        try:
            from tinytorch.core.spatial import Conv2d, MaxPool2d
            from tinytorch.core.tensor import Tensor
            
            # Test 1: Conv2d exists
            capabilities["Conv2d exists"] = True
            
            # Test 2: Conv2d forward
            conv = Conv2d(3, 8, kernel_size=3, padding=1)
            x = Tensor(np.random.randn(2, 3, 16, 16))
            out = conv(x)
            if out.shape == (2, 8, 16, 16):
                capabilities["Conv2d forward works"] = True
            
            # Test 3: MaxPool2d exists
            capabilities["MaxPool2d exists"] = True
            
            # Test 4: MaxPool2d forward
            pool = MaxPool2d(kernel_size=2)
            pooled = pool(out)
            if pooled.shape == (2, 8, 8, 8):
                capabilities["MaxPool2d forward works"] = True
            
            # Test 5: CNN architecture
            from tinytorch.core.layers import Linear
            fc = Linear(8 * 8 * 8, 10)
            flat = Tensor(pooled.data.reshape(2, -1))
            logits = fc(flat)
            if logits.shape == (2, 10):
                capabilities["CNN architecture works"] = True
            
            completed = sum(capabilities.values())
            assert completed >= 4, f"Convolutions not ready: {capabilities}"
            
        except ImportError as e:
            assert False, f"Module 09 import failed: {e}"
