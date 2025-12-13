#!/usr/bin/env python3
"""
TinyTorch Integration Tests - Test-First Approach

These tests define EXACTLY what must work for our examples to succeed.
Written based on analyzing examples/*/train_*_modern_api.py

Test scenarios match our example usage:
1. XOR Network (nn.Module + nn.Linear + F.relu + optim.SGD + MSELoss)
2. MNIST MLP (nn.Module + nn.Linear + F.relu + F.flatten + optim.Adam + CrossEntropyLoss)
3. CIFAR-10 CNN (Complete modern API integration)

If these tests pass, the examples WILL work.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

# Test imports - these MUST work for examples to succeed
try:
    import tinytorch.nn as nn
    import tinytorch.nn.functional as F
    import tinytorch.optim as optim
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.autograd import Variable
    from tinytorch.core.training import CrossEntropyLoss, MeanSquaredError as MSELoss
    from tinytorch.core.dataloader import DataLoader
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Import failed: {e}")
    IMPORTS_AVAILABLE = False

def test_imports_available():
    """Test that all required imports work."""
    assert IMPORTS_AVAILABLE, "Required imports must work for examples to succeed"

class TestXORIntegration:
    """Test XOR network integration - matches examples/xornet/train_xor_modern_api.py"""

    def test_xor_network_creation(self):
        """Test creating XOR network like in the example."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Imports not available")

        class XORNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden = nn.Linear(2, 4)
                self.output = nn.Linear(4, 1)

            def forward(self, x):
                x = F.relu(self.hidden(x))
                x = self.output(x)
                return x

        # This MUST work for XOR example to succeed
        model = XORNet()
        assert hasattr(model, 'parameters'), "nn.Module must provide parameters() method"
        assert len(list(model.parameters())) > 0, "Model must have parameters for optimizer"

    def test_xor_forward_pass(self):
        """Test XOR forward pass works."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Imports not available")

        class XORNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden = nn.Linear(2, 4)
                self.output = nn.Linear(4, 1)

            def forward(self, x):
                x = F.relu(self.hidden(x))
                x = self.output(x)
                return x

        model = XORNet()

        # Test with XOR data
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        inputs = Variable(Tensor(X), requires_grad=False)

        # This MUST work for XOR example to succeed
        outputs = model(inputs)  # Calls model.forward(inputs)
        assert hasattr(outputs, 'data'), "Model output must be a Variable with data"
        assert outputs.data.shape == (4, 1), f"Expected shape (4, 1), got {outputs.data.shape}"

    def test_xor_training_loop(self):
        """Test XOR training loop components work together."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Imports not available")

        class XORNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden = nn.Linear(2, 4)
                self.output = nn.Linear(4, 1)

            def forward(self, x):
                x = F.relu(self.hidden(x))
                x = self.output(x)
                return x

        # Components that MUST work together
        model = XORNet()
        optimizer = optim.SGD(model.parameters(), learning_rate=0.1)
        criterion = MSELoss()

        # XOR dataset
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([[0], [1], [1], [0]], dtype=np.float32)

        # Training step - this MUST work
        inputs = Variable(Tensor(X), requires_grad=False)
        targets = Variable(Tensor(y), requires_grad=False)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Verify training components work
        assert hasattr(loss, 'data'), "Loss must be a Variable with data"
        assert hasattr(loss, 'backward'), "Loss must support backward()"

class TestMNISTIntegration:
    """Test MNIST MLP integration - matches examples/mnist/train_mlp_modern_api.py"""

    def test_mlp_creation(self):
        """Test creating MLP like in the example."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Imports not available")

        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden1 = nn.Linear(784, 128)
                self.hidden2 = nn.Linear(128, 64)
                self.output = nn.Linear(64, 10)

            def forward(self, x):
                x = F.flatten(x, start_dim=1)
                x = F.relu(self.hidden1(x))
                x = F.relu(self.hidden2(x))
                x = self.output(x)
                return x

        # This MUST work for MNIST example to succeed
        model = SimpleMLP()
        optimizer = optim.Adam(model.parameters(), learning_rate=0.001)
        criterion = CrossEntropyLoss()

        assert len(list(model.parameters())) > 0, "Model must have parameters"

    def test_mlp_forward_pass(self):
        """Test MLP forward pass with flatten."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Imports not available")

        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden1 = nn.Linear(784, 128)
                self.hidden2 = nn.Linear(128, 64)
                self.output = nn.Linear(64, 10)

            def forward(self, x):
                x = F.flatten(x, start_dim=1)
                x = F.relu(self.hidden1(x))
                x = F.relu(self.hidden2(x))
                x = self.output(x)
                return x

        model = SimpleMLP()

        # Test with MNIST-like data
        batch_size = 32
        X = np.random.randn(batch_size, 784).astype(np.float32) * 0.1
        inputs = Variable(Tensor(X), requires_grad=False)

        # This MUST work for MNIST example to succeed
        outputs = model(inputs)
        assert outputs.data.shape == (batch_size, 10), f"Expected shape ({batch_size}, 10), got {outputs.data.shape}"

    def test_mlp_training_with_crossentropy(self):
        """Test MLP training with CrossEntropyLoss."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Imports not available")

        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden1 = nn.Linear(784, 128)
                self.output = nn.Linear(128, 10)

            def forward(self, x):
                x = F.flatten(x, start_dim=1)
                x = F.relu(self.hidden1(x))
                x = self.output(x)
                return x

        model = SimpleMLP()
        optimizer = optim.Adam(model.parameters(), learning_rate=0.001)
        criterion = CrossEntropyLoss()

        # Sample data
        batch_size = 8
        X = np.random.randn(batch_size, 784).astype(np.float32) * 0.1
        y = np.random.randint(0, 10, batch_size).astype(np.int64)

        inputs = Variable(Tensor(X), requires_grad=False)
        targets = Variable(Tensor(y.astype(np.float32)), requires_grad=False)

        # Training step - this MUST work
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert hasattr(loss, 'data'), "CrossEntropyLoss must return Variable with data"

class TestCIFAR10Integration:
    """Test CIFAR-10 CNN integration - matches examples/cifar10/train_cnn_modern_api.py"""

    def test_cnn_creation(self):
        """Test creating CNN like in the example."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Imports not available")

        class ModernCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, (3, 3))
                self.conv2 = nn.Conv2d(32, 64, (3, 3))
                self.fc1 = nn.Linear(2304, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, (2, 2))
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, (2, 2))
                x = F.flatten(x)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        # This MUST work for CIFAR-10 example to succeed
        model = ModernCNN()
        assert len(list(model.parameters())) > 0, "CNN must have parameters"

    def test_cnn_forward_pass(self):
        """Test CNN forward pass with convolution and pooling."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Imports not available")

        # Simplified CNN for testing
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 8, (3, 3))
                self.fc1 = nn.Linear(8 * 30 * 30, 10)  # Simplified size calculation

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.flatten(x)
                x = self.fc1(x)
                return x

        model = SimpleCNN()

        # Test with CIFAR-10-like data (3x32x32)
        batch_size = 4
        X = np.random.randn(batch_size, 3, 32, 32).astype(np.float32) * 0.1
        inputs = Variable(Tensor(X), requires_grad=False)

        # This MUST work for CIFAR-10 example to succeed
        try:
            outputs = model(inputs)
            assert outputs.data.shape == (batch_size, 10), f"Expected shape ({batch_size}, 10), got {outputs.data.shape}"
        except Exception as e:
            # If this fails, we know what needs to be fixed
            print(f"❌ CNN forward pass failed: {e}")
            assert False, f"CNN forward pass must work for CIFAR-10 example: {e}"

class TestModernAPIComponents:
    """Test individual modern API components work correctly."""

    def test_nn_module_base_class(self):
        """Test nn.Module provides required functionality."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Imports not available")

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(2, 1)

            def forward(self, x):
                return self.layer(x)

        model = TestModule()

        # Required functionality for examples
        assert hasattr(model, 'parameters'), "nn.Module must have parameters() method"
        assert hasattr(model, '__call__'), "nn.Module must be callable"
        assert callable(model), "model() must call model.forward()"

        # Test parameter collection
        params = list(model.parameters())
        assert len(params) > 0, "parameters() must return model parameters"

    def test_functional_interface(self):
        """Test nn.functional components work."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Imports not available")

        # Test data
        X = np.array([[-1, 0, 1, 2]], dtype=np.float32)
        tensor_x = Variable(Tensor(X), requires_grad=False)

        # F.relu must work
        relu_out = F.relu(tensor_x)
        assert hasattr(relu_out, 'data'), "F.relu must return Variable"

        # F.flatten must work
        X_2d = np.random.randn(2, 3, 4).astype(np.float32)
        tensor_2d = Variable(Tensor(X_2d), requires_grad=False)
        flat_out = F.flatten(tensor_2d)
        assert flat_out.data.shape[0] == 2, "F.flatten must preserve batch dimension"

    def test_optimizer_integration(self):
        """Test optimizers work with nn.Module."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Imports not available")

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(2, 1)

            def forward(self, x):
                return self.layer(x)

        model = SimpleModel()

        # Both optimizers must work with model.parameters()
        adam_opt = optim.Adam(model.parameters(), learning_rate=0.001)
        sgd_opt = optim.SGD(model.parameters(), learning_rate=0.01)

        # Test optimizer functionality
        assert hasattr(adam_opt, 'step'), "Optimizer must have step() method"
        assert hasattr(adam_opt, 'zero_grad'), "Optimizer must have zero_grad() method"

def run_integration_tests():
    """Run all integration tests and report results."""
    print("🧪 Running TinyTorch Integration Tests")
    print("=" * 50)
    print("These tests define what MUST work for examples to succeed.")
    print()

    test_results = []

    # Test categories
    test_classes = [
        TestXORIntegration,
        TestMNISTIntegration,
        TestCIFAR10Integration,
        TestModernAPIComponents
    ]

    for test_class in test_classes:
        print(f"🔍 Testing {test_class.__name__}...")

        instance = test_class()
        methods = [method for method in dir(instance) if method.startswith('test_')]

        for method_name in methods:
            method = getattr(instance, method_name)
            try:
                method()
                print(f"  ✅ {method_name}")
                test_results.append((method_name, True, None))
            except Exception as e:
                print(f"  ❌ {method_name}: {e}")
                test_results.append((method_name, False, str(e)))
        print()

    # Summary
    passed = sum(1 for _, success, _ in test_results if success)
    total = len(test_results)

    print("📊 Integration Test Results")
    print("=" * 30)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All integration tests passed!")
        print("✅ Examples should work correctly!")
    else:
        print("❌ Some integration tests failed.")
        print("⚠️  Examples will NOT work until these are fixed:")
        for test_name, success, error in test_results:
            if not success:
                print(f"   • {test_name}: {error}")

    print()
    print("🎯 Next Steps:")
    if passed < total:
        print("1. Fix failing integration tests")
        print("2. Update module unit tests to support integration")
        print("3. Export modules to tinytorch package")
        print("4. Re-run integration tests")
        print("5. Test examples")
    else:
        print("1. Export modules to tinytorch package")
        print("2. Test examples work end-to-end")

    return passed == total

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
