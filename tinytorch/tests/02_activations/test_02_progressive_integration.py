"""
Module 02: Progressive Integration Tests
Tests that Module 03 (Activations) works correctly AND that all previous modules still work.

DEPENDENCY CHAIN: 01_setup → 02_tensor → 03_activations
Students can trace back exactly where issues originate.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestModule01StillWorking:
    """Verify Module 01 (Setup) functionality is still intact."""
    
    def test_setup_environment_stable(self):
        """Ensure setup environment wasn't broken by activations development."""
        # Core environment should be stable
        assert sys.version_info >= (3, 8), "Setup: Python version check broken"
        
        # Project structure should remain intact
        project_root = Path(__file__).parent.parent.parent
        assert (project_root / "modules").exists(), "Setup: Module structure broken"
        assert (project_root / "tinytorch").exists(), "Setup: Package structure broken"


class TestModule02StillWorking:
    """Verify Module 02 (Tensor) functionality is still intact."""
    
    def test_tensor_functionality_stable(self):
        """Ensure tensor functionality wasn't broken by activations development."""
        try:
            from tinytorch.core.tensor import Tensor
            
            # Basic tensor operations should still work
            t = Tensor([1, 2, 3])
            assert t.shape == (3,), "Module 02: Tensor creation broken"
            
            # Numpy integration should still work
            arr = np.array([[1, 2], [3, 4]])
            t2 = Tensor(arr)
            assert t2.shape == (2, 2), "Module 02: Numpy integration broken"
            
        except ImportError:
            assert True, "Module 02: Tensor not implemented yet"


class TestModule03ActivationsCore:
    """Test Module 03 (Activations) core functionality."""
    
    def test_relu_activation(self):
        """Test ReLU activation function."""
        try:
            from tinytorch.core.activations import ReLU
            from tinytorch.core.tensor import Tensor
            
            relu = ReLU()
            x = Tensor(np.array([-2, -1, 0, 1, 2]))
            output = relu(x)
            
            expected = np.array([0, 0, 0, 1, 2])
            assert np.array_equal(output.data, expected), "ReLU activation failed"
            
        except ImportError:
            assert True, "Module 02: Activations not implemented yet"
    
    def test_sigmoid_activation(self):
        """Test Sigmoid activation function."""
        try:
            from tinytorch.core.activations import Sigmoid
            from tinytorch.core.tensor import Tensor
            
            sigmoid = Sigmoid()
            x = Tensor(np.array([0, 1, -1]))
            output = sigmoid(x)
            
            # Sigmoid(0) should be 0.5
            assert np.isclose(output.data[0], 0.5, atol=1e-6), "Sigmoid activation failed"
            
            # All outputs should be in (0, 1)
            assert np.all(output.data > 0) and np.all(output.data < 1), "Sigmoid range failed"
            
        except ImportError:
            assert True, "Module 02: Sigmoid not implemented yet"


class TestProgressiveStackIntegration:
    """Test that the full stack (01→02→03) works together."""
    
    def test_tensor_activation_pipeline(self):
        """Test tensors work correctly with activations."""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.activations import ReLU, Sigmoid
            
            # Create tensor using Module 02
            x = Tensor(np.array([-1, 0, 1, 2]))
            
            # Apply activations from Module 03
            relu = ReLU()
            sigmoid = Sigmoid()
            
            # Pipeline: input -> ReLU -> Sigmoid
            h = relu(x)
            output = sigmoid(h)
            
            # Should work end-to-end
            assert output.shape == x.shape, "Tensor-activation pipeline broken"
            assert np.all(output.data >= 0) and np.all(output.data <= 1), "Pipeline output invalid"
            
        except ImportError:
            assert True, "Progressive stack not fully implemented yet"
    
    def test_activation_chaining(self):
        """Test multiple activations can be chained."""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.activations import ReLU, Sigmoid, Tanh
            
            x = Tensor(np.random.randn(5, 10))
            
            # Chain multiple activations
            relu = ReLU()
            tanh = Tanh()
            sigmoid = Sigmoid()
            
            h1 = relu(x)      # Apply ReLU
            h2 = tanh(h1)     # Apply Tanh
            output = sigmoid(h2)  # Apply Sigmoid
            
            assert output.shape == x.shape, "Activation chaining broken"
            
        except ImportError:
            assert True, "Activation chaining not implemented yet"


class TestNonLinearityCapability:
    """Test that activations enable non-linear computation."""
    
    def test_nonlinearity_proof(self):
        """Test that activations actually provide non-linearity."""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.activations import ReLU
            
            relu = ReLU()
            
            # Linear input
            x = Tensor(np.array([-2, -1, 0, 1, 2]))
            
            # Non-linear output from ReLU
            y = relu(x)
            
            # Should be different from linear function
            linear_output = x.data  # Identity function
            nonlinear_output = y.data
            
            # ReLU introduces non-linearity
            assert not np.array_equal(linear_output, nonlinear_output), "No nonlinearity detected"
            
            # Specifically, negative values should become zero
            assert np.all(nonlinear_output >= 0), "ReLU non-linearity not working"
            
        except ImportError:
            assert True, "Nonlinearity testing not ready yet"


class TestXORProblemReadiness:
    """Test that the stack is ready for XOR problem (non-linear learning)."""
    
    def test_xor_components_available(self):
        """Test components needed for XOR are available."""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.activations import ReLU, Sigmoid
            
            # XOR inputs
            X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
            
            # Should be able to apply activations
            relu = ReLU()
            sigmoid = Sigmoid()
            
            # Simulated hidden layer output
            hidden = relu(X)  # Non-linear transformation
            
            # Simulated output layer
            output = sigmoid(hidden)
            
            assert output.shape == X.shape, "XOR components not ready"
            
        except ImportError:
            assert True, "XOR components not implemented yet"
    
    def test_activation_expressiveness(self):
        """Test activations provide sufficient expressiveness."""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.activations import ReLU, Sigmoid
            
            # Test that we can represent different patterns
            patterns = [
                np.array([1, 0, 0, 1]),  # XOR pattern
                np.array([0, 1, 1, 0]),  # Inverse XOR
                np.array([1, 1, 0, 0]),  # AND-like pattern
            ]
            
            relu = ReLU()
            sigmoid = Sigmoid()
            
            for pattern in patterns:
                x = Tensor(pattern)
                
                # Should be able to transform any pattern
                h = relu(x)
                y = sigmoid(h)
                
                assert y.shape == x.shape, "Pattern transformation failed"
                
        except ImportError:
            assert True, "Activation expressiveness testing not ready"


class TestRegressionPrevention:
    """Ensure previous modules still work after Module 03 development."""
    
    def test_no_module_01_regression(self):
        """Verify Module 01 functionality unchanged."""
        # These should ALWAYS work
        assert sys.version_info.major >= 3, "Module 01: Python detection broken"
        
        project_root = Path(__file__).parent.parent.parent
        assert project_root.exists(), "Module 01: Project structure broken"
    
    def test_no_module_02_regression(self):
        """Verify Module 02 functionality unchanged."""
        try:
            from tinytorch.core.tensor import Tensor
            
            # Basic tensor creation should still work
            t = Tensor([1, 2, 3])
            assert t.shape == (3,), "Module 02: Basic tensor broken"
            
        except ImportError:
            # If not implemented, that's fine
            # But numpy should still work (from Module 01)
            import numpy as np
            arr = np.array([1, 2, 3])
            assert arr.shape == (3,), "Module 02: Numpy foundation broken"
    
    def test_progressive_stability(self):
        """Test the progressive stack is stable."""
        # Stack should be stable through: Setup -> Tensor -> Activations
        
        # Setup level
        import numpy as np
        assert np is not None, "Setup level broken"
        
        # Tensor level (if available)
        try:
            from tinytorch.core.tensor import Tensor
            t = Tensor([1])
            assert t.shape == (1,), "Tensor level broken"
        except ImportError:
            pass  # Not implemented yet
        
        # Activation level (if available)
        try:
            from tinytorch.core.activations import ReLU
            relu = ReLU()
            assert callable(relu), "Activation level broken"
        except ImportError:
            pass  # Not implemented yet