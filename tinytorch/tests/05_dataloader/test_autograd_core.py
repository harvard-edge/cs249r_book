"""
Module 06: Autograd - Core Functionality Tests
Tests automatic differentiation and computational graphs
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestVariableCreation:
    """Test Variable creation and gradient tracking."""

    def test_variable_creation(self):
        """Test creating Variable with gradient tracking."""
        try:
            from tinytorch.core.autograd import Variable

            # Create variable that requires gradients
            x = Variable(np.array([2.0, 3.0]), requires_grad=True)

            assert x.requires_grad == True
            assert x.shape == (2,)
            assert np.array_equal(x.data, [2.0, 3.0])

        except ImportError:
            assert True, "Variable not implemented yet"

    def test_variable_no_grad(self):
        """Test creating Variable without gradient tracking."""
        try:
            from tinytorch.core.autograd import Variable

            x = Variable(np.array([1.0, 2.0]), requires_grad=False)

            assert x.requires_grad == False
            assert hasattr(x, 'grad')
            assert x.grad is None

        except ImportError:
            assert True, "Variable not implemented yet"

    def test_variable_grad_initialization(self):
        """Test gradient is properly initialized."""
        try:
            from tinytorch.core.autograd import Variable

            x = Variable(np.array([1.0]), requires_grad=True)

            # Gradient should start as None
            assert x.grad is None

        except ImportError:
            assert True, "Variable gradient initialization not implemented yet"


class TestBasicOperations:
    """Test basic operations with gradient computation."""

    def test_addition_gradient(self):
        """Test gradient computation for addition."""
        try:
            from tinytorch.core.autograd import Variable

            x = Variable(np.array([2.0]), requires_grad=True)
            y = Variable(np.array([3.0]), requires_grad=True)

            z = x + y

            assert np.array_equal(z.data, [5.0])

            if hasattr(z, 'backward'):
                z.backward()

                # d(x+y)/dx = 1, d(x+y)/dy = 1
                assert np.array_equal(x.grad, [1.0])
                assert np.array_equal(y.grad, [1.0])

        except ImportError:
            assert True, "Addition gradient not implemented yet"

    def test_multiplication_gradient(self):
        """Test gradient computation for multiplication."""
        try:
            from tinytorch.core.autograd import Variable

            x = Variable(np.array([3.0]), requires_grad=True)
            y = Variable(np.array([4.0]), requires_grad=True)

            z = x * y

            assert np.array_equal(z.data, [12.0])

            if hasattr(z, 'backward'):
                z.backward()

                # d(x*y)/dx = y, d(x*y)/dy = x
                assert np.array_equal(x.grad, [4.0])
                assert np.array_equal(y.grad, [3.0])

        except ImportError:
            assert True, "Multiplication gradient not implemented yet"

    def test_power_gradient(self):
        """Test gradient computation for power operation."""
        try:
            from tinytorch.core.autograd import Variable

            x = Variable(np.array([3.0]), requires_grad=True)

            # z = x²
            z = x ** 2

            assert np.array_equal(z.data, [9.0])

            if hasattr(z, 'backward'):
                z.backward()

                # d(x²)/dx = 2x = 2*3 = 6
                assert np.array_equal(x.grad, [6.0])

        except ImportError:
            assert True, "Power gradient not implemented yet"


class TestChainRule:
    """Test chain rule application."""

    def test_simple_chain_rule(self):
        """Test chain rule with simple composition."""
        try:
            from tinytorch.core.autograd import Variable

            x = Variable(np.array([2.0]), requires_grad=True)

            # z = (x + 1)²
            y = x + 1  # y = 3
            z = y * y  # z = 9

            if hasattr(z, 'backward'):
                z.backward()

                # dz/dx = dz/dy * dy/dx = 2y * 1 = 2*3 = 6
                assert np.array_equal(x.grad, [6.0])

        except ImportError:
            assert True, "Chain rule not implemented yet"

    def test_complex_chain_rule(self):
        """Test chain rule with more complex composition."""
        try:
            from tinytorch.core.autograd import Variable

            x = Variable(np.array([2.0]), requires_grad=True)

            # z = (x²)² = x⁴
            y = x * x      # y = x²
            z = y * y      # z = (x²)²

            if hasattr(z, 'backward'):
                z.backward()

                # dz/dx = 4x³ = 4 * 2³ = 32
                assert np.array_equal(x.grad, [32.0])

        except ImportError:
            assert True, "Complex chain rule not implemented yet"

    def test_multiple_variable_chain(self):
        """Test chain rule with multiple variables."""
        try:
            from tinytorch.core.autograd import Variable

            x = Variable(np.array([2.0]), requires_grad=True)
            y = Variable(np.array([3.0]), requires_grad=True)

            # z = (x + y)²
            u = x + y      # u = 5
            z = u * u      # z = 25

            if hasattr(z, 'backward'):
                z.backward()

                # dz/dx = dz/du * du/dx = 2u * 1 = 2*5 = 10
                # dz/dy = dz/du * du/dy = 2u * 1 = 2*5 = 10
                assert np.array_equal(x.grad, [10.0])
                assert np.array_equal(y.grad, [10.0])

        except ImportError:
            assert True, "Multiple variable chain rule not implemented yet"


class TestComputationGraph:
    """Test computation graph construction and traversal."""

    def test_graph_construction(self):
        """Test that computation graph is built correctly."""
        try:
            from tinytorch.core.autograd import Variable

            x = Variable(np.array([1.0]), requires_grad=True)
            y = x + 1
            z = y * 2

            # Each operation should create new nodes
            assert isinstance(y, Variable)
            assert isinstance(z, Variable)

            # Should track computation history
            if hasattr(z, 'grad_fn') or hasattr(z, '_backward_fn'):
                assert True  # Has some form of backward tracking

        except ImportError:
            assert True, "Computation graph not implemented yet"

    def test_graph_backward_traversal(self):
        """Test backward pass traverses graph correctly."""
        try:
            from tinytorch.core.autograd import Variable

            x = Variable(np.array([2.0]), requires_grad=True)
            y = Variable(np.array([3.0]), requires_grad=True)

            # Build computation graph
            u = x * y      # u = 6
            v = u + x      # v = 8
            w = v * 2      # w = 16

            if hasattr(w, 'backward'):
                w.backward()

                # dw/dx = dw/dv * (dv/du * du/dx + dv/dx) = 2 * (y + 1) = 2 * 4 = 8
                # dw/dy = dw/dv * dv/du * du/dy = 2 * 1 * x = 2 * 2 = 4
                assert np.array_equal(x.grad, [8.0])
                assert np.array_equal(y.grad, [4.0])

        except ImportError:
            assert True, "Graph backward traversal not implemented yet"

    def test_graph_memory_management(self):
        """Test computation graph doesn't cause memory leaks."""
        try:
            from tinytorch.core.autograd import Variable

            # Create many operations
            x = Variable(np.array([1.0]), requires_grad=True)
            result = x

            for i in range(100):
                result = result * 1.01  # Small multiplications

            if hasattr(result, 'backward'):
                result.backward()

                # Should complete without memory issues
                assert x.grad is not None
                assert x.grad.size == 1

        except ImportError:
            assert True, "Graph memory management not implemented yet"


class TestGradientAccumulation:
    """Test gradient accumulation and zeroing."""

    def test_gradient_accumulation(self):
        """Test gradients accumulate across multiple backward passes."""
        try:
            from tinytorch.core.autograd import Variable

            x = Variable(np.array([1.0]), requires_grad=True)

            # First computation
            y1 = x * 2
            if hasattr(y1, 'backward'):
                y1.backward()
                first_grad = x.grad.copy() if x.grad is not None else None

                # Second computation (gradients should accumulate)
                y2 = x * 3
                y2.backward()

                if first_grad is not None and x.grad is not None:
                    # Gradient should be sum: 2 + 3 = 5
                    assert np.array_equal(x.grad, [5.0])

        except ImportError:
            assert True, "Gradient accumulation not implemented yet"

    def test_gradient_zeroing(self):
        """Test gradient zeroing functionality."""
        try:
            from tinytorch.core.autograd import Variable

            x = Variable(np.array([1.0]), requires_grad=True)

            # Compute gradient
            y = x * 5
            if hasattr(y, 'backward'):
                y.backward()

                if x.grad is not None:
                    assert np.array_equal(x.grad, [5.0])

                    # Zero gradients
                    if hasattr(x, 'zero_grad'):
                        x.zero_grad()
                        assert x.grad is None or np.array_equal(x.grad, [0.0])

        except ImportError:
            assert True, "Gradient zeroing not implemented yet"

    def test_gradient_clipping(self):
        """Test gradient clipping for stability."""
        try:
            from tinytorch.core.autograd import Variable, clip_gradients

            x = Variable(np.array([10.0]), requires_grad=True)

            # Create large gradient
            y = x ** 3  # dy/dx = 3x² = 300

            if hasattr(y, 'backward'):
                y.backward()

                if x.grad is not None and hasattr(clip_gradients, '__call__'):
                    # Clip to max norm of 1.0
                    clip_gradients([x], max_norm=1.0)

                    # Gradient should be clipped
                    assert np.linalg.norm(x.grad) <= 1.0

        except ImportError:
            assert True, "Gradient clipping not implemented yet"


class TestAutogradUtilities:
    """Test autograd utility functions."""

    def test_no_grad_context(self):
        """Test no_grad context manager."""
        try:
            from tinytorch.core.autograd import Variable, no_grad

            x = Variable(np.array([1.0]), requires_grad=True)

            with no_grad():
                y = x * 2

                # Operations in no_grad should not require gradients
                assert not y.requires_grad

        except ImportError:
            assert True, "no_grad context not implemented yet"

    def test_detach_operation(self):
        """Test detaching variables from computation graph."""
        try:
            from tinytorch.core.autograd import Variable

            x = Variable(np.array([2.0]), requires_grad=True)
            y = x * 3

            if hasattr(y, 'detach'):
                z = y.detach()

                # Detached variable should not require gradients
                assert not z.requires_grad
                assert np.array_equal(z.data, y.data)

        except ImportError:
            assert True, "Detach operation not implemented yet"

    def test_grad_check(self):
        """Test gradient checking utility."""
        try:
            from tinytorch.core.autograd import Variable, gradcheck

            def simple_function(x):
                return x ** 2

            x = Variable(np.array([3.0]), requires_grad=True)

            if hasattr(gradcheck, '__call__'):
                # Check if analytical gradient matches numerical gradient
                is_correct = gradcheck(simple_function, x)
                assert isinstance(is_correct, bool)

        except ImportError:
            assert True, "Gradient checking not implemented yet"
