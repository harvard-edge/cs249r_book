"""
Module 03: Progressive Integration Tests
Tests that Module 04 (Layers) works correctly AND that the foundation stack (01→03) still works.

DEPENDENCY CHAIN: 01_setup → 02_tensor → 03_activations → 04_layers
This is where we create reusable building blocks for neural networks.

🎯 WHAT THIS TESTS:
- Module 03: Layer base class and interface design
- Integration: Layers work with tensors and activations from previous modules
- Regression: Previous modules (01→03) still work correctly
- Preparation: Foundation ready for Module 05 (DataLoader)

💡 FOR STUDENTS: If tests fail, check:
1. Does your Layer base class exist in tinytorch.core.layers?
2. Does Layer have a forward() method?
3. Is Layer callable (has __call__ method)?
4. Do layers work with Tensor objects from Module 02?
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPriorModulesStillWork:
    """
    🔄 REGRESSION CHECK: Verify Modules 01→03 still work after Layer development.

    💡 If these fail: You may have broken something in the foundation while working on layers.
    🔧 Fix: Check that your layer code doesn't interfere with basic tensor/activation functionality.
    """

    def test_setup_environment_stable(self):
        """
        ✅ TEST: Module 01 (Setup) - Environment should still work

        📋 CHECKS:
        - Python 3.8+ available
        - NumPy working correctly
        - Project structure intact

        🚨 IF FAILS: Your development environment is broken
        🔧 FIX: Check Python installation, reinstall NumPy if needed
        """
        try:
            # Environment checks
            assert sys.version_info >= (3, 8), \
                "❌ Python 3.8+ required. Current: Python {}.{}".format(
                    sys.version_info.major, sys.version_info.minor)

            # NumPy functionality
            import numpy as np
            test_array = np.array([1, 2, 3])
            assert test_array.shape == (3,), \
                "❌ NumPy broken. Cannot create basic arrays."

            # Project structure
            project_root = Path(__file__).parent.parent.parent
            assert (project_root / "modules").exists(), \
                "❌ Project structure broken. Missing 'modules' directory."

        except Exception as e:
            assert False, f"""
            ❌ MODULE 01 REGRESSION: Setup environment broken!

            🔍 ERROR: {str(e)}

            🔧 HOW TO FIX:
            1. Check Python version: python --version (need 3.8+)
            2. Reinstall NumPy: pip install numpy
            3. Verify project structure exists
            4. Run Module 01 tests separately: python tests/run_all_modules.py --module module_01
            """

    def test_tensor_operations_stable(self):
        """
        ✅ TEST: Module 02 (Tensor) - Tensors should still work

        📋 CHECKS:
        - Tensor class can be imported
        - Basic tensor creation works
        - Tensor operations function correctly

        🚨 IF FAILS: Tensor implementation is broken
        🔧 FIX: Check your tensor implementation, ensure exports are correct
        """
        try:
            from tinytorch.core.tensor import Tensor

            # Basic tensor creation
            t = Tensor([1, 2, 3])
            assert t.shape == (3,), \
                f"❌ Tensor shape broken. Expected (3,), got {t.shape}"

            # Multi-dimensional tensors
            t2 = Tensor(np.random.randn(4, 5))
            assert t2.shape == (4, 5), \
                f"❌ Multi-dim tensor broken. Expected (4, 5), got {t2.shape}"

        except ImportError as e:
            assert False, f"""
            ❌ MODULE 02 REGRESSION: Cannot import Tensor!

            🔍 IMPORT ERROR: {str(e)}

            🔧 HOW TO FIX:
            1. Implement Tensor class in modules/02_tensor/
            2. Export module: tito module complete 02_tensor
            3. Check tinytorch.core.tensor exists
            4. Verify Tensor class is exported correctly

            📚 EXPECTED STRUCTURE:
            class Tensor:
                def __init__(self, data): ...
                @property
                def shape(self): ...
            """
        except Exception as e:
            assert False, f"""
            ❌ MODULE 02 REGRESSION: Tensor functionality broken!

            🔍 ERROR: {str(e)}

            🔧 HOW TO FIX:
            1. Check Tensor.__init__ accepts data parameter
            2. Ensure Tensor.shape property returns correct tuple
            3. Test with: t = Tensor([1,2,3]); print(t.shape)
            4. Run Module 02 tests: python tests/run_all_modules.py --module module_02
            """

    def test_activations_stable(self):
        """
        ✅ TEST: Module 03 (Activations) - Activations should still work

        📋 CHECKS:
        - Activation functions can be imported
        - ReLU and Sigmoid work correctly
        - Activations work with tensors

        🚨 IF FAILS: Activation implementation is broken
        🔧 FIX: Check activation function implementations
        """
        try:
            from tinytorch.core.activations import ReLU, Sigmoid
            from tinytorch.core.tensor import Tensor

            # ReLU functionality
            relu = ReLU()
            x = Tensor(np.array([-1, 0, 1]))
            output = relu(x)

            expected = np.array([0, 0, 1])
            assert np.array_equal(output.data, expected), \
                f"❌ ReLU broken. Expected {expected}, got {output.data}"

            # Sigmoid functionality
            sigmoid = Sigmoid()
            x = Tensor(np.array([0]))
            output = sigmoid(x)

            assert np.isclose(output.data[0], 0.5, atol=1e-6), \
                f"❌ Sigmoid broken. Expected ~0.5, got {output.data[0]}"

        except ImportError as e:
            assert False, f"""
            ❌ MODULE 03 REGRESSION: Cannot import activations!

            🔍 IMPORT ERROR: {str(e)}

            🔧 HOW TO FIX:
            1. Implement ReLU and Sigmoid in modules/03_activations/
            2. Export module: tito module complete 03_activations
            3. Check tinytorch.core.activations exists
            4. Verify activation classes are exported

            📚 EXPECTED STRUCTURE:
            class ReLU:
                def __call__(self, x): return max(0, x)
            class Sigmoid:
                def __call__(self, x): return 1/(1+exp(-x))
            """
        except Exception as e:
            assert False, f"""
            ❌ MODULE 03 REGRESSION: Activation functionality broken!

            🔍 ERROR: {str(e)}

            🔧 HOW TO FIX:
            1. Check ReLU returns max(0, x) element-wise
            2. Check Sigmoid returns 1/(1+exp(-x))
            3. Ensure activations are callable: relu(tensor)
            4. Test activations work with Tensor objects
            5. Run Module 03 tests: python tests/run_all_modules.py --module module_03
            """


class TestModule04LayersCore:
    """
    🆕 NEW FUNCTIONALITY: Test Module 04 (Layers) core implementation.

    💡 What you're implementing: Base Layer class that all neural network layers inherit from.
    🎯 Goal: Create a standard interface for all layers (Dense, Conv2D, etc.)
    """

    def test_layer_base_class_exists(self):
        """
        ✅ TEST: Layer base class - Foundation for all neural network layers

        📋 WHAT YOU NEED TO IMPLEMENT:
        class Layer:
            def forward(self, x): raise NotImplementedError
            def __call__(self, x): return self.forward(x)

        🚨 IF FAILS: Layer base class doesn't exist or missing methods
        """
        try:
            from tinytorch.core.layers import Layer

            # Layer class should exist
            assert hasattr(Layer, 'forward'), \
                "❌ Layer missing forward() method. All layers need forward(x) -> output"

            assert hasattr(Layer, '__call__'), \
                "❌ Layer missing __call__() method. Layers should be callable: layer(x)"

            # Test instantiation
            layer = Layer()
            assert layer is not None, \
                "❌ Cannot create Layer instance"

        except ImportError as e:
            assert False, f"""
            ❌ LAYER BASE CLASS MISSING!

            🔍 IMPORT ERROR: {str(e)}

            🔧 HOW TO IMPLEMENT:

            1. Create in modules/04_layers/04_layers.py:

            class Layer:
                '''Base class for all neural network layers.'''

                def forward(self, x):
                    '''Forward pass through the layer.'''
                    raise NotImplementedError("Subclasses must implement forward()")

                def __call__(self, x):
                    '''Make layer callable: layer(x) calls layer.forward(x)'''
                    return self.forward(x)

            2. Export the module:
               tito module complete 04_layers

            📚 WHY THIS MATTERS:
            - All layers (Dense, Conv2D, etc.) will inherit from Layer
            - Provides consistent interface: layer(input) -> output
            - Enables polymorphism: treat all layers the same way
            """
        except Exception as e:
            assert False, f"""
            ❌ LAYER BASE CLASS BROKEN!

            🔍 ERROR: {str(e)}

            🔧 HOW TO FIX:
            1. Check Layer class has forward() method
            2. Check Layer class has __call__() method
            3. Ensure Layer can be instantiated
            4. Test with: layer = Layer(); print(type(layer))
            """

    def test_layer_interface_design(self):
        """
        ✅ TEST: Layer interface - Proper object-oriented design

        📋 CHECKS:
        - Layer is callable (can use layer(x) syntax)
        - forward() method raises NotImplementedError (abstract method)
        - __call__ delegates to forward()

        🎯 DESIGN GOAL: Clean, consistent interface for all layer types
        """
        try:
            from tinytorch.core.layers import Layer

            layer = Layer()

            # Test that forward raises NotImplementedError (abstract method)
            try:
                layer.forward(None)
                assert False, "❌ Layer.forward() should raise NotImplementedError"
            except NotImplementedError:
                pass  # This is expected

            # Test that __call__ delegates to forward
            assert callable(layer), \
                "❌ Layer should be callable. Add __call__ method."

            # Test __call__ behavior
            try:
                layer(None)  # Should call forward and raise NotImplementedError
                assert False, "❌ layer(x) should raise NotImplementedError (calls forward)"
            except NotImplementedError:
                pass  # This is expected

        except ImportError:
            assert False, """
            ❌ Cannot test layer interface - Layer class missing!
            See test_layer_base_class_exists for implementation guide.
            """
        except Exception as e:
            assert False, f"""
            ❌ LAYER INTERFACE DESIGN BROKEN!

            🔍 ERROR: {str(e)}

            🔧 HOW TO FIX:

            Your Layer class should look like:

            class Layer:
                def forward(self, x):
                    raise NotImplementedError("Subclasses must implement forward()")

                def __call__(self, x):
                    return self.forward(x)

            📝 DESIGN PRINCIPLES:
            - forward(): Abstract method, subclasses override this
            - __call__(): Makes layer callable, delegates to forward()
            - Inheritance: All layers inherit this interface
            """

    def test_custom_layer_implementation(self):
        """
        ✅ TEST: Custom layer example - Verify inheritance works

        📋 WHAT THIS TESTS:
        - Can create custom layers that inherit from Layer
        - Custom forward() method works correctly
        - Layer interface is flexible and extensible

        💡 FOR STUDENTS: This shows how your Linear layer works in Module 03
        """
        try:
            from tinytorch.core.layers import Layer
            from tinytorch.core.tensor import Tensor

            # Create a simple custom layer for testing
            class TestLayer(Layer):
                def __init__(self, multiplier=2):
                    self.multiplier = multiplier

                def forward(self, x):
                    # Simple operation: multiply input by constant
                    return Tensor(x.data * self.multiplier)

            # Test custom layer
            test_layer = TestLayer(multiplier=3)

            # Test with tensor input
            x = Tensor([1, 2, 3])
            output = test_layer(x)  # Should call __call__ -> forward()

            expected = np.array([3, 6, 9])
            assert np.array_equal(output.data, expected), \
                f"❌ Custom layer broken. Expected {expected}, got {output.data}"

            # Test that it's using the Layer interface
            assert isinstance(test_layer, Layer), \
                "❌ Custom layer should inherit from Layer"

        except ImportError:
            assert False, """
            ❌ Cannot test custom layer - missing dependencies!
            Need both Layer (Module 04) and Tensor (Module 02) working.
            """
        except Exception as e:
            assert False, f"""
            ❌ CUSTOM LAYER IMPLEMENTATION BROKEN!

            🔍 ERROR: {str(e)}

            🔧 POSSIBLE ISSUES:
            1. Layer.__call__ not working correctly
            2. Layer inheritance broken
            3. forward() method implementation issues
            4. Tensor compatibility problems

            💡 DEBUG STEPS:
            1. Test Layer base class separately
            2. Check Tensor operations work
            3. Verify inheritance: isinstance(custom_layer, Layer)
            4. Test method calls: custom_layer.forward(x)
            """


class TestProgressiveStackIntegration:
    """
    🔗 INTEGRATION TEST: Layer + Tensor + Activations working together.

    💡 This tests the "progressive stack" - ensuring all modules 01→04 work together.
    🎯 Goal: Verify you can build neural network components using everything so far.
    """

    def test_layer_tensor_integration(self):
        """
        ✅ TEST: Layers work correctly with Tensors

        📋 INTEGRATION POINTS:
        - Layer.forward() accepts Tensor objects
        - Layer returns Tensor objects
        - Tensor operations work within layers

        💡 This is crucial for Module 05 (DataLoader layers)
        """
        try:
            from tinytorch.core.layers import Layer
            from tinytorch.core.tensor import Tensor

            # Create a layer that actually works with tensors
            class IdentityLayer(Layer):
                def forward(self, x):
                    # Return input unchanged (identity function)
                    return Tensor(x.data)

            layer = IdentityLayer()

            # Test with different tensor shapes
            test_cases = [
                ([1, 2, 3], (3,)),
                ([[1, 2], [3, 4]], (2, 2)),
                (np.random.randn(5, 10), (5, 10))
            ]

            for input_data, expected_shape in test_cases:
                x = Tensor(input_data)
                output = layer(x)

                assert isinstance(output, Tensor), \
                    f"❌ Layer should return Tensor, got {type(output)}"

                assert output.shape == expected_shape, \
                    f"❌ Wrong output shape. Expected {expected_shape}, got {output.shape}"

                assert np.array_equal(output.data, x.data), \
                    "❌ Identity layer should preserve data"

        except Exception as e:
            assert False, f"""
            ❌ LAYER-TENSOR INTEGRATION BROKEN!

            🔍 ERROR: {str(e)}

            🔧 COMMON ISSUES:
            1. Layer.forward() doesn't accept Tensor objects
            2. Layer doesn't return Tensor objects
            3. Tensor.data access broken
            4. Shape handling incorrect

            💡 INTEGRATION REQUIREMENTS:
            - Layers must accept Tensor inputs
            - Layers must return Tensor outputs
            - Preserve tensor properties (shape, data)

            🧪 DEBUG TEST:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Layer

            t = Tensor([1, 2, 3])
            print(f"Tensor shape: {{t.shape}}")
            print(f"Tensor data: {{t.data}}")
            """

    def test_layer_activation_integration(self):
        """
        ✅ TEST: Layers work with activation functions

        📋 INTEGRATION POINTS:
        - Layer output can be fed to activations
        - Activations can be chained with layers
        - Combined operations preserve tensor properties

        💡 This is the foundation for neural networks: layer -> activation -> layer
        """
        try:
            from tinytorch.core.layers import Layer
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.activations import ReLU

            # Simple layer for testing
            class ScaleLayer(Layer):
                def forward(self, x):
                    # Scale input by 2, then subtract 1 (some values will be negative)
                    return Tensor(x.data * 2 - 1)

            # Test layer -> activation chain
            layer = ScaleLayer()
            relu = ReLU()

            # Input that will produce both positive and negative values after scaling
            x = Tensor([0, 0.5, 1.0])  # After scaling: [-1, 0, 1]

            # Layer -> Activation chain
            layer_output = layer(x)
            final_output = relu(layer_output)

            # Check the chain works
            expected_after_layer = np.array([-1, 0, 1])
            expected_after_relu = np.array([0, 0, 1])

            assert np.array_equal(layer_output.data, expected_after_layer), \
                f"❌ Layer output wrong. Expected {expected_after_layer}, got {layer_output.data}"

            assert np.array_equal(final_output.data, expected_after_relu), \
                f"❌ ReLU after layer wrong. Expected {expected_after_relu}, got {final_output.data}"

        except Exception as e:
            assert False, f"""
            ❌ LAYER-ACTIVATION INTEGRATION BROKEN!

            🔍 ERROR: {str(e)}

            🔧 INTEGRATION REQUIREMENTS:
            1. Layer output should be valid Tensor
            2. Activation should accept layer output
            3. Chain: input -> layer -> activation -> output

            💡 NEURAL NETWORK BUILDING BLOCKS:
            This integration is crucial because neural networks are:
            input -> layer -> activation -> layer -> activation -> ... -> output

            🧪 DEBUG TEST:
            x = Tensor([1, 2, 3])
            layer_out = layer(x)
            activation_out = relu(layer_out)
            print(f"Layer: {{layer_out.data}}")
            print(f"ReLU: {{activation_out.data}}")
            """

    def test_complete_stack_readiness(self):
        """
        ✅ TEST: Complete stack (01→04) ready for neural networks

        📋 COMPREHENSIVE CHECK:
        - Environment (01) working
        - Tensors (02) working
        - Activations (03) working
        - Layers (04) working
        - All components work together

        🎯 MILESTONE: Ready for Module 05 (DataLoader Networks)!
        """
        try:
            # Import all components
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.activations import ReLU, Sigmoid
            from tinytorch.core.layers import Layer

            # Create complete neural network building blocks
            class LinearLayer(Layer):
                def forward(self, x):
                    # Simple linear transformation: y = x + 1
                    return Tensor(x.data + 1)

            # Test complete pipeline
            x = Tensor([-2, -1, 0, 1, 2])

            # Build mini neural network: input -> layer -> relu -> layer -> sigmoid
            layer1 = LinearLayer()   # x + 1: [-1, 0, 1, 2, 3]
            relu = ReLU()           # max(0, x): [0, 0, 1, 2, 3]
            layer2 = LinearLayer()   # x + 1: [1, 1, 2, 3, 4]
            sigmoid = Sigmoid()      # 1/(1+exp(-x)): ~[0.73, 0.73, 0.88, 0.95, 0.98]

            # Forward pass through complete stack
            h1 = layer1(x)
            h1_act = relu(h1)
            h2 = layer2(h1_act)
            output = sigmoid(h2)

            # Verify each step
            assert h1.shape == x.shape, "❌ Layer 1 output shape wrong"
            assert h1_act.shape == x.shape, "❌ ReLU output shape wrong"
            assert h2.shape == x.shape, "❌ Layer 2 output shape wrong"
            assert output.shape == x.shape, "❌ Sigmoid output shape wrong"

            # Verify sigmoid output range
            assert np.all(output.data >= 0) and np.all(output.data <= 1), \
                "❌ Sigmoid output not in range [0,1]"

            # Verify the complete computation chain worked
            assert len(output.data) == 5, "❌ Complete stack pipeline broken"

        except Exception as e:
            assert False, f"""
            ❌ COMPLETE STACK NOT READY!

            🔍 ERROR: {str(e)}

            🎯 STACK REQUIREMENTS (Modules 01→04):
            ✅ Module 01: Python environment, NumPy, project structure
            ✅ Module 02: Tensor class with data and shape
            ✅ Module 03: ReLU and Sigmoid activation functions
            ✅ Module 03: Layer base class with forward() and __call__()

            🔧 FIX EACH MODULE:
            1. Run individual module tests to isolate the issue
            2. Check exports: tito module complete XX_modulename
            3. Verify imports work: from tinytorch.core.* import *

            💡 DEBUGGING STRATEGY:
            Test each component separately:
            - Tensor: t = Tensor([1,2,3]); print(t.shape)
            - ReLU: r = ReLU(); print(r(Tensor([-1,1])).data)
            - Layer: l = Layer(); print(callable(l))
            """


class TestNeuralNetworkReadiness:
    """
    🧠 NEURAL NETWORK READINESS: Test foundation is ready for real neural networks.

    💡 These tests verify you have everything needed for Module 05 (DataLoader Networks).
    🎯 Goal: Confirm the foundation supports building actual neural networks.
    """

    def test_parameter_handling_foundation(self):
        """
        ✅ TEST: Foundation supports neural network parameters (weights, biases)

        📋 WHAT NEURAL NETWORKS NEED:
        - Store parameters (weights, biases)
        - Update parameters during training
        - Initialize parameters properly

        💡 This prepares for Dense layer implementation
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Layer

            # Test parameter storage and manipulation
            class ParameterizedLayer(Layer):
                def __init__(self, input_size, output_size):
                    # Neural network parameters
                    self.weight = Tensor(np.random.randn(input_size, output_size))
                    self.bias = Tensor(np.random.randn(output_size))

                def forward(self, x):
                    # Simple linear transformation: y = x @ W + b
                    # Note: Using numpy for now, will use Tensor ops later
                    output_data = np.dot(x.data, self.weight.data) + self.bias.data
                    return Tensor(output_data)

            # Test parameterized layer
            layer = ParameterizedLayer(3, 2)  # 3 inputs -> 2 outputs

            # Check parameters exist and have correct shapes
            assert hasattr(layer, 'weight'), "❌ Layer missing weights parameter"
            assert hasattr(layer, 'bias'), "❌ Layer missing bias parameter"
            assert layer.weight.shape == (3, 2), \
                f"❌ Wrong weight shape. Expected (3, 2), got {layer.weight.shape}"
            assert layer.bias.shape == (2,), \
                f"❌ Wrong bias shape. Expected (2,), got {layer.bias.shape}"

            # Test forward pass
            x = Tensor([[1, 2, 3], [4, 5, 6]])  # 2 samples, 3 features each
            output = layer(x)

            assert output.shape == (2, 2), \
                f"❌ Wrong output shape. Expected (2, 2), got {output.shape}"

        except Exception as e:
            assert False, f"""
            ❌ PARAMETER HANDLING FOUNDATION BROKEN!

            🔍 ERROR: {str(e)}

            🔧 NEURAL NETWORK REQUIREMENTS:
            1. Layers must store parameters (weights, bias)
            2. Parameters must be Tensor objects
            3. Forward pass must use parameters correctly
            4. Matrix operations must work

            💡 WHAT THIS ENABLES:
            - Linear/Dense layers (Module 03)
            - Convolutional layers (Module 09)
            - All parameterized neural network components

            🧪 DEBUG CHECKLIST:
            □ Can create Tensor with random data?
            □ Can access Tensor.shape property?
            □ Can do matrix multiplication with numpy?
            □ Can store Tensors as instance variables?
            """

    def test_batch_processing_foundation(self):
        """
        ✅ TEST: Foundation supports batch processing (multiple samples at once)

        📋 BATCH PROCESSING:
        - Process multiple inputs simultaneously
        - Maintain correct dimensions
        - Enable efficient training

        💡 Essential for real neural network training
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.activations import ReLU

            # Test batch processing
            batch_size = 32
            feature_dim = 10

            # Create batch of data
            batch_data = Tensor(np.random.randn(batch_size, feature_dim))

            # Test batch processing through activation
            relu = ReLU()
            batch_output = relu(batch_data)

            # Verify batch dimensions preserved
            assert batch_output.shape == (batch_size, feature_dim), \
                f"❌ Batch processing broken. Expected {(batch_size, feature_dim)}, got {batch_output.shape}"

            # Verify ReLU applied element-wise to entire batch
            assert np.all(batch_output.data >= 0), \
                "❌ ReLU not applied correctly to batch"

            # Test that we can process different batch sizes
            for test_batch_size in [1, 16, 64]:
                test_batch = Tensor(np.random.randn(test_batch_size, feature_dim))
                test_output = relu(test_batch)
                assert test_output.shape == (test_batch_size, feature_dim), \
                    f"❌ Batch size {test_batch_size} processing broken"

        except Exception as e:
            assert False, f"""
            ❌ BATCH PROCESSING FOUNDATION BROKEN!

            🔍 ERROR: {str(e)}

            🔧 BATCH PROCESSING REQUIREMENTS:
            1. Tensors must support 2D+ arrays (batch_size, features)
            2. Operations must work element-wise on batches
            3. Shape preservation through operations
            4. Support variable batch sizes

            💡 WHY BATCH PROCESSING MATTERS:
            - Training uses batches of data (e.g., 32 samples at once)
            - Much more efficient than processing one sample at a time
            - Essential for modern deep learning

            🧪 DEBUG TEST:
            batch = Tensor(np.random.randn(4, 3))  # 4 samples, 3 features
            print(f"Batch shape: {{batch.shape}}")
            relu = ReLU()
            output = relu(batch)
            print(f"Output shape: {{output.shape}}")
            """

    def test_neural_network_workflow_ready(self):
        """
        ✅ TEST: Complete neural network workflow foundation

        📋 WORKFLOW COMPONENTS:
        - Data input (batches)
        - Layer processing
        - Activation functions
        - Output generation

        🎯 MILESTONE: Ready for Module 05 DataLoader Networks!
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Layer
            from tinytorch.core.activations import ReLU, Sigmoid

            # Simulate complete neural network workflow
            class MockDenseLayer(Layer):
                def __init__(self, in_features, out_features):
                    self.weight = Tensor(np.random.randn(in_features, out_features) * 0.1)
                    self.bias = Tensor(np.random.randn(out_features) * 0.1)

                def forward(self, x):
                    # Dense layer: y = x @ W + b
                    output = np.dot(x.data, self.weight.data) + self.bias.data
                    return Tensor(output)

            # Build mini neural network: 784 -> 128 -> 64 -> 10 (like MNIST)
            layer1 = MockDenseLayer(784, 128)
            layer2 = MockDenseLayer(128, 64)
            layer3 = MockDenseLayer(64, 10)
            relu = ReLU()
            sigmoid = Sigmoid()

            # Simulate MNIST-like input
            batch_size = 32
            x = Tensor(np.random.randn(batch_size, 784))  # Flattened 28x28 images

            # Forward pass through network
            h1 = relu(layer1(x))      # 32 x 128
            h2 = relu(layer2(h1))     # 32 x 64
            logits = layer3(h2)       # 32 x 10
            output = sigmoid(logits)  # 32 x 10

            # Verify complete workflow
            assert output.shape == (32, 10), \
                f"❌ Neural network output shape wrong. Expected (32, 10), got {output.shape}"

            # Verify output is valid probabilities (after sigmoid)
            assert np.all(output.data >= 0) and np.all(output.data <= 1), \
                "❌ Neural network output not in valid range [0, 1]"

            # Verify no NaN or infinity values
            assert not np.any(np.isnan(output.data)), \
                "❌ Neural network produced NaN values"
            assert not np.any(np.isinf(output.data)), \
                "❌ Neural network produced infinity values"

        except Exception as e:
            assert False, f"""
            ❌ NEURAL NETWORK WORKFLOW NOT READY!

            🔍 ERROR: {str(e)}

            🎯 COMPLETE WORKFLOW REQUIREMENTS:
            1. ✅ Layers can be chained together
            2. ✅ Activations work between layers
            3. ✅ Batch processing throughout
            4. ✅ Proper shapes maintained
            5. ✅ Numerical stability

            🎉 SUCCESS MEANS:
            You can build real neural networks like:
            - MNIST digit classification (784 -> 128 -> 64 -> 10)
            - Image classification CNNs
            - Language models with transformers

            🔧 IF FAILING:
            Check each component individually:
            1. Layer parameter storage and forward pass
            2. Activation function batch processing
            3. Tensor shape handling
            4. Matrix multiplication correctness

            💡 YOU'RE ALMOST THERE!
            This failure means you're very close to building real neural networks.
            Fix the specific error above and you'll have a working foundation!
            """


class TestModuleCompletionReadiness:
    """
    ✅ COMPLETION CHECK: Module 04 ready and foundation set for Module 05.

    🎯 This is the final validation that everything works before moving to Dense Networks.
    """

    def test_module_04_complete(self):
        """
        ✅ FINAL TEST: Module 04 (Layers) is complete and working

        📋 COMPLETION CHECKLIST:
        □ Layer base class implemented
        □ Proper interface design (__call__, forward)
        □ Integration with previous modules
        □ Ready for inheritance (Dense, Conv2D, etc.)

        🎯 SUCCESS = Ready for Module 05: DataLoader!
        """
        completion_checklist = {
            "Layer base class exists": False,
            "Layer has forward() method": False,
            "Layer is callable": False,
            "Layer works with Tensors": False,
            "Layer chains with activations": False,
            "Foundation supports parameters": False,
            "Foundation supports batches": False,
            "Ready for neural networks": False
        }

        try:
            # Check 1: Layer base class
            from tinytorch.core.layers import Layer
            completion_checklist["Layer base class exists"] = True

            # Check 2: forward method
            layer = Layer()
            assert hasattr(layer, 'forward')
            completion_checklist["Layer has forward() method"] = True

            # Check 3: callable
            assert callable(layer)
            completion_checklist["Layer is callable"] = True

            # Check 4: works with tensors
            from tinytorch.core.tensor import Tensor

            class TestLayer(Layer):
                def forward(self, x):
                    return Tensor(x.data)

            test_layer = TestLayer()
            x = Tensor([1, 2, 3])
            output = test_layer(x)
            assert isinstance(output, Tensor)
            completion_checklist["Layer works with Tensors"] = True

            # Check 5: chains with activations
            from tinytorch.core.activations import ReLU
            relu = ReLU()
            chained_output = relu(output)
            assert isinstance(chained_output, Tensor)
            completion_checklist["Layer chains with activations"] = True

            # Check 6: supports parameters
            class ParameterLayer(Layer):
                def __init__(self):
                    self.weight = Tensor([0.5])

                def forward(self, x):
                    return Tensor(x.data * self.weight.data)

            param_layer = ParameterLayer()
            param_output = param_layer(x)
            completion_checklist["Foundation supports parameters"] = True

            # Check 7: supports batches
            batch_x = Tensor(np.random.randn(4, 3))
            batch_output = test_layer(batch_x)
            assert batch_output.shape == (4, 3)
            completion_checklist["Foundation supports batches"] = True

            # Check 8: ready for neural networks
            completion_checklist["Ready for neural networks"] = True

        except Exception as e:
            # Show progress even if not complete
            completed_count = sum(completion_checklist.values())
            total_count = len(completion_checklist)

            progress_report = "\n🔍 COMPLETION PROGRESS:\n"
            for check, completed in completion_checklist.items():
                status = "✅" if completed else "❌"
                progress_report += f"  {status} {check}\n"

            progress_report += f"\n📊 Progress: {completed_count}/{total_count} checks passed"

            assert False, f"""
            ❌ MODULE 04 NOT COMPLETE!

            🔍 ERROR: {str(e)}

            {progress_report}

            🔧 NEXT STEPS:
            1. Fix the failing check above
            2. Re-run this test
            3. When all ✅, you're ready for Module 05!

            💡 ALMOST THERE!
            You've completed {completed_count}/{total_count} requirements.
            Just fix the error above and you'll have a complete Layer foundation!
            """

        # If we get here, everything passed!
        assert True, """
        🎉 MODULE 04 COMPLETE! 🎉

        ✅ Layer base class implemented
        ✅ Proper interface design
        ✅ Integration with foundation
        ✅ Ready for neural networks

        🚀 READY FOR MODULE 05: DATALOADER!

        💡 What you can now do:
        - Inherit from Layer to create Dense layers
        - Build multi-layer neural networks
        - Train on real datasets
        - Solve non-linear problems like XOR

        🎯 Next: Implement Losses in Module 04!
        """


# Note: No regression prevention section needed here since we ARE testing regression
# by checking that modules 01-03 still work after implementing module 04.
