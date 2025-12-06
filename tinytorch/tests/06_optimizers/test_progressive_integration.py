"""
Module 06: Progressive Integration Tests
Tests that Module 06 (Spatial/CNN Operations) works correctly AND that the foundation stack (01→05) still works.

DEPENDENCY CHAIN: 01_setup → 02_tensor → 03_activations → 04_layers → 05_dense → 06_spatial
This is where we enable spatial processing for images and computer vision.

🎯 WHAT THIS TESTS:
- Module 06: Convolutional layers, pooling operations, spatial processing
- Integration: CNNs work with tensors, layers, and activations from previous modules
- Regression: Foundation stack (01→05) still works correctly
- Preparation: Ready for advanced architectures (attention, training, etc.)

💡 FOR STUDENTS: If tests fail, check:
1. Does your Conv2D class exist in tinytorch.core.spatial?
2. Does Conv2D inherit from Layer (Module 04)?
3. Do convolution operations work with Tensor objects?
4. Are spatial dimensions handled correctly?

🔧 DEBUGGING HELP:
- Conv2D input: (batch_size, channels, height, width)
- Conv2D output: (batch_size, out_channels, out_height, out_width)
- Pooling reduces spatial dimensions but preserves channels
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestFoundationStackStillWorks:
    """
    🔄 REGRESSION CHECK: Verify foundation stack (01→05) still works after spatial development.
    
    💡 If these fail: You may have broken something in the foundation while working on CNN operations.
    🔧 Fix: Check that your spatial code doesn't interfere with basic neural network functionality.
    """
    
    def test_foundation_pipeline_stable(self):
        """
        ✅ TEST: Complete foundation pipeline (01→05) should still work
        
        📋 FOUNDATION COMPONENTS:
        - Setup environment working
        - Tensor operations working
        - Activation functions working
        - Layer base class working
        - Dense networks working
        
        🚨 IF FAILS: Core foundation broken by spatial development
        """
        try:
            # Test foundation components still work
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            
            # Create simple neural network
            dense = Linear(10, 5)
            relu = ReLU()
            
            # Test forward pass
            x = Tensor(np.random.randn(4, 10))
            h = dense(x)
            output = relu(h)
            
            assert output.shape == (4, 5), \
                f"❌ Foundation broken. Expected (4, 5), got {output.shape}"
            
            assert np.all(output.data >= 0), \
                "❌ ReLU not working in foundation"
            
        except ImportError as e:
            assert False, f"""
            ❌ FOUNDATION IMPORT BROKEN!
            
            🔍 IMPORT ERROR: {str(e)}
            
            🔧 HOW TO FIX:
            1. Check all foundation modules are exported correctly
            2. Run: tito module complete 02_tensor
            3. Run: tito module complete 04_layers  
            4. Run: tito module complete 05_dense
            5. Test imports individually:
               from tinytorch.core.tensor import Tensor
               from tinytorch.core.layers import Linear
               from tinytorch.core.activations import ReLU
            
            💡 FOUNDATION REQUIREMENTS:
            - Tensor: Basic tensor operations
            - Dense: Fully connected layers
            - ReLU: Non-linear activations
            - Layer: Base class for all layers
            """
        except Exception as e:
            assert False, f"""
            ❌ FOUNDATION FUNCTIONALITY BROKEN!
            
            🔍 ERROR: {str(e)}
            
            🔧 POSSIBLE CAUSES:
            1. Dense layer forward pass broken
            2. ReLU activation function broken
            3. Tensor operations corrupted
            4. Layer inheritance issues
            
            💡 DEBUG STEPS:
            1. Test each component separately
            2. Check Dense layer: dense = Linear(5, 3); print(linear.weight.shape)
            3. Check ReLU: relu = ReLU(); print(relu(Tensor([-1, 1])).data)
            4. Run foundation tests: python tests/run_all_modules.py --module module_05
            """
    
    def test_neural_network_capability_stable(self):
        """
        ✅ TEST: Can still build neural networks after adding spatial operations
        
        📋 NEURAL NETWORK CAPABILITY:
        - Multi-layer networks
        - Non-linear problem solving
        - Batch processing
        - Parameter management
        
        🎯 This ensures spatial additions don't break core ML functionality
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU, Sigmoid
            
            # Build 3-layer network for XOR problem
            layer1 = Linear(2, 4, bias=True)
            layer2 = Linear(4, 1, bias=True)
            relu = ReLU()
            sigmoid = Sigmoid()
            
            # XOR problem inputs
            X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
            
            # Forward pass through complete network
            h = relu(layer1(X))         # Non-linear hidden layer
            logits = layer2(h)          # Output layer
            predictions = sigmoid(logits)  # Probabilities
            
            assert predictions.shape == (4, 1), \
                f"❌ Neural network shape broken. Expected (4, 1), got {predictions.shape}"
            
            assert np.all(predictions.data >= 0) and np.all(predictions.data <= 1), \
                "❌ Neural network output not in valid range [0, 1]"
            
            # Network should have capacity for XOR (non-linear problem)
            param_count = layer1.weight.data.size + layer1.bias.data.size + \
                         layer2.weight.data.size + layer2.bias.data.size
            
            assert param_count >= 9, \
                f"❌ Network has insufficient parameters for XOR. Need ≥9, got {param_count}"
            
        except Exception as e:
            assert False, f"""
            ❌ NEURAL NETWORK CAPABILITY BROKEN!
            
            🔍 ERROR: {str(e)}
            
            🔧 NEURAL NETWORK REQUIREMENTS:
            1. Dense layers must work correctly
            2. Activations must chain properly
            3. Multi-layer networks must function
            4. Batch processing must work
            5. Parameter storage must be intact
            
            💡 XOR PROBLEM TEST:
            This is a key capability test because XOR requires:
            - Non-linear activation functions
            - Multi-layer architecture  
            - Sufficient parameters
            
            🧪 DEBUG CHECKLIST:
            □ Dense layer creates correct weight/bias shapes?
            □ ReLU applies element-wise to all inputs?
            □ Sigmoid produces values in [0, 1] range?
            □ Layer chaining preserves tensor operations?
            """


class TestModule06SpatialCore:
    """
    🆕 NEW FUNCTIONALITY: Test Module 06 (Spatial/CNN) core implementation.
    
    💡 What you're implementing: Convolutional and pooling operations for computer vision.
    🎯 Goal: Enable processing of images and spatial data with CNNs.
    """
    
    def test_conv2d_layer_exists(self):
        """
        ✅ TEST: Conv2D layer - Core of convolutional neural networks
        
        📋 WHAT YOU NEED TO IMPLEMENT:
        class Conv2D(Layer):
            def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
                # Initialize convolutional weights and bias
            def forward(self, x):
                # Perform 2D convolution operation
        
        🚨 IF FAILS: Conv2D layer doesn't exist or missing components
        """
        try:
            from tinytorch.core.spatial import Conv2D
            from tinytorch.core.layers import Layer
            
            # Conv2D should inherit from Layer
            assert issubclass(Conv2D, Layer), \
                "❌ Conv2D must inherit from Layer base class"
            
            # Test Conv2D creation
            conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
            
            # Should have convolutional parameters
            assert hasattr(conv, 'weight') or hasattr(conv, 'kernel'), \
                "❌ Conv2D missing convolution weights/kernel"
            
            # Should be callable (inherits from Layer)
            assert callable(conv), \
                "❌ Conv2D should be callable (inherit __call__ from Layer)"
            
            # Check parameter shapes (basic validation)
            if hasattr(conv, 'weight'):
                weights = conv.weights
                expected_shape = (16, 3, 3, 3)  # (out_channels, in_channels, kernel_h, kernel_w)
                assert weights.shape == expected_shape, \
                    f"❌ Conv2D weights wrong shape. Expected {expected_shape}, got {weights.shape}"
            
        except ImportError as e:
            assert False, f"""
            ❌ CONV2D LAYER MISSING!
            
            🔍 IMPORT ERROR: {str(e)}
            
            🔧 HOW TO IMPLEMENT:
            
            1. Create in modules/06_spatial/06_spatial.py:
            
            from tinytorch.core.layers import Layer
            from tinytorch.core.tensor import Tensor
            import numpy as np
            
            class Conv2D(Layer):
                '''2D Convolutional layer for computer vision.'''
                
                def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
                    self.in_channels = in_channels
                    self.out_channels = out_channels
                    self.kernel_size = kernel_size
                    self.stride = stride
                    self.padding = padding
                    
                    # Initialize convolution weights
                    # Shape: (out_channels, in_channels, kernel_size, kernel_size)
                    self.weights = Tensor(np.random.randn(
                        out_channels, in_channels, kernel_size, kernel_size
                    ) * 0.1)
                    
                    # Initialize bias
                    self.bias = Tensor(np.random.randn(out_channels) * 0.1)
                
                def forward(self, x):
                    # Implement 2D convolution
                    # Input: (batch_size, in_channels, height, width)
                    # Output: (batch_size, out_channels, out_height, out_width)
                    
                    # For now, simplified implementation
                    batch_size, in_ch, height, width = x.shape
                    
                    # Calculate output dimensions
                    out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
                    out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
                    
                    # Placeholder implementation (you'll implement actual convolution)
                    output_shape = (batch_size, self.out_channels, out_height, out_width)
                    output_data = np.random.randn(*output_shape)  # Replace with real convolution
                    
                    return Tensor(output_data)
            
            2. Export the module:
               tito module complete 06_spatial
            
            📚 CONVOLUTION CONCEPTS:
            - Kernel/Filter: Small weight matrix that slides over input
            - Stride: How much kernel moves each step
            - Padding: Zero-padding around input edges
            - Output size: (input + 2*padding - kernel) / stride + 1
            """
        except Exception as e:
            assert False, f"""
            ❌ CONV2D LAYER BROKEN!
            
            🔍 ERROR: {str(e)}
            
            🔧 CONV2D REQUIREMENTS:
            1. Must inherit from Layer base class
            2. Must have __init__ with (in_channels, out_channels, kernel_size)
            3. Must have weights with shape (out_ch, in_ch, k_size, k_size)
            4. Must have forward() method
            5. Must be callable via Layer.__call__()
            
            💡 COMPUTER VISION FOUNDATION:
            Conv2D is the core building block for:
            - Image classification (ResNet, VGG)
            - Object detection (YOLO, R-CNN)
            - Image generation (GANs, VAEs)
            - Medical imaging, autonomous driving, etc.
            """
    
    def test_pooling_operations(self):
        """
        ✅ TEST: Pooling operations - Reduce spatial dimensions in CNNs
        
        📋 POOLING TYPES:
        - MaxPool2D: Take maximum value in each region
        - AvgPool2D: Take average value in each region
        - Used to reduce overfitting and computational cost
        
        🎯 Essential for efficient CNN architectures
        """
        try:
            from tinytorch.core.spatial import MaxPool2D
            from tinytorch.core.tensor import Tensor
            
            # Test MaxPool2D creation
            pool = MaxPool2D(kernel_size=2, stride=2)
            
            # Test pooling operation
            # Input: 4x4 image, pooling 2x2 -> 2x2 output
            x = Tensor(np.array([[[[1, 2, 3, 4],
                                  [5, 6, 7, 8],
                                  [9, 10, 11, 12],
                                  [13, 14, 15, 16]]]], dtype=np.float32))  # (1, 1, 4, 4)
            
            output = pool(x)
            
            # MaxPool 2x2 should take max of each 2x2 region
            expected_shape = (1, 1, 2, 2)
            assert output.shape == expected_shape, \
                f"❌ MaxPool output shape wrong. Expected {expected_shape}, got {output.shape}"
            
            # Check values (max of each 2x2 region)
            expected_values = np.array([[[[6, 8], [14, 16]]]])  # Max of each 2x2 block
            assert np.array_equal(output.data, expected_values), \
                f"❌ MaxPool values wrong. Expected {expected_values}, got {output.data}"
            
        except ImportError as e:
            assert False, f"""
            ❌ POOLING OPERATIONS MISSING!
            
            🔍 IMPORT ERROR: {str(e)}
            
            🔧 HOW TO IMPLEMENT MaxPool2D:
            
            class MaxPool2D:
                '''2D Max pooling for downsampling spatial dimensions.'''
                
                def __init__(self, kernel_size, stride=None):
                    self.kernel_size = kernel_size
                    self.stride = stride if stride is not None else kernel_size
                
                def __call__(self, x):
                    # Input: (batch_size, channels, height, width)
                    batch_size, channels, height, width = x.shape
                    
                    # Calculate output dimensions
                    out_height = height // self.stride
                    out_width = width // self.stride
                    
                    # Perform max pooling (simplified implementation)
                    output = np.zeros((batch_size, channels, out_height, out_width))
                    
                    for b in range(batch_size):
                        for c in range(channels):
                            for h in range(out_height):
                                for w in range(out_width):
                                    h_start = h * self.stride
                                    w_start = w * self.stride
                                    h_end = h_start + self.kernel_size
                                    w_end = w_start + self.kernel_size
                                    
                                    # Take maximum in this region
                                    region = x.data[b, c, h_start:h_end, w_start:w_end]
                                    output[b, c, h, w] = np.max(region)
                    
                    return Tensor(output)
            
            💡 POOLING PURPOSE:
            - Reduces spatial dimensions (4x4 -> 2x2)
            - Reduces parameters and computation
            - Provides translation invariance
            - Prevents overfitting
            """
        except Exception as e:
            assert False, f"""
            ❌ POOLING OPERATIONS BROKEN!
            
            🔍 ERROR: {str(e)}
            
            🔧 POOLING REQUIREMENTS:
            1. MaxPool2D takes kernel_size and stride parameters
            2. Input shape: (batch, channels, height, width)
            3. Output shape: (batch, channels, out_height, out_width)
            4. Operation: take max value in each kernel_size x kernel_size region
            5. Stride determines how much to move kernel each step
            
            🧪 DEBUG TEST:
            x = Tensor(np.arange(16).reshape(1, 1, 4, 4))  # 0-15 in 4x4
            pool = MaxPool2D(kernel_size=2)
            y = pool(x)
            print(f"Input: {{x.data}}")
            print(f"Output: {{y.data}}")  # Should be max of each 2x2 region
            """
    
    def test_spatial_tensor_operations(self):
        """
        ✅ TEST: Spatial operations work correctly with 4D tensors
        
        📋 4D TENSOR FORMAT:
        - Dimension 0: Batch size (number of images)
        - Dimension 1: Channels (RGB = 3, grayscale = 1)
        - Dimension 2: Height (image height in pixels)
        - Dimension 3: Width (image width in pixels)
        
        💡 This is the standard format for computer vision
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.spatial import Conv2D
            
            # Test 4D tensor creation and manipulation
            batch_size, channels, height, width = 2, 3, 32, 32
            
            # Create batch of RGB images
            images = Tensor(np.random.randn(batch_size, channels, height, width))
            
            assert images.shape == (2, 3, 32, 32), \
                f"❌ 4D tensor creation broken. Expected (2, 3, 32, 32), got {images.shape}"
            
            # Test convolution with 4D tensors
            conv = Conv2D(in_channels=3, out_channels=16, kernel_size=5, padding=2)
            conv_output = conv(images)
            
            # With padding=2 and kernel_size=5, spatial dimensions should be preserved
            expected_shape = (2, 16, 32, 32)
            assert conv_output.shape == expected_shape, \
                f"❌ Conv2D with 4D tensors broken. Expected {expected_shape}, got {conv_output.shape}"
            
            # Test different spatial sizes
            small_images = Tensor(np.random.randn(1, 1, 8, 8))
            small_conv = Conv2D(in_channels=1, out_channels=4, kernel_size=3)
            small_output = small_conv(small_images)
            
            # 8x8 input with 3x3 kernel -> 6x6 output
            expected_small_shape = (1, 4, 6, 6)
            assert small_output.shape == expected_small_shape, \
                f"❌ Small Conv2D broken. Expected {expected_small_shape}, got {small_output.shape}"
            
        except Exception as e:
            assert False, f"""
            ❌ SPATIAL TENSOR OPERATIONS BROKEN!
            
            🔍 ERROR: {str(e)}
            
            🔧 4D TENSOR REQUIREMENTS:
            1. Support (batch, channels, height, width) format
            2. Convolution preserves batch and channel semantics
            3. Spatial dimensions computed correctly:
               output_size = (input_size + 2*padding - kernel_size) / stride + 1
            4. Handle different input sizes correctly
            
            💡 COMPUTER VISION TENSOR FORMAT:
            - MNIST: (batch, 1, 28, 28) - grayscale 28x28 images
            - CIFAR-10: (batch, 3, 32, 32) - RGB 32x32 images  
            - ImageNet: (batch, 3, 224, 224) - RGB 224x224 images
            
            🧪 DEBUG SPATIAL DIMENSIONS:
            Input: H_in = 32, W_in = 32
            Kernel: K = 5, Padding: P = 2, Stride: S = 1
            Output: H_out = (32 + 2*2 - 5) / 1 + 1 = 32
            
            Test this calculation in your implementation!
            """


class TestSpatialIntegration:
    """
    🔗 INTEGRATION TEST: Spatial operations + Foundation stack working together.
    
    💡 Test that CNNs can be built using the complete progressive stack.
    🎯 Goal: Build convolutional neural networks for computer vision.
    """
    
    def test_cnn_architecture_building(self):
        """
        ✅ TEST: Can build complete CNN architectures
        
        📋 CNN ARCHITECTURE:
        input -> conv -> relu -> pool -> conv -> relu -> pool -> dense -> output
        
        💡 This is the foundation for all computer vision models
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.spatial import Conv2D, MaxPool2D
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU, Softmax
            
            # Build mini CNN for CIFAR-10 style classification
            # Input: 32x32 RGB images, Output: 10 classes
            
            # Convolutional layers
            conv1 = Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1)  # 32x32 -> 32x32
            pool1 = MaxPool2D(kernel_size=2, stride=2)  # 32x32 -> 16x16
            conv2 = Conv2D(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # 16x16 -> 16x16
            pool2 = MaxPool2D(kernel_size=2, stride=2)  # 16x16 -> 8x8
            
            # Dense layers (after flattening)
            # 32 channels * 8 * 8 = 2048 features
            fc1 = Linear(32 * 8 * 8, 128)
            fc2 = Linear(128, 10)
            
            # Activations
            relu = ReLU()
            softmax = Softmax()
            
            # Test forward pass through complete CNN
            batch_size = 4
            x = Tensor(np.random.randn(batch_size, 3, 32, 32))  # Batch of CIFAR-10 images
            
            # Convolutional feature extraction
            h1 = relu(conv1(x))      # (4, 16, 32, 32)
            h1_pool = pool1(h1)      # (4, 16, 16, 16)
            h2 = relu(conv2(h1_pool)) # (4, 32, 16, 16)
            h2_pool = pool2(h2)      # (4, 32, 8, 8)
            
            # Flatten for dense layers
            flattened = Tensor(h2_pool.data.reshape(batch_size, -1))  # (4, 2048)
            
            # Classification layers
            h3 = relu(fc1(flattened))  # (4, 128)
            logits = fc2(h3)          # (4, 10)
            output = softmax(logits)   # (4, 10)
            
            # Verify complete CNN pipeline
            assert output.shape == (4, 10), \
                f"❌ CNN output shape wrong. Expected (4, 10), got {output.shape}"
            
            # Verify softmax probabilities
            prob_sums = np.sum(output.data, axis=1)
            assert np.allclose(prob_sums, 1.0), \
                f"❌ CNN softmax broken. Probabilities don't sum to 1: {prob_sums}"
            
            # Verify feature extraction pipeline
            assert h1.shape == (4, 16, 32, 32), "❌ Conv1 output shape wrong"
            assert h1_pool.shape == (4, 16, 16, 16), "❌ Pool1 output shape wrong"
            assert h2.shape == (4, 32, 16, 16), "❌ Conv2 output shape wrong"
            assert h2_pool.shape == (4, 32, 8, 8), "❌ Pool2 output shape wrong"
            
        except Exception as e:
            assert False, f"""
            ❌ CNN ARCHITECTURE BUILDING BROKEN!
            
            🔍 ERROR: {str(e)}
            
            🔧 CNN PIPELINE REQUIREMENTS:
            1. ✅ Spatial operations (Conv2D, MaxPool2D)
            2. ✅ Foundation operations (Dense, ReLU, Softmax)
            3. ✅ 4D tensor handling throughout
            4. ✅ Shape preservation and transformation
            5. ✅ Integration between spatial and dense layers
            
            💡 CNN ARCHITECTURE PATTERN:
            [Input Images] 
                ↓ 
            [Conv2D + ReLU] → Extract spatial features
                ↓
            [MaxPool2D] → Reduce spatial dimensions
                ↓
            [Conv2D + ReLU] → Extract higher-level features  
                ↓
            [MaxPool2D] → Further dimension reduction
                ↓
            [Flatten] → Convert to 1D for dense layers
                ↓
            [Dense + ReLU] → Classification features
                ↓
            [Dense + Softmax] → Class probabilities
            
            🧪 DEBUG CNN SHAPES:
            Input: (batch=4, channels=3, height=32, width=32)
            Conv1: (4, 16, 32, 32) - 16 feature maps
            Pool1: (4, 16, 16, 16) - halved spatial size
            Conv2: (4, 32, 16, 16) - 32 feature maps
            Pool2: (4, 32, 8, 8) - halved again
            Flatten: (4, 2048) - 32*8*8 = 2048 features
            Dense: (4, 10) - 10 class scores
            """
    
    def test_image_processing_pipeline(self):
        """
        ✅ TEST: Complete image processing pipeline
        
        📋 IMAGE PROCESSING:
        - Load and preprocess images
        - Extract features with CNNs
        - Make predictions
        - Handle different image sizes
        
        🎯 Real-world computer vision workflow
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.spatial import Conv2D, MaxPool2D
            from tinytorch.core.activations import ReLU
            
            # Simulate different image processing scenarios
            
            # Scenario 1: MNIST-style grayscale images
            mnist_images = Tensor(np.random.randn(8, 1, 28, 28))  # 8 images, 1 channel, 28x28
            mnist_conv = Conv2D(in_channels=1, out_channels=8, kernel_size=5)
            mnist_features = mnist_conv(mnist_images)
            
            expected_mnist_shape = (8, 8, 24, 24)  # 28-5+1 = 24
            assert mnist_features.shape == expected_mnist_shape, \
                f"❌ MNIST processing broken. Expected {expected_mnist_shape}, got {mnist_features.shape}"
            
            # Scenario 2: CIFAR-10 style RGB images
            cifar_images = Tensor(np.random.randn(16, 3, 32, 32))  # 16 images, 3 channels, 32x32
            cifar_conv = Conv2D(in_channels=3, out_channels=64, kernel_size=3, padding=1)
            cifar_pool = MaxPool2D(kernel_size=2)
            
            cifar_features = cifar_conv(cifar_images)
            cifar_pooled = cifar_pool(cifar_features)
            
            assert cifar_features.shape == (16, 64, 32, 32), "❌ CIFAR conv broken"
            assert cifar_pooled.shape == (16, 64, 16, 16), "❌ CIFAR pooling broken"
            
            # Scenario 3: Multi-scale feature extraction
            relu = ReLU()
            
            # Small features (fine details)
            small_conv = Conv2D(in_channels=3, out_channels=32, kernel_size=3)
            small_features = relu(small_conv(cifar_images))
            
            # Large features (global patterns)  
            large_conv = Conv2D(in_channels=3, out_channels=32, kernel_size=7)
            large_features = relu(large_conv(cifar_images))
            
            # Both should extract meaningful features
            assert small_features.shape[1] == 32, "❌ Small feature extraction broken"
            assert large_features.shape[1] == 32, "❌ Large feature extraction broken"
            assert np.all(small_features.data >= 0), "❌ Small features ReLU broken"
            assert np.all(large_features.data >= 0), "❌ Large features ReLU broken"
            
        except Exception as e:
            assert False, f"""
            ❌ IMAGE PROCESSING PIPELINE BROKEN!
            
            🔍 ERROR: {str(e)}
            
            🔧 IMAGE PROCESSING REQUIREMENTS:
            1. Handle different image formats (grayscale, RGB)
            2. Support various image sizes (28x28, 32x32, etc.)
            3. Extract features at different scales
            4. Maintain spatial relationships
            5. Work with batches of images
            
            💡 REAL-WORLD APPLICATIONS:
            - Medical imaging: X-rays, MRIs, CT scans
            - Autonomous driving: Camera feeds, object detection
            - Security: Face recognition, surveillance
            - Entertainment: Photo filters, style transfer
            - Science: Satellite imagery, microscopy
            
            🧪 IMAGE PROCESSING CHECKLIST:
            □ MNIST (28x28 grayscale): Medical imaging, digit recognition
            □ CIFAR-10 (32x32 RGB): Object classification
            □ ImageNet (224x224 RGB): General computer vision
            □ Multi-scale features: Fine details + global patterns
            """
    
    def test_cnn_spatial_hierarchies(self):
        """
        ✅ TEST: CNNs build spatial feature hierarchies
        
        📋 FEATURE HIERARCHIES:
        - Early layers: Edges, corners, simple patterns
        - Middle layers: Shapes, textures, objects parts
        - Late layers: Complete objects, complex patterns
        
        💡 This is why CNNs work so well for computer vision
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.spatial import Conv2D, MaxPool2D
            from tinytorch.core.activations import ReLU
            
            # Build hierarchical CNN feature extractor
            relu = ReLU()
            
            # Layer 1: Low-level features (edges, corners)
            conv1 = Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1)
            pool1 = MaxPool2D(kernel_size=2)
            
            # Layer 2: Mid-level features (shapes, textures)
            conv2 = Conv2D(in_channels=16, out_channels=32, kernel_size=3, padding=1)
            pool2 = MaxPool2D(kernel_size=2)
            
            # Layer 3: High-level features (object parts)
            conv3 = Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            pool3 = MaxPool2D(kernel_size=2)
            
            # Test feature hierarchy with realistic image
            x = Tensor(np.random.randn(1, 3, 64, 64))  # Single 64x64 RGB image
            
            # Extract features at each level
            # Level 1: 64x64 -> 32x32 (low-level features)
            features_1 = relu(conv1(x))      # (1, 16, 64, 64)
            pooled_1 = pool1(features_1)     # (1, 16, 32, 32)
            
            # Level 2: 32x32 -> 16x16 (mid-level features)
            features_2 = relu(conv2(pooled_1)) # (1, 32, 32, 32)
            pooled_2 = pool2(features_2)      # (1, 32, 16, 16)
            
            # Level 3: 16x16 -> 8x8 (high-level features)
            features_3 = relu(conv3(pooled_2)) # (1, 64, 16, 16)
            pooled_3 = pool3(features_3)      # (1, 64, 8, 8)
            
            # Verify hierarchical feature extraction
            assert features_1.shape == (1, 16, 64, 64), "❌ Level 1 features broken"
            assert pooled_1.shape == (1, 16, 32, 32), "❌ Level 1 pooling broken"
            assert features_2.shape == (1, 32, 32, 32), "❌ Level 2 features broken"
            assert pooled_2.shape == (1, 32, 16, 16), "❌ Level 2 pooling broken"
            assert features_3.shape == (1, 64, 16, 16), "❌ Level 3 features broken"
            assert pooled_3.shape == (1, 64, 8, 8), "❌ Level 3 pooling broken"
            
            # Verify feature complexity increases (more channels, smaller spatial)
            channel_progression = [16, 32, 64]
            spatial_progression = [(32, 32), (16, 16), (8, 8)]
            
            for i, (channels, spatial) in enumerate(zip(channel_progression, spatial_progression)):
                level = i + 1
                assert channels > (8 if i == 0 else channel_progression[i-1]), \
                    f"❌ Level {level}: Feature complexity not increasing"
                
                h, w = spatial
                assert h < (64 if i == 0 else spatial_progression[i-1][0]), \
                    f"❌ Level {level}: Spatial size not decreasing"
            
        except Exception as e:
            assert False, f"""
            ❌ CNN SPATIAL HIERARCHIES BROKEN!
            
            🔍 ERROR: {str(e)}
            
            🔧 HIERARCHICAL CNN REQUIREMENTS:
            1. Early layers extract simple features (edges, corners)
            2. Later layers extract complex features (objects, patterns)
            3. Spatial resolution decreases through network
            4. Feature complexity (channels) increases through network
            5. Each level builds on previous level features
            
            💡 CNN FEATURE HIERARCHY:
            
            Level 1 (64x64 → 32x32):
            - 16 channels detect edges, corners, simple patterns
            - High spatial resolution preserves fine details
            
            Level 2 (32x32 → 16x16):  
            - 32 channels detect shapes, textures, object parts
            - Medium spatial resolution focuses on local patterns
            
            Level 3 (16x16 → 8x8):
            - 64 channels detect complete objects, complex patterns
            - Low spatial resolution captures global structure
            
            🧠 WHY THIS WORKS:
            This mimics the human visual system:
            - Retina → edges and motion
            - V1 → oriented edges and bars  
            - V2 → shapes and textures
            - V4 → objects and faces
            """


class TestComputerVisionCapabilities:
    """
    🖼️ COMPUTER VISION CAPABILITIES: Test real-world CV applications.
    
    💡 Verify the spatial foundation enables actual computer vision tasks.
    🎯 Goal: Show students can now build real CV systems.
    """
    
    def test_image_classification_capability(self):
        """
        ✅ TEST: Can build image classification systems
        
        📋 IMAGE CLASSIFICATION:
        - Input: Images
        - Output: Class probabilities
        - Applications: Medical diagnosis, quality control, content moderation
        
        💡 This is the "Hello World" of computer vision
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.spatial import Conv2D, MaxPool2D
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU, Softmax
            
            # Build classifier for 10 classes (CIFAR-10 style)
            class ImageClassifier:
                def __init__(self, num_classes=10):
                    # Feature extraction (convolutional layers)
                    self.conv1 = Conv2D(3, 32, kernel_size=3, padding=1)
                    self.pool1 = MaxPool2D(kernel_size=2)
                    self.conv2 = Conv2D(32, 64, kernel_size=3, padding=1)
                    self.pool2 = MaxPool2D(kernel_size=2)
                    
                    # Classification (dense layers)
                    self.fc1 = Linear(64 * 8 * 8, 128)  # Assuming 32x32 input
                    self.fc2 = Linear(128, num_classes)
                    
                    # Activations
                    self.relu = ReLU()
                    self.softmax = Softmax()
                
                def __call__(self, x):
                    # Feature extraction
                    h1 = self.relu(self.conv1(x))     # Extract low-level features
                    h1_pool = self.pool1(h1)          # Downsample
                    h2 = self.relu(self.conv2(h1_pool)) # Extract high-level features
                    h2_pool = self.pool2(h2)          # Downsample
                    
                    # Flatten for classification
                    batch_size = h2_pool.shape[0]
                    flattened = Tensor(h2_pool.data.reshape(batch_size, -1))
                    
                    # Classification
                    h3 = self.relu(self.fc1(flattened))
                    logits = self.fc2(h3)
                    probabilities = self.softmax(logits)
                    
                    return probabilities
            
            # Test image classifier
            classifier = ImageClassifier(num_classes=10)
            
            # Batch of test images
            test_images = Tensor(np.random.randn(5, 3, 32, 32))
            predictions = classifier(test_images)
            
            # Verify classifier output
            assert predictions.shape == (5, 10), \
                f"❌ Classifier shape wrong. Expected (5, 10), got {predictions.shape}"
            
            # Verify probabilities sum to 1
            prob_sums = np.sum(predictions.data, axis=1)
            assert np.allclose(prob_sums, 1.0, atol=1e-6), \
                f"❌ Classifier probabilities don't sum to 1: {prob_sums}"
            
            # Verify probabilities in valid range
            assert np.all(predictions.data >= 0) and np.all(predictions.data <= 1), \
                "❌ Classifier probabilities not in [0, 1] range"
            
            # Test prediction extraction (most likely class)
            predicted_classes = np.argmax(predictions.data, axis=1)
            assert len(predicted_classes) == 5, "❌ Prediction extraction broken"
            assert all(0 <= cls < 10 for cls in predicted_classes), \
                "❌ Predicted classes out of range"
            
        except Exception as e:
            assert False, f"""
            ❌ IMAGE CLASSIFICATION CAPABILITY BROKEN!
            
            🔍 ERROR: {str(e)}
            
            🔧 IMAGE CLASSIFICATION REQUIREMENTS:
            1. CNN feature extraction (Conv2D + pooling)
            2. Dense classification layers
            3. Softmax probability output
            4. Batch processing support
            5. End-to-end differentiable pipeline
            
            💡 REAL-WORLD APPLICATIONS:
            
            🏥 Medical Imaging:
            - X-ray diagnosis (pneumonia detection)
            - Skin cancer classification
            - Retinal disease detection
            
            🚗 Autonomous Vehicles:
            - Traffic sign recognition
            - Pedestrian detection
            - Lane boundary detection
            
            🏭 Quality Control:
            - Defect detection in manufacturing
            - Food quality assessment
            - Product sorting and grading
            
            📱 Consumer Applications:
            - Photo tagging and search
            - Content moderation
            - Augmented reality filters
            """
    
    def test_feature_extraction_capability(self):
        """
        ✅ TEST: Can extract meaningful visual features
        
        📋 FEATURE EXTRACTION:
        - Low-level: Edges, corners, textures
        - High-level: Objects, shapes, patterns
        - Transfer learning: Features from one task help another
        
        💡 Feature extraction is the foundation of all computer vision
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.spatial import Conv2D, MaxPool2D
            from tinytorch.core.activations import ReLU
            
            # Build feature extractor
            class FeatureExtractor:
                def __init__(self):
                    # Multi-scale feature extraction
                    self.small_features = Conv2D(3, 16, kernel_size=3, padding=1)  # Fine details
                    self.medium_features = Conv2D(3, 16, kernel_size=5, padding=2)  # Medium patterns
                    self.large_features = Conv2D(3, 16, kernel_size=7, padding=3)   # Large patterns
                    
                    # Feature refinement
                    self.refine = Conv2D(48, 32, kernel_size=1)  # 1x1 conv for feature fusion
                    self.pool = MaxPool2D(kernel_size=2)
                    self.relu = ReLU()
                
                def extract_features(self, x):
                    # Extract features at multiple scales
                    small = self.relu(self.small_features(x))
                    medium = self.relu(self.medium_features(x))
                    large = self.relu(self.large_features(x))
                    
                    # Concatenate multi-scale features
                    # In real implementation, would use tensor concatenation
                    # For now, simulate by combining channels
                    combined_data = np.concatenate([small.data, medium.data, large.data], axis=1)
                    combined = Tensor(combined_data)
                    
                    # Refine combined features
                    refined = self.relu(self.refine(combined))
                    pooled = self.pool(refined)
                    
                    return pooled
            
            # Test feature extraction
            extractor = FeatureExtractor()
            
            # Test with different types of images
            test_cases = [
                ("Natural images", np.random.randn(3, 3, 64, 64)),
                ("Medical images", np.random.randn(2, 3, 128, 128)),
                ("Satellite images", np.random.randn(1, 3, 256, 256))
            ]
            
            for name, image_data in test_cases:
                images = Tensor(image_data)
                features = extractor.extract_features(images)
                
                batch_size = images.shape[0]
                expected_channels = 32
                expected_spatial = (images.shape[2] // 2, images.shape[3] // 2)  # Halved by pooling
                
                assert features.shape[0] == batch_size, f"❌ {name}: Batch size wrong"
                assert features.shape[1] == expected_channels, f"❌ {name}: Feature channels wrong"
                assert features.shape[2:] == expected_spatial, f"❌ {name}: Spatial dimensions wrong"
                
                # Features should be meaningful (not all zeros)
                assert not np.allclose(features.data, 0), f"❌ {name}: Features are all zeros"
                
                # ReLU should ensure non-negative features
                assert np.all(features.data >= 0), f"❌ {name}: Features contain negative values"
            
        except Exception as e:
            assert False, f"""
            ❌ FEATURE EXTRACTION CAPABILITY BROKEN!
            
            🔍 ERROR: {str(e)}
            
            🔧 FEATURE EXTRACTION REQUIREMENTS:
            1. Multi-scale feature detection (small, medium, large)
            2. Feature combination and refinement
            3. Spatial dimension handling
            4. Meaningful feature representations
            5. Transfer learning capability
            
            💡 FEATURE EXTRACTION APPLICATIONS:
            
            🔬 Scientific Research:
            - Analyzing microscopy images
            - Identifying cellular structures
            - Tracking biological processes
            
            🛰️ Remote Sensing:
            - Land use classification
            - Environmental monitoring
            - Disaster response planning
            
            🎨 Creative Applications:
            - Style transfer (artistic filters)
            - Image enhancement
            - Content-aware editing
            
            🤖 Robotics:
            - Object recognition and grasping
            - Navigation and mapping
            - Human-robot interaction
            
            💡 TRANSFER LEARNING:
            Features learned on one dataset (ImageNet) transfer to:
            - Medical imaging with small datasets
            - Specialized domains (satellite, microscopy)
            - New tasks with limited training data
            """
    
    def test_spatial_understanding_capability(self):
        """
        ✅ TEST: CNNs understand spatial relationships
        
        📋 SPATIAL UNDERSTANDING:
        - Local patterns: Textures, edges within small regions
        - Global structure: Object layout, scene composition
        - Translation invariance: Same object anywhere in image
        - Scale invariance: Objects at different sizes
        
        💡 This is what makes CNNs powerful for vision
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.spatial import Conv2D, MaxPool2D
            from tinytorch.core.activations import ReLU
            
            # Test spatial understanding with different spatial patterns
            relu = ReLU()
            
            # Pattern detector
            pattern_detector = Conv2D(1, 8, kernel_size=3, padding=1)
            spatial_pool = MaxPool2D(kernel_size=2)
            
            # Create test images with known spatial patterns
            batch_size = 4
            
            # Pattern 1: Vertical stripes
            vertical_stripes = np.zeros((1, 1, 16, 16))
            vertical_stripes[0, 0, :, ::2] = 1  # Every other column
            
            # Pattern 2: Horizontal stripes  
            horizontal_stripes = np.zeros((1, 1, 16, 16))
            horizontal_stripes[0, 0, ::2, :] = 1  # Every other row
            
            # Pattern 3: Checkerboard
            checkerboard = np.zeros((1, 1, 16, 16))
            for i in range(16):
                for j in range(16):
                    if (i + j) % 2 == 0:
                        checkerboard[0, 0, i, j] = 1
            
            # Pattern 4: Center blob
            center_blob = np.zeros((1, 1, 16, 16))
            center_blob[0, 0, 6:10, 6:10] = 1
            
            # Combine patterns into batch
            patterns = np.concatenate([vertical_stripes, horizontal_stripes, 
                                     checkerboard, center_blob], axis=0)
            pattern_tensor = Tensor(patterns)
            
            # Extract features for each pattern
            features = relu(pattern_detector(pattern_tensor))
            pooled_features = spatial_pool(features)
            
            # Test spatial pattern detection
            assert features.shape == (4, 8, 16, 16), \
                f"❌ Pattern features shape wrong. Expected (4, 8, 16, 16), got {features.shape}"
            
            assert pooled_features.shape == (4, 8, 8, 8), \
                f"❌ Pooled features shape wrong. Expected (4, 8, 8, 8), got {pooled_features.shape}"
            
            # Features should be different for different patterns
            for i in range(4):
                for j in range(i+1, 4):
                    pattern_i_features = features.data[i].flatten()
                    pattern_j_features = features.data[j].flatten()
                    
                    # Patterns should produce different features
                    assert not np.allclose(pattern_i_features, pattern_j_features, rtol=0.1), \
                        f"❌ Patterns {i} and {j} produce identical features"
            
            # Test translation invariance (same pattern, different location)
            shifted_blob = np.zeros((1, 1, 16, 16))
            shifted_blob[0, 0, 2:6, 2:6] = 1  # Same blob, different position
            
            original_blob_tensor = Tensor(center_blob)
            shifted_blob_tensor = Tensor(shifted_blob)
            
            original_features = relu(pattern_detector(original_blob_tensor))
            shifted_features = relu(pattern_detector(shifted_blob_tensor))
            
            # After pooling, features should be similar (translation invariance)
            original_pooled = spatial_pool(original_features)
            shifted_pooled = spatial_pool(shifted_features)
            
            # Global feature similarity (though not exact due to edge effects)
            original_global = np.mean(original_pooled.data)
            shifted_global = np.mean(shifted_pooled.data)
            
            assert abs(original_global - shifted_global) < 0.5, \
                "❌ Translation invariance broken: shifted pattern too different"
            
        except Exception as e:
            assert False, f"""
            ❌ SPATIAL UNDERSTANDING CAPABILITY BROKEN!
            
            🔍 ERROR: {str(e)}
            
            🔧 SPATIAL UNDERSTANDING REQUIREMENTS:
            1. Pattern detection: Different spatial patterns produce different features
            2. Translation invariance: Same pattern different locations → similar features
            3. Local processing: Convolution respects spatial neighborhoods
            4. Hierarchical understanding: Local → global feature extraction
            5. Spatial pooling: Reduce spatial resolution while preserving features
            
            💡 SPATIAL UNDERSTANDING ENABLES:
            
            🖼️ Image Analysis:
            - Object detection: "Where is the cat in the image?"
            - Semantic segmentation: "Which pixels belong to the road?"
            - Instance segmentation: "Separate the two cars in the image"
            
            🏥 Medical Imaging:
            - Tumor localization: "Where is the abnormal tissue?"
            - Anatomical structure identification
            - Disease progression tracking over time
            
            🚗 Autonomous Navigation:
            - Lane detection: "Where are the road boundaries?"
            - Obstacle avoidance: "What objects are in my path?"
            - Traffic sign recognition: "What does this sign mean?"
            
            🎮 Augmented Reality:
            - Object tracking in real-time
            - Spatial registration of virtual objects
            - Hand gesture recognition
            """


class TestModule06Completion:
    """
    ✅ COMPLETION CHECK: Module 06 ready and foundation set for advanced architectures.
    
    🎯 Final validation that spatial operations work and foundation supports computer vision.
    """
    
    def test_computer_vision_foundation_complete(self):
        """
        ✅ FINAL TEST: Complete computer vision foundation ready
        
        📋 CV FOUNDATION CHECKLIST:
        □ Convolutional operations (Conv2D)
        □ Pooling operations (MaxPool2D)
        □ 4D tensor handling (batch, channels, height, width)
        □ Spatial feature hierarchies
        □ Integration with dense layers
        □ Image classification capability
        □ Feature extraction capability  
        □ Spatial understanding
        
        🎯 SUCCESS = Ready for advanced CV architectures!
        """
        cv_capabilities = {
            "Conv2D operations": False,
            "Pooling operations": False,
            "4D tensor handling": False,
            "CNN architecture building": False,
            "Image classification": False,
            "Feature extraction": False,
            "Spatial understanding": False,
            "Foundation integration": False
        }
        
        try:
            # Test 1: Conv2D operations
            from tinytorch.core.spatial import Conv2D
            conv = Conv2D(3, 16, kernel_size=3)
            cv_capabilities["Conv2D operations"] = True
            
            # Test 2: Pooling operations
            from tinytorch.core.spatial import MaxPool2D
            pool = MaxPool2D(kernel_size=2)
            cv_capabilities["Pooling operations"] = True
            
            # Test 3: 4D tensor handling
            from tinytorch.core.tensor import Tensor
            x = Tensor(np.random.randn(2, 3, 32, 32))
            conv_out = conv(x)
            assert len(conv_out.shape) == 4
            cv_capabilities["4D tensor handling"] = True
            
            # Test 4: CNN architecture building
            from tinytorch.core.activations import ReLU
            from tinytorch.core.layers import Linear
            
            relu = ReLU()
            h1 = relu(conv_out)
            h1_pool = pool(h1)
            
            # Flatten and connect to dense
            flattened = Tensor(h1_pool.data.reshape(2, -1))
            dense = Linear(flattened.shape[1], 10)
            output = dense(flattened)
            
            assert output.shape == (2, 10)
            cv_capabilities["CNN architecture building"] = True
            
            # Test 5: Image classification capability
            from tinytorch.core.activations import Softmax
            softmax = Softmax()
            probs = softmax(output)
            
            prob_sums = np.sum(probs.data, axis=1)
            assert np.allclose(prob_sums, 1.0)
            cv_capabilities["Image classification"] = True
            
            # Test 6: Feature extraction
            features = relu(conv(x))
            assert np.all(features.data >= 0)  # ReLU features
            assert not np.allclose(features.data, 0)  # Non-trivial features
            cv_capabilities["Feature extraction"] = True
            
            # Test 7: Spatial understanding
            small_x = Tensor(np.random.randn(1, 3, 8, 8))
            small_conv = Conv2D(3, 8, kernel_size=3)
            small_features = small_conv(small_x)
            assert small_features.shape == (1, 8, 6, 6)  # Correct spatial calculation
            cv_capabilities["Spatial understanding"] = True
            
            # Test 8: Foundation integration
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear, Layer
            from tinytorch.core.activations import ReLU
            
            # All foundation components should work together
            assert issubclass(Conv2D, Layer)  # Inherits from Layer
            cv_capabilities["Foundation integration"] = True
            
        except Exception as e:
            # Show progress even if not complete
            completed_count = sum(cv_capabilities.values())
            total_count = len(cv_capabilities)
            
            progress_report = "\n🔍 COMPUTER VISION PROGRESS:\n"
            for capability, completed in cv_capabilities.items():
                status = "✅" if completed else "❌"
                progress_report += f"  {status} {capability}\n"
            
            progress_report += f"\n📊 Progress: {completed_count}/{total_count} capabilities ready"
            
            assert False, f"""
            ❌ COMPUTER VISION FOUNDATION NOT COMPLETE!
            
            🔍 ERROR: {str(e)}
            
            {progress_report}
            
            🔧 NEXT STEPS:
            1. Fix the failing capability above
            2. Re-run this test
            3. When all ✅, you have complete computer vision foundation!
            
            💡 ALMOST THERE!
            You've completed {completed_count}/{total_count} CV capabilities.
            Just fix the error above and you'll be ready for advanced vision architectures!
            """
        
        # If we get here, everything passed!
        assert True, f"""
        🎉 COMPUTER VISION FOUNDATION COMPLETE! 🎉
        
        ✅ Conv2D convolutional operations
        ✅ MaxPool2D pooling operations  
        ✅ 4D tensor handling (batch, channels, height, width)
        ✅ CNN architecture building
        ✅ Image classification capability
        ✅ Feature extraction capability
        ✅ Spatial understanding and processing
        ✅ Complete foundation integration
        
        🚀 READY FOR ADVANCED COMPUTER VISION!
        
        💡 What you can now build:
        - Image classifiers (MNIST, CIFAR-10, ImageNet)
        - Object detection systems
        - Medical image analysis
        - Autonomous vehicle vision
        - Artistic style transfer
        - And much more!
        
        🎯 Next modules will add:
        - Attention mechanisms (Module 07)
        - Data loading pipelines (Module 08)  
        - Training loops (Module 11)
        - Advanced optimizations (Module 13)
        
        🏆 ACHIEVEMENT UNLOCKED: Computer Vision Engineer!
        """


# Note: No separate regression prevention class needed - we test foundation stability above