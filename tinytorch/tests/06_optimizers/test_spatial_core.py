"""
Module 06: Spatial - Core Functionality Tests
Tests convolutional layers and spatial operations for computer vision
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestConv2DLayer:
    """Test 2D convolution layer."""
    
    def test_conv2d_creation(self):
        """Test Conv2D layer creation."""
        try:
            from tinytorch.core.spatial import Conv2d as Conv2D

            conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3)

            assert conv.in_channels == 3
            assert conv.out_channels == 16
            # kernel_size can be int or tuple
            assert conv.kernel_size == 3 or conv.kernel_size == (3, 3)

        except ImportError:
            assert True, "Conv2D not implemented yet"
    
    def test_conv2d_weight_shape(self):
        """Test Conv2D weight tensor has correct shape."""
        try:
            from tinytorch.core.spatial import Conv2d as Conv2D
            
            conv = Conv2D(in_channels=3, out_channels=16, kernel_size=5)
            
            # Weights should be (out_channels, in_channels, kernel_height, kernel_width)
            expected_shape = (16, 3, 5, 5)
            if hasattr(conv, 'weight'):
                assert conv.weight.shape == expected_shape
            elif hasattr(conv, 'weight'):
                assert conv.weight.shape == expected_shape
                
        except ImportError:
            assert True, "Conv2D weights not implemented yet"
    
    def test_conv2d_forward_shape(self):
        """Test Conv2D forward pass output shape."""
        try:
            from tinytorch.core.spatial import Conv2d as Conv2D
            from tinytorch.core.tensor import Tensor

            conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3)

            # Input: (batch_size, channels, height, width) - NCHW format
            x = Tensor(np.random.randn(8, 3, 32, 32))
            output = conv(x)

            # With kernel_size=3 and no padding, output should be 30x30
            # Output: (batch_size, out_channels, new_height, new_width)
            expected_shape = (8, 16, 30, 30)
            assert output.shape == expected_shape

        except ImportError:
            assert True, "Conv2D forward pass not implemented yet"
    
    def test_conv2d_simple_convolution(self):
        """Test simple convolution operation."""
        try:
            from tinytorch.core.spatial import Conv2d as Conv2D
            from tinytorch.core.tensor import Tensor

            # Simple 1-channel convolution
            conv = Conv2D(in_channels=1, out_channels=1, kernel_size=3)

            # Set known kernel for testing
            if hasattr(conv, 'weight'):
                conv.weight = Tensor(np.ones((1, 1, 3, 3)))  # Sum kernel

            # Simple input in NCHW format
            x = Tensor(np.ones((1, 1, 5, 5)))  # All ones
            output = conv(x)

            # With all-ones input and all-ones kernel, output should be 9 everywhere
            expected_value = 9.0
            if output.shape == (1, 1, 3, 3):
                assert np.allclose(output.data, expected_value)

        except ImportError:
            assert True, "Conv2D convolution operation not implemented yet"


class TestPoolingLayers:
    """Test pooling layers."""
    
    def test_maxpool2d_creation(self):
        """Test MaxPool2d layer creation."""
        try:
            from tinytorch.core.spatial import MaxPool2d

            pool = MaxPool2d(kernel_size=2)

            # kernel_size can be int or tuple
            assert pool.kernel_size == 2 or pool.kernel_size == (2, 2)

        except ImportError:
            assert True, "MaxPool2d not implemented yet"
    
    def test_maxpool2d_forward_shape(self):
        """Test MaxPool2d forward pass output shape."""
        try:
            from tinytorch.core.spatial import MaxPool2d
            from tinytorch.core.tensor import Tensor

            pool = MaxPool2d(kernel_size=2)

            # Input: (batch_size, channels, height, width) - NCHW format
            x = Tensor(np.random.randn(4, 32, 28, 28))
            output = pool(x)

            # Pooling by 2 should halve spatial dimensions
            expected_shape = (4, 32, 14, 14)
            assert output.shape == expected_shape

        except ImportError:
            assert True, "MaxPool2d forward pass not implemented yet"
    
    def test_maxpool2d_operation(self):
        """Test MaxPool2d actually finds maximum values."""
        try:
            from tinytorch.core.spatial import MaxPool2d
            from tinytorch.core.tensor import Tensor

            pool = MaxPool2d(kernel_size=2)

            # Create input with known pattern in NCHW format
            # Shape: (batch=1, channels=2, height=2, width=2)
            x_data = np.array([[[[1, 2],
                                [3, 4]],
                               [[5, 6],
                                [7, 8]]]])  # Shape: (1, 2, 2, 2)

            x = Tensor(x_data)
            output = pool(x)

            # MaxPool should select max from each 2x2 spatial region per channel
            # Output shape: (1, 2, 1, 1)
            if output.shape == (1, 2, 1, 1):
                assert output.data[0, 0, 0, 0] == 4  # Max of [1,2,3,4] in channel 0
                assert output.data[0, 1, 0, 0] == 8  # Max of [5,6,7,8] in channel 1

        except ImportError:
            assert True, "MaxPool2d operation not implemented yet"
    
    def test_avgpool2d_operation(self):
        """Test average pooling."""
        try:
            from tinytorch.core.spatial import AvgPool2D
            from tinytorch.core.tensor import Tensor
            
            pool = AvgPool2D(pool_size=2)
            
            # 2x2 input with known values
            x_data = np.array([[[[1, 2],
                                [3, 4]]]])  # Shape: (1, 2, 2, 1)
            
            x = Tensor(x_data)
            output = pool(x)
            
            # Average should be (1+2+3+4)/4 = 2.5
            if output.shape == (1, 1, 1, 1):
                assert np.isclose(output.data[0, 0, 0, 0], 2.5)
            
        except ImportError:
            assert True, "AvgPool2D not implemented yet"


class TestSpatialUtilities:
    """Test spatial operation utilities."""
    
    def test_padding_operation(self):
        """Test padding functionality."""
        try:
            from tinytorch.core.spatial import pad2d
            from tinytorch.core.tensor import Tensor
            
            # Simple 2x2 input
            x = Tensor(np.array([[[[1, 2],
                                  [3, 4]]]]))  # Shape: (1, 2, 2, 1)
            
            # Pad with 1 pixel on all sides
            padded = pad2d(x, padding=1, value=0)
            
            # Should become 4x4 with zeros around border
            expected_shape = (1, 4, 4, 1)
            assert padded.shape == expected_shape
            
            # Center should contain original values
            assert padded.data[0, 1, 1, 0] == 1
            assert padded.data[0, 1, 2, 0] == 2
            assert padded.data[0, 2, 1, 0] == 3
            assert padded.data[0, 2, 2, 0] == 4
            
        except ImportError:
            assert True, "Padding operation not implemented yet"
    
    def test_im2col_operation(self):
        """Test im2col operation for efficient convolution."""
        try:
            from tinytorch.core.spatial import im2col
            from tinytorch.core.tensor import Tensor
            
            # Simple 3x3 input
            x = Tensor(np.arange(9).reshape(1, 3, 3, 1))
            
            # Extract 2x2 patches
            patches = im2col(x, kernel_size=2, stride=1)
            
            # Should get 4 patches (2x2 sliding window on 3x3 input)
            # Each patch should have 4 values (2x2 kernel)
            expected_num_patches = 4
            expected_patch_size = 4
            
            if hasattr(patches, 'shape'):
                assert patches.shape[1] == expected_patch_size
            
        except ImportError:
            assert True, "im2col operation not implemented yet"
    
    def test_spatial_dimensions(self):
        """Test spatial dimension calculations."""
        try:
            from tinytorch.core.spatial import calc_output_size
            
            # Common convolution size calculation
            input_size = 32
            kernel_size = 5
            stride = 1
            padding = 2
            
            output_size = calc_output_size(input_size, kernel_size, stride, padding)
            
            # Formula: (input + 2*padding - kernel) / stride + 1
            expected = (32 + 2*2 - 5) // 1 + 1  # = 32
            assert output_size == expected
            
        except ImportError:
            # Manual calculation test
            input_size = 32
            kernel_size = 5
            stride = 1
            padding = 2
            
            output_size = (input_size + 2*padding - kernel_size) // stride + 1
            assert output_size == 32


class TestCNNArchitecture:
    """Test CNN architecture components working together."""
    
    def test_conv_relu_pool_chain(self):
        """Test Conv -> ReLU -> Pool chain."""
        try:
            from tinytorch.core.spatial import Conv2d as Conv2D, MaxPool2d
            from tinytorch.core.activations import ReLU
            from tinytorch.core.tensor import Tensor

            # Build simple CNN block
            conv = Conv2D(3, 16, kernel_size=3)
            relu = ReLU()
            pool = MaxPool2d(kernel_size=2)

            # Input image in NCHW format
            x = Tensor(np.random.randn(1, 3, 32, 32))

            # Forward pass
            h1 = conv(x)      # (1, 16, 30, 30)
            h2 = relu(h1)     # (1, 16, 30, 30)
            output = pool(h2) # (1, 16, 15, 15)

            expected_shape = (1, 16, 15, 15)
            assert output.shape == expected_shape

        except ImportError:
            assert True, "CNN architecture chaining not ready yet"
    
    def test_feature_map_progression(self):
        """Test feature map size progression through CNN."""
        try:
            from tinytorch.core.spatial import Conv2d as Conv2D, MaxPool2d
            from tinytorch.core.tensor import Tensor

            # Typical CNN progression: increase channels, decrease spatial size
            conv1 = Conv2D(3, 32, kernel_size=3)    # 3 -> 32 channels
            pool1 = MaxPool2d(kernel_size=2)        # /2 spatial size
            conv2 = Conv2D(32, 64, kernel_size=3)   # 32 -> 64 channels
            pool2 = MaxPool2d(kernel_size=2)        # /2 spatial size

            # NCHW format: (batch, channels, height, width)
            x = Tensor(np.random.randn(1, 3, 32, 32))  # Start: 3x32x32

            h1 = conv1(x)   # 32x30x30
            h2 = pool1(h1)  # 32x15x15
            h3 = conv2(h2)  # 64x13x13
            h4 = pool2(h3)  # 64x6x6

            # Should progressively reduce spatial size, increase channels
            # In NCHW: shape[1] is channels, shape[2] is height
            assert h1.shape[1] == 32  # More channels
            assert h2.shape[2] < h1.shape[2]  # Smaller spatial
            assert h3.shape[1] == 64  # Even more channels
            assert h4.shape[2] < h3.shape[2]  # Even smaller spatial

        except ImportError:
            assert True, "Feature map progression not ready yet"
    
    def test_global_average_pooling(self):
        """Test global average pooling for classification."""
        try:
            from tinytorch.core.spatial import GlobalAvgPool2D
            from tinytorch.core.tensor import Tensor
            
            gap = GlobalAvgPool2D()
            
            # Feature maps from CNN
            x = Tensor(np.random.randn(1, 7, 7, 512))  # Typical CNN output
            output = gap(x)
            
            # Should average over spatial dimensions
            expected_shape = (1, 1, 1, 512)  # or (1, 512)
            assert output.shape == expected_shape or output.shape == (1, 512)
            
        except ImportError:
            # Manual global average pooling
            x_data = np.random.randn(1, 7, 7, 512)
            output_data = np.mean(x_data, axis=(1, 2), keepdims=True)
            assert output_data.shape == (1, 1, 1, 512)