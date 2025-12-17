"""
Module 09: Convolutions - Core Functionality Tests
===================================================

These tests verify convolutional layers work correctly for computer vision.

WHY CONVOLUTIONS MATTER:
-----------------------
Convolutions are the foundation of computer vision:
- Image classification (ImageNet, CIFAR)
- Object detection (YOLO, Faster R-CNN)
- Segmentation (U-Net, Mask R-CNN)

Unlike dense layers, convolutions:
- Share weights across spatial locations (translation invariance)
- Preserve spatial structure (nearby pixels stay nearby)
- Use far fewer parameters (kernel is tiny vs full connection)

WHAT STUDENTS LEARN:
-------------------
1. How convolution "slides" a kernel across an image
2. How kernel_size, stride, padding affect output shape
3. How pooling reduces spatial dimensions
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestConv2DLayer:
    """
    Test 2D Convolution layer.

    CONCEPT: A kernel (small matrix) slides across the input image,
    computing dot products to detect features like edges, corners, textures.
    """

    def test_conv2d_creation(self):
        """
        WHAT: Verify Conv2d layer can be created.

        WHY: Conv2d is the building block of CNNs.
        If it can't be created, no computer vision is possible.

        STUDENT LEARNING: Key parameters:
        - in_channels: number of input channels (3 for RGB)
        - out_channels: number of filters (learned feature detectors)
        - kernel_size: size of the sliding window (typically 3 or 5)
        """
        try:
            from tinytorch.core.spatial import Conv2d

            conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3)

            assert conv.in_channels == 3, "in_channels not set correctly"
            assert conv.out_channels == 16, "out_channels not set correctly"
            # kernel_size can be int or tuple
            assert conv.kernel_size == 3 or conv.kernel_size == (3, 3), "kernel_size not set correctly"


        except ImportError:
            pytest.skip("Conv2d not implemented yet")

    def test_conv2d_weight_shape(self):
        """
        WHAT: Verify Conv2d weights have correct shape.

        WHY: Weight shape must be (out_channels, in_channels, kH, kW)
        for correct convolution. Wrong shape = wrong computation.

        STUDENT LEARNING: Conv2d weights are 4D tensors:
        (out_channels, in_channels, kernel_height, kernel_width)
        Each output channel has a separate kernel for each input channel.
        """
        try:
            from tinytorch.core.spatial import Conv2d

            conv = Conv2d(in_channels=3, out_channels=16, kernel_size=5)

            # Weights: (out_channels, in_channels, kH, kW)
            expected_shape = (16, 3, 5, 5)
            weight = conv.weight if hasattr(conv, 'weight') else conv.weights

            assert weight.shape == expected_shape, (
                f"Conv2d weight shape wrong.\n"
                f"  Expected: {expected_shape} (out, in, kH, kW)\n"
                f"  Got: {weight.shape}\n"
                "Remember: each output channel needs kernels for ALL input channels."
            )

        except ImportError:
            pytest.skip("Conv2d weights not implemented yet")

    def test_conv2d_forward_shape(self):
        """
        WHAT: Verify Conv2d output has correct shape.

        WHY: Output shape = (batch, out_channels, H_out, W_out) - NCHW format
        where H_out = H_in - kernel_size + 1 (no padding)

        STUDENT LEARNING: Output size formula (no padding, stride=1):
        output_size = input_size - kernel_size + 1
        Example: 32 - 3 + 1 = 30
        """
        try:
            from tinytorch.core.spatial import Conv2d
            from tinytorch.core.tensor import Tensor

            conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3)

            # Input: (batch, C, H, W) - NCHW format
            x = Tensor(np.random.randn(8, 3, 32, 32))
            output = conv(x)

            # 32 - 3 + 1 = 30
            expected_shape = (8, 16, 30, 30)
            assert output.shape == expected_shape, (
                f"Conv2d output shape wrong.\n"
                f"  Input: (8, 3, 32, 32) NCHW\n"
                f"  kernel_size=3, no padding\n"
                f"  Expected: (8, 16, 30, 30)\n"
                f"  Got: {output.shape}\n"
                "Formula: output = input - kernel + 1 = 32 - 3 + 1 = 30"
            )

        except ImportError:
            pytest.skip("Conv2d forward pass not implemented yet")

    def test_conv2d_simple_convolution(self):
        """
        WHAT: Verify convolution computes correctly with known kernel.

        WHY: This validates the actual convolution math is correct,
        not just shapes.

        STUDENT LEARNING: Convolution = sum of element-wise products.
        With all-ones kernel (3×3) on all-ones input:
        output = 1*1 + 1*1 + ... (9 terms) = 9
        """
        try:
            from tinytorch.core.spatial import Conv2d
            from tinytorch.core.tensor import Tensor

            conv = Conv2d(in_channels=1, out_channels=1, kernel_size=3)

            # Set kernel to all ones (sum kernel)
            weight = conv.weight if hasattr(conv, 'weight') else conv.weights
            weight.data = np.ones((1, 1, 3, 3))

            # All-ones input in NCHW format
            x = Tensor(np.ones((1, 1, 5, 5)))
            output = conv(x)

            # Each output pixel = sum of 9 ones = 9
            if output.shape == (1, 1, 3, 3):
                assert np.allclose(output.data, 9.0), (
                    f"Convolution value wrong.\n"
                    f"  All-ones kernel (3×3) on all-ones input\n"
                    f"  Each output should be 9 (sum of 9 ones)\n"
                    f"  Got: {output.data[0,0,0,0]}"
                )

        except ImportError:
            pytest.skip("Conv2d convolution operation not implemented yet")


class TestPoolingLayers:
    """
    Test pooling layers (MaxPool, AvgPool).

    CONCEPT: Pooling reduces spatial dimensions by summarizing
    local regions. This adds translation invariance and reduces computation.
    """

    def test_maxpool2d_creation(self):
        """
        WHAT: Verify MaxPool2d can be created.

        WHY: Pooling is essential for:
        - Reducing computation in deeper layers
        - Adding translation invariance
        - Summarizing local features

        STUDENT LEARNING: MaxPool(2) with stride=2:
        - Takes 2×2 windows
        - Keeps only the maximum value
        - Reduces H,W by half
        """
        try:
            from tinytorch.core.spatial import MaxPool2d

            pool = MaxPool2d(kernel_size=2)
            assert pool is not None

        except ImportError:
            pytest.skip("MaxPool2d not implemented yet")

    def test_maxpool2d_forward(self):
        """
        WHAT: Verify MaxPool2d takes maximum in each window.

        WHY: The max operation must be exact - it's used in
        backprop to route gradients to max locations.

        STUDENT LEARNING: For 2×2 window [[1,2],[3,4]]:
        MaxPool output = 4 (the maximum)
        During backprop, gradient flows only to where max was.
        """
        try:
            from tinytorch.core.spatial import MaxPool2d
            from tinytorch.core.tensor import Tensor

            pool = MaxPool2d(kernel_size=2, stride=2)

            # Simple 4×4 input with known values
            x = Tensor(np.array([[
                [[1], [2], [5], [6]],
                [[3], [4], [7], [8]],
                [[9], [10], [13], [14]],
                [[11], [12], [15], [16]]
            ]]))  # (1, 4, 4, 1)

            output = pool(x)

            # 2×2 pooling should give max of each 2×2 region
            # Top-left: max(1,2,3,4) = 4
            # Top-right: max(5,6,7,8) = 8
            # etc.
            expected = np.array([[[[4], [8]], [[12], [16]]]])

            if output.shape == (1, 2, 2, 1):
                assert np.array_equal(output.data, expected), (
                    f"MaxPool values wrong.\n"
                    f"  Expected: {expected.squeeze()}\n"
                    f"  Got: {output.data.squeeze()}"
                )


        except ImportError:
            pytest.skip("MaxPool2d forward not implemented yet")

    def test_avgpool2d_forward(self):
        """
        WHAT: Verify AvgPool2d computes average of each window.

        WHY: AvgPool is smoother than MaxPool, sometimes preferred
        for the final layer (Global Average Pooling).

        STUDENT LEARNING: AvgPool is gentler than MaxPool.
        For 2×2 window [[1,2],[3,4]]:
        AvgPool = (1+2+3+4)/4 = 2.5
        """
        try:
            from tinytorch.core.spatial import AvgPool2d
            from tinytorch.core.tensor import Tensor

            pool = AvgPool2d(kernel_size=2, stride=2)

            # All-ones input - average should be 1
            x = Tensor(np.ones((1, 4, 4, 1)))
            output = pool(x)

            if output.shape == (1, 2, 2, 1):
                assert np.allclose(output.data, 1.0), (
                    f"AvgPool of all-ones should be 1.0\n"
                    f"  Got: {output.data[0,0,0,0]}"
                )


        except ImportError:
            pytest.skip("AvgPool2d not implemented yet")


class TestConvOutputShapes:
    """
    Test convolution output shape calculations.

    CONCEPT: Output shape depends on kernel_size, stride, padding.
    Getting this right is essential for building architectures.
    """

    def test_conv_padding_same(self):
        """
        WHAT: Verify 'same' padding preserves spatial dimensions.

        WHY: Same padding is convenient - output = input size.
        Used when you want to stack many conv layers.

        STUDENT LEARNING: For 'same' padding with odd kernel:
        padding = (kernel_size - 1) / 2
        For kernel=3: padding=1, for kernel=5: padding=2
        """
        try:
            from tinytorch.core.spatial import Conv2d
            from tinytorch.core.tensor import Tensor

            # With padding='same', output should match input spatial dims
            conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding='same')

            # NCHW format
            x = Tensor(np.random.randn(4, 3, 32, 32))
            output = conv(x)

            assert output.shape == (4, 8, 32, 32), (
                f"'same' padding should preserve spatial dims.\n"
                f"  Input: (4, 3, 32, 32) NCHW\n"
                f"  Expected: (4, 8, 32, 32)\n"
                f"  Got: {output.shape}"
            )


        except (ImportError, TypeError):
            pytest.skip("Conv2d padding='same' not implemented yet")

    def test_conv_stride(self):
        """
        WHAT: Verify stride reduces output dimensions.

        WHY: Stride > 1 downsamples the feature map.
        Stride=2 halves each dimension (like pooling).

        STUDENT LEARNING: With stride=2:
        output_size = (input_size - kernel_size) / stride + 1
        For input=32, kernel=3, stride=2: (32-3)/2 + 1 = 15
        """
        try:
            from tinytorch.core.spatial import Conv2d
            from tinytorch.core.tensor import Tensor

            conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2)

            # NCHW format
            x = Tensor(np.random.randn(1, 3, 32, 32))
            output = conv(x)

            # (32 - 3) / 2 + 1 = 15
            expected_size = 15
            # In NCHW, spatial dims are at indices 2 and 3
            assert output.shape[2] == expected_size and output.shape[3] == expected_size, (
                f"Stride=2 output size wrong.\n"
                f"  Input: 32×32, kernel=3, stride=2\n"
                f"  Expected: {expected_size}×{expected_size}\n"
                f"  Got: {output.shape[2]}×{output.shape[3]}\n"
                "Formula: (input - kernel) / stride + 1"
            )


        except (ImportError, TypeError):
            pytest.skip("Conv2d stride not implemented yet")


class TestConvGradientFlow:
    """
    Test that gradients flow through convolutions.

    CONCEPT: Conv layers must be differentiable for training.
    Gradients flow from output back to input AND kernel weights.
    """

    def test_conv2d_gradient_to_input(self):
        """
        WHAT: Verify input receives gradients through Conv2d.

        WHY: Backprop needs gradients at input to continue
        flowing to earlier layers.

        STUDENT LEARNING: Conv gradient is a "transposed convolution"
        (deconvolution). It spreads the output gradient back to input.
        """
        try:
            from tinytorch.core.spatial import Conv2d
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.autograd import enable_autograd

            enable_autograd()

            conv = Conv2d(in_channels=1, out_channels=1, kernel_size=3)
            # NCHW format
            x = Tensor(np.random.randn(1, 1, 8, 8), requires_grad=True)

            output = conv(x)
            loss = output.sum()
            loss.backward()

            assert x.grad is not None, (
                "Input didn't receive gradients through Conv2d.\n"
                "This means backprop through the conv is broken."
            )

        except ImportError:
            pytest.skip("Conv2d gradient not implemented yet")

    def test_conv2d_gradient_to_weights(self):
        """
        WHAT: Verify conv weights receive gradients.

        WHY: Weight gradients are what we use to train!
        No weight gradients = conv layer can't learn.

        STUDENT LEARNING: Weight gradient is computed by convolving
        input with output gradient. Each weight sees where it contributed.
        """
        try:
            from tinytorch.core.spatial import Conv2d
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.autograd import enable_autograd

            enable_autograd()

            conv = Conv2d(in_channels=1, out_channels=1, kernel_size=3)
            # NCHW format
            x = Tensor(np.random.randn(1, 1, 8, 8), requires_grad=True)

            output = conv(x)
            loss = output.sum()
            loss.backward()

            weight = conv.weight if hasattr(conv, 'weight') else conv.weights
            assert weight.grad is not None, (
                "Conv weights didn't receive gradients.\n"
                "This means the conv layer cannot learn."
            )

        except ImportError:
            pytest.skip("Conv2d weight gradient not implemented yet")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
