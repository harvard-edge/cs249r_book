"""
Checkpoint 5: Learning (After Module 6 - Spatial)
Question: "Can I process spatial data like images with convolutional operations?"
"""

import numpy as np
import pytest

def test_checkpoint_05_learning():
    """
    Checkpoint 5: Learning

    Validates that students can apply spatial operations like convolution to
    process image-like data efficiently - the foundation of computer vision
    and spatial pattern recognition.
    """
    print("\n👁️ Checkpoint 5: Learning")
    print("=" * 50)

    try:
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.spatial import Conv2d as Conv2D, MaxPool2d
        from tinytorch.core.activations import ReLU
    except ImportError as e:
        pytest.fail(f"❌ Cannot import required classes - complete Modules 2-6 first: {e}")

    # Test 1: Basic convolution operation
    print("🔍 Testing convolution operation...")
    conv = Conv2D(in_channels=1, out_channels=3, kernel_size=3)

    # Create a simple "image" (single channel, 5x5 pixels)
    image = Tensor(np.random.randn(1, 1, 5, 5))  # batch=1, channels=1, height=5, width=5

    conv_output = conv(image)
    expected_shape = (1, 3, 3, 3)  # Output size depends on kernel size and padding

    assert conv_output.shape == expected_shape, f"Convolution output should be {expected_shape}, got {conv_output.shape}"
    print(f"✅ Convolution: {image.shape} → {conv_output.shape}")

    # Test 2: Pooling operation
    print("📉 Testing pooling operation...")
    pool = MaxPool2d(kernel_size=2)

    # Create larger feature map for pooling
    feature_map = Tensor(np.random.randn(1, 3, 4, 4))
    pooled = pool(feature_map)

    expected_pool_shape = (1, 3, 2, 2)  # Pooling reduces spatial dimensions
    assert pooled.shape == expected_pool_shape, f"Pooling output should be {expected_pool_shape}, got {pooled.shape}"
    print(f"✅ Pooling: {feature_map.shape} → {pooled.shape}")

    # Test 3: CNN building block (Conv + ReLU + Pool)
    print("🏗️ Testing CNN building block...")
    relu = ReLU()

    # Simulate a small CNN layer
    input_image = Tensor(np.random.randn(2, 1, 8, 8))  # 2 images, 1 channel, 8x8

    # CNN forward pass: Conv → ReLU → Pool
    conv_out = conv(input_image)
    activated = relu(conv_out)
    final_output = pool(activated)

    print(f"✅ CNN block: {input_image.shape} → Conv → ReLU → Pool → {final_output.shape}")

    # Test 4: Multi-channel processing
    print("🎨 Testing multi-channel processing...")
    # RGB image processing
    rgb_conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
    rgb_image = Tensor(np.random.randn(1, 3, 32, 32))  # RGB image 32x32

    rgb_features = rgb_conv(rgb_image)
    expected_rgb_shape = (1, 16, 30, 30)  # 16 feature maps

    assert rgb_features.shape == expected_rgb_shape, f"RGB processing should output {expected_rgb_shape}, got {rgb_features.shape}"
    print(f"✅ Multi-channel: {rgb_image.shape} → {rgb_features.shape}")

    # Test 5: Spatial hierarchy (multiple conv layers)
    print("🏔️ Testing spatial hierarchy...")
    conv1 = Conv2D(in_channels=1, out_channels=8, kernel_size=3)
    conv2 = Conv2D(in_channels=8, out_channels=16, kernel_size=3)
    pool = MaxPool2d(kernel_size=2)

    # Build spatial hierarchy
    x = Tensor(np.random.randn(1, 1, 16, 16))

    # Layer 1: Conv → ReLU → Pool
    h1 = relu(conv1(x))
    p1 = pool(h1)

    # Layer 2: Conv → ReLU → Pool
    h2 = relu(conv2(p1))
    p2 = pool(h2)

    print(f"✅ Spatial hierarchy: {x.shape} → {h1.shape} → {p1.shape} → {h2.shape} → {p2.shape}")

    # Test 6: Feature map visualization concept
    print("🖼️ Testing feature map properties...")

    # Test that convolution preserves important properties
    test_image = Tensor(np.ones((1, 1, 5, 5)))  # All ones image
    test_conv = Conv2D(in_channels=1, out_channels=1, kernel_size=3)

    # Set simple kernel for predictable output
    test_conv.weight.data = np.ones((1, 1, 3, 3)) * 0.1  # Simple averaging kernel
    test_conv.bias.data = np.zeros((1,))

    result = test_conv(test_image)

    # All outputs should be similar since input was uniform
    output_std = np.std(result.data)
    assert output_std < 0.1, f"Uniform input should produce low variance output, got std={output_std}"
    print(f"✅ Feature extraction: uniform input → low variance output (std={output_std:.4f})")

    # Test 7: Edge detection demonstration
    print("🔍 Testing edge detection capability...")

    # Create a simple edge pattern
    edge_image = Tensor(np.array([
        [[[0, 0, 0, 1, 1],
          [0, 0, 0, 1, 1],
          [0, 0, 0, 1, 1],
          [0, 0, 0, 1, 1],
          [0, 0, 0, 1, 1]]]], dtype=np.float32))

    # Simple edge detection kernel
    edge_conv = Conv2D(in_channels=1, out_channels=1, kernel_size=3)
    edge_conv.weight.data = np.array([[[[-1, 0, 1],
                                       [-1, 0, 1],
                                       [-1, 0, 1]]]], dtype=np.float32)
    edge_conv.bias.data = np.zeros((1,), dtype=np.float32)

    edge_response = edge_conv(edge_image)

    # Should detect the vertical edge
    assert edge_response.shape[2:] == (3, 3), f"Edge detection output should be 3x3, got {edge_response.shape[2:]}"
    print(f"✅ Edge detection: {edge_image.shape} → detected features → {edge_response.shape}")

    print("\n🎉 Learning Complete!")
    print("📝 You can now process spatial data like images with convolutional operations")
    print("🔧 Built capabilities: Convolution, pooling, CNN blocks, multi-channel processing")
    print("🧠 Breakthrough: You can now extract spatial features from images!")
    print("🎯 Next: Build attention mechanisms for sequence processing")

if __name__ == "__main__":
    test_checkpoint_05_learning()
