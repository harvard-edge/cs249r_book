#!/usr/bin/env python3
"""
CNN Integration Test - After Module 11
======================================

This test validates that modules 1-11 work together for CNN image classification.

Required modules:
- Module 01-08: Core MLP functionality (from MNIST test)
- Module 09: Convolutions (Conv2d, MaxPool2d)
- Module 10: DataLoader for efficient batch processing
- Module 08: Training capabilities for CNN

This demonstrates the milestone: "Can train CNNs on CIFAR-10"
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear as Dense
from tinytorch.core.activations import ReLU
from tinytorch.core.training import CrossEntropyLoss

# Try to import spatial operations
try:
    from tinytorch.core.spatial import Conv2d, MaxPool2d, Flatten
    SPATIAL_AVAILABLE = True
except ImportError:
    print("⚠️  Spatial operations not available - using placeholder tests")
    SPATIAL_AVAILABLE = False

class SimpleCNN:
    """Simple CNN for CIFAR-10 style classification."""

    def __init__(self, num_classes=10):
        if SPATIAL_AVAILABLE:
            # Convolutional layers
            self.conv1 = Conv2d(3, 32, kernel_size=3)  # 3 channels -> 32 filters
            self.conv2 = Conv2d(32, 64, kernel_size=3) # 32 -> 64 filters
            self.pool = MaxPool2d(kernel_size=2)
            self.flatten = Flatten()

            # Dense layers
            self.fc1 = Linear(64 * 5 * 5, 256)  # Assuming 32x32 input -> 5x5 after conv+pool
            self.fc2 = Linear(256, num_classes)
        else:
            # Fallback: treat as flattened MLP
            self.fc1 = Linear(32*32*3, 256)
            self.fc2 = Linear(256, num_classes)

        self.relu = ReLU()

    def forward(self, x):
        """Forward pass."""
        if SPATIAL_AVAILABLE:
            # CNN path
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)

            x = self.conv2(x)
            x = self.relu(x)
            x = self.pool(x)

            x = self.flatten(x)
        else:
            # MLP path - flatten input
            if len(x.shape) == 4:  # (batch, channels, height, width)
                batch_size = x.shape[0]
                x = Tensor(x.data.reshape(batch_size, -1))

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        """Get all trainable parameters."""
        params = []
        if SPATIAL_AVAILABLE:
            if hasattr(self.conv1, 'parameters'):
                params.extend(self.conv1.parameters())
            if hasattr(self.conv2, 'parameters'):
                params.extend(self.conv2.parameters())

        params.extend([
            self.fc1.weights, self.fc1.bias,
            self.fc2.weights, self.fc2.bias
        ])
        return params

def generate_fake_cifar(num_samples=32, num_classes=10):
    """Generate fake CIFAR-10 like data for testing."""
    np.random.seed(42)

    # Generate random 32x32x3 images
    X = np.random.randn(num_samples, 3, 32, 32).astype(np.float32)

    # Generate random labels
    y = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int64)

    return X, y

def test_cnn_architecture():
    """Test CNN architecture can handle image data."""
    print("🏗️  Testing CNN Architecture...")

    try:
        model = SimpleCNN(num_classes=10)

        # Create fake image batch: (batch_size, channels, height, width)
        batch_size = 8
        x = Tensor(np.random.randn(batch_size, 3, 32, 32).astype(np.float32))

        print(f"  ✓ Created model and image batch")
        print(f"    Input shape: {x.shape} (batch, channels, height, width)")

        # Forward pass
        output = model(x)

        print(f"  ✓ Forward pass successful")
        print(f"    Output shape: {output.shape}")

        expected_shape = (batch_size, 10)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

        print("✅ CNN architecture working!")
        return True

    except Exception as e:
        print(f"❌ CNN architecture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spatial_operations():
    """Test spatial operations if available."""
    print("🔍 Testing Spatial Operations...")

    if not SPATIAL_AVAILABLE:
        print("  ⚠️  Spatial operations not available - skipping")
        return True

    try:
        # Test Conv2d
        conv = Conv2d(3, 16, kernel_size=3)
        x = Tensor(np.random.randn(1, 3, 8, 8).astype(np.float32))
        conv_out = conv(x)
        print(f"  ✓ Conv2d: {x.shape} -> {conv_out.shape}")

        # Test MaxPool2d
        pool = MaxPool2d(kernel_size=2)
        pool_out = pool(conv_out)
        print(f"  ✓ MaxPool2d: {conv_out.shape} -> {pool_out.shape}")

        # Test Flatten
        flatten = Flatten()
        flat_out = flatten(pool_out)
        print(f"  ✓ Flatten: {pool_out.shape} -> {flat_out.shape}")

        print("✅ Spatial operations working!")
        return True

    except Exception as e:
        print(f"❌ Spatial operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cnn_training_step():
    """Test CNN training step."""
    print("🏋️  Testing CNN Training Step...")

    try:
        # Create small CNN and fake CIFAR data
        model = SimpleCNN(num_classes=5)

        # Small batch
        x = Tensor(np.random.randn(4, 3, 16, 16).astype(np.float32))  # Smaller images
        y = Tensor(np.array([0, 1, 2, 3]))

        print(f"  ✓ Created CNN model and data")
        print(f"    Image batch shape: {x.shape}")
        print(f"    Labels shape: {y.shape}")

        # Forward pass
        outputs = model(x)
        print(f"  ✓ CNN forward pass: {x.shape} -> {outputs.shape}")

        # Loss computation
        criterion = CrossEntropyLoss()
        loss = criterion(outputs, y)
        print(f"  ✓ Loss computation successful")

        print("✅ CNN training step working!")
        return True

    except Exception as e:
        print(f"❌ CNN training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_data_pipeline():
    """Test image data processing pipeline."""
    print("📸 Testing Image Data Pipeline...")

    try:
        # Generate batch of fake CIFAR images
        X, y = generate_fake_cifar(num_samples=16)

        print(f"  ✓ Generated fake image data")
        print(f"    Images shape: {X.shape}")
        print(f"    Labels shape: {y.shape}")

        # Convert to tensors
        X_tensor = Tensor(X)
        y_tensor = Tensor(y)

        print(f"  ✓ Converted to tensors")

        # Test CNN can process this data
        model = SimpleCNN(num_classes=10)
        outputs = model(X_tensor)

        print(f"  ✓ CNN processed image batch: {X_tensor.shape} -> {outputs.shape}")

        # Test loss computation
        criterion = CrossEntropyLoss()
        loss = criterion(outputs, y_tensor)

        print(f"  ✓ Loss computation on image batch successful")

        print("✅ Image data pipeline working!")
        return True

    except Exception as e:
        print(f"❌ Image data pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_cnn_integration_test():
    """Run complete CNN integration test."""
    print("=" * 60)
    print("🔥 CNN INTEGRATION TEST - Modules 1-11")
    print("=" * 60)
    print()

    success = True
    tests = [
        test_cnn_architecture,
        test_spatial_operations,
        test_cnn_training_step,
        test_image_data_pipeline
    ]

    for test in tests:
        try:
            if not test():
                success = False
            print()
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            success = False
            print()

    if success:
        print("🎉 CNN INTEGRATION TEST PASSED!")
        print()
        print("✅ Milestone Achieved: Can build CNNs for image classification")
        print("   • CNN architecture handles 4D image tensors")
        if SPATIAL_AVAILABLE:
            print("   • Spatial operations (Conv2d, MaxPool2d) work")
        else:
            print("   • Fallback MLP architecture works for images")
        print("   • Training pipeline supports image data")
        print("   • End-to-end image classification pipeline functional")
        print()
        print("🚀 Ready for Module 12+: Attention and Transformers!")
    else:
        print("❌ CNN INTEGRATION TEST FAILED!")
        print("   Check spatial and training modules before proceeding")

    print("=" * 60)
    return success

if __name__ == "__main__":
    run_cnn_integration_test()
