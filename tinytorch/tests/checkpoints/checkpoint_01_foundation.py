"""
Checkpoint 1: Foundation (After Module 2 - Tensor)
Question: "Can I create and manipulate the building blocks of ML?"
"""

import numpy as np
import pytest

def test_checkpoint_01_foundation():
    """
    Checkpoint 1: Foundation

    Validates that students can create and manipulate multi-dimensional tensors,
    perform arithmetic operations, and understand tensor shapes - the foundation
    of all machine learning computations.
    """
    print("\n🏁 Checkpoint 1: Foundation")
    print("=" * 50)

    try:
        from tinytorch.core.tensor import Tensor
    except ImportError:
        pytest.fail("❌ Cannot import Tensor - complete Module 2 first")

    # Test 1: Basic tensor creation
    print("📊 Testing tensor creation...")
    x = Tensor([[1, 2], [3, 4]])
    y = Tensor([[5, 6], [7, 8]])

    assert x.shape == (2, 2), f"Expected shape (2, 2), got {x.shape}"
    assert y.shape == (2, 2), f"Expected shape (2, 2), got {y.shape}"
    print(f"✅ Created tensors with shapes: {x.shape}")

    # Test 2: Arithmetic operations
    print("🧮 Testing arithmetic operations...")
    result = x + y * 2  # Should be [[1+10, 2+12], [3+14, 4+16]] = [[11, 14], [17, 20]]

    expected = np.array([[11, 14], [17, 20]])
    assert np.allclose(result.data, expected), f"Expected {expected}, got {result.data}"
    print(f"✅ Arithmetic operations working: {result.data}")

    # Test 3: Different tensor shapes
    print("📐 Testing different shapes...")
    vector = Tensor([1, 2, 3, 4, 5])
    scalar = Tensor(42)
    matrix_3x3 = Tensor(np.random.randn(3, 3))

    assert vector.shape == (5,), f"Vector shape should be (5,), got {vector.shape}"
    assert scalar.shape == (), f"Scalar shape should be (), got {scalar.shape}"
    assert matrix_3x3.shape == (3, 3), f"Matrix shape should be (3, 3), got {matrix_3x3.shape}"
    print(f"✅ Multiple shapes supported: vector{vector.shape}, scalar{scalar.shape}, matrix{matrix_3x3.shape}")

    # Test 4: Data type handling
    print("🔢 Testing data types...")
    float_tensor = Tensor([1.5, 2.7, 3.14])
    int_tensor = Tensor([1, 2, 3])

    assert hasattr(float_tensor, 'dtype'), "Tensor should have dtype attribute"
    assert hasattr(int_tensor, 'dtype'), "Tensor should have dtype attribute"
    print(f"✅ Data types: float_tensor.dtype={float_tensor.dtype}, int_tensor.dtype={int_tensor.dtype}")

    print("\n🎉 Foundation Complete!")
    print("📝 You can now create and manipulate the building blocks of ML")
    print("🔧 Built capabilities: Tensor creation, arithmetic, shapes, dtypes")
    print("🎯 Next: Add intelligence with activation functions")

if __name__ == "__main__":
    test_checkpoint_01_foundation()
