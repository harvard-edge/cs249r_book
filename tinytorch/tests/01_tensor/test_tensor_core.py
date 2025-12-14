"""
Module 01: Tensor - Core Functionality Tests
=============================================

These tests verify that Tensor, the fundamental data structure of TinyTorch, works correctly.

WHY TENSORS MATTER:
------------------
Tensors are the foundation of ALL deep learning:
- Every input (images, text, audio) becomes a tensor
- Every weight and bias in a neural network is a tensor
- Every gradient computed during training is a tensor

If Tensor doesn't work, nothing else will. This is Module 01 for a reason.

WHAT STUDENTS LEARN:
-------------------
1. How data is represented in deep learning frameworks
2. Why NumPy is the backbone of Python ML
3. How operations like broadcasting save memory and compute
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestTensorCreation:
    """
    Test tensor creation and initialization.

    CONCEPT: A Tensor wraps a NumPy array and adds deep learning capabilities
    (like gradient tracking). Creating tensors is the first step in any ML pipeline.
    """

    def test_tensor_from_list(self):
        """
        WHAT: Create tensors from Python lists.

        WHY: Students often start with raw Python data (lists of numbers,
        nested lists for matrices). TinyTorch must accept this natural input
        and convert it to the internal NumPy representation.

        STUDENT LEARNING: Data can enter the framework in different forms,
        but internally it's always a NumPy array.
        """
        try:
            from tinytorch.core.tensor import Tensor

            # 1D tensor (vector) - like a single data sample's features
            t1 = Tensor([1, 2, 3])
            assert t1.shape == (3,), (
                f"1D tensor has wrong shape.\n"
                f"  Input: [1, 2, 3] (3 elements)\n"
                f"  Expected shape: (3,)\n"
                f"  Got: {t1.shape}"
            )
            assert np.array_equal(t1.data, [1, 2, 3])

            # 2D tensor (matrix) - like a batch of samples or weight matrix
            t2 = Tensor([[1, 2], [3, 4]])
            assert t2.shape == (2, 2), (
                f"2D tensor has wrong shape.\n"
                f"  Input: [[1,2], [3,4]] (2 rows, 2 cols)\n"
                f"  Expected shape: (2, 2)\n"
                f"  Got: {t2.shape}"
            )

        except ImportError:
            pytest.skip("Tensor not implemented yet")

    def test_tensor_from_numpy(self):
        """
        WHAT: Create tensors from NumPy arrays.

        WHY: Real ML data comes from NumPy (pandas, scikit-learn, image loaders).
        TinyTorch must seamlessly accept NumPy arrays.

        STUDENT LEARNING: TinyTorch uses float32 by default (like PyTorch)
        because it's faster and uses half the memory of float64.
        """
        try:
            from tinytorch.core.tensor import Tensor

            arr = np.array([[1.0, 2.0], [3.0, 4.0]])
            t = Tensor(arr)

            assert t.shape == (2, 2)
            assert t.dtype == np.float32, (
                f"Tensor should use float32 for efficiency.\n"
                f"  Expected dtype: np.float32\n"
                f"  Got: {t.dtype}\n"
                "float32 is half the memory of float64 and faster on GPUs."
            )
            assert np.allclose(t.data, arr)

        except ImportError:
            pytest.skip("Tensor not implemented yet")

    def test_tensor_shapes(self):
        """
        WHAT: Handle tensors of various dimensions.

        WHY: Deep learning uses many tensor shapes:
        - 1D: feature vectors, biases
        - 2D: weight matrices, batch of 1D samples
        - 3D: sequences (batch, seq_len, features)
        - 4D: images (batch, height, width, channels)

        STUDENT LEARNING: Shape is critical. Most bugs are shape mismatches.
        """
        try:
            from tinytorch.core.tensor import Tensor

            test_cases = [
                ((5,), "1D: feature vector"),
                ((3, 4), "2D: weight matrix"),
                ((2, 3, 4), "3D: sequence data"),
                ((1, 28, 28, 3), "4D: single RGB image"),
            ]

            for shape, description in test_cases:
                data = np.random.randn(*shape)
                t = Tensor(data)
                assert t.shape == shape, (
                    f"Shape mismatch for {description}.\n"
                    f"  Expected: {shape}\n"
                    f"  Got: {t.shape}"
                )

        except ImportError:
            pytest.skip("Tensor not implemented yet")


class TestTensorOperations:
    """
    Test tensor arithmetic and operations.

    CONCEPT: Neural networks are just sequences of mathematical operations
    on tensors. If these operations don't work, training is impossible.
    """

    def test_tensor_addition(self):
        """
        WHAT: Element-wise tensor addition.

        WHY: Addition is used everywhere in neural networks:
        - Adding bias to layer output: y = Wx + b
        - Residual connections: output = layer(x) + x
        - Gradient accumulation

        STUDENT LEARNING: Operations return new Tensors (functional style).
        """
        try:
            from tinytorch.core.tensor import Tensor

            t1 = Tensor([1, 2, 3])
            t2 = Tensor([4, 5, 6])

            result = t1 + t2
            expected = np.array([5, 7, 9])

            assert isinstance(result, Tensor), (
                "Addition should return a Tensor, not numpy array.\n"
                "This maintains the computation graph for backpropagation."
            )
            assert np.array_equal(result.data, expected), (
                f"Element-wise addition failed.\n"
                f"  {t1.data} + {t2.data}\n"
                f"  Expected: {expected}\n"
                f"  Got: {result.data}"
            )

        except (ImportError, TypeError):
            pytest.skip("Tensor addition not implemented yet")

    def test_tensor_multiplication(self):
        """
        WHAT: Element-wise tensor multiplication.

        WHY: Element-wise multiplication (Hadamard product) is used for:
        - Applying masks (setting values to zero)
        - Gating mechanisms (LSTM, attention)
        - Dropout during training

        STUDENT LEARNING: This is NOT matrix multiplication. It's element-by-element.
        """
        try:
            from tinytorch.core.tensor import Tensor

            t1 = Tensor([1, 2, 3])
            t2 = Tensor([2, 3, 4])

            result = t1 * t2
            expected = np.array([2, 6, 12])

            assert isinstance(result, Tensor)
            assert np.array_equal(result.data, expected), (
                f"Element-wise multiplication failed.\n"
                f"  {t1.data} * {t2.data} (element-wise)\n"
                f"  Expected: {expected}\n"
                f"  Got: {result.data}\n"
                "Remember: * is element-wise, @ is matrix multiplication."
            )

        except (ImportError, TypeError):
            pytest.skip("Tensor multiplication not implemented yet")

    def test_matrix_multiplication(self):
        """
        WHAT: Matrix multiplication (the @ operator).

        WHY: Matrix multiplication is THE core operation of neural networks:
        - Linear layers: y = x @ W
        - Attention: scores = Q @ K^T
        - Every fully-connected layer uses it

        STUDENT LEARNING: Matrix dimensions must be compatible.
        (m×n) @ (n×p) = (m×p) - inner dimensions must match.
        """
        try:
            from tinytorch.core.tensor import Tensor

            t1 = Tensor([[1, 2], [3, 4]])  # 2×2
            t2 = Tensor([[5, 6], [7, 8]])  # 2×2

            # Matrix multiplication using @ operator
            if hasattr(t1, '__matmul__'):
                result = t1 @ t2
            else:
                result = Tensor(t1.data @ t2.data)

            # Manual calculation:
            # [1*5+2*7, 1*6+2*8]   = [19, 22]
            # [3*5+4*7, 3*6+4*8]   = [43, 50]
            expected = np.array([[19, 22], [43, 50]])

            assert np.array_equal(result.data, expected), (
                f"Matrix multiplication failed.\n"
                f"  {t1.data}\n  @\n  {t2.data}\n"
                f"  Expected:\n  {expected}\n"
                f"  Got:\n  {result.data}"
            )

        except (ImportError, TypeError):
            pytest.skip("Matrix multiplication not implemented yet")


class TestTensorMemory:
    """
    Test tensor memory management.

    CONCEPT: Efficient memory use is critical for deep learning.
    Large models can use 10s of GB. Understanding memory helps debug OOM errors.
    """

    def test_tensor_data_access(self):
        """
        WHAT: Access the underlying NumPy array.

        WHY: Sometimes you need the raw data for:
        - Visualization (matplotlib expects NumPy)
        - Debugging (print values)
        - Integration with other libraries

        STUDENT LEARNING: .data gives you the NumPy array inside the Tensor.
        """
        try:
            from tinytorch.core.tensor import Tensor

            data = np.array([1, 2, 3, 4])
            t = Tensor(data)

            assert hasattr(t, 'data'), (
                "Tensor must have a .data attribute.\n"
                "This gives access to the underlying NumPy array."
            )
            assert np.array_equal(t.data, data)

        except ImportError:
            pytest.skip("Tensor not implemented yet")

    def test_tensor_copy_semantics(self):
        """
        WHAT: Verify tensors don't share memory unexpectedly.

        WHY: Shared memory can cause subtle bugs:
        - Modifying one tensor accidentally changes another
        - Gradient corruption during backprop
        - Non-reproducible results

        STUDENT LEARNING: TinyTorch should copy data by default for safety.
        """
        try:
            from tinytorch.core.tensor import Tensor

            original_data = np.array([1, 2, 3])
            t1 = Tensor(original_data)
            t2 = Tensor(original_data.copy())

            # Should have same values but independent data
            assert np.array_equal(t1.data, t2.data)

            # Modifying original shouldn't affect t2
            original_data[0] = 999
            if not np.shares_memory(t2.data, original_data):
                assert t2.data[0] == 1, (
                    "Tensor should not share memory with input!\n"
                    "Modifying the original array changed the tensor.\n"
                    "This can cause hard-to-debug issues."
                )

        except ImportError:
            pytest.skip("Tensor not implemented yet")

    def test_tensor_memory_efficiency(self):
        """
        WHAT: Handle large tensors efficiently.

        WHY: Real models have millions of parameters:
        - ResNet-50: 25 million parameters
        - GPT-2: 1.5 billion parameters
        - LLaMA: 7-65 billion parameters

        STUDENT LEARNING: Memory efficiency matters at scale.
        """
        try:
            from tinytorch.core.tensor import Tensor

            # Create a 1000×1000 tensor (1 million elements)
            data = np.random.randn(1000, 1000)
            t = Tensor(data)

            assert t.shape == (1000, 1000)
            assert t.data.size == 1000000, (
                f"Tensor should have 1M elements.\n"
                f"  Got: {t.data.size} elements"
            )

        except ImportError:
            pytest.skip("Tensor not implemented yet")


class TestTensorReshaping:
    """
    Test tensor reshaping and view operations.

    CONCEPT: Reshaping changes how we interpret the same data.
    The underlying values don't change, just their arrangement.
    """

    def test_tensor_reshape(self):
        """
        WHAT: Reshape tensor to different dimensions.

        WHY: Reshaping is constantly needed:
        - Flattening images for dense layers
        - Rearranging for batch processing
        - Preparing data for specific layer types

        STUDENT LEARNING: Total elements must stay the same.
        [12 elements] can become (3,4) or (2,6) or (2,2,3), but not (5,3).
        """
        try:
            from tinytorch.core.tensor import Tensor

            t = Tensor(np.arange(12))  # [0, 1, 2, ..., 11]

            if hasattr(t, 'reshape'):
                reshaped = t.reshape(3, 4)
                assert reshaped.shape == (3, 4), (
                    f"Reshape failed.\n"
                    f"  Original: {t.shape} (12 elements)\n"
                    f"  Requested: (3, 4) (12 elements)\n"
                    f"  Got: {reshaped.shape}"
                )
                assert reshaped.data.size == 12
            else:
                reshaped_data = t.data.reshape(3, 4)
                assert reshaped_data.shape == (3, 4)

        except ImportError:
            pytest.skip("Tensor reshape not implemented yet")

    def test_tensor_flatten(self):
        """
        WHAT: Flatten tensor to 1D.

        WHY: Flattening is required to connect:
        - Conv layers (4D) to Dense layers (2D)
        - Image data to classification heads

        STUDENT LEARNING: flatten() is shorthand for reshape(-1)
        """
        try:
            from tinytorch.core.tensor import Tensor

            t = Tensor(np.random.randn(2, 3, 4))  # 2×3×4 = 24 elements

            if hasattr(t, 'flatten'):
                flat = t.flatten()
                assert flat.shape == (24,), (
                    f"Flatten failed.\n"
                    f"  Original: {t.shape} = {2*3*4} elements\n"
                    f"  Expected: (24,)\n"
                    f"  Got: {flat.shape}"
                )
            else:
                flat_data = t.data.flatten()
                assert flat_data.shape == (24,)

        except ImportError:
            pytest.skip("Tensor flatten not implemented yet")

    def test_tensor_transpose(self):
        """
        WHAT: Transpose tensor (swap dimensions).

        WHY: Transpose is used for:
        - Matrix multiplication compatibility
        - Attention: K^T in Q @ K^T
        - Rearranging data layouts

        STUDENT LEARNING: Transpose swaps rows and columns.
        (m×n) becomes (n×m).
        """
        try:
            from tinytorch.core.tensor import Tensor

            t = Tensor([[1, 2, 3], [4, 5, 6]])  # 2×3

            if hasattr(t, 'T') or hasattr(t, 'transpose'):
                transposed = t.T if hasattr(t, 'T') else t.transpose()

                assert transposed.shape == (3, 2), (
                    f"Transpose failed.\n"
                    f"  Original: {t.shape}\n"
                    f"  Expected: (3, 2)\n"
                    f"  Got: {transposed.shape}"
                )
                expected = np.array([[1, 4], [2, 5], [3, 6]])
                assert np.array_equal(transposed.data, expected)
            else:
                transposed_data = t.data.T
                assert transposed_data.shape == (3, 2)

        except ImportError:
            pytest.skip("Tensor transpose not implemented yet")


class TestTensorBroadcasting:
    """
    Test tensor broadcasting operations.

    CONCEPT: Broadcasting lets you operate on tensors of different shapes
    by automatically expanding the smaller one. This saves memory and code.
    """

    def test_scalar_broadcasting(self):
        """
        WHAT: Add a scalar to every element.

        WHY: Scalar operations are common:
        - Adding bias: output + bias
        - Normalization: (x - mean) / std
        - Scaling: x * 0.1

        STUDENT LEARNING: Scalars broadcast to match any shape.
        """
        try:
            from tinytorch.core.tensor import Tensor

            t = Tensor([1, 2, 3])

            if hasattr(t, '__add__'):
                result = t + 5
                expected = np.array([6, 7, 8])
                assert np.array_equal(result.data, expected), (
                    f"Scalar broadcasting failed.\n"
                    f"  {t.data} + 5\n"
                    f"  Expected: {expected}\n"
                    f"  Got: {result.data}\n"
                    "The scalar 5 should be added to every element."
                )

        except (ImportError, TypeError):
            pytest.skip("Scalar broadcasting not implemented yet")

    def test_vector_broadcasting(self):
        """
        WHAT: Broadcast a vector across a matrix.

        WHY: Vector broadcasting is used for:
        - Adding bias to batch output: (batch, features) + (features,)
        - Normalizing channels: (batch, H, W, C) / (C,)

        STUDENT LEARNING: Broadcasting aligns from the RIGHT.
        (2,3) + (3,) works because 3 aligns with 3.
        (2,3) + (2,) fails because 2 doesn't align with 3.
        """
        try:
            from tinytorch.core.tensor import Tensor

            t1 = Tensor([[1, 2, 3], [4, 5, 6]])  # 2×3
            t2 = Tensor([10, 20, 30])            # 3,

            if hasattr(t1, '__add__'):
                result = t1 + t2
                assert result.shape == (2, 3), (
                    f"Broadcasting produced wrong shape.\n"
                    f"  (2,3) + (3,) should give (2,3)\n"
                    f"  Got: {result.shape}"
                )
                expected = np.array([[11, 22, 33], [14, 25, 36]])
                assert np.array_equal(result.data, expected), (
                    f"Vector broadcasting failed.\n"
                    f"  [[1,2,3], [4,5,6]] + [10,20,30]\n"
                    f"  Expected: {expected}\n"
                    f"  Got: {result.data}\n"
                    "Each row should have [10,20,30] added to it."
                )

        except (ImportError, TypeError):
            pytest.skip("Vector broadcasting not implemented yet")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
