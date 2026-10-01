import pytest
import numpy as np
from tinytorch.core.tensor import Tensor
import tinytorch.extensions as ext
from tinytorch.extensions import (
    simd_matmul,
    simd_fused_bias_gelu,
    has_simd_support,
    simd_build_info,
    mps_matmul,
    has_mps_support,
    triton_fused_gelu,
    has_triton_support,
    LoRALinear,
    LossScaler,
    checkpoint,
    compile_graph,
)


def test_extensions_exports():
    """Verify all 7 extensions are exported in __all__."""
    expected = [
        "simd_matmul",
        "simd_fused_bias_gelu",
        "has_simd_support",
        "simd_build_info",
        "triton_fused_gelu",
        "has_triton_support",
        "mps_matmul",
        "has_mps_support",
        "LoRALinear",
        "LossScaler",
        "checkpoint",
        "compile_graph",
    ]
    for name in expected:
        assert hasattr(ext, name), f"Missing export: {name}"
        assert name in ext.__all__, f"{name} not in __all__"


def test_simd_matmul_numpy_and_tensor():
    """Test simd_matmul works interchangeably with numpy arrays and TinyTorch Tensors."""
    A_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    expected = np.matmul(A_np, B_np)

    # NumPy path
    C_np = simd_matmul(A_np, B_np)
    assert isinstance(C_np, np.ndarray)
    np.testing.assert_allclose(C_np, expected, rtol=1e-4, atol=1e-4)

    # Tensor path
    A_t = Tensor(A_np)
    B_t = Tensor(B_np)
    C_t = simd_matmul(A_t, B_t)
    assert isinstance(C_t, Tensor)
    np.testing.assert_allclose(C_t.data, expected, rtol=1e-4, atol=1e-4)

    # Mixed path (Tensor + NumPy)
    C_mixed = simd_matmul(A_t, B_np)
    assert isinstance(C_mixed, Tensor)
    np.testing.assert_allclose(C_mixed.data, expected, rtol=1e-4, atol=1e-4)


def test_simd_matmul_dimension_mismatch():
    """Test dimension mismatch raises ValueError."""
    A = Tensor(np.ones((2, 3), dtype=np.float32))
    B = Tensor(np.ones((4, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="Incompatible matrix dimensions"):
        simd_matmul(A, B)


def test_simd_fused_bias_gelu_numpy_and_tensor():
    """Test simd_fused_bias_gelu handles both numpy and Tensor, including large values."""
    rng = np.random.default_rng(42)
    X_np = (rng.standard_normal((16, 32)) * 30).astype(np.float32)
    bias_np = rng.standard_normal((32,)).astype(np.float32)

    u = X_np + bias_np
    ref = 0.5 * u * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (u + 0.044715 * np.power(u, 3))))

    # NumPy path
    out_np = simd_fused_bias_gelu(X_np, bias_np)
    assert isinstance(out_np, np.ndarray)
    assert np.isfinite(out_np).all()
    np.testing.assert_allclose(out_np, ref, rtol=1e-4, atol=1e-4)

    # Tensor path
    X_t = Tensor(X_np)
    bias_t = Tensor(bias_np)
    out_t = simd_fused_bias_gelu(X_t, bias_t)
    assert isinstance(out_t, Tensor)
    assert np.isfinite(out_t.data).all()
    np.testing.assert_allclose(out_t.data, ref, rtol=1e-4, atol=1e-4)


def test_mps_matmul_numpy_and_tensor():
    """Test mps_matmul with numpy and Tensor."""
    A_np = np.ones((4, 4), dtype=np.float32)
    B_np = np.eye(4, dtype=np.float32)

    C_np = mps_matmul(A_np, B_np)
    assert isinstance(C_np, np.ndarray)
    np.testing.assert_allclose(C_np, A_np)

    A_t = Tensor(A_np)
    B_t = Tensor(B_np)
    C_t = mps_matmul(A_t, B_t)
    assert isinstance(C_t, Tensor)
    np.testing.assert_allclose(C_t.data, A_np)


def test_triton_fused_gelu_numpy_and_tensor():
    """Test triton_fused_gelu with numpy and Tensor."""
    rng = np.random.default_rng(101)
    X_np = rng.standard_normal((8, 16)).astype(np.float32)
    bias_np = rng.standard_normal((16,)).astype(np.float32)

    u = X_np + bias_np
    ref = 0.5 * u * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (u + 0.044715 * np.power(u, 3))))

    # With bias
    out_np = triton_fused_gelu(X_np, bias_np)
    assert isinstance(out_np, np.ndarray)
    np.testing.assert_allclose(out_np, ref, rtol=1e-4, atol=1e-4)

    X_t = Tensor(X_np)
    bias_t = Tensor(bias_np)
    out_t = triton_fused_gelu(X_t, bias_t)
    assert isinstance(out_t, Tensor)
    np.testing.assert_allclose(out_t.data, ref, rtol=1e-4, atol=1e-4)

    # Without bias
    u_nobias = X_np
    ref_nobias = 0.5 * u_nobias * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (u_nobias + 0.044715 * np.power(u_nobias, 3))))
    out_nobias_t = triton_fused_gelu(X_t)
    assert isinstance(out_nobias_t, Tensor)
    np.testing.assert_allclose(out_nobias_t.data, ref_nobias, rtol=1e-4, atol=1e-4)
