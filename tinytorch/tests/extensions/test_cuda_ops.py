import pytest
import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.extensions import cuda_ops
from tinytorch.extensions.cuda_ops import cuda_matmul, has_cuda_support, cuda_build_info, STAGES


@pytest.mark.parametrize("stage", STAGES)
def test_cuda_matmul_numpy_and_tensor(stage):
    """Each stage works interchangeably with numpy arrays and TinyTorch Tensors."""
    A_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    expected = np.matmul(A_np, B_np)

    C_np = cuda_matmul(A_np, B_np, stage=stage)
    assert isinstance(C_np, np.ndarray)
    np.testing.assert_allclose(C_np, expected, rtol=1e-4, atol=1e-4)

    C_t = cuda_matmul(Tensor(A_np), Tensor(B_np), stage=stage)
    assert isinstance(C_t, Tensor)
    np.testing.assert_allclose(C_t.data, expected, rtol=1e-4, atol=1e-4)

    C_mixed = cuda_matmul(Tensor(A_np), B_np, stage=stage)
    assert isinstance(C_mixed, Tensor)
    np.testing.assert_allclose(C_mixed.data, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("M,K,N", [(1, 1, 1), (32, 32, 32), (127, 65, 255), (33, 97, 31), (64, 0, 64), (0, 5, 7)])
def test_cuda_matmul_shapes(stage, M, K, N):
    """Partial tiles, non-square, K=0 and empty outputs all match NumPy."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K), dtype=np.float32)
    B = rng.standard_normal((K, N), dtype=np.float32)
    C = cuda_matmul(A, B, stage=stage)
    assert C.shape == (M, N)
    np.testing.assert_allclose(C, A @ B, rtol=1e-4, atol=1e-3)


def test_cuda_matmul_dimension_mismatch():
    """Test dimension mismatch raises ValueError."""
    A = Tensor(np.ones((2, 3), dtype=np.float32))
    B = Tensor(np.ones((4, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="Incompatible matrix dimensions"):
        cuda_matmul(A, B)


def test_cuda_matmul_unknown_stage():
    A = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="stage must be one of"):
        cuda_matmul(A, A, stage="fast")


def test_cuda_matmul_fallback_without_cuda(monkeypatch):
    """With no CUDA library, results still come from np.matmul."""
    monkeypatch.setattr(cuda_ops, "compile_and_load_cuda", lambda: None)
    A = np.arange(6, dtype=np.float32).reshape(2, 3)
    B = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_allclose(cuda_matmul(A, B), A @ B)


def test_cuda_build_info():
    info = cuda_build_info()
    assert set(info) >= {"built", "devices", "command", "error"}
    assert has_cuda_support() == (info["devices"] > 0 and info["error"] is None)
