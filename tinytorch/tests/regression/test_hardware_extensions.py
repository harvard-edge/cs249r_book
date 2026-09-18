"""Tests and performance evaluations for TinyTorch Hardware Extensions.

Verifies:
1. C++ SIMD matrix multiply and fused bias + GELU (compiled on first use)
2. Triton fused bias + GELU (NumPy fallback when there is no NVIDIA GPU)
3. MPS matrix multiply (NumPy fallback without PyTorch or an Apple GPU)
4. Numerical parity against NumPy reference
5. Latency benchmarking and performance evaluation
"""

import shutil
import time
import pytest
import numpy as np
from tinytorch.extensions import (
    has_simd_support,
    simd_matmul,
    simd_fused_bias_gelu,
    simd_build_info,
    has_triton_support,
    triton_fused_gelu,
    has_mps_support,
    mps_matmul,
)


class TestHardwareExtensions:
    """Validate hardware accelerator extensions and fallbacks."""

    def test_simd_matmul_numerical_parity(self):
        """Test C++ SIMD GEMM matches NumPy within floating-point tolerance."""
        rng = np.random.default_rng(42)
        M, K, N = 64, 128, 64
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)

        C_ref = np.matmul(A, B)
        C_simd = simd_matmul(A, B)

        np.testing.assert_allclose(
            C_simd,
            C_ref,
            rtol=1e-4,
            atol=1e-4,
            err_msg="SIMD GEMM output diverges from NumPy reference",
        )

    @pytest.mark.skipif(shutil.which("c++") is None, reason="no C++ compiler on PATH")
    def test_simd_library_builds(self):
        """With a compiler present, the C++ path must really load, not fall back silently."""
        assert has_simd_support(), simd_build_info().get("error")
        info = simd_build_info()
        assert info["threads"] >= 1

    def test_simd_fused_bias_gelu_parity(self):
        """C++ fused bias + GELU matches NumPy, including inputs large enough to
        overflow exp(2z); the kernel used exp-based tanh and returned NaN there
        before 2026-09-18."""
        rng = np.random.default_rng(7)
        X = (rng.standard_normal((32, 96)) * 40).astype(np.float32)
        bias = rng.standard_normal((96,)).astype(np.float32)
        u = X + bias
        ref = 0.5 * u * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (u + 0.044715 * np.power(u, 3))))

        out = simd_fused_bias_gelu(X, bias)
        assert np.isfinite(out).all()
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)

    def test_simd_matmul_shape_mismatch(self):
        """Test that invalid inner dimensions raise ValueError."""
        A = np.ones((10, 20), dtype=np.float32)
        B = np.ones((25, 30), dtype=np.float32)

        with pytest.raises(ValueError, match="Incompatible matrix dimensions"):
            simd_matmul(A, B)

    def test_triton_gelu_numerical_parity(self):
        """Test Triton Fused GELU matches mathematical reference."""
        rng = np.random.default_rng(123)
        X = rng.standard_normal((32, 64)).astype(np.float32)
        bias = rng.standard_normal((64,)).astype(np.float32)

        # Mathematical reference: GELU(x + bias)
        u = X + bias
        ref = 0.5 * u * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (u + 0.044715 * np.power(u, 3))))

        out = triton_fused_gelu(X, bias)

        np.testing.assert_allclose(
            out,
            ref,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Triton/CPU reference GELU diverges from mathematical formulation",
        )

    def test_triton_gelu_without_bias(self):
        """Test GELU without bias vector."""
        rng = np.random.default_rng(456)
        X = rng.standard_normal((16, 32)).astype(np.float32)

        u = X
        ref = 0.5 * u * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (u + 0.044715 * np.power(u, 3))))

        out = triton_fused_gelu(X, bias=None)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)

    def test_mps_matmul_parity(self):
        """Test Apple Silicon MPS matmul if supported on this host."""
        rng = np.random.default_rng(789)
        A = rng.standard_normal((128, 128)).astype(np.float32)
        B = rng.standard_normal((128, 128)).astype(np.float32)

        C_ref = np.matmul(A, B)
        C_mps = mps_matmul(A, B)

        np.testing.assert_allclose(
            C_mps,
            C_ref,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MPS GEMM output diverges from NumPy reference",
        )

    def test_performance_evaluation(self):
        """Benchmark extensions and evaluate runtime performance."""
        rng = np.random.default_rng(2026)
        M, K, N = 256, 256, 256
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)

        # Warmup
        _ = np.matmul(A, B)
        _ = simd_matmul(A, B)

        # Measure NumPy time
        t0 = time.perf_counter()
        for _ in range(20):
            _ = np.matmul(A, B)
        numpy_ms = (time.perf_counter() - t0) * 1000 / 20

        # Measure SIMD time
        t0 = time.perf_counter()
        for _ in range(20):
            _ = simd_matmul(A, B)
        simd_ms = (time.perf_counter() - t0) * 1000 / 20

        print(f"\n[Performance Benchmark 256x256 GEMM]")
        print(f"  NumPy BLAS: {numpy_ms:.3f} ms")
        print(f"  SIMD GEMM:  {simd_ms:.3f} ms")

        # Verify correctness and finite values
        C_numpy = np.matmul(A, B)
        C_simd = simd_matmul(A, B)
        np.testing.assert_allclose(C_simd, C_numpy, rtol=1e-4, atol=1e-4)
        assert C_simd.shape == (M, N)
        assert np.isfinite(C_simd).all()
        assert 0 < simd_ms < 5000, f"SIMD GEMM time {simd_ms} ms out of expected bounds"
        assert 0 < numpy_ms < 5000, f"NumPy BLAS time {numpy_ms} ms out of expected bounds"
