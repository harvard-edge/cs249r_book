"""C++ SIMD Matrix Multiplication Bridge for TinyTorch.

Compiles and loads the native C++ SIMD kernel via ctypes for zero-overhead execution.
"""

import ctypes
import os
import subprocess
import platform
import numpy as np

_LIB_PATH = os.path.join(os.path.dirname(__file__), "libtinytorch_simd.so")
_CPP_SOURCE = os.path.join(os.path.dirname(__file__), "cpp_simd_gemm.cpp")
_simd_lib = None


def compile_and_load_simd():
    """Compiles the C++ SIMD GEMM kernel if not already compiled."""
    global _simd_lib
    if _simd_lib is not None:
        return _simd_lib

    if not os.path.exists(_LIB_PATH):
        try:
            # Detect compiler flags
            flags = ["-O3", "-shared", "-fPIC", "-std=c++17"]
            if platform.system() == "Darwin":
                # Clang on macOS
                flags += ["-Xpreprocessor", "-fopenmp", "-lomp"]
            else:
                # GCC / Linux
                flags += ["-fopenmp", "-mavx2", "-mfma"]

            cmd = ["c++"] + flags + [_CPP_SOURCE, "-o", _LIB_PATH]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                # Fallback without OpenMP flags if libomp is missing on Mac
                cmd_fallback = ["c++", "-O3", "-shared", "-fPIC", "-std=c++17", _CPP_SOURCE, "-o", _LIB_PATH]
                subprocess.run(cmd_fallback, check=True)
        except Exception as e:
            return None

    try:
        lib = ctypes.CDLL(_LIB_PATH)
        # Setup signatures
        # void tinytorch_cpp_gemm(const float* A, const float* B, float* C, int M, int K, int N)
        lib.tinytorch_cpp_gemm.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.tinytorch_cpp_gemm.restype = None

        # void tinytorch_cpp_fused_bias_gelu(const float* X, const float* bias, float* Y, int total, int inner)
        lib.tinytorch_cpp_fused_bias_gelu.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.tinytorch_cpp_fused_bias_gelu.restype = None

        _simd_lib = lib
        return _simd_lib
    except Exception:
        return None


def has_simd_support() -> bool:
    """Returns True if C++ SIMD library is available and compiled."""
    return compile_and_load_simd() is not None


def simd_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Executes cache-blocked multi-threaded SIMD matrix multiplication.
    
    Args:
        a: 2D numpy array [M, K] of dtype float32
        b: 2D numpy array [K, N] of dtype float32
        
    Returns:
        c: 2D numpy array [M, N] of dtype float32
    """
    lib = compile_and_load_simd()
    if lib is None:
        # Fallback to pure numpy
        return np.matmul(a, b)

    a_c = np.ascontiguousarray(a, dtype=np.float32)
    b_c = np.ascontiguousarray(b, dtype=np.float32)

    M, K = a_c.shape
    K_b, N = b_c.shape
    assert K == K_b, f"Matrix dimension mismatch: ({M}, {K}) x ({K_b}, {N})"

    c = np.empty((M, N), dtype=np.float32)

    ptr_a = a_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ptr_b = b_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ptr_c = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.tinytorch_cpp_gemm(ptr_a, ptr_b, ptr_c, M, K, N)
    return c
