# Hand-maintained extension, NOT generated. This file is not produced by
# `tito dev export` or `tito module complete`; it is tracked in git and
# edited directly here. It is the CUDA SGEMM bridge (see cuda_sgemm.cu);
# no src/ module or student notebook exports to it.
"""CUDA SGEMM Bridge for TinyTorch.

Compiles cuda_sgemm.cu with nvcc on first use and calls it through ctypes.
Three teaching kernels are selected with stage: "naive", "coalesced", "tiled".
Each call copies A and B to the GPU and C back, so timings include PCIe.

Needs nvcc and an NVIDIA GPU, not PyTorch. Without them, cuda_matmul falls
back to np.matmul and cuda_build_info() says why.
"""

import ctypes
import os
import shutil
import subprocess
import warnings
import numpy as np

_DIR = os.path.dirname(__file__)
_LIB_PATH = os.path.join(_DIR, "libtinytorch_cuda.so")
_CU_SOURCE = os.path.join(_DIR, "cuda_sgemm.cu")
_cuda_lib = None
_build_info = {"built": False, "devices": 0, "command": None, "error": None}

STAGES = ("naive", "coalesced", "tiled")


def _compile():
    """Build the shared library with nvcc. Returns True on success."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        _build_info["error"] = "nvcc not found on PATH"
        return False

    tmp = _LIB_PATH + f".{os.getpid()}.tmp"
    # Older nvcc has no -arch=native; retry with its default architecture
    for arch in (["-arch=native"], []):
        cmd = [nvcc, "-O3", "-shared", "-Xcompiler", "-fPIC", *arch, _CU_SOURCE, "-o", tmp]
        res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if res.returncode == 0:
            os.replace(tmp, _LIB_PATH)  # atomic, so a concurrent import never sees half a file
            _build_info.update(built=True, command=" ".join(cmd[:-2] + [_LIB_PATH]))
            return True
        _build_info["error"] = res.stderr.strip()
    return False


def compile_and_load_cuda():
    """Compile the CUDA kernels if needed and load them. Returns None if unavailable."""
    global _cuda_lib
    if _cuda_lib is not None:
        return _cuda_lib

    stale = (not os.path.exists(_LIB_PATH)
             or os.path.getmtime(_LIB_PATH) < os.path.getmtime(_CU_SOURCE))
    if stale and not _compile():
        return None

    try:
        lib = ctypes.CDLL(_LIB_PATH)
    except OSError as e:  # built elsewhere, or no CUDA runtime here
        _build_info["error"] = str(e)
        return None

    f32p = ctypes.POINTER(ctypes.c_float)
    for stage in STAGES:
        fn = getattr(lib, f"tinytorch_cuda_gemm_{stage}")
        fn.argtypes = [f32p, f32p, f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        fn.restype = ctypes.c_int
    lib.tinytorch_cuda_device_count.restype = ctypes.c_int

    _build_info["built"] = True
    _build_info["devices"] = lib.tinytorch_cuda_device_count()
    if _build_info["devices"] == 0:
        _build_info["error"] = "no CUDA device found"
        return None
    _cuda_lib = lib
    return _cuda_lib


def has_cuda_support() -> bool:
    """Returns True if the CUDA library loaded and found a GPU."""
    return compile_and_load_cuda() is not None


def cuda_build_info() -> dict:
    """GPUs found, the nvcc command (None if an earlier process built it), and the last error."""
    compile_and_load_cuda()
    return dict(_build_info)


def _ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _is_tensor(obj):
    return obj is not None and type(obj).__name__ == "Tensor"


def cuda_matmul(a, b, stage="tiled"):
    """Matrix multiply on an NVIDIA GPU with one of the three teaching kernels.

    Args:
        a: 2D array or Tensor [M, K]
        b: 2D array or Tensor [K, N]
        stage: "naive", "coalesced" or "tiled"

    Returns:
        2D float32 array or Tensor [M, N]. Falls back to np.matmul if CUDA is unavailable.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")

    is_tensor = _is_tensor(a) or _is_tensor(b)
    a_arr = a.data if _is_tensor(a) else a
    b_arr = b.data if _is_tensor(b) else b

    a_c = np.ascontiguousarray(a_arr, dtype=np.float32)
    b_c = np.ascontiguousarray(b_arr, dtype=np.float32)
    M, K = a_c.shape
    K_b, N = b_c.shape
    if K != K_b:
        raise ValueError(f"Incompatible matrix dimensions: ({M}, {K}) x ({K_b}, {N})")

    out = None
    lib = compile_and_load_cuda()
    if lib is not None:
        c = np.empty((M, N), dtype=np.float32)
        err = getattr(lib, f"tinytorch_cuda_gemm_{stage}")(_ptr(a_c), _ptr(b_c), _ptr(c), M, K, N)
        if err == 0:
            out = c
        else:
            # Warn rather than fall back silently: a timing of NumPy labeled CUDA would mislead
            _build_info["error"] = f"cudaError_t {err} from stage {stage!r}"
            warnings.warn(f"cuda_matmul: CUDA error {err}; fell back to np.matmul", RuntimeWarning)
    if out is None:
        out = np.matmul(a_c, b_c).astype(np.float32)

    if is_tensor:
        from tinytorch.core.tensor import Tensor
        return Tensor(out)
    return out
