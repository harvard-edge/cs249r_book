# Hand-maintained extension, NOT generated. This file is not produced by
# `tito dev export` or `tito module complete`; it is tracked in git and
# edited directly here. It is the CUDA SGEMM bridge (see cuda_sgemm.cu), an
# example of adding a GPU backend through the C ABI that students can read,
# run, and time against their own Module 17 kernels; no src/ module or student
# notebook exports to it.
"""CUDA SGEMM Bridge for TinyTorch: an example extension.

This is an optional, hand-written extension: it is tracked in git as-is and is
not generated from src/. Read it as a worked example of the silicon boundary:
how a few lines of Python hand NumPy memory to a GPU kernel you wrote yourself,
with no framework in between.

How it works, end to end:

1. Build on first use. The first call runs nvcc on cuda_sgemm.cu and writes
   libtinytorch_cuda.so next to it. Later calls reuse the library until the
   .cu file is edited, which triggers a rebuild.
2. Load through ctypes. ctypes.CDLL opens the library, and the argtypes lines
   below declare each C function's signature, so Python passes raw float
   pointers and ints exactly as the C code expects them.
3. Hand over the memory. cuda_matmul makes A and B contiguous float32 arrays
   and passes their raw pointers, plus a NumPy buffer for C to be written into.
   TinyTorch Tensors and plain arrays are both accepted, and the result comes
   back in the same kind.
4. Run on the GPU. The C side copies A and B to the GPU, launches the kernel
   for the chosen stage, copies C back, and returns a CUDA error code.

The three stages ("naive", "coalesced", "tiled") compute the same product and
differ only in how threads map onto memory; cuda_sgemm.cu explains each one.
Every call includes both PCIe copies, so time large matrices if you want to
see the kernels rather than the bus.

It needs nvcc and an NVIDIA GPU, not PyTorch. Without them, cuda_matmul falls
back to np.matmul and cuda_build_info() says why. If a kernel fails at run
time, it warns before falling back, so a NumPy timing is never mistaken for a
GPU one.

Try it:
    from tinytorch.extensions.cuda_ops import benchmark_stages
    benchmark_stages(2048)
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
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
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



def benchmark_stages(n: int = 2048, repeats: int = 5) -> list:
    """Time the three stages on one n x n product and check each against NumPy.

    Prints one line per stage and returns them as dicts. Each timing is the
    median of `repeats` calls after one warm-up call (the first call also pays
    for CUDA context setup), and includes both PCIe copies. Returns an empty
    list, after saying why, when CUDA is not available.
    """
    import time

    if not has_cuda_support():
        print(f"CUDA not available ({cuda_build_info()['error']}); nothing to time.")
        return []

    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n), dtype=np.float32)
    B = rng.standard_normal((n, n), dtype=np.float32)
    reference = A @ B
    flops = 2 * n ** 3

    print(f"SGEMM {n}x{n}x{n}, median of {repeats}, PCIe copies included")
    results = []
    for stage in STAGES:
        cuda_matmul(A, B, stage=stage)  # warm-up
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            C = cuda_matmul(A, B, stage=stage)
            times.append(time.perf_counter() - start)
        seconds = sorted(times)[len(times) // 2]
        ok = bool(np.allclose(C, reference, rtol=1e-3, atol=1e-2))
        gflops = flops / seconds / 1e9
        print(f"  {stage:<10} {seconds * 1e3:9.2f} ms {gflops:9.1f} GFLOP/s   "
              f"{'matches NumPy' if ok else 'MISMATCH'}")
        results.append({"stage": stage, "ms": seconds * 1e3, "gflops": gflops, "matches_numpy": ok})
    return results
