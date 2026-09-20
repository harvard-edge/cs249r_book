# TinyTorch hardware extensions

Optional kernels that take TinyTorch past NumPy onto the hardware underneath.
None of the 20 modules needs them. They are hand-written and tracked in git as
they are; unlike the rest of `tinytorch/`, nothing here is exported from `src/`,
so edit these files directly.

Each extension falls back to NumPy when its compiler, library, or hardware is
missing, so importing `tinytorch.extensions` never fails.

| File | What it runs | Needs | Fallback |
|------|--------------|-------|----------|
| `cpp_simd_gemm.cpp`, `simd_ops.py` | Cache-blocked matrix multiply and fused bias + GELU in C++, called through `ctypes` | A C++ compiler; OpenMP for more than one core | `np.matmul`, NumPy GELU |
| `triton_gelu.py` | Fused bias + GELU as one Triton kernel | PyTorch, Triton, an NVIDIA GPU | NumPy GELU |
| `mps_ops.py` | Matrix multiply on the Apple GPU through MPS | PyTorch on an Apple M-series Mac | `np.matmul` |

## C++ SIMD kernels

`simd_ops.py` compiles `cpp_simd_gemm.cpp` into `libtinytorch_simd.so` the
first time you call it, and again whenever the `.cpp` file is newer than the
library. The GEMM walks the matrices in 64 × 64 tiles so each tile stays in
cache, and its inner loop runs along contiguous rows so the compiler vectorizes
it (AVX2 on x86, NEON on ARM).

The build tries OpenMP first. Apple clang has no OpenMP runtime of its own, so
on a Mac the kernel builds single-threaded unless Homebrew's `libomp` is
installed (`brew install libomp`). `simd_build_info()` reports what you got:

```python
from tinytorch.extensions import simd_matmul, simd_build_info
import numpy as np

A = np.random.randn(512, 512).astype(np.float32)
B = np.random.randn(512, 512).astype(np.float32)
C = simd_matmul(A, B)
print(simd_build_info())   # {'built': True, 'openmp': False, 'threads': 1, ...}
```

## Triton fused bias + GELU

Adding a bias and applying GELU as two separate operations writes `x + bias`
to GPU memory and reads it back. The Triton kernel does both in one pass, so
that intermediate never leaves registers.

```python
from tinytorch.extensions import triton_fused_gelu
Y = triton_fused_gelu(X, bias)   # X: [..., D], bias: [D]
```

## Apple MPS matrix multiply

`mps_matmul` hands the multiply to the Apple GPU through PyTorch's Metal
Performance Shaders backend. It runs on the GPU, not the Neural Engine. The
CPU and GPU share memory, but moving the arrays to and from the `mps` device
still copies them, and those copies count when you time it.

## Ideas for going further

1. Give the C++ GEMM a second level of tiling for the L2 cache, then register
   tiles, and measure each step against `np.matmul`.
2. Write a Triton kernel for INT8 matrix multiply that reads the weights
   produced by Module 15.
3. Wire `simd_matmul` into TinyTorch's `Linear` layer behind a flag, and
   profile a training step with Module 14's profiler.
