# TinyTorch Hardware Extensions & Acceleration Backends

This directory contains production-grade hardware extension examples showing how to bridge TinyTorch to native hardware accelerators, compiling custom compute kernels, and interfacing with accelerator runtimes.

---

## 1. Architecture Overview

TinyTorch is architected like `xv6`: clear, minimal, and extensible. When you need performance beyond interpreted Python, you can plug in custom kernels at three distinct layers:

```
+-------------------------------------------------------------+
|                      TinyTorch Tensor                       |
+-------------------------------------------------------------+
         |                       |                     |
         v                       v                     v
+------------------+   +-------------------+  +------------------+
|  C++ SIMD Kernel |   | OpenAI Triton JIT |  | Apple MPS / Metal|
|  (AVX2 / NEON)   |   | (SRAM Tiling GPU) |  | (Unified Memory) |
+------------------+   +-------------------+  +------------------+
         |                       |                     |
         v                       v                     v
+------------------+   +-------------------+  +------------------+
| Intel / AMD / ARM|   | NVIDIA Tensor Core|  | Apple M-Series   |
| CPU Execution    |   | HBM3 High Bandw.  |  | GPU / ANE Core   |
+------------------+   +-------------------+  +------------------+
```

---

## 2. Included Extension Modules

### `cpp_simd_gemm.cpp` & `simd_ops.py`
- **What it does**: Native C++ cache-blocked GEMM kernel ($64\times 64$ L1/L2 tile hierarchy) with OpenMP multi-threading and auto-vectorized SIMD inner loops.
- **Compilation**: Automatically compiled on first run via `c++ -O3 -shared -fPIC -std=c++17 cpp_simd_gemm.cpp -o libtinytorch_simd.so`.
- **Usage**:
  ```python
  from tinytorch.extensions import simd_matmul
  import numpy as np

  A = np.random.randn(512, 512).astype(np.float32)
  B = np.random.randn(512, 512).astype(np.float32)
  C = simd_matmul(A, B)
  ```

### `triton_gelu.py`
- **What it does**: OpenAI Triton GPU kernel fusing elementwise Bias addition + Gaussian Error Linear Unit (GELU) into a single SRAM register pass, eliminating round-trip DRAM memory bandwidth bottlenecks.
- **Hardware Target**: NVIDIA GPUs (A100, H100, RTX 3090/4090).
- **Usage**:
  ```python
  from tinytorch.extensions import triton_fused_gelu
  import numpy as np

  X = np.random.randn(1024, 768).astype(np.float32)
  bias = np.random.randn(768).astype(np.float32)
  Y = triton_fused_gelu(X, bias)
  ```

### `mps_ops.py`
- **What it does**: Apple Silicon Metal Performance Shaders (MPS) dispatch exploiting unified memory architecture without host-device PCIe copying penalties.
- **Hardware Target**: Apple M1/M2/M3/M4 Max/Ultra.

---

## 3. Extending TinyTorch: Student Assignment Ideas

1. **FlashAttention-2 Kernel in Triton**: Replace TinyTorch's attention loop with an online-softmax tiled kernel in Triton.
2. **INT4 Weight-Only Dequantization**: Implement a SIMD kernel that unpacks nibbles (4-bit integers) into FP16 registers on the fly.
3. **Rust PyO3 Tensor Core**: Rewrite `core/autograd.py` using Rust `pyo3` and `rayon` for zero-overhead graph traversal.
