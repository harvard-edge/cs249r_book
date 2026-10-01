# Milestone 07: Custom Kernels (2024)

## Historical Context

Between 2021 and 2024, deep learning systems underwent a fundamental shift: the boundary between high-level frameworks and low-level hardware accelerators dissolved.

For years, writing custom GPU kernels required navigating millions of lines of proprietary CUDA C++, manually managing warp divergence, shared memory bank conflicts, and register pressure. In 2021, OpenAI introduced **Triton** (Philippe Tillet), enabling systems engineers to write high-performance GPU kernels in block-SPMD Python syntax that compile directly into PTX. Apple Silicon's unified memory let **Metal Performance Shaders (MPS)** run GPU kernels without a separate device memory.

Every one of those kernels rests on the ideas you implemented in Module 17: blocking a matrix multiply into cache-sized tiles, lowering convolution to one GEMM, and fusing an elementwise chain into one pass.

## What You're Running

**`01_custom_kernels.py`** has two parts.

1. **Part A, the pass gate: YOUR Module 17 kernels.** `tiled_matmul`, `fused_gelu`, `im2col`, `im2col_conv2d`, and `col2im` run on many shapes, including sizes that are not a multiple of the tile, stride 2, and padding. Each must match a NumPy reference within float32 tolerance.
2. **Part B, context: bundled native kernels.** The same GEMM and bias + GELU run on the kernels that ship with TinyTorch in `tinytorch/extensions/` (C++ SIMD when a compiler is available; Apple Metal MPS or OpenAI Triton when the hardware is present). The script prints measured timings next to yours. These kernels are reference points, not your work, and they never decide the result.

## Required Modules

| Module | Component | What It Provides |
| :--- | :--- | :--- |
| **Module 01** | Tensor | The tensor type the kernels take and return |
| **Module 06** | Autograd | Backward pass for the im2col convolution |
| **Module 09** | Convolutions | The `Conv2d` reference that im2col must reproduce |
| **Module 14** | Profiling | Latency and memory measurement |
| **Module 17** | Acceleration | YOUR tiled matmul, fused GELU, and im2col lowering |
| Optional | C++ compiler, Apple GPU, or CUDA + Triton | Native reference points in Part B |

## Running the Milestone

**Run via TITO (recommended):**

```bash
tito milestone run 07
```

**Or run directly:**

```bash
python3 milestones/07_2024_kernels/01_custom_kernels.py
```

## Expected Results

| Check | Criterion |
| :--- | :--- |
| `tiled_matmul`, `im2col_conv2d` | `allclose(rtol=1e-4, atol=1e-3)` vs NumPy on every shape and tile size |
| `fused_gelu` | `allclose(rtol=1e-5, atol=1e-5)` vs the float64 tanh formula |
| `col2im` | adjoint of `im2col`: `<im2col(x), g> == <x, col2im(g)>` |
| Native kernels | Timed and reported; a mismatch prints a warning (a TinyTorch bug, not yours) |

The milestone passes when Part A passes, with or without a compiler or GPU.

## Achievement Unlocked

After completing this milestone, you will understand:
- Why a blocked matmul must handle the ragged last tile on every axis.
- How im2col turns a sliding-window convolution into one matrix multiply.
- What a compiled SIMD kernel changes relative to NumPy, measured on your machine.
- Why modern frameworks compile computational graphs down to custom native kernels.
