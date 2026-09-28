# Milestone 07: Custom GPU Kernels (2024)

## Historical Context

Between 2021 and 2024, deep learning systems underwent a fundamental shift: the boundary between high-level frameworks and low-level hardware accelerators dissolved.

For years, writing custom GPU kernels required navigating millions of lines of proprietary CUDA C++, manually managing warp divergence, shared memory bank conflicts, and register pressure. In 2021, OpenAI introduced **Triton** (Philippe Tillet), enabling systems engineers to write high-performance GPU kernels in block-SPMD Python syntax that compile directly into bare-metal PTX assembly. Simultaneously, Apple Silicon unified memory introduced **Metal Performance Shaders (MPS)**, eliminating PCIe host-to-device memory transfer penalties.

Now it is your turn to cross the foreign function interface (FFI) and execute native kernels on bare silicon!

## What You're Running

**`01_custom_kernels.py` (Default):**
Executes and validates the three native hardware acceleration paths implemented in Chapter 21:
1. **C++ SIMD GEMM:** Passing raw contiguous DRAM pointers to AVX2/NEON vector units executing 8 to 16 floating-point operations per cycle.
2. **Apple Metal Performance Shaders (MPS):** Dispatching matrix multiplications directly to the Apple Silicon GPU across unified memory.
3. **OpenAI Triton GPU Shaders:** Running a fused bias + GELU block-SPMD kernel that eliminates DRAM round trips.

## Required Modules

**Run after Module 21** (Native Hardware Extensions)

| Module | Component | What It Provides |
| :--- | :--- | :--- |
| **Module 01** | Tensor | Flat DRAM memory buffers and contiguity checks |
| **Module 02** | Activations | GELU activation formula |
| **Module 03** | Layers | Linear layer projection math |
| **Module 21** | Extensions | C++ SIMD, Apple Metal MPS, and OpenAI Triton kernels |

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

| Backend | Platform | Verification Criteria |
| :--- | :--- | :--- |
| **C++ SIMD** | Any CPU (x86/ARM) | Max absolute error < 1e-4 vs NumPy float32 |
| **Apple Metal (MPS)** | macOS Apple Silicon | Max absolute error < 1e-4 vs NumPy float32 |
| **OpenAI Triton** | NVIDIA GPU / Fallback | Max absolute error < 1e-4 vs reference formula |

## Achievement Unlocked

After completing this milestone, you will understand:
- How the `ctypes` foreign function interface passes memory pointers without data copies.
- How SIMD vector registers execute data-parallel arithmetic in a single clock cycle.
- How OpenAI Triton compiles block-level SPMD programs with automatic memory coalescing.
- Why modern frameworks compile computational graphs down to custom native kernels.

**You have crossed from framework builder to hardware systems architect!**
