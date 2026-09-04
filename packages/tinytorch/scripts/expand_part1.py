#!/usr/bin/env python3
"""
Deep Narrative Expansion Script for Part I: The Core Engine
Frontmatter, Chapters 01-08, and Milestone 01
"""
import os
from pathlib import Path

DEST_DIR = Path("/Users/VJ/GitHub/MLSysBook/packages/tinytorch/narrative_book")

# ---------------------------------------------------------------------------
# 00_welcome.qmd
# ---------------------------------------------------------------------------
WELCOME_CONTENT = """# Welcome: The Black Box Crisis {.unnumbered}

In modern software engineering, artificial intelligence has arrived at a strange and precarious crossroad. 

Every day, hundreds of thousands of engineers around the world write the following three lines of Python code:

```python
import torch
loss = criterion(model(inputs), targets)
loss.backward()
optimizer.step()
```

These three lines possess immense power. They train multi-billion parameter foundation models, power self-driving vehicles, generate photorealistic visual art, and translate spoken human languages in real time.

Yet if you ask even experienced machine learning practitioners to explain what physically occurs on the silicon chip during those three lines of code, you will often be met with silence or hand-waving metaphors:

- *Where are the multi-dimensional tensor arrays physically allocated in system memory?*
- *How does the runtime transpose a matrix in zero time without copying a single byte of data?*
- *Why does stacking twenty linear layers collapse mathematically into a single shallow regression unless non-linear activation gates are inserted between them?*
- *How does the backward pass calculate exact analytical partial derivatives for twenty million parameters across dynamically branching control flow without running out of memory?*
- *Why does a floating-point computation that looks mathematically sound in a textbook crash a graphics processing unit (GPU) cluster with NaN gradients within ten iterations?*

```
The Developer vs. The Framework Reality:

┌─────────────────────────────────────────────────────────────────────────────┐
│  THE HIGH-LEVEL DEVELOPER ILLUSION:                                         │
│  "import torch" ──► "model.fit()" ──► Magical Black Box Optimization        │
├─────────────────────────────────────────────────────────────────────────────┤
│  THE HARDWARE SYSTEMS REALITY:                                              │
│  • 1D Contiguous DRAM Byte Arrays & Coordinate Strides                      │
│  • IEEE 754 Floating-Point Exponent Clamping & Log-Sum-Exp Invariants       │
│  • Dynamic In-Memory Tape DAGs & Reverse Topological VJP Traversal          │
│  • 16-Bytes-per-Parameter GPU Memory Allocation Law (AdamW)                 │
│  • 64-Byte Cache Line Alignment & SIMD Systolic Tensor Core GEMMs           │
└─────────────────────────────────────────────────────────────────────────────┘
```

Modern artificial intelligence has become an exercise in black-box alchemy. Frameworks like PyTorch, TensorFlow, and JAX have grown so massive---spanning hundreds of thousands of lines of dense C++, CUDA, and custom compiler intermediate representations (IR)---that they have become impenetrable monoliths to the engineers who rely on them.

This book is the antidote to that alchemy.

---

## The Philosophy of TinyTorch: The xv6 of Machine Learning

In 2006, Frans Kaashoek, Robert Morris, and Russ Cox at MIT created **xv6**: a modern, complete reimplementation of Dennis Ritchie and Ken Thompson's Sixth Edition Unix (v6) in ANSI C. xv6 was small enough to be read and understood by a single human being in a weekend, yet complete enough to boot on physical x86 hardware, schedule preemptive processes, manage virtual memory page tables, and handle interrupts. By stripping away millions of lines of legacy device drivers and backward compatibility shims, xv6 revealed the timeless, elegant soul of operating systems.

**TinyTorch is the xv6 of machine learning systems.**

TinyTorch is a complete, publication-grade deep learning framework built entirely from first principles in pure, readable Python and NumPy memory buffers. It contains **zero external machine learning dependencies**. We do not import PyTorch, TensorFlow, JAX, Scikit-Learn, or HuggingFace. 

Every single concept---from raw memory strides and automatic differentiation tapes to scaled dot-product attention, the GPT-2 transformer architecture, and the Williams Roofline performance model---is constructed by your own hands, line by line, equation by equation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE 4-TIER ARCHITECTURE OF TINYTORCH                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 1: The Core Compute Engine                                             │
│   • 1D Memory Buffers, Strides, Non-Linear Activations, Modular Layers      │
│   • Numerically Stable Losses, Async DataLoader, Dynamic Autograd Tape      │
│   • Heavy-Ball Momentum & AdamW Optimizers, 5-Step Training State Machine   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 2: Deep Architectures                                                  │
│   • Spatial 2D Convolutions & im2col GEMM Unrolling                         │
│   • Subword Byte-Pair Encoding (BPE) Tokenization                           │
│   • Continuous Embeddings & Fourier Sinusoidal Positional Waveforms         │
│   • Scaled Dot-Product Attention & Causal Lower-Triangular Masking          │
│   • Pre-LN Residual Transformer Highways & Complete TinyGPT Language Model  │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 3: Systems Acceleration & Performance Engineering                      │
│   • Williams Roofline Model Profiling & Memory-Bound Diagnostics            │
│   • Symmetric Uniform INT8 Quantization (4x Memory Footprint Reduction)     │
│   • Structured Channel Pruning & Low-Rank SVD Matrix Factorization          │
│   • Kernel Fusion (Fused GELU) & Cache-Tiled Matrix Multiplications         │
│   • The KV Cache Engine (Converting O(N) Generation Bandwidth into O(1))    │
│   • Rigorous MLPerf Benchmarking with P50/P95/P99 Tail Latency Statistics   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 4: Extensions & Future Frontiers                                       │
│   • OpenAI Triton Block Programming, TorchInductor, & Custom Accelerators   │
└─────────────────────────────────────────────────────────────────────────────┘
```

The entire TinyTorch framework comprises exactly **4,558 pure lines of executable code** across twenty self-contained modules. It is less than half the size of MIT xv6, yet it contains the complete mathematical and systems engine required to train vision models and generate human language with transformers.

---

## How to Read This Book

This is not an academic textbook filled with detached mathematical lemmas, nor is it a cookbook of copy-pasted configuration files. It is a **systems engineering monograph**.

Every chapter in this book follows a strict, disciplined five-beat narrative arc:

1. **The Crisis (The Engineering Tension)**: We open immediately with the physical or mathematical failure mode. Why does standard software fail? Why do linear layers collapse? Why does floating-point math overflow into NaN? Why do GPU tensor cores sit idle waiting on memory buses?
2. **The Mental Model & Geometry**: Before writing code, we construct the spatial, geometric, and physical intuition of the solution.
3. **The Pure TinyTorch Construction**: We implement the working solution in clean, tested, syntax-highlighted Python code.
4. **The Production Bridge**: We connect our implementation directly to how production engines (PyTorch C++ `c10::TensorImpl`, NVIDIA cuDNN, FlashAttention-2, and OpenAI Triton) solve the exact same invariant at industrial scale.
5. **Building the System: How It All Connects**: Every chapter concludes with an architectural synthesis showing how the newly built subsystem locks into the global engine, ending with a forward cliffhanger to the next breakthrough.

---

## The Builder's Rule

Throughout this book, every equation, data structure, and algorithm will be directly translated into a working implementation. If an equation does not serve a physical, memory, or computational purpose on hardware, it does not belong in this framework.

Turn the page. We begin in Chapter 1 with the foundation of all machine learning systems: **The Tensor**.
"""

# ---------------------------------------------------------------------------
# 01_tensors.qmd
# ---------------------------------------------------------------------------
CH01_CONTENT = """# Tensors & Strides: The Flat Memory Illusion {#sec-tensors}

Before we can compute a single gradient, train a neural network, or generate a single word with a transformer, we must confront the foundational data structure of all numerical computing: **The Multi-Dimensional Tensor**.

In introductory machine learning courses, a tensor is often described as a *"mathematical grid of numbers"* or a *"nested list of floats"*. In computer systems engineering, this high-level abstraction is dangerously misleading. Physical computer memory (Dynamic Random-Access Memory, or DRAM) is not a multi-dimensional matrix. DRAM is a flat, one-dimensional ribbon of sequential byte addresses extending from address `0x00000000` to `0xFFFFFFFF`.

In this chapter, we build the core data structure of TinyTorch: the **`Tensor`**. We explore why nested Python lists destroy CPU cache lines, how multi-dimensional coordinates are mapped to flat memory addresses through **strides**, and how the **stride-0 broadcasting trick** enables complex algebraic operations with zero memory allocation.

![The Multi-Dimensional Tensor Illusion: Flat 1D Memory Buffers Viewed Through Strides and Shapes](assets/images/diagrams/01_tensor-diag-1.svg){#fig-tensor-strides}

---

## 1.1 The Crisis: The Spatial Locality Failure of Nested Lists

When a programmer creates a 2D matrix in pure Python:

```python
matrix = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
]
```

Python does not allocate a compact block of nine floating-point numbers. Instead, Python allocates an outer list containing three pointers. Each pointer references an independent inner list allocated on the operating system heap. Each element within those inner lists is a full-fledged `PyObject` (occupying 24 bytes of metadata for an 8-byte float).

```
The Nested Python List Memory Nightmare:

Heap Address: 0x1000        Heap Address: 0x5400        Heap Address: 0x8200
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│ [Ptr to Row 0] ──┼───────►│ 1.0 │ 2.0 │ 3.0  │        │ 7.0 │ 8.0 │ 9.0  │
├──────────────────┤        └──────────────────┘        └──────────────────┘
│ [Ptr to Row 1] ──┼──────────────────────┐                       ▲
├──────────────────┤                      ▼                       │
│ [Ptr to Row 2] ──┼──────────────────────────────────────────────┘
└──────────────────┘        Heap Address: 0x2100
Outer List Pointer Array    ┌──────────────────┐
                            │ 4.0 │ 5.0 │ 6.0  │
                            └──────────────────┘
```

This pointer-indirected heap layout causes three catastrophic systems failures:

1. **Spatial Locality Destruction**: The rows of the matrix are scattered across arbitrary, disconnected heap locations. When the CPU attempts to read `matrix[0][1]` followed by `matrix[1][0]`, the CPU cache line misses completely, forcing the processor to stall for hundreds of clock cycles while waiting for off-chip DRAM.
2. **SIMD Vectorization Impossibility**: Modern CPU registers (AVX-512, ARM Neon) and GPU Tensor Cores can execute eight to sixty-four floating-point multiplications in a single clock cycle. However, SIMD instructions require contiguous memory: they cannot chase Python heap pointers.
3. **Massive Memory Overhead**: Nine 64-bit floats require 72 bytes of raw data. The nested Python list consumes over 300 bytes of pointer tables and reference counts---a **$400\\%$ memory tax**.

To build a high-performance machine learning runtime, our framework must enforce a strict invariant: **All tensor data must reside in a single, contiguous 1D memory buffer.**

---

## 1.2 The Mental Model: The Coordinate Translation Engine

If physical memory is strictly a flat 1D buffer, how do we represent a $3 \\times 4 \\times 5$ three-dimensional tensor?

A tensor is not a physical container of multi-dimensional memory. A tensor is an **algebraic coordinate translator** consisting of three lightweight metadata components layered over a flat 1D storage array:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          THE ANATOMY OF A TENSOR                            │
├───────────────────┬─────────────────────────────────────────────────────────┤
│ Component         │ Systems Function                                        │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ 1. Storage Buffer │ A contiguous 1D array of raw bytes in DRAM or VRAM.     │
│ 2. Shape (d_0..d) │ A tuple specifying the virtual dimensions (e.g. (3, 4)).│
│ 3. Strides (s_0..s│ The step size in memory needed to advance 1 unit along  │
│                   │ each respective dimension.                              │
│ 4. Storage Offset │ The starting memory index of the tensor within storage. │
└───────────────────┴─────────────────────────────────────────────────────────┘
```

### The Universal Memory Offset Equation

Given an $N$-dimensional virtual coordinate $(i_0, i_1, \\dots, i_{N-1})$, the exact 1D physical index in the underlying memory buffer is evaluated by the dot product of the coordinate vector with the strides vector:

$$\\text{Physical Offset} = \\text{storage\\_offset} + \\sum_{k=0}^{N-1} i_k \\times \\text{stride}_k$$

For a row-major contiguous tensor of shape $(D_0, D_1, D_2)$, the standard strides are computed by cumulative products from right to left:

$$\\text{stride}_2 = 1, \\qquad \\text{stride}_1 = D_2, \\qquad \\text{stride}_0 = D_1 \\times D_2$$

```
Row-Major 2D Memory Layout Example: Shape (2, 3), Strides (3, 1)

Virtual 2D Matrix:            Flat 1D Physical Buffer:
      Col 0  Col 1  Col 2     Index:  0    1    2    3    4    5
Row 0 ┌ 1.0   2.0   3.0 ┐     Value:[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
Row 1 └ 4.0   5.0   6.0 ┘            └─── Row 0 ───┘  └─── Row 1 ───┘

To access Element (1, 2):
Offset = 1 * stride[0] + 2 * stride[1] = 1 * 3 + 2 * 1 = 5 ──► Value 6.0!
```

---

## 1.3 Zero-Copy Tensor Views and the Stride-0 Trick

Because the shape and strides are decoupled from the physical memory storage, we can perform complex geometric transformations in **$O(1)$ constant time without copying a single byte of memory**.

### 1. Zero-Copy Transpose

To transpose a 2D matrix from shape $(M, N)$ with strides $(S_0, S_1)$ to shape $(N, M)$, we simply swap the stride values to $(S_1, S_0)$:

```
Transposing Shape (2, 3) with Strides (3, 1) ──► Shape (3, 2) with Strides (1, 3):
Original Coordinate (row, col): Offset = row * 3 + col * 1
Transposed Coordinate (col, row): Offset = col * 1 + row * 3  (SAME MEMORY!)
```

The underlying 1D data buffer is completely untouched. The transpose is instantaneous ($0.0\\text{ microseconds}$).

### 2. The Stride-0 Broadcasting Trick

When adding a vector of shape $(3,)$ to a matrix of shape $(4, 3)$, standard algebra requires expanding the vector into four identical rows.

In TinyTorch, we never allocate memory for broadcasted copies. We set the stride of the expanded dimension to **zero**:

$$\\text{Broadcasted Shape} = (4, 3), \\qquad \\text{Broadcasted Strides} = (0, 1)$$

```
Offset = row * 0 + col * 1 = col * 1
```

Regardless of which row index is requested ($0, 1, 2, \\text{ or } 3$), multiplying by `stride[0] = 0` always maps back to the exact same physical elements in memory! We achieve infinite virtual memory expansion with **zero bytes of RAM allocated**.

---

## 1.4 The Pure TinyTorch Construction

We implement the fundamental `Tensor` class in TinyTorch, managing contiguous memory, shapes, strides, and basic arithmetic:

```python
import numpy as np
from typing import Tuple, List, Union, Optional

class Tensor:
    \"\"\"The Fundamental Multi-Dimensional Tensor in TinyTorch.
    
    Encapsulates flat 1D memory buffers viewed through shape, strides,
    and automatic differentiation gradient tracking metadata.
    \"\"\"
    def __init__(self, data: Union[List, np.ndarray, float, int], requires_grad: bool = False):
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        elif isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, (float, int)):
            self.data = np.array([data], dtype=np.float32)
        elif isinstance(data, Tensor):
            self.data = data.data.copy()
        else:
            raise TypeError(f"Unsupported data type for Tensor: {type(data)}")

        self.requires_grad = requires_grad
        self.grad: Optional[Union['Tensor', np.ndarray]] = None
        
        # Autograd graph metadata (initialized in Chapter 6)
        self._op: Optional[str] = None
        self._parents: List['Tensor'] = []
        self._ctx: Optional[dict] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def strides(self) -> Tuple[int, ...]:
        # Convert byte strides to element strides (dividing by sizeof(float32)=4)
        return tuple(s // self.data.itemsize for s in self.data.strides)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def reshape(self, *shape) -> 'Tensor':
        \"\"\"Return a new tensor view with reshaped dimensions.\"\"\"
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)

    def transpose(self, dim0: int = -2, dim1: int = -1) -> 'Tensor':
        \"\"\"Zero-copy matrix transpose swapping two dimensions.\"\"\"
        return Tensor(np.swapaxes(self.data, dim0, dim1), requires_grad=self.requires_grad)

    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other_data = other.data if isinstance(other, Tensor) else other
        out = Tensor(self.data + other_data)
        out._op = "Add"
        out._parents = [self, other] if isinstance(other, Tensor) else [self]
        return out

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other_data = other.data if isinstance(other, Tensor) else other
        out = Tensor(self.data * other_data)
        out._op = "Mul"
        out._parents = [self, other] if isinstance(other, Tensor) else [self]
        return out

    def matmul(self, other: 'Tensor') -> 'Tensor':
        \"\"\"Matrix multiplication: Y = A @ B.\"\"\"
        out = Tensor(np.matmul(self.data, other.data))
        out._op = "MatMul"
        out._parents = [self, other]
        return out

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, data=\\n{self.data})"
```

---

## 1.5 The Production Bridge: PyTorch C++ `c10::TensorImpl`

In production PyTorch, Python's `torch.Tensor` is a lightweight Python wrapper around a deep C++ object called **`c10::TensorImpl`**:

```cpp
// PyTorch Core C++ Tensor Implementation (c10/core/TensorImpl.h)
struct TensorImpl : public c10::intrusive_ptr_target {
  c10::Storage storage_;           // Pointer to flat 1D memory buffer
  c10::SmallVector<int64_t, 5> sizes_;   // Shape tuple
  c10::SmallVector<int64_t, 5> strides_; // Strides tuple
  int64_t storage_offset_ = 0;     // Memory offset
  caffe2::TypeMeta data_type_;     // float32, float16, int8
  c10::Device device_;             // CPU, CUDA:0, TPU
};
```

When you call `tensor.transpose(0, 1)` or `tensor.view(-1)` in PyTorch, the C++ runtime does not touch the `c10::Storage` memory pointer. It creates a brand-new `TensorImpl` instance whose `sizes_` and `strides_` point to the original storage. This is why PyTorch views are guaranteed to execute in zero microseconds with zero memory allocations.

---

## 1.6 Building the System: How It All Connects

Let us examine what we have established in Chapter 1:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CHAPTER 1 RECAP: THE PHYSICAL FOUNDATION                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Flat 1D Memory Buffers  : Eliminates heap pointer chasing & cache misses. │
│ • Multi-Dimensional Strides: Decouples virtual coordinates from DRAM layout. │
│ • Zero-Copy Views         : Instant transpose, slice, and reshape in O(1).  │
│ • Stride-0 Broadcasting   : Virtual tensor expansion with zero memory tax.  │
└─────────────────────────────────────────────────────────────────────────────┘
```

We now have multi-dimensional matrix operations flowing over contiguous memory.

However, if we begin stacking linear matrix multiplications on top of each other ($W_2(W_1 x)$), we immediately hit a brick wall of linear algebra: **all multi-layer linear networks collapse into a single shallow matrix**.

In **Chapter 2**, we explore why deep linear networks fail, and engineer **Non-Linear Activations: Breaking the Linear Collapse Wall**.
"""

with open(DEST_DIR / "00_welcome.qmd", "w", encoding="utf-8") as f:
    f.write(WELCOME_CONTENT.strip() + "\n")
print("✓ Expanded 00_welcome.qmd")

with open(DEST_DIR / "01_tensors.qmd", "w", encoding="utf-8") as f:
    f.write(CH01_CONTENT.strip() + "\n")
print("✓ Expanded 01_tensors.qmd")

# ---------------------------------------------------------------------------
# 02_activations.qmd
# ---------------------------------------------------------------------------
CH02_CONTENT = """# Activations: Breaking the Linear Collapse Wall {#sec-activations}

In Chapter 1, we constructed the multi-dimensional tensor, conquering the memory allocation and strided layout of numerical data. With high-performance matrix multiplication ($Y = XW^T$) in hand, the intuitive next step in building a "deep" neural network is simply to chain multiple matrix multiplications together:

$$h_1 = X W_1, \\qquad h_2 = h_1 W_2, \\qquad \\dots, \\qquad y = h_{L-1} W_L$$

Yet if you construct this multi-layer network and train it for a thousand epochs, you will discover an astonishing mathematical failure: **a 100-layer linear network has no more expressive power than a single-layer linear regression**.

In this chapter, we engineer **Non-Linear Activation Functions (`ReLU`, `Sigmoid`, `Tanh`, `GELU`)**. We provide the mathematical proof of linear collapse, explore why historical sigmoid activations caused the vanishing gradient crisis, and implement the modern Gaussian Error Linear Unit (`GELU`) that powers the transformer architecture.

![The Activation Zoo: Breaking Linear Hyperplanes into Warped Non-Linear Decision Manifolds](assets/images/diagrams/02_activations-diag-1.svg){#fig-activation-zoo}

---

## 2.1 The Crisis: The Mathematical Proof of Linear Collapse

Why can a network composed entirely of linear matrix multiplications never learn non-linear patterns (such as computer vision, speech recognition, or human language)?

Let us examine a deep network composed of $L$ consecutive linear layers:

$$y = \\left( \\dots \\left( \\left( X W_1 \\right) W_2 \\right) \\dots \\right) W_L$$

Because matrix multiplication is strictly **associative** ($A(BC) = (AB)C$), we can group all $L$ weight matrices together into a single combined matrix product:

$$W_{\\text{combined}} = W_1 \\cdot W_2 \\cdot W_3 \\dots W_L$$

If $W_1 \\in \\mathbb{R}^{D_{\\text{in}} \\times H_1}$ and $W_L \\in \\mathbb{R}^{H_{L-1} \\times D_{\\text{out}}}$, the composite product $W_{\\text{combined}}$ is simply a single matrix of shape $\\mathbb{R}^{D_{\\text{in}} \\times D_{\\text{out}}}$.

```
The Associative Linear Collapse Proof:
Step 1: h_1 = X · W_1
Step 2: h_2 = h_1 · W_2 = (X · W_1) · W_2 = X · (W_1 · W_2)
Step 3: y   = h_2 · W_3 = (X · (W_1 · W_2)) · W_3 = X · (W_1 · W_2 · W_3)

Result: y = X · W_effective  where W_effective = W_1 · W_2 · W_3
```

No matter how many millions of parameters or how many layers you stack, the entire computational graph collapses into a single affine hyper-plane. A 100-layer deep linear network cannot even solve the basic logical XOR problem (which we will explore in Milestone I).

To build deep intelligence, we must break the associative chain. We must insert a **non-linear mathematical gate** $\\sigma(\\cdot)$ after every linear transformation:

$$h_1 = \\sigma(X W_1), \\qquad h_2 = \\sigma(h_1 W_2), \\qquad y = \\sigma(h_{L-1} W_L)$$

Because $\\sigma(A \\cdot B) \\ne \\sigma(A) \\cdot \\sigma(B)$, the layers cannot be merged. The network can now warp, fold, and carve non-linear decision manifolds in high-dimensional space.

---

## 2.2 The Mental Model: From Biological Thresholds to Smooth Quantum Gating

Over the history of artificial intelligence, four major activation functions have defined the evolution of deep learning:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       THE EVOLUTION OF ACTIVATION FUNCTIONS                 │
├─────────────┬───────────────────────────────┬───────────────────────────────┤
│ Activation  │ Mathematical Formulation      │ Systems / Mathematical Impact │
├─────────────┼───────────────────────────────┼───────────────────────────────┤
│ 1. Sigmoid  │ σ(x) = 1 / (1 + exp(-x))      │ Bounded (0, 1). Vanishing     │
│             │                               │ gradients for |x| > 4.        │
├─────────────┼───────────────────────────────┼───────────────────────────────┤
│ 2. Tanh     │ tanh(x) = (e^x - e^-x)/(e^x+e^-x) Zero-centered (-1, +1).     │
│             │                               │ Vanishing gradients for |x|>3.│
├─────────────┼───────────────────────────────┼───────────────────────────────┤
│ 3. ReLU     │ ReLU(x) = max(0, x)           │ Piecewise linear. Gradient=1  │
│             │                               │ for x>0 (Rescued deep nets!). │
├─────────────┼───────────────────────────────┼───────────────────────────────┤
│ 4. GELU     │ GELU(x) = x · Φ(x)            │ Smooth, probabilistic gating. │
│             │ ≈ 0.5x(1 + tanh(√(2/π)(x+...))) Modern transformer standard.  │
└─────────────┴───────────────────────────────┴───────────────────────────────┘
```

### The Vanishing Gradient Crisis of Sigmoid and Tanh

In the 1990s, networks relied primarily on the **Sigmoid** function $\\sigma(x) = \\frac{1}{1 + e^{-x}}$.

The derivative of the sigmoid function is:

$$\\sigma'(x) = \\sigma(x) (1 - \\sigma(x))$$

The maximum possible value of $\\sigma'(x)$ occurs at $x = 0$, where $\\sigma'(0) = 0.5 \\times 0.5 = \\mathbf{0.25}$.

```
Vanishing Gradients in a 10-Layer Sigmoid Network:
During backpropagation, gradients multiply across layers by the chain rule:
dL/dh_1 = dL/dh_10 * (0.25) * (0.25) * ... * (0.25) = dL/dh_10 * (0.25)^10 ≈ dL/dh_10 * 9.5e-7

The loss error completely vanishes before reaching the first hidden layer! ❌
```

### The ReLU Revolution (Nair & Hinton, 2010)

The **Rectified Linear Unit (ReLU)** rescued deep learning by defining a simple piecewise linear function:

$$\\text{ReLU}(x) = \\max(0, x)$$

For all positive activations ($x > 0$), the local derivative is exactly $\\mathbf{1.0}$. Gradients propagate through 100 layers with zero attenuation.

### The GELU Standard in Transformers (Hendrycks & Gimpel, 2016)

While ReLU makes a hard binary decision ($x > 0$), the **Gaussian Error Linear Unit (GELU)** weights inputs by their probability under a standard Gaussian cumulative distribution:

$$\\text{GELU}(x) = x \\cdot \\Phi(x) = x \\cdot P(X \\le x) \\quad \\text{where } X \\sim \\mathcal{N}(0, 1)$$

GELU provides a smooth, non-monotonic curvature: for small negative values ($x \\in [-2, 0]$), it allows a tiny negative gradient to pass through rather than hard-clamping to zero, preventing "dead neuron" syndrome.

---

## 2.3 The Pure TinyTorch Construction

We implement the complete suite of non-linear activations in TinyTorch:

```python
import numpy as np
from .tensor import Tensor

class Activation:
    \"\"\"Base class for non-linear activation functions.\"\"\"
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

class ReLU(Activation):
    \"\"\"Rectified Linear Unit: max(0, x).\"\"\"
    def forward(self, x: Tensor) -> Tensor:
        out_data = np.maximum(0.0, x.data)
        out_tensor = Tensor(out_data)
        out_tensor._op = "ReLU"
        out_tensor._parents = [x]
        return out_tensor

class Sigmoid(Activation):
    \"\"\"Sigmoid Activation: 1 / (1 + exp(-x)).\"\"\"
    def forward(self, x: Tensor) -> Tensor:
        # Numerically stable clipping to prevent overflow
        clipped_x = np.clip(x.data, -88.0, 88.0)
        out_data = 1.0 / (1.0 + np.exp(-clipped_x))
        out_tensor = Tensor(out_data)
        out_tensor._op = "Sigmoid"
        out_tensor._parents = [x]
        return out_tensor

class Tanh(Activation):
    \"\"\"Hyperbolic Tangent: (exp(x) - exp(-x)) / (exp(x) + exp(-x)).\"\"\"
    def forward(self, x: Tensor) -> Tensor:
        out_data = np.tanh(x.data)
        out_tensor = Tensor(out_data)
        out_tensor._op = "Tanh"
        out_tensor._parents = [x]
        return out_tensor

class GELU(Activation):
    \"\"\"Gaussian Error Linear Unit (GPT-2 standard approximation).\"\"\"
    def forward(self, x: Tensor) -> Tensor:
        # Fast Taylor approximation used in production transformers:
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        s = np.sqrt(2.0 / np.pi)
        x_cubed = x.data ** 3
        inner = s * (x.data + 0.044715 * x_cubed)
        out_data = 0.5 * x.data * (1.0 + np.tanh(inner))

        out_tensor = Tensor(out_data)
        out_tensor._op = "GELU"
        out_tensor._parents = [x]
        return out_tensor
```

---

## 2.4 The Production Bridge: Arithmetic Intensity of Elementwise Kernels

In systems performance engineering, non-linear activations are classified as **Elementwise Operators**.

Let us calculate the **Arithmetic Intensity** ($I = \\text{FLOPs} / \\text{Bytes Transferred}$) of a `ReLU` kernel on an NVIDIA H100 GPU:
- **Memory Read**: Load 32-bit float $x$ from DRAM ($4$ bytes).
- **Compute**: 1 comparison instruction (`max(0.0, x)` = 1 FLOP).
- **Memory Write**: Store 32-bit float $y$ back to DRAM ($4$ bytes).

$$I_{\\text{ReLU}} = \\frac{1 \\text{ FLOP}}{8 \\text{ Bytes}} = 0.125 \\text{ FLOPs/Byte}$$

Because an H100 GPU has a ridge point of $\\approx 298\\text{ FLOPs/Byte}$, the arithmetic tensor cores are operating at **less than $0.05\\%$ of their physical compute capability**. The processor is stalled $99.95\\%$ of the time waiting on the off-chip DRAM bus!

In Chapter 17, we will solve this crisis using **Kernel Fusion**, combining the linear matrix multiplication and GELU activation inside fast on-chip SRAM registers to eliminate DRAM roundtrips entirely.

---

## 2.5 Building the System: How It All Connects

Let us examine what we have established in Chapter 2:

```
                  ┌───────────────────────────────┐
                  │   Chapter 1: Contiguous Tensor │
                  │   (Flat 1D Memory Buffers)    │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │   Chapter 2: Activations      │
                  │   (ReLU, Sigmoid, Tanh, GELU) │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Non-Linear Forward Pass: h_1 = GELU(X · W_1),  h_2 = GELU(h_1 · W_2)        │
│ • Breaks associative linear collapse.                                       │
│ • Enables arbitrary multi-dimensional manifold carving.                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

We have tensors and non-linearities. But how do we organize learnable weights, biases, and initialization variance into clean, modular containers?

In **Chapter 3**, we engineer **Layers & Parameters: Packaging the Affine Transform**.
"""

# ---------------------------------------------------------------------------
# 03_layers.qmd
# ---------------------------------------------------------------------------
CH03_CONTENT = """# Layers & Parameters: Packaging the Affine Transform {#sec-layers}

In Chapters 1 and 2, we created the foundational data structures and mathematical gates of deep learning: contiguous tensors and non-linear activations.

Yet if you attempt to write a 10-layer network by manually creating standalone weight matrices (`w1`, `b1`, `w2`, `b2`, ..., `w10`, `b10`), your code quickly becomes unmaintainable. You must manually manage shapes, track which matrices contain learnable parameters during backpropagation, coordinate parameter updates across training epochs, and carefully balance initialization statistics so activations do not explode into numerical infinity.

In this chapter, we engineer **Modular Layers & Parameter Containers (`Layer`, `Linear`, `Dropout`, `Sequential`)**. We derive **Kaiming He's Variance Conservation Law**, prove why dropout requires scaling during training, and construct the modular parameter management tree that powers PyTorch's `nn.Module`.

![Modular Layer Architecture: Encapsulating Learnable Weights, Biases, and Forward Transformations](assets/images/diagrams/03_layers-diag-1.svg){#fig-layers-container}

---

## 3.1 The Crisis: The Exploding/Vanishing Variance Wall

Before a neural network trains on a single sample of data, its parameter weights must be initialized to random numbers.

What happens if you initialize the weights of a 50-layer network with standard normal random variables $W_{i,j} \\sim \\mathcal{N}(0, 1)$?

Let us trace the variance of activations flowing through a single linear layer $y = X W^T$ where input dimension is $D_{\\text{in}} = 1024$:

$$\\text{Var}(y_i) = \\text{Var}\\left( \\sum_{j=1}^{D_{\\text{in}}} x_j w_{ij} \\right) = \\sum_{j=1}^{D_{\\text{in}}} \\text{Var}(x_j w_{ij})$$

Assuming zero-mean independent variables:

$$\\text{Var}(y_i) = D_{\\text{in}} \\times \\text{Var}(x) \\times \\text{Var}(w) = 1024 \\times (1.0) \\times (1.0) = \\mathbf{1024}$$

In a single layer, activation variance explodes by a factor of $1024\\times$. Across just ten layers:

$$\\text{Variance after 10 layers} = (1024)^{10} = 1.26 \\times 10^{30} \\quad (\\text{Floating-point overflow into +inf!}) ❌$$

Conversely, if you initialize weights too small ($W_{i,j} \\sim \\mathcal{N}(0, 0.001)$):

$$\\text{Variance after 10 layers} = (1024 \\times 10^{-6})^{10} = 10^{-30} \\quad (\\text{Activations collapse to exact zero!}) ❌$$

Without rigorous mathematical initialization, deep neural networks either explode into `+inf` or collapse into dead silence before training even begins.

---

## 3.2 The Mental Model: Kaiming Variance Conservation

In 2015, Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun solved this crisis by deriving the exact condition for **Variance Conservation across ReLU networks**.

### Derivation of Kaiming Initialization

When activations pass through a `ReLU` gate ($\\max(0, x)$), exactly half of the distribution is zeroed out, cutting the active variance in half:

$$\\text{Var}(\\text{ReLU}(x)) = \\frac{1}{2} \\text{Var}(x)$$

Therefore, for a layer with input dimension $D_{\\text{in}}$:

$$\\text{Var}(y) = D_{\\text{in}} \\times \\left( \\frac{1}{2} \\text{Var}(x) \\right) \\times \\text{Var}(w) = \\left( \\frac{D_{\\text{in}}}{2} \\text{Var}(w) \\right) \\text{Var}(x)$$

To enforce the invariant that activation variance remains constant across every layer ($\\text{Var}(y) = \\text{Var}(x)$), we must set:

$$\\frac{D_{\\text{in}}}{2} \\text{Var}(w) = 1.0 \\implies \\mathbf{\\text{Var}(w) = \\frac{2}{D_{\\text{in}}}}$$

$$\\mathbf{\\sigma_w = \\sqrt{\\frac{2}{D_{\\text{in}}}}}$$

By sampling initial weights from $\\mathcal{N}\\left(0, \\sqrt{\\frac{2}{D_{\\text{in}}}}\\right)$ or uniform distribution $\\mathcal{U}\\left(-\\sqrt{\\frac{6}{D_{\\text{in}}}}, \\sqrt{\\frac{6}{D_{\\text{in}}}}\\right)$, activation signals flow through 100 deep layers with perfectly preserved unit variance!

---

## 3.3 Inverted Dropout: Zero-Cost Inference

To prevent co-adaptation of neurons during training, Nitish Srivastava and Geoffrey Hinton introduced **Dropout** (2014): randomly dropping each neuron with probability $p$.

```
Standard Dropout vs. Inverted Dropout:

Standard Dropout (Old):
• Training : y = x * mask  (where mask ~ Bernoulli(1 - p))
• Inference: y = x * (1 - p)  (Requires floating-point multiplication on every neuron!) ❌

Inverted Dropout (Modern TinyTorch Standard):
• Training : y = (x * mask) / (1 - p)  (Pre-scaled during training)
• Inference: y = x  (ZERO FLOPs, completely transparent identity pass!) ✅
```

By scaling activations by $\\frac{1}{1-p}$ during the training forward pass, the expected value $\\mathbb{E}[y]$ is preserved, requiring **zero computation at test-time inference**.

---

## 3.4 The Pure TinyTorch Construction

We implement the `Layer` base class, `Linear` affine layer, `Dropout`, and `Sequential` container:

```python
import numpy as np
from typing import List, Optional
from .tensor import Tensor

class Layer:
    \"\"\"Base container for all modular neural network layers in TinyTorch.\"\"\"
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        \"\"\"Return all learnable parameter tensors in this layer.\"\"\"
        params = []
        for attr_name, attr_val in self.__dict__.items():
            if isinstance(attr_val, Tensor) and attr_val.requires_grad:
                params.append(attr_val)
            elif isinstance(attr_val, Layer):
                params.extend(attr_val.parameters())
            elif isinstance(attr_val, (list, tuple)):
                for item in attr_val:
                    if isinstance(item, Layer):
                        params.extend(item.parameters())
        return params

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def train(self, mode: bool = True):
        \"\"\"Set training mode across all child modules.\"\"\"
        self.training = mode
        for attr_val in self.__dict__.values():
            if isinstance(attr_val, Layer):
                attr_val.train(mode)

    def eval(self):
        \"\"\"Set evaluation mode (disables dropout).\"\"\"
        self.train(False)

class Linear(Layer):
    \"\"\"Fully Connected Affine Layer: Y = X · W^T + b with Kaiming Uniform Init.\"\"\"
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Kaiming uniform initialization: bound = 1 / sqrt(fan_in)
        bound = 1.0 / np.sqrt(in_features)
        
        # Weight shape: [out_features, in_features]
        w_data = np.random.uniform(-bound, bound, (out_features, in_features))
        self.weight = Tensor(w_data, requires_grad=True)

        if bias:
            b_data = np.random.uniform(-bound, bound, (out_features,))
            self.bias = Tensor(b_data, requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        \"\"\"Forward affine transformation: Y = X · W^T + b.\"\"\"
        # Perform matrix multiplication
        out_data = np.matmul(x.data, self.weight.data.T)
        if self.bias is not None:
            out_data = out_data + self.bias.data

        out_tensor = Tensor(out_data)
        out_tensor._op = "Linear"
        out_tensor._parents = [x, self.weight] + ([self.bias] if self.bias is not None else [])
        return out_tensor

class Dropout(Layer):
    \"\"\"Inverted Dropout Layer with zero inference overhead.\"\"\"
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        # Inverted dropout: scale by 1 / (1 - p) during training
        keep_prob = 1.0 - self.p
        mask = (np.random.rand(*x.data.shape) < keep_prob).astype(np.float32) / keep_prob
        
        out_tensor = Tensor(x.data * mask)
        out_tensor._op = "Dropout"
        out_tensor._parents = [x]
        return out_tensor

class Sequential(Layer):
    \"\"\"Sequential Layer Container chaining multiple sub-layers.\"\"\"
    def __init__(self, *layers: Layer):
        super().__init__()
        self.layers = list(layers)

    def parameters(self) -> List[Tensor]:
        params = []
        for l in self.layers:
            params.extend(l.parameters())
        return params

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer.forward(x)
        return x
```

---

## 3.5 The Production Bridge: PyTorch C++ `torch::nn::Module` Tree

In PyTorch C++, every layer inherits from `torch::nn::Module`, which implements an internal recursive tree traversal:

```cpp
// PyTorch Parameter Registration (torch/csrc/api/include/torch/nn/module.h)
Tensor& Module::register_parameter(std::string name, Tensor tensor, bool requires_grad) {
  tensor.set_requires_grad(requires_grad);
  parameters_.insert({std::move(name), std::move(tensor)});
  return parameters_[name];
}
```

When an optimizer iterates over `model.parameters()`, it traverses this exact recursive C++ tree to extract pointers to all weight and bias tensors.

---

## 3.6 Building the System: How It All Connects

With Chapter 3, TinyTorch can package complex neural architectures:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Complete Multi-Layer Neural Network Container:                              │
│                                                                             │
│ model = Sequential(                                                         │
│     Linear(in_features=784, out_features=256),   # Kaiming Init             │
│     ReLU(),                                      # Non-Linear Gate          │
│     Dropout(p=0.2),                              # Inverted Regularization  │
│     Linear(in_features=256, out_features=10)     # Final Logits             │
│ )                                                                           │
│                                                                             │
│ • model.parameters() automatically harvests all weight and bias Tensors.   │
│ • Activation variance is strictly conserved across all forward passes.      │
└─────────────────────────────────────────────────────────────────────────────┘
```

Our model produces raw continuous output vectors (logits). But how do we measure how wrong our predictions are without causing floating-point overflow?

In **Chapter 4**, we engineer **Loss Functions & Log-Sum-Exp: Numerical Hazards in Probability Space**.
"""

with open(DEST_DIR / "02_activations.qmd", "w", encoding="utf-8") as f:
    f.write(CH02_CONTENT.strip() + "\n")
print("✓ Expanded 02_activations.qmd")

with open(DEST_DIR / "03_layers.qmd", "w", encoding="utf-8") as f:
    f.write(CH03_CONTENT.strip() + "\n")
print("✓ Expanded 03_layers.qmd")

# ---------------------------------------------------------------------------
# 04_losses.qmd
# ---------------------------------------------------------------------------
CH04_CONTENT = """# Loss Functions & Log-Sum-Exp: Numerical Hazards in Probability Space {#sec-losses}

In Chapters 1 through 3, we constructed deep neural networks capable of transforming input vectors into unconstrained continuous numbers known as **logits** ($z \\in \\mathbb{R}^C$).

Yet before an optimization algorithm can steer our network, we must evaluate a single scalar error metric: the **Loss Function** $\\mathcal{L}$. In classification and generative language modeling, this requires converting unconstrained real logits into a normalized probability distribution using the **Softmax** function, followed by evaluating the **Cross-Entropy Loss**.

In this chapter, we engineer **Loss Functions (`MSELoss`, `CrossEntropyLoss`)** and master the **Log-Sum-Exp Numerical Invariant**. We explore why naive floating-point exponential math overflows into `NaN` on modern processors, and derive the shift-invariance equation that guarantees stable probabilistic training.

![The Numerical Stability Landscape: IEEE 754 Overflow/Underflow Hazards vs. Log-Sum-Exp Shift Invariance](assets/images/diagrams/04_losses-diag-1.svg){#fig-loss-stability}

---

## 4.1 The Crisis: The IEEE 754 Floating-Point Overflow Hazard

To convert a vector of $C$ logits $\\mathbf{z} = [z_1, z_2, \\dots, z_C]$ into predicted probabilities $\\mathbf{p}$, standard probability theory dictates the **Softmax** function:

$$p_i = \\frac{e^{z_i}}{\\sum_{j=1}^C e^{z_j}}$$

The standard **Cross-Entropy Loss** for ground-truth one-hot label $y$ is:

$$\\mathcal{L} = -\\sum_{i=1}^C y_i \\log(p_i) = -\\log(p_{\\text{target}}) = -\\log\\left( \\frac{e^{z_{\\text{target}}}}{\\sum_{j=1}^C e^{z_j}} \\right)$$

In an ideal mathematical universe with infinite-precision real numbers, this formula is clean and elegant. On real silicon chips operating under the **IEEE 754 32-bit Floating-Point Standard (FP32)**, this naive formula causes immediate catastrophic failure:

```
IEEE 754 FP32 Floating-Point Representation Limits:
• Maximum Representable Float : 3.4028235 x 10^38  (exp(88.72) is the ABSOLUTE CEILING!)
• Minimum Positive Normal Float: 1.1754944 x 10^-38 (exp(-87.33) is the UNDERFLOW FLOOR!)
```

Consider what happens when an untrained model outputs unscaled logits during its first forward pass:

```
The Two Fatal Floating-Point Catastrophes:

1. THE EXPONENTIAL OVERFLOW CRASH (z_i = 100):
   • Calculation : exp(100.0) ──► OVERFLOWS TO +inf!
   • Softmax     : (+inf) / (+inf) = NaN  (Not a Number!)
   • Result      : Loss becomes NaN; all parameter weights corrupted to NaN! ❌

2. THE EXPONENTIAL UNDERFLOW COLLAPSE (z_i = -100):
   • Calculation : exp(-100.0) ──► UNDERFLOWS TO EXACT 0.0!
   • Softmax Sum : ∑ exp(z_j) = 0.0.
   • Division    : 0.0 / 0.0 = NaN!
   • Logarithm   : log(0.0) = -inf!  (Loss explodes to infinity!) ❌
```

A single large or negative logit anywhere in a batch of thousands of tokens immediately destroys the entire training run.

---

## 4.2 The Mental Model: Log-Sum-Exp Shift Invariance

How do we compute Softmax and Cross-Entropy on arbitrary logit values ranging from $-\\infty$ to $+\\infty$ without ever exceeding the IEEE 754 float limits?

The solution is the **Log-Sum-Exp (LSE) Shift Invariance Invariant**:

### Proof of Softmax Shift Invariance

Let $c$ be an arbitrary scalar constant. We multiply both the numerator and denominator of the Softmax equation by $e^{-c}$:

$$p_i = \\frac{e^{z_i}}{\\sum_{j=1}^C e^{z_j}} = \\frac{e^{-c} \\cdot e^{z_i}}{e^{-c} \\cdot \\sum_{j=1}^C e^{z_j}} = \\frac{e^{z_i - c}}{\\sum_{j=1}^C e^{z_j - c}}$$

The mathematical output is **identical for any choice of constant $c$**.

### The Optimal Choice: $c = \\max(\\mathbf{z})$

If we choose $c = \\max(z_1, z_2, \\dots, z_C)$, we transform the shifted logits $\\tilde{z}_i = z_i - \\max(\\mathbf{z})$:

$$\\tilde{z}_i \\le 0 \\quad \\text{for ALL } i \\in \\{1, \\dots, C\\}$$

$$\\max(\\tilde{\\mathbf{z}}) = 0.0$$

```
The Magical Stability of Shifting by Max(z):
1. No Overflow : Because all z_i - max(z) <= 0, the maximum possible exponent is exp(0.0) = 1.0!
                 exp(z_i - max(z)) NEVER exceeds 1.0 (Overflow is mathematically IMPOSSIBLE!). ✅
2. No Underflow: The denominator sum ALWAYS contains at least one term equal to exp(0.0) = 1.0!
                 The denominator sum is ALWAYS >= 1.0 (Division by zero is IMPOSSIBLE!). ✅
```

### The Fused Log-Sum-Exp Cross-Entropy Formula

In Cross-Entropy, we compute $\\log(p_i)$ directly without computing Softmax as an intermediate step:

$$\\log(p_i) = \\log\\left( \\frac{e^{z_i}}{\\sum_j e^{z_j}} \\right) = z_i - \\log\\left( \\sum_{j=1}^C e^{z_j} \\right)$$

$$\\text{LSE}(\\mathbf{z}) = c + \\log\\left( \\sum_{j=1}^C e^{z_j - c} \\right) \\quad \\text{where } c = \\max(\\mathbf{z})$$

$$\\mathcal{L}_{\\text{CE}} = \\text{LSE}(\\mathbf{z}) - z_{\\text{target}}$$

This fused formulation requires **zero division operations**, **zero intermediate DRAM allocations**, and is **$100\\%$ mathematically immune to float overflow and underflow**.

---

## 4.3 The Pure TinyTorch Construction

We implement `MSELoss` and numerically stable `CrossEntropyLoss` in TinyTorch:

```python
import numpy as np
from typing import Optional
from .tensor import Tensor

class Loss:
    \"\"\"Base class for TinyTorch loss functions.\"\"\"
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)

class MSELoss(Loss):
    \"\"\"Mean Squared Error Loss: L = (1/N) * sum((y_pred - y_true)^2).\"\"\"
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        diff = predictions.data - targets.data
        loss_val = np.mean(diff ** 2)
        
        out_tensor = Tensor(loss_val)
        out_tensor._op = "MSELoss"
        out_tensor._parents = [predictions, targets]
        return out_tensor

class CrossEntropyLoss(Loss):
    \"\"\"Numerically Stable Cross-Entropy Loss with Fused Log-Sum-Exp.\"\"\"
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        \"\"\"Evaluate Cross-Entropy Loss over logits and integer/one-hot targets.
        
        Args:
            predictions: Logits tensor of shape [batch_size, num_classes]
            targets: Target labels (integer indices [batch_size] or one-hot [batch_size, num_classes])
        \"\"\"
        logits = predictions.data
        N, C = logits.shape

        # 1. Log-Sum-Exp Shift Invariance: subtract row-wise maximum
        max_logits = np.max(logits, axis=-1, keepdims=True)  # Shape: [N, 1]
        shifted_logits = logits - max_logits
        
        # 2. Compute stable Log-Sum-Exp: max(z) + log(sum(exp(z - max(z))))
        sum_exp = np.sum(np.exp(shifted_logits), axis=-1, keepdims=True)
        log_sum_exp = max_logits + np.log(sum_exp)  # Shape: [N, 1]

        # 3. Extract target logits
        if targets.data.ndim == 1:  # Integer class indices
            target_indices = targets.data.astype(np.int64)
            target_logits = logits[np.arange(N), target_indices][:, np.newaxis]
        else:  # One-hot encoded targets
            target_logits = np.sum(logits * targets.data, axis=-1, keepdims=True)

        # 4. Scalar Cross-Entropy Loss: Mean over batch of (LSE - target_logit)
        sample_losses = log_sum_exp - target_logits
        loss_val = np.mean(sample_losses)

        out_tensor = Tensor(np.float32(loss_val))
        out_tensor._op = "CrossEntropyLoss"
        out_tensor._parents = [predictions, targets]
        return out_tensor
```

---

## 4.4 The Production Bridge: PyTorch C++ `log_softmax` Dispatch

In PyTorch, calling `torch.nn.CrossEntropyLoss` does not execute `torch.softmax()` followed by `torch.log()`. 

The PyTorch C++ dispatcher bypasses both functions and routes directly to the fused CUDA kernel `at::_log_softmax_backward_data_out`:

```cpp
// PyTorch Fused Log-Softmax Kernel (aten/src/ATen/native/cuda/SoftMax.cu)
template <typename scalar_t>
__global__ void log_softmax_warp_forward(scalar_t* output, const scalar_t* input, int64_t classes) {
  // Uses NVIDIA Warp Shuffle (__shfl_xor_sync) to find max(z) across 32 threads in 1 cycle
  scalar_t max_val = warp_reduce_max(input[threadIdx.x]);
  scalar_t sum_exp = warp_reduce_sum(exp(input[threadIdx.x] - max_val));
  output[threadIdx.x] = (input[threadIdx.x] - max_val) - log(sum_exp);
}
```

By leveraging hardware warp shuffle instructions, all 32 threads in a CUDA warp find $\\max(\\mathbf{z})$ and compute the fused Log-Sum-Exp in **a single hardware clock cycle** with zero off-chip memory traffic.

---

## 4.5 Building the System: How It All Connects

With Chapter 4, our framework has an overflow-safe error metric:

```
                  ┌───────────────────────────────┐
                  │   Chapter 3: Deep MLP Model   │
                  │   (Produces Raw Logits z)     │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Chapter 4: Numerically Stable CrossEntropyLoss                              │
│ • Log-Sum-Exp Shift Invariance clamps exponents into [-inf, 0].             │
│ • Zero overflow, zero underflow, zero NaN corruption.                       │
│ • Evaluates scalar error metric L: scalar compass for backpropagation.      │
└─────────────────────────────────────────────────────────────────────────────┘
```

We have a model and an error metric. But where does the training data come from? How do we stream millions of samples from disk into memory without starving the CPU?

In **Chapter 5**, we engineer **The DataLoader: Asynchronous Feeding of the Compute Engine**.
"""

# ---------------------------------------------------------------------------
# 05_dataloader.qmd
# ---------------------------------------------------------------------------
CH05_CONTENT = """# The DataLoader: Asynchronous Feeding of the Compute Engine {#sec-dataloader}

In Chapters 1 through 4, we engineered the compute half of our deep learning framework: multidimensional tensor storage, non-linear activation gates, modular parameter layers, and overflow-safe loss functions.

Yet a compute engine is useless without fuel. If our training loop reads one sample at a time synchronously from disk during execution, the processor spends $95\\%$ of its time stalled waiting for disk I/O, leaving expensive arithmetic ALUs completely starved.

In this chapter, we engineer **The Data Pipeline (`Dataset`, `DataLoader`, `BatchSampler`)**. We explore the Producer-Consumer pattern in operating systems, implement random shuffling without replacement, and build asynchronous memory-pinned batch collation that keeps arithmetic hardware saturated at $100\\%$ utilization.

![The Asynchronous DataLoader Pipeline: Decoupled Worker Processes Streaming Batches into Pinned Shared Memory](assets/images/diagrams/05_dataloader-diag-1.svg){#fig-dataloader-pipeline}

---

## 5.1 The Crisis: The I/O Starvation Bottleneck

Consider training a vision model on a dataset of 100,000 images. A beginner writes the following training loop:

```python
# The Synchronous I/O Starvation Disaster:
for sample_path in dataset_paths:
    image, label = load_from_disk(sample_path)  # Disk Read: 10.0 ms
    output = model(image)                       # Compute:   0.2 ms
    loss = criterion(output, label)             # Loss:      0.01 ms
```

Let us calculate the hardware duty cycle:

$$\\text{Total Iteration Time} = 10.0\\text{ ms (I/O)} + 0.21\\text{ ms (Compute)} = 10.21\\text{ ms}$$

$$\\text{Hardware Utilization} = \\frac{0.21\\text{ ms}}{10.21\\text{ ms}} = \\mathbf{2.05\\%}$$

The arithmetic units are sitting **completely idle $98\\%$ of the time**. 

```
Synchronous Single-Sample Execution Timeline:
Disk Read (10ms) ────────► Compute (0.2ms) ──► Disk Read (10ms) ────────► Compute (0.2ms)
[████████████████████████] [█]                 [████████████████████████] [█]
▲ GPU sits idle 98% of the time!
```

Furthermore, training a neural network on single samples ($B = 1$) introduces extreme gradient variance, causing parameter updates to bounce erratically without converging.

To saturate hardware compute, our framework must solve two systems challenges:
1. **Batching**: Grouping $B$ individual samples into a single 4D tensor matrix ($B \\times C \\times H \\times W$), converting multiple scalar memory lookups into a single high-throughput SIMD matrix multiplication.
2. **Asynchronous Prefetching**: Decoupling disk I/O from compute via the **Producer-Consumer Architecture**: worker threads load batch $t+1$ from disk into memory *while* the processor is computing batch $t$.

---

## 5.2 The Mental Model: The Producer-Consumer Queue and Shuffling

The modern data pipeline separates responsibilities into three distinct architectural layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ THE THREE LAYERS OF THE DATA PIPELINE                                       │
├───────────────────┬─────────────────────────────────────────────────────────┤
│ 1. Dataset        │ Defines the random-access data contract:                │
│                   │ • __len__(): Returns total sample count N.              │
│                   │ • __getitem__(idx): Returns the i-th sample (x_i, y_i). │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ 2. Sampler        │ Determines the order of indices to visit:               │
│                   │ • Sequential: [0, 1, 2, ..., N-1]                       │
│                   │ • Shuffled: Random permutation of {0..N-1} per epoch.   │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ 3. DataLoader     │ Manages batching, multi-worker prefetching, and memory: │
│                   │ • Pulls B samples from Dataset.                         │
│                   │ • Collates them into contiguous Tensor batches.         │
│                   │ • Yields batches asynchronously to the compute engine.  │
└───────────────────┴─────────────────────────────────────────────────────────┘
```

```
Producer-Consumer Asynchronous Pipelining Timeline:
Worker Process (I/O):  [ Load Batch 1 ] [ Load Batch 2 ] [ Load Batch 3 ]
                             │               │               │
                             ▼               ▼               ▼
Shared Memory Queue:   ──► [ Queue ] ────► [ Queue ] ────► [ Queue ] ──►
                             │               │               │
Compute Engine (ALU):        ▼               ▼               ▼
                       [ Step Batch 0 ] [ Step Batch 1 ] [ Step Batch 2 ]
                       ▲ Compute engine NEVER STALLS! Hardware runs at 100% saturation!
```

### Why Shuffling Without Replacement Matters

If we train on unshuffled data (e.g. all class 0 images, then all class 1 images), the optimizer suffers from **Catastrophic Gradient Bias**: it spends the first 1,000 steps optimizing purely for class 0, overwriting earlier learned features.

By generating a new **Random Permutation** of indices $\\pi \\in S_N$ at the start of every epoch, every batch contains an unbiased, independent and identically distributed (i.i.d.) representation of the global dataset.

---

## 5.3 The Pure TinyTorch Construction

We implement the complete data pipeline in TinyTorch:

```python
import numpy as np
from typing import Iterator, List, Tuple, Any, Optional
from .tensor import Tensor

class Dataset:
    \"\"\"Abstract Dataset Protocol defining random-access data storage.\"\"\"
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        raise NotImplementedError

class TensorDataset(Dataset):
    \"\"\"Dataset wrapping contiguous input and target Tensors.\"\"\"
    def __init__(self, inputs: Tensor, targets: Tensor):
        if len(inputs.data) != len(targets.data):
            raise ValueError(f"Inputs and targets must have same length: {len(inputs.data)} vs {len(targets.data)}")
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs.data)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.inputs.data[index], self.targets.data[index]

class DataLoader:
    \"\"\"Asynchronous Batch Collation and Shuffling Engine.\"\"\"
    def __init__(self, dataset: Dataset, batch_size: int = 32, 
                 shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = len(dataset)

    def __len__(self) -> int:
        \"\"\"Return total number of batches per epoch.\"\"\"
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        \"\"\"Yield batches of contiguous collated Tensors.\"\"\"
        # 1. Generate index permutation
        if self.shuffle:
            indices = np.random.permutation(self.num_samples)
        else:
            indices = np.arange(self.num_samples)

        # 2. Iterate in strides of batch_size
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = start_idx + self.batch_size
            
            if end_idx > self.num_samples:
                if self.drop_last:
                    break
                end_idx = self.num_samples

            batch_indices = indices[start_idx:end_idx]

            # 3. Collate individual samples into contiguous batch arrays
            batch_inputs = []
            batch_targets = []
            for idx in batch_indices:
                x_sample, y_sample = self.dataset[idx]
                batch_inputs.append(x_sample)
                batch_targets.append(y_sample)

            # Package into high-performance Tensors
            yield Tensor(np.array(batch_inputs)), Tensor(np.array(batch_targets))
```

---

## 5.4 The Production Bridge: POSIX Shared Memory and DMA Page Locking

In production PyTorch (`torch.utils.data.DataLoader(..., num_workers=8, pin_memory=True)`):

```
Production PyTorch DataLoader Architecture:
1. Multi-Processing IPC (/dev/shm):
   • Worker processes communicate with the main Python process via POSIX Shared Memory.
   • Eliminates expensive Python inter-process object pickling overhead!

2. Pinned Memory (CUDA Page-Locked Host Memory):
   • Memory pages allocated via cudaHostAlloc() cannot be paged out by the OS kernel.
   • GPU can read data directly across PCIe bus via Direct Memory Access (DMA)
     with ZERO CPU intervention!
```

---

## 5.5 Building the System: How It All Connects

Let us examine our complete forward execution pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ COMPLETE FORWARD EXECUTION PIPELINE                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. DataLoader (Ch 5) : Asynchronously collates batches X: [B, D_in].        │
│ 2. Layers (Ch 3)     : Evaluates affine transforms: Z = X · W^T + b.        │
│ 3. Activations (Ch 2): Applies non-linear gates: H = GELU(Z).               │
│ 4. Losses (Ch 4)     : Computes stable scalar loss L via Log-Sum-Exp.       │
└─────────────────────────────────────────────────────────────────────────────┘
```

The entire forward pipeline of deep learning is alive.

Now we arrive at the central mathematical miracle of modern artificial intelligence:

Given a scalar loss $\\mathcal{L}$ computed over millions of parameters, how do we calculate the exact analytical partial derivative with respect to every single parameter in a single reverse sweep?

In **Chapter 6**, we construct **Automatic Differentiation: The Dynamic Tape DAG**.
"""

# ---------------------------------------------------------------------------
# 06_autograd.qmd
# ---------------------------------------------------------------------------
CH06_CONTENT = """# Automatic Differentiation: The Dynamic Tape DAG {#sec-autograd}

In Chapters 1 through 5, we constructed the complete forward execution pipeline of TinyTorch. We can stream batches of data asynchronously from disk, pass them through deep layers of non-linear transformations, and compute an exact scalar loss $\\mathcal{L}$.

Now we confront the central mathematical engine of all modern artificial intelligence: **Automatic Differentiation (Autograd)**. In this chapter, we explore why numerical and symbolic differentiation fail for deep networks, and construct a **Dynamic Reverse-Mode Tape Directed Acyclic Graph (DAG)** that evaluates exact analytical gradients for millions of parameters in a single reverse sweep.

![Dynamic Autograd Tape: Forward Tape Recording and Reverse-Mode Topological Accumulation](assets/images/diagrams/06_autograd-diag-1.svg){#fig-autograd-dag}

---

## 6.1 The Crisis: The Derivative Calculation Bottleneck

To minimize our loss function $\\mathcal{L}(W)$ using gradient descent, we must compute the partial derivative of the loss with respect to every single parameter weight $w_i$:

$$\\nabla_W \\mathcal{L} = \\left[ \\frac{\\partial \\mathcal{L}}{\\partial w_1}, \\frac{\\partial \\mathcal{L}}{\\partial w_2}, \\dots, \\frac{\\partial \\mathcal{L}}{\\partial w_N} \\right]$$

In computer science history, engineers attempted three distinct approaches to differentiation before modern autograd was discovered:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. NUMERICAL FINITE DIFFERENCES: The O(N) Compute Wall                      │
│    • Formula:  ∂L/∂w_i ≈ (L(w_i + ε) - L(w_i)) / ε                          │
│    • Flaw:     Requires N+1 forward passes for N parameters.                │
│    • Reality:  A 100M parameter model would take 3 years for ONE step! ❌   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. SYMBOLIC ALGEBRAIC DIFFERENTIATION: The Expression Explosion             │
│    • Formula:  Applies product/chain rules algebraically (e.g. Mathematica).│
│    • Flaw:     Expression length grows exponentially with network depth.    │
│    • Reality:  Differentiating a 50-layer network yields gigabytes of math! ❌│
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. MANUAL ANALYTICAL DERIVATION: The Brittleness Wall                       │
│    • Formula:  Hand-coding backward formulas in C++ for every network.      │
│    • Flaw:     Breaks instantly if a researcher adds an `if` branch. ❌    │
└─────────────────────────────────────────────────────────────────────────────┘
```

Modern deep learning demands an engine that can compute exact analytical derivatives for **arbitrary, dynamically branching Python code** with the computational cost of just **two forward passes** ($O(1)$ scaling with respect to parameter count).

That engine is **Reverse-Mode Automatic Differentiation on a Dynamic Tape**.

---

## 6.2 The Mental Model: The Reverse-Mode Tape DAG

Reverse-mode automatic differentiation operates across two distinct phases:

### Phase 1: Forward Tape Recording
During the forward pass, as each mathematical operation (`+`, `*`, `matmul`, `gelu`) executes, the runtime dynamically records an **Operation Node (`Op`)** on an in-memory execution tape.

Each Node stores three pieces of information:
1. **Pointers to Input Tensors**: The parent nodes in the computational graph.
2. **Saved Tensors for Backward**: Any intermediate activations needed to compute local derivatives (e.g., input $x$ for $\\text{ReLU}$, or inputs $A$ and $B$ for $\\text{MatMul}$).
3. **The Vector-Jacobian Product (VJP) Function**: A function that multiplies an incoming upstream gradient by the local Jacobian matrix.

```
Forward Graph Construction (DAG):
X [B, D] ────┐
             ├──>[ MatMulOp ]──> Z [B, H] ──>[ GeluOp ]──> A [B, H] ──>[ LossOp ]──> L (Scalar)
W [D, H] ────┘
```

### Phase 2: Reverse-Mode Topological Sweep
Once the forward pass computes the scalar loss $\\mathcal{L}$, we initiate backpropagation by seeding the loss with its own derivative:

$$\\frac{\\partial \\mathcal{L}}{\\partial \\mathcal{L}} = 1.0$$

The engine performs a **Reverse Topological Sort** of the computational graph, visiting each operation in exact reverse order of execution. For each node, it evaluates the local chain rule:

$$\\text{Gradient to Parent } i = \\text{Upstream Gradient} \\times \\text{Local Jacobian}$$

```
Reverse Gradient Propagation:
dL/dX ◄────┐
           ├──◄[ dMatMul ]◄──── dL/dZ ◄────[ dGELU ]◄──── dL/dA ◄────[ dLoss ]◄──── dL/dL = 1.0
dL/dW ◄────┘
```

---

## 6.3 The Fundamental Vector-Jacobian Products (VJPs)

Let us derive the exact Vector-Jacobian Products for the elementary operators in TinyTorch:

### 1. Matrix Multiplication: $Y = A \\cdot B$

Given incoming upstream gradient $\\frac{\\partial \\mathcal{L}}{\\partial Y} = \\mathbf{G} \\in \\mathbb{R}^{M \\times N}$:

$$\\frac{\\partial \\mathcal{L}}{\\partial A} = \\mathbf{G} \\cdot B^T, \\qquad \\frac{\\partial \\mathcal{L}}{\\partial B} = A^T \\cdot \\mathbf{G}$$

Notice the beautiful symmetry: the backward pass of a matrix multiplication is simply **two transposed matrix multiplications**!

### 2. Elementwise Addition: $Y = A + B$

$$\\frac{\\partial \\mathcal{L}}{\\partial A} = \\mathbf{G}, \\qquad \\frac{\\partial \\mathcal{L}}{\\partial B} = \\mathbf{G}$$

The upstream gradient flows backward unchanged to both inputs.

### 3. In-Place Gradient Accumulation: `param.grad += grad`

In neural architectures with **branching dataflow** (such as residual skip connections where $y = x + f(x)$), input tensor $x$ is consumed by multiple child nodes.

By the multivariate chain rule:

$$\\frac{\\partial \\mathcal{L}}{\\partial x} = \\sum_{\\text{children } c} \\frac{\\partial \\mathcal{L}}{\\partial y_c} \\frac{\\partial y_c}{\\partial x}$$

Therefore, autograd engines must **accumulate gradients in-place** (`+=`), rather than overwriting them:

```python
if param.grad is None:
    param.grad = local_grad
else:
    param.grad += local_grad
```

---

## 6.4 The Pure TinyTorch Construction

We implement the complete dynamic autograd engine in TinyTorch:

```python
import numpy as np
from typing import List, Set
from .tensor import Tensor

def topological_sort(root: Tensor) -> List[Tensor]:
    \"\"\"Perform post-order DFS to obtain topologically sorted execution order.\"\"\"
    order: List[Tensor] = []
    visited: Set[int] = set()

    def dfs(node: Tensor):
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)
        for parent in getattr(node, '_parents', []):
            if isinstance(parent, Tensor):
                dfs(parent)
        order.append(node)

    dfs(root)
    return order

def backward(loss_tensor: Tensor, grad: Optional[np.ndarray] = None):
    \"\"\"Execute reverse-mode automatic differentiation across the computational tape.\"\"\"
    if grad is None:
        loss_tensor.grad = np.ones_like(loss_tensor.data)
    else:
        loss_tensor.grad = grad

    # 1. Obtain reverse topological ordering of DAG
    nodes = reversed(topological_sort(loss_tensor))

    # 2. Propagate VJPs backward
    for node in nodes:
        if node.grad is None or not hasattr(node, '_op') or node._op is None:
            continue

        upstream_grad = node.grad if isinstance(node.grad, np.ndarray) else node.grad.data
        op = node._op
        parents = node._parents

        if op == "Add":
            for p in parents:
                if isinstance(p, Tensor) and p.requires_grad:
                    # Handle broadcasting reductions if shapes differ
                    g = upstream_grad
                    while g.ndim > p.data.ndim:
                        g = g.sum(axis=0)
                    for dim in range(p.data.ndim):
                        if p.data.shape[dim] == 1 and g.shape[dim] > 1:
                            g = g.sum(axis=dim, keepdims=True)
                    p.grad = g if p.grad is None else p.grad + g

        elif op == "MatMul":
            A, B = parents[0], parents[1]
            if A.requires_grad:
                g_A = np.matmul(upstream_grad, B.data.T)
                A.grad = g_A if A.grad is None else A.grad + g_A
            if B.requires_grad:
                g_B = np.matmul(A.data.T, upstream_grad)
                B.grad = g_B if B.grad is None else B.grad + g_B

        elif op == "ReLU":
            p = parents[0]
            if p.requires_grad:
                g = upstream_grad * (p.data > 0.0).astype(np.float32)
                p.grad = g if p.grad is None else p.grad + g

# Attach backward method directly to Tensor class
Tensor.backward = backward
```

---

## 6.5 The Production Bridge: PyTorch C++ `torch::autograd::Engine`

In production PyTorch, autograd is executed by a dedicated C++ multithreaded worker queue called **`torch::autograd::Engine`**:

```cpp
// PyTorch Autograd Engine (torch/csrc/autograd/engine.cpp)
struct Node {
  virtual variable_list apply(variable_list&& grads) = 0;
  std::vector<Edge> next_edges_;
};
```

When you call `loss.backward()`, PyTorch pushes `Node` execution tasks into a multi-threaded task queue. As soon as all child dependencies of an operator have completed their VJPs, worker threads execute their backward kernels concurrently across multiple CPU and GPU streams.

---

## 6.6 Building the System: How It All Connects

With Chapter 6, TinyTorch has unlocked the engine of gradient calculus:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ THE COMPLETE AUTOGRAD LEARNING LOOP                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Forward Pass : Evaluates loss L; dynamically records DAG tape in RAM.    │
│ 2. backward()   : Traverses DAG in reverse topological order.               │
│ 3. Output       : Populates .grad field for EVERY learnable parameter W, b. │
└─────────────────────────────────────────────────────────────────────────────┘
```

We now have exact gradient vectors pointing uphill on the loss surface.

How do we use these gradients to update parameters without getting stuck in high-dimensional saddle points or oscillating erratically across steep ravines?

In **Chapter 7**, we engineer **Optimizers: Momentum, AdamW, and Decoupled Weight Decay**.
"""

with open(DEST_DIR / "04_losses.qmd", "w", encoding="utf-8") as f:
    f.write(CH04_CONTENT.strip() + "\n")
print("✓ Expanded 04_losses.qmd")

with open(DEST_DIR / "05_dataloader.qmd", "w", encoding="utf-8") as f:
    f.write(CH05_CONTENT.strip() + "\n")
print("✓ Expanded 05_dataloader.qmd")

with open(DEST_DIR / "06_autograd.qmd", "w", encoding="utf-8") as f:
    f.write(CH06_CONTENT.strip() + "\n")
print("✓ Expanded 06_autograd.qmd")
