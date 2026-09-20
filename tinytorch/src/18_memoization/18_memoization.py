# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
r"""
# Module 18: Memoization - Computational Reuse for Inference

Welcome to Module 18! In this module, we transition from silicon execution primitives to algorithmic state caching: eliminating redundant matrix operations during autoregressive token generation via pre-allocated Key-Value (KV) cache buffers.

## 🔗 Prerequisites & Progress

**You've Built**: A complete transformer stack with multi-head attention (`12_attention`) and an autoregressive `GPT` (`13_transformers`), plus the measurement and optimization tier of profiling (`14_profiling`), quantization (`15_quantization`), compression (`16_compression`), and kernel acceleration (`17_acceleration`).
**You'll Build**: A pre-allocated `KVCache` with pointer-advanced writes, a non-invasive `CachedAttention` stand-in, and a two-phase cached generation loop.
**You'll Enable**: Token-by-token decoding whose per-step projection cost no longer grows with the conversation, which is the optimization that makes interactive language model serving affordable.

<div align="center">
  <img src="memoization_blueprint.svg" width="380px" alt="TinyTorch Framework Blueprint: Module 18 Memoization" />
</div>

### Architectural Roadmap

| Optimization Stage | Core Technique | Hardware & Algorithmic Focus | Primary Target |
|:---|:---|:---|:---|
| **14. Profiling** | Microsecond Benchmarks & Tracing | Profiler timer loops, Roofline bounds | Identify compute vs memory bottlenecks |
| **15. Quantization** | Symmetric/Asymmetric INT8 | 8-bit scale & zero-point arithmetic | 4× weight footprint & memory bus bandwidth |
| **16. Compression** | Magnitude Pruning & Distillation | Weight sparsity & student distillation | Redundant parameter elimination |
| **17. Acceleration** | SIMD GEMM, Fusion, `im2col` | Memory traffic elimination & systolic arrays | Kernel overhead & hardware utilization |
| **18. Memoization** *(Active)* | **Static KV Cache Buffers** | **$\mathcal{O}(1)$ projections per decode step & zero recomputation** | **Autoregressive decoding latency** |
| **19–20. Serving & Capstone** | End-to-End Pipeline Integration | End-to-end throughput & TTFT/ITL serving | Production inference deployment |

## 🎯 Learning Objectives
By the end of this module, you will:
1. Formulate memoization as a general systems optimization pattern that trades persistent DRAM memory capacity for $\mathcal{O}(S)$ computational reuse.
2. Mathematically derive the $\mathcal{O}(S^2) \to \mathcal{O}(S)$ reduction in key and value projection operations during autoregressive generation.
3. Construct a production-grade `KVCache` class featuring contiguous buffer pre-allocation, pointer advancement, and zero dynamic heap reallocation.
4. Implement a non-invasive `CachedAttention` wrapper that preserves forward compatibility and clean model encapsulation.
5. Budget the cache's memory footprint across model configurations and measure the wall-clock speedup it delivers as generation length grows.

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/18_memoization/memoization.ipynb`  
**Building Side:** Code exports to `tinytorch.perf.memoization`

<div align="center">
  <img src="memoization_source_card.svg" width="260px" alt="Source Code Mapping Card for Module 18 Memoization" />
</div>

```python
# How to use this module:
from tinytorch.perf.memoization import KVCache, enable_kv_cache, disable_kv_cache
```

## 📋 Module Dependencies

| Dependency Module | Exported Abstraction | Consumed Functional Role | Memory & Architectural Invariant |
|:---|:---|:---|:---|
| **Module 01 (`01_tensor`)** | `Tensor` | Contiguous N-D array storage & slicing | Pre-allocated float32 buffers without autograd overhead |
| **Module 12 (`12_attention`)** | `MultiHeadAttention`, `MASK_VALUE` | Attention projections, head transformations, masking sentinel | Splits `Q, K, V` into `(B, H, S, D)`, recombines output, and reuses one masking constant |
| **Module 13 (`13_transformers`)** | `GPT` | Autoregressive language model backbone | Blocks are reached by duck typing through `model.blocks`, so `TransformerBlock` is never imported or subclassed |
| **Module 14 (`14_profiling`)** | `Profiler` | High-resolution microsecond timer | Quantifies latency scaling with and without cache |
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": false}
#| default_exp perf.memoization
#| export

import numpy as np
rng = np.random.default_rng(7)
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from contextlib import contextmanager

# Import TinyTorch components from previous modules
from tinytorch.core.tensor import Tensor
from tinytorch.core.attention import MASK_VALUE

# Internal constants for memory calculations (not exported)
_BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
_BYTES_PER_MB = 1_000_000  # Decimal megabyte: this module quotes MB and GB as 10^6 and 10^9 bytes
_MB_TO_BYTES = 1024 * 1024  # Binary MiB, kept for KVCache.get_memory_usage's reported figure

# %% [markdown]
r"""
## 💡 Introduction: Why Memoization Matters for Transformers

Before implementing KV caching, let us profile naive autoregressive generation to isolate the fundamental computational bottleneck of modern transformer inference.

<div align="center">
  <img src="memoization_generation_overview.svg" width="680px" alt="Autoregressive KV Cache Generation Overview" />
</div>

### Computational Paradigm Comparison

| Execution Attribute | Naive Decoding (No Cache) | Memoized Decoding (`KVCache`) | Systems Rationale |
|:---|:---|:---|:---|
| **Key & Value Projections** | Recomputed for all $t$ past tokens | Computed **only** for single incoming token $t$ | Previous token representations never change |
| **Projection Complexity** | $\sum_{t=1}^S t = \frac{S(S+1)}{2} \in \mathcal{O}(S^2)$ | $\sum_{t=1}^S 1 = S \in \mathcal{O}(S)$ | $(S+1)/2$ factor reduction in projection FLOPs |
| **Memory Allocation** | Dynamic allocations per sequence step | Single static pre-allocation up front | Eliminates heap fragmentation & malloc stalls |
| **Attention Query Shape** | Full sequence $(B, H, S, d_k)$ | Single token vector $(B, H, 1, d_k)$ | Query vector attends over cached key prefix |
| **Causal Mask Overhead** | Requires lower-triangular causal mask | **Zero** causal mask needed during decode | Single query can strictly attend to past tokens |

**Key Architectural Insight**: In autoregressive generation, keys $\mathbf{K}$ and values $\mathbf{V}$ for past tokens $1 \dots t-1$ are strictly static and immutable. Naively re-projecting historical tokens wastes memory bandwidth and processor ALUs on dead computation.
"""

# %% nbgrader={"grade": false, "grade_id": "motivation-profile", "solution": false}
def profile_naive_generation():
    """
    Profile transformer generation to discover the O(n²) bottleneck.

    Educational Purpose:
        Demonstrates why KV caching is necessary by showing concrete
        measurements of quadratic growth in generation time.

    This function runs ONLY when the module is executed directly,
    not when imported (avoiding side effects during imports).
    """
    from tinytorch.perf.profiling import Profiler

    profiler = Profiler()

    class NaiveAttentionStep:
        """One generation step without a cache: attention over the whole sequence."""

        def forward(self, x):
            # Without caching, every step recomputes K and V for ALL tokens so far
            # (projections omitted: the O(seq_len²) part is the score matrix)
            q, k, v = x.data, x.data, x.data
            scores = np.matmul(q, np.transpose(k, (0, 2, 1)))  # (1, seq_len, seq_len)
            return Tensor(np.matmul(scores, v))

    step = NaiveAttentionStep()

    # Profile at increasing sequence lengths
    print("🧪 Profiling Transformer Generation (Without Caching):\n")
    print("   Seq Len  |  Latency (ms)  |  Growth")
    print("   ---------|----------------|----------")

    sequence_lengths = [64, 128, 256, 512, 1024]
    latencies = []

    for seq_len in sequence_lengths:
        x = Tensor(rng.standard_normal((1, seq_len, 64)))
        latency = profiler.measure_latency(step, x, warmup=5, iterations=20)
        latencies.append(latency)

        # Calculate growth rate
        if len(latencies) > 1:
            growth = latencies[-1] / latencies[-2]
            print(f"   {seq_len:5d}    |  {latency:6.2f}        |  {growth:.2f}×")
        else:
            print(f"   {seq_len:5d}    |  {latency:6.2f}        |  baseline")

    print("\n💡 Key Observations:")
    print("   • Latency grows QUADRATICALLY with sequence length")
    print("   • Each new token forces recomputation of ALL previous K,V pairs")
    print("   • Doubling the sequence MORE than quadruples the time (5-6× measured above),")
    print("     because the S×S score matrix also has to be allocated and streamed")

    print("\n🎯 The Problem:")
    print("   K and V values for previous tokens NEVER change,")
    print("   yet we recompute them every single step!")

    print("\n✨ The Solution:")
    print("   CACHE the K,V values! (That's memoization)")
    print("   • First compute: Calculate and store K,V")
    print("   • Later steps: Reuse stored K,V")
    print("   • Complexity: O(n²) → O(n)")
    print("   • K,V projection ratio: (n+1)/2 for n generated tokens (≈50× at n=100)\n")

if __name__ == "__main__":
    profile_naive_generation()

# %% [markdown]
r"""
## 📐 Foundations: Understanding the Autoregressive Generation Problem

### The Core Inefficiency

When generating text token by token, causal transformers face a fundamental computational bottleneck. Let us analyze what occurs during naive generation without memoization:

At autoregressive decoding step $t$, the model processes an incoming prompt or previous generated token $\mathbf{x}_t \in \mathbb{R}^{1 \times d_{\text{model}}}$. Without caching, the model must re-feed the entire historical context $\mathbf{X}_{1:t} = [\mathbf{x}_1; \mathbf{x}_2; \dots; \mathbf{x}_t] \in \mathbb{R}^{t \times d_{\text{model}}}$ through all transformer layers:

$$\mathbf{Q}_{1:t} = \mathbf{X}_{1:t} \mathbf{W}_Q, \quad \mathbf{K}_{1:t} = \mathbf{X}_{1:t} \mathbf{W}_K, \quad \mathbf{V}_{1:t} = \mathbf{X}_{1:t} \mathbf{W}_V$$

$$\text{Attention}(\mathbf{Q}_{1:t}, \mathbf{K}_{1:t}, \mathbf{V}_{1:t}) = \text{softmax}\left(\frac{\mathbf{Q}_{1:t} \mathbf{K}_{1:t}^T}{\sqrt{d_k}} + \mathbf{M}_{\text{causal}}\right) \mathbf{V}_{1:t}$$

Notice that the $i$-th row of $\mathbf{K}_{1:t}$ and $\mathbf{V}_{1:t}$ (for any prior step $i < t$) is mathematically identical to what was computed at step $i$:

$$\mathbf{k}_i = \mathbf{x}_i \mathbf{W}_K \in \mathbb{R}^{1 \times d_k}, \quad \mathbf{v}_i = \mathbf{x}_i \mathbf{W}_V \in \mathbb{R}^{1 \times d_v}$$

Because weight matrices $\mathbf{W}_K, \mathbf{W}_V$ and token representations $\mathbf{x}_i$ are static during inference, recomputing historical keys and values at step $t$ performs 100% redundant matrix multiplications.

### Computational Complexity Analysis

Let $S$ denote the total sequence length. Across $S$ generation steps, the cumulative number of key and value vector projections computed under naive execution is:

$$\mathcal{N}_{\text{naive}}(S) = \sum_{t=1}^S t = \frac{S(S+1)}{2} \approx \frac{1}{2} S^2 \in \mathcal{O}(S^2)$$

For a sequence of $S = 100$ tokens:
$$\mathcal{N}_{\text{naive}}(100) = \frac{100 \times 101}{2} = 5,050 \text{ projections}$$

Only $100$ projections are mathematically necessary (one per token). The remaining $4,950$ projections ($98.02\%$ of all projection FLOPs) are redundant recomputation!

### The Key-Value Caching Insight

By caching $\mathbf{K}_{1:t-1}$ and $\mathbf{V}_{1:t-1}$ from prior steps in high-speed GPU/CPU memory, step $t$ only needs to project the single newest token:

$$\mathbf{q}_t = \mathbf{x}_t \mathbf{W}_Q \in \mathbb{R}^{1 \times d_k}, \quad \mathbf{k}_t = \mathbf{x}_t \mathbf{W}_K \in \mathbb{R}^{1 \times d_k}, \quad \mathbf{v}_t = \mathbf{x}_t \mathbf{W}_V \in \mathbb{R}^{1 \times d_v}$$

The new key $\mathbf{k}_t$ and value $\mathbf{v}_t$ are appended in-place to pre-allocated buffers:

$$\mathbf{K}_{1:t} = \begin{bmatrix} \mathbf{K}_{\text{cached}} \\ \mathbf{k}_t \end{bmatrix} \in \mathbb{R}^{t \times d_k}, \quad \mathbf{V}_{1:t} = \begin{bmatrix} \mathbf{V}_{\text{cached}} \\ \mathbf{v}_t \end{bmatrix} \in \mathbb{R}^{t \times d_v}$$

Attention scoring simplifies from matrix-matrix multiplication to vector-matrix multiplication:

$$\mathbf{a}_t = \text{softmax}\left(\frac{\mathbf{q}_t \mathbf{K}_{1:t}^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{1 \times t}$$

$$\mathbf{o}_t = \mathbf{a}_t \mathbf{V}_{1:t} \in \mathbb{R}^{1 \times d_v}$$

The total projection complexity collapses to:

$$\mathcal{N}_{\text{cached}}(S) = \sum_{t=1}^S 1 = S \in \mathcal{O}(S)$$

### Memory vs Compute Trade-Off Analysis

| Execution Model | Persistent KV Storage | Projection FLOPs ($S$ tokens) | Attention Score FLOPs ($S$ tokens) | Recomputation Redundancy |
|:---|:---|:---|:---|:---|
| **Naive (No Cache)** | $\mathbf{0} \text{ bytes}$ (stateless) | $\frac{S(S+1)}{2} \cdot 4 d_{\text{model}} d_k \in \mathcal{O}(S^2)$ | $\sum_{t=1}^S t^2 \approx \frac{1}{3} S^3 \in \mathcal{O}(S^3)$ | $1 - \frac{2}{S+1} \approx 98\%$ wasted FLOPs |
| **Cached (`KVCache`)** | $2 \cdot L \cdot B \cdot H \cdot S_{\max} \cdot d_k \cdot 4 \text{ B}$ | $S \cdot 4 d_{\text{model}} d_k \in \mathcal{O}(S)$ | $\sum_{t=1}^S t = \frac{S(S+1)}{2} \in \mathcal{O}(S^2)$ | **0% redundant projection FLOPs** |

### Why a Decode Step Is Bound by Bytes, Not FLOPs

One more fact decides how much the cache is actually worth. A decode step emits a single token, and to emit it the processor must read the *entire* weight set out of DRAM: every projection matrix and every MLP matrix, in every layer. That is hundreds of megabytes of traffic against a few megaflops of arithmetic, so the step finishes when the bytes arrive, not when the ALUs are done. Decoding is memory-bandwidth-bound, while the prefill of a long prompt, which reuses each loaded weight across many tokens, is compute-bound.

The cache therefore removes redundant *bytes moved* as much as redundant FLOPs, and that is why its measured payoff is far smaller than its operation counts. The 📊 analysis at the end of this module makes the gap visible: the attention score-pair ratio climbs from $8.8\times$ to roughly $69\times$ while the measured wall-clock speedup only moves from $1.2\times$ to a little over $2\times$. Nothing is wrong with either number. The ratio counts arithmetic the cache eliminates; the clock also pays for weight reads, the MLP, prefix copies, and Python interpreter overhead, none of which the cache touches.

**Systems Takeaway**: DRAM *capacity* is the cheap resource; DRAM *bandwidth* is the scarce one and DRAM *latency* is the wall. Trading $\mathcal{O}(S)$ persistent cache capacity for a large reduction in redundant bytes and FLOPs per token is the standard architecture of LLM serving.
"""

# %% [markdown]
r"""
## 🏗️ Implementation: KVCache Class

### Core Architectural Requirements

To serve high-performance inference pipelines, `KVCache` must satisfy five systems invariants:
1. **Multi-Layer Isolation**: Independent contiguous key and value buffers per transformer layer $l \in [0, L-1]$.
2. **Multi-Head Layout**: Dedicated dimensions for batch, heads, sequence length, and head dimension $(B, H, S_{\max}, d_k)$.
3. **Static Pre-Allocation**: Contiguous allocation of the full context window $S_{\max}$ up front to prevent dynamic memory allocation and heap fragmentation.
4. **$\mathcal{O}(1)$ Update Semantics**: In-place indexed assignment at write cursor `seq_pos` without array recreation.
5. **Prefix Retrieval**: Exposing the valid token history $\mathbf{K}[:, :, :t, :]$ for immediate vector-matrix GEMV attention. The NumPy slice itself is a view, but `get()` wraps it in a `Tensor`, which copies, so retrieval costs $\mathcal{O}(t)$ bytes per call rather than being free. A production serving stack avoids that copy by handing the kernel the buffer and a length; TinyTorch pays it to keep the `Tensor` API uniform.

<div align="center">
  <img src="kv_cache_state_machine.svg" width="680px" alt="KV Cache Buffer State Machine" />
</div>

<div align="center">
  <img src="kv_cache_buffer_card.svg" width="320px" alt="KV Cache Buffer Card" />
</div>

### Buffer Dimension Specification

| Cache Tensor | Dimension Order | Shape | Element Type | Role in Attention GEMV |
|:---|:---|:---|:---|:---|
| **Key Cache ($\mathbf{K}$)** | `(batch, heads, seq, dim)` | $(B, H, S_{\max}, d_k)$ | `float32` (4 bytes) | Multiplied by query vector $\mathbf{q}_t$ to compute attention scores |
| **Value Cache ($\mathbf{V}$)** | `(batch, heads, seq, dim)` | $(B, H, S_{\max}, d_v)$ | `float32` (4 bytes) | Linearly combined by attention probabilities $\mathbf{a}_t$ |

### Buffer Write & Retrieval Progression

| Operational Step | Buffer Action | Tensor Slice Expression | Systems Mechanism |
|:---|:---|:---|:---|
| **Initialization** | Pre-allocate zeroed buffers | `zeros((B, H, S_max, D))` | Contiguous single malloc per layer |
| **Incoming Token $t$** | In-place slot write | `cache[:, :, seq_pos:seq_pos+1, :] = new_token` | $\mathcal{O}(1)$ strided memory copy into DRAM |
| **Attention Query** | Slice active prefix | `cache[:, :, :valid_len, :]` | Contiguous memory read up to cursor |
| **Pointer Step** | Advance write cursor | `self.seq_pos += 1` | Zero cost integer increment |
"""

# %% nbgrader={"grade": false, "grade_id": "kvcache-class", "solution": true}
#| export
class KVCache:
    """
    Efficient key-value cache for autoregressive generation.

    Stores K,V matrices for each transformer layer to avoid recomputation
    during sequential token generation. This is THE critical optimization
    that makes production language model serving economically viable.

    ⚠️  IMPORTANT: INFERENCE-ONLY (No Gradient Tracking)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    KV caching is designed ONLY for inference (generation), NOT training.
    - During generation: Cache operations detach values; eval() alone does not disable gradients
    - Cache operations use .data (no gradient tracking)
    - This is correct and intentional for maximum speed
    - DO NOT use caching during training (use standard forward pass)

    Architecture:
    - Pre-allocates cache tensors with maximum sequence length
    - Tracks current sequence position for efficient O(1) updates
    - Provides update() method to append new K,V pairs without reallocating storage
    - Provides get() method to retrieve cached values for attention
    - Handles multiple layers and attention heads properly

    Memory Layout:
    ```
    Layer 0: [Key_cache, Value_cache]  # Shape: (batch, num_heads, max_seq, head_dim)
    Layer 1: [Key_cache, Value_cache]
    ...
    Layer N: [Key_cache, Value_cache]
    ```

    Performance:
    - Update: constant in context length; copies one token across all heads
    - Get: O(context length) - Tensor construction copies the sliced prefix
    - Memory: O(num_layers × batch × heads × max_seq × head_dim)
    """

    def __init__(self, batch_size: int, max_seq_len: int, num_layers: int,
                 num_heads: int, head_dim: int):
        """
        Initialize KV cache for efficient generation.

        TODO: Set up pre-allocated cache storage for all transformer layers

        APPROACH:
        1. Store configuration parameters (batch_size, max_seq_len, etc.)
        2. Initialize sequence position counter to 0
        3. Create empty list for cache storage
        4. For each layer, pre-allocate zero-filled key and value caches
        5. Store each layer's (key_cache, value_cache) tuple in the list

        Args:
            batch_size: Number of sequences to generate simultaneously
            max_seq_len: Maximum sequence length to support
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            head_dim: Dimension of each attention head

        EXAMPLE:
        >>> cache = KVCache(batch_size=2, max_seq_len=128, num_layers=4,
        ...                 num_heads=8, head_dim=64)
        >>> cache.seq_pos  # 0 (no tokens cached yet)
        >>> len(cache.caches)  # 4 (one per layer)
        >>> cache.caches[0][0].shape  # (2, 8, 128, 64) - key cache for layer 0

        HINTS:
        - Cache shape: (batch_size, num_heads, max_seq_len, head_dim)
        - Use Tensor(np.zeros(...)) to create cache tensors
        - Store caches as list of tuples: [(key_0, val_0), (key_1, val_1), ...]
        - Pre-allocation avoids dynamic resizing overhead during generation
        """
        ### BEGIN SOLUTION role="scaffold"
        for name, value in (("batch_size", batch_size), ("max_seq_len", max_seq_len),
                            ("num_layers", num_layers), ("num_heads", num_heads),
                            ("head_dim", head_dim)):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Current sequence position (how many tokens are cached)
        self.seq_pos = 0
        self._generation_active = False

        # Cache storage: list of (key_cache, value_cache) tuples per layer
        self.caches = []

        for layer_idx in range(num_layers):
            # Pre-allocate cache tensors with maximum size
            # Shape: (batch_size, num_heads, max_seq_len, head_dim)
            key_cache = Tensor(np.zeros((batch_size, num_heads, max_seq_len, head_dim)))
            value_cache = Tensor(np.zeros((batch_size, num_heads, max_seq_len, head_dim)))

            self.caches.append((key_cache, value_cache))
        ### END SOLUTION

    def update(self, layer_idx: int, key: Tensor, value: Tensor) -> None:
        """
        Update cache with new key-value pairs for given layer.

        TODO: Efficiently append new K,V to cache without reallocating the cache

        APPROACH:
        1. Validate layer_idx is in range [0, num_layers-1]
        2. Validate seq_pos hasn't exceeded max_seq_len
        3. Retrieve the (key_cache, value_cache) tuple for this layer
        4. Write new key to position seq_pos in key_cache using indexed assignment
        5. Write new value to position seq_pos in value_cache using indexed assignment
        6. Note: seq_pos is advanced externally via advance() after all layers

        This is the core caching operation - efficiently append new K,V
        to the cache without recomputation. This operation is O(1) because
        it's just an indexed assignment.

        IMPORTANT: KV caching is designed for INFERENCE (generation) only,
        not training. During generation, gradients are not computed. If you
        need gradients, don't use caching (use standard forward pass instead).

        Args:
            layer_idx: Which transformer layer (0 to num_layers-1)
            key: New key tensor, shape (batch_size, num_heads, 1, head_dim)
            value: New value tensor, shape (batch_size, num_heads, 1, head_dim)

        EXAMPLE:
        >>> cache = KVCache(batch_size=1, max_seq_len=10, num_layers=2,
        ...                 num_heads=4, head_dim=64)
        >>> new_k = Tensor(rng.standard_normal((1, 4, 1, 64)))
        >>> new_v = Tensor(rng.standard_normal((1, 4, 1, 64)))
        >>> cache.update(layer_idx=0, key=new_k, value=new_v)
        >>> cache.seq_pos  # Still 0 (update doesn't advance position)
        >>> cache.advance()
        >>> cache.seq_pos  # Now 1

        HINTS:
        - Use slicing: cache[:, :, seq_pos:seq_pos+1, :] to write to position
        - Use .data for direct NumPy access (no gradient tracking needed)
        - Raise ValueError with helpful messages for invalid inputs
        - This is an in-place operation (modifies cache, returns None)

        Raises:
            ValueError: If layer_idx is out of range or sequence is full
        """
        ### BEGIN SOLUTION
        if not isinstance(layer_idx, (int, np.integer)) or not 0 <= layer_idx < self.num_layers:
            raise ValueError(
                f"Invalid layer index for cache update\n"
                f"  ❌ layer_idx={layer_idx} is out of range [0, {self.num_layers - 1}]\n"
                f"  💡 KVCache was initialized with num_layers={self.num_layers}, so valid indices are 0 to {self.num_layers - 1}\n"
                f"  🔧 Check your transformer block loop: for layer_idx in range({self.num_layers})"
            )

        if self.seq_pos >= self.max_seq_len:
            raise ValueError(
                f"KV cache is full - cannot add more tokens\n"
                f"  ❌ Current position {self.seq_pos} has reached max_seq_len={self.max_seq_len}\n"
                f"  💡 The cache was pre-allocated for {self.max_seq_len} tokens maximum. Autoregressive generation cannot exceed this limit.\n"
                f"  🔧 Either: (1) call cache.reset() to start a new sequence, or (2) create a larger cache with max_seq_len > {self.max_seq_len}"
            )

        expected_shape = (self.batch_size, self.num_heads, 1, self.head_dim)
        if key.shape != expected_shape or value.shape != expected_shape:
            raise ValueError(f"Cache update requires key and value shape {expected_shape}")

        # Get cache for this layer
        key_cache, value_cache = self.caches[layer_idx]

        # Update cache at current position (efficient O(1) write)
        # Note: We use .data here because caching is inference-only (no gradients needed)
        # This avoids gradient tracking overhead during generation
        key_cache.data[:, :, self.seq_pos:self.seq_pos+1, :] = key.data
        value_cache.data[:, :, self.seq_pos:self.seq_pos+1, :] = value.data

        # Note: seq_pos is advanced externally via advance() after all layers process
        ### END SOLUTION

    def get(self, layer_idx: int) -> Tuple[Tensor, Tensor]:
        """
        Retrieve cached key-value pairs for attention computation.

        TODO: Return only the valid cached portion for this layer

        APPROACH:
        1. Validate layer_idx is in range
        2. Retrieve the (key_cache, value_cache) tuple for this layer
        3. Calculate valid_len = seq_pos (number of tokens currently cached)
        4. Slice key_cache to get [:, :, :valid_len, :] (only filled portion)
        5. Slice value_cache to get [:, :, :valid_len, :] (only filled portion)
        6. Wrap sliced data in new Tensor objects and return

        Returns only the valid portion of the cache (up to current seq_pos).
        The NumPy slice is a view, but wrapping it in Tensor copies the prefix.
        Retrieval therefore grows with the number of cached tokens.

        IMPORTANT: Returns Tensors without gradient tracking since caching
        is inference-only. The returned tensors can be used in attention
        computation but won't propagate gradients backward.

        Args:
            layer_idx: Which transformer layer to get cache for

        Returns:
            (cached_keys, cached_values): Tensors shaped for attention
            Keys: (batch_size, num_heads, seq_pos, head_dim)
            Values: (batch_size, num_heads, seq_pos, head_dim)

        EXAMPLE:
        >>> cache = KVCache(batch_size=1, max_seq_len=100, num_layers=2,
        ...                 num_heads=4, head_dim=64)
        >>> # After processing 3 tokens
        >>> cache.seq_pos = 3
        >>> cached_k, cached_v = cache.get(layer_idx=0)
        >>> cached_k.shape  # (1, 4, 3, 64) - only first 3 positions
        >>> cached_v.shape  # (1, 4, 3, 64)

        HINTS:
        - valid_len = self.seq_pos (how many tokens have been cached so far)
        - Use slicing: cache.data[:, :, :valid_len, :] to get valid portion
        - Wrap result in Tensor() for consistency with TinyTorch API
        - If seq_pos=0, returns empty cache (shape with 0 in sequence dimension)

        Raises:
            ValueError: If layer_idx is out of range
        """
        ### BEGIN SOLUTION
        if not isinstance(layer_idx, (int, np.integer)) or not 0 <= layer_idx < self.num_layers:
            raise ValueError(
                f"Invalid layer index for cache retrieval\n"
                f"  ❌ layer_idx={layer_idx} is out of range [0, {self.num_layers - 1}]\n"
                f"  💡 KVCache was initialized with num_layers={self.num_layers}, so valid indices are 0 to {self.num_layers - 1}\n"
                f"  🔧 Check your transformer block loop: for layer_idx in range({self.num_layers})"
            )

        # Get cache for this layer
        key_cache, value_cache = self.caches[layer_idx]

        # Return only the valid portion (up to current sequence position)
        # seq_pos tracks where to write next, so we have seq_pos valid tokens
        valid_len = self.seq_pos

        # Note: Creating new Tensors from .data (no gradient tracking)
        # This is correct for inference-only caching
        cached_keys = Tensor(key_cache.data[:, :, :valid_len, :])
        cached_values = Tensor(value_cache.data[:, :, :valid_len, :])

        return cached_keys, cached_values
        ### END SOLUTION

    @contextmanager
    def generation(self) -> Iterator["KVCache"]:
        """Use cached attention only inside this inference scope.

        Reset before a new sequence and advance after each token's layers.
        Ordinary model forwards outside the scope never consume cached history.
        Nested scopes restore the previous state, including after exceptions.
        """
        previous = self._generation_active
        self._generation_active = True
        try:
            yield self
        finally:
            self._generation_active = previous

    def advance(self) -> None:
        """
        Advance sequence position after processing current token.

        Call this after all layers have processed the current token and
        updated their caches. This moves the write pointer forward.
        """
        if self.seq_pos >= self.max_seq_len:
            raise ValueError("Cannot advance: KV cache is full")
        self.seq_pos += 1

    def reset(self) -> None:
        """
        Reset cache for new generation sequence.

        Call this when starting a new generation (new prompt).
        Resets the sequence position counter and optionally zeros cache data.
        """
        self.seq_pos = 0

        # Zero out caches for clean state (helps with debugging)
        for layer_idx in range(self.num_layers):
            key_cache, value_cache = self.caches[layer_idx]
            key_cache.data.fill(0.0)
            value_cache.data.fill(0.0)

    def get_memory_usage(self) -> Dict[str, Union[int, float]]:
        """
        Calculate memory usage of the cache system.

        Unit note: this method reports BINARY megabytes (MiB, 2^20 bytes), which
        is what the key name 'total_mb' has always held. Every table and every
        printed figure elsewhere in this module quotes DECIMAL MB and GB
        (10^6 and 10^9 bytes), so the same buffer reads about 4.8% smaller here.

        Returns:
            Dictionary with 'total_mb' and 'per_layer_mb' in MiB, plus the
            integer 'cache_tensors' and 'total_elements' counts
        """
        # Calculate size of one cache tensor
        cache_size = self.batch_size * self.num_heads * self.max_seq_len * self.head_dim

        # Each layer has key_cache + value_cache
        total_cache_tensors = self.num_layers * 2
        total_elements = cache_size * total_cache_tensors
        total_bytes = total_elements * _BYTES_PER_FLOAT32
        total_mb = total_bytes / _MB_TO_BYTES

        return {
            'total_mb': total_mb,
            'per_layer_mb': total_mb / self.num_layers,
            'cache_tensors': total_cache_tensors,
            'total_elements': total_elements
        }

# %% [markdown]
r"""
### 🧪 Unit Test: KVCache Implementation

This test validates that our cache correctly stores and retrieves key-value pairs across multiple layers and sequence positions.

**What we're testing**: KVCache initialization, update, get, and reset operations
**Why it matters**: Cache must work correctly for generation to produce coherent output
**Expected**: Cache stores and retrieves values correctly, tracks sequence position
"""

# %% nbgrader={"grade": true, "grade_id": "test-kvcache", "locked": true, "points": 10}
def test_unit_kvcache():
    """🧪 Unit Test: KVCache Implementation"""
    print("🧪 Unit Test: KVCache Implementation...")

    # Test parameters (small transformer for testing)
    batch_size, max_seq_len = 2, 8
    num_layers, num_heads, head_dim = 3, 4, 16

    # Create cache
    cache = KVCache(batch_size, max_seq_len, num_layers, num_heads, head_dim)

    # Test 1: Initial state
    assert cache.seq_pos == 0, "Cache should start at position 0"
    mem_usage = cache.get_memory_usage()
    assert mem_usage['total_mb'] > 0, "Cache should have non-zero memory usage"
    print(f"   Cache initialized: {mem_usage['total_mb']:.2f} MiB")

    # Test 2: Single token update and retrieval
    key1 = Tensor(rng.standard_normal((batch_size, num_heads, 1, head_dim)))
    value1 = Tensor(rng.standard_normal((batch_size, num_heads, 1, head_dim)))

    # Update layer 0 with first token
    cache.update(0, key1, value1)

    # Before advance, get() should return empty (seq_pos=0)
    cached_k, cached_v = cache.get(0)
    assert cached_k.shape == (batch_size, num_heads, 0, head_dim), "Before advance, cache should be empty"

    # Advance position
    cache.advance()

    # Now cache should have 1 token
    cached_k, cached_v = cache.get(0)
    assert cached_k.shape == (batch_size, num_heads, 1, head_dim), f"Expected shape (2,4,1,16), got {cached_k.shape}"
    assert cached_v.shape == (batch_size, num_heads, 1, head_dim), f"Expected shape (2,4,1,16), got {cached_v.shape}"

    # Test 3: Multi-token sequence
    key2 = Tensor(rng.standard_normal((batch_size, num_heads, 1, head_dim)))
    value2 = Tensor(rng.standard_normal((batch_size, num_heads, 1, head_dim)))
    cache.update(0, key2, value2)
    cache.advance()

    cached_k, cached_v = cache.get(0)
    assert cached_k.shape == (batch_size, num_heads, 2, head_dim), "Should have 2 tokens cached"
    assert cached_v.shape == (batch_size, num_heads, 2, head_dim), "Should have 2 tokens cached"

    # Test 4: Multiple layers
    cache.reset()
    key_test = Tensor(rng.standard_normal((batch_size, num_heads, 1, head_dim)))
    value_test = Tensor(rng.standard_normal((batch_size, num_heads, 1, head_dim)))

    # Update all layers with same token
    cache.update(0, key_test, value_test)  # Layer 0
    cache.update(1, key_test, value_test)  # Layer 1
    cache.update(2, key_test, value_test)  # Layer 2
    cache.advance()

    # Each layer should have the cached token
    for layer_idx in range(num_layers):
        cached_k, cached_v = cache.get(layer_idx)
        assert cached_k.shape[2] == 1, f"Layer {layer_idx} should have 1 token"

    # Test 5: Reset functionality
    cache.reset()
    assert cache.seq_pos == 0, "Reset should clear sequence position"
    cached_k, cached_v = cache.get(0)
    assert cached_k.shape == (batch_size, num_heads, 0, head_dim), "Reset should clear cache"

    print("✅ KVCache implementation works correctly!")

if __name__ == "__main__":
    test_unit_kvcache()

# %% [markdown]
r"""
## 🏗️ Cache-Aware Generation: Wrapping Attention Without Editing It

### Integration Strategy

Now we need a clean way to enable KV caching in our existing transformer models without editing Module 12 or Module 13. We'll write an `enable_kv_cache()` function that:

1. Creates a KVCache instance sized for the model
2. Puts a `CachedAttention` stand-in in front of each block's attention layer
3. Returns the cache for manual control if needed

The stand-in owns the original attention layer and decides, on every call, which of two paths to take:
1. A full sequence (training, or an ordinary forward pass) goes straight to the original attention
2. A single new token is projected to K,V once, written into the cache, and attended against everything cached so far

`disable_kv_cache()` removes the stand-ins and the model is exactly as it was. Nothing inside the model is patched: `block.attention` simply points at a different object for a while, the same wrapping pattern PyTorch uses for `DataParallel` and quantization stubs, which stand in front of a module and forward to it.

### Generation Flow Comparison

```
Without Cache (Current):
for each new token:
    input_seq = [all tokens so far]        # Length grows: 1, 2, 3, ...
    logits = model.forward(input_seq)       # Recomputes everything!
    next_token = sample(logits[-1])
    append next_token

With Cache (New):
cache = enable_kv_cache(model)
with cache.generation():
    for each prompt or generated token:
        input_token = [just new token]      # Length always 1
        logits = model.forward(input_token, start_pos=cache.seq_pos)
        cache.advance()                     # All layers have written their K,V
        next_token = sample(logits[-1])
```

**Key Difference**: Input changes from growing sequence to single token, with cache providing history.
"""

# %% [markdown]
r"""
## 🔧 Integration: Non-Invasive Model Enhancement

### The Challenge

We built KV caching in Module 18 (this module), but our transformer (Modules 12-13) doesn't know about it!

**❌ BAD Solution**: Go back and modify Module 12 (MultiHeadAttention)
- Breaks "forward-only" learning (students shouldn't revisit old modules)
- Makes Module 12 depend on Module 18 (wrong dependency direction!)
- Violates clean module boundaries

**✅ GOOD Solution**: Module 18 ADDS caching to existing models without modification!
- Use composition (wrap the model, keep its classes untouched)
- Module 18 wraps/enhances Module 12, not modifies it
- Students learn systems engineering: "Add capabilities, don't break old code"

### Using KV Cache in Practice

To use KV caching in your transformer generation:

**Before Generation:**
1. Enable caching with `enable_kv_cache(model)`
2. Cache is automatically sized for your model architecture
3. Verify memory usage is acceptable

**During Generation:**
1. Enter `with cache.generation():` (the `_cached_generate` helper does this for you)
2. Process prompt tokens **one at a time** at `start_pos=cache.seq_pos` so each one populates the cache. Prefill here is sequential; handing the whole prompt in as one multi-token forward silently bypasses the cache and leaves it empty
3. For subsequent tokens:
   - Only process the NEW token (not entire sequence)
   - Cache is automatically updated with new K,V pairs
   - Cached values are automatically used in attention
   - Cache position advances after all layers

**After Generation:**
1. Reset cache if generating another sequence: `model._kv_cache.reset()`
2. Disable caching if needed: `disable_kv_cache(model)`
3. Monitor memory usage for production deployment

### Theoretical Scaling: Projection Operations Avoided

| Generated Tokens ($S$) | Naive Projections ($\frac{S(S+1)}{2}$) | Cached Projections ($S$) | Reduction Ratio ($\frac{S+1}{2}$) | Computational Savings |
|:---|:---|:---|:---|:---|
| **10 tokens** | $55 \text{ ops}$ | $10 \text{ ops}$ | $5.5\times$ | $81.8\%$ |
| **25 tokens** | $325 \text{ ops}$ | $25 \text{ ops}$ | $13.0\times$ | $92.3\%$ |
| **50 tokens** | $1,275 \text{ ops}$ | $50 \text{ ops}$ | $25.5\times$ | $96.1\%$ |
| **100 tokens** | $5,050 \text{ ops}$ | $100 \text{ ops}$ | $50.5\times$ | $98.0\%$ |
| **512 tokens** | $131,328 \text{ ops}$ | $512 \text{ ops}$ | $256.5\times$ | $99.6\%$ |
| **2048 tokens** | $2,098,176 \text{ ops}$ | $2,048 \text{ ops}$ | $1,024.5\times$ | $99.9\%$ |

These counts describe key and value projections ($\frac{S(S+1)}{2}$ vs $S$), not total wall-clock time. Attention still scores every cached key for each new query: full-sequence forwards score $t^2$ pairs at step $t$, while a cached forward scores $t$ pairs. End-to-end latency also encompasses feed-forward MLP projections, token sampling, prefix copies, and Python runtime overhead.

### Production Serving Memory Formulas

$$\mathcal{M}_{\text{cache}} = 2 \cdot L \cdot B \cdot H \cdot S_{\max} \cdot d_k \cdot 4 \text{ bytes}$$

| Model Architecture | Layers ($L$) | Heads ($H$) | Head Dim ($d_k$) | Context ($S_{\max}$) | Cache Memory per Stream ($B=1$) |
|:---|:---|:---|:---|:---|:---|
| **GPT-2 Small (124M)** | 12 | 12 | 64 | 1,024 | $\approx 75.5 \text{ MB}$ |
| **GPT-2 XL (1.5B)** | 48 | 25 | 64 | 1,024 | $\approx 629.1 \text{ MB}$ |
| **Llama 2 (7B)** | 32 | 32 | 128 | 4,096 | $\approx 4.29 \text{ GB}$ |
| **GPT-3 (175B)** | 96 | 96 | 128 | 2,048 | $\approx 19.33 \text{ GB}$ |

Every row above is float32 at $B = 1$, in **decimal** units ($1 \text{ MB} = 10^6$ bytes, $1 \text{ GB} = 10^9$ bytes), which is the convention this module's prose and printed tables use throughout. Work the Llama 2 row yourself as a check: $2 \times 32 \times 1 \times 32 \times 4096 \times 128 \times 4 = 4{,}294{,}967{,}296$ bytes. Halving any row gives the float16 figure, and serving stacks quote float16 far more often than float32, so always ask which precision a published cache number assumes. The one place this module reports binary units is `KVCache.get_memory_usage`, whose `total_mb` is MiB ($2^{20}$ bytes) and is labeled MiB wherever it is printed.
"""

# %% nbgrader={"grade": false, "grade_id": "cached-generation-step", "solution": false}
#| exporti
def _cached_generation_step(x: Tensor, attention: Any, cache_obj: "KVCache",
                            layer_idx: int, mask: Optional[Tensor] = None) -> Tensor:
    """
    Execute a single cached generation step for one new token.

    This helper function isolates the core KV-cache logic, making it:
    - Testable independently
    - Reusable across different attention implementations
    - Clear about what happens during cached generation

    Args:
        x: Input tensor for new token, shape (batch, 1, embed_dim)
        attention: Attention layer with q_proj, k_proj, v_proj, out_proj
        cache_obj: KVCache instance holding previous K,V pairs
        layer_idx: Which transformer layer (for cache indexing)
        mask: Optional binary mask broadcastable to (batch, heads, 1, prefix_length)

    Returns:
        Output tensor, shape (batch, 1, embed_dim)

    Algorithm:
        1. Project x to Q, K, V for this single new token
        2. Reshape to multi-head format
        3. Update cache with new K, V
        4. Retrieve all cached K, V (history + new)
        5. Compute attention: softmax(Q @ K^T / sqrt(d)) @ V
        6. Reshape and project to output
    """
    batch_size = x.shape[0]
    num_heads = attention.num_heads
    head_dim = attention.head_dim

    # Step 1: Project new token to Q, K, V
    Q_new = attention.q_proj.forward(x)  # (batch, 1, embed_dim)
    K_new = attention.k_proj.forward(x)
    V_new = attention.v_proj.forward(x)

    # Step 2: Split into heads (batch, num_heads, 1, head_dim), reusing Module 12's helper
    Q_heads = attention._split_heads(Q_new, batch_size, 1)
    K_heads = attention._split_heads(K_new, batch_size, 1)
    V_heads = attention._split_heads(V_new, batch_size, 1)

    # Step 3: Update cache with new K, V
    cache_obj.update(layer_idx, K_heads, V_heads)

    # Step 4: Retrieve ALL cached K, V (includes history + new token)
    # cache_obj.get() only returns tokens already made visible by advance()
    # (by design -- see KVCache.get()), which does NOT yet include the token
    # just written by update() above. Append this step's own K/V, which we
    # already have in hand, so the new token attends to itself too.
    K_history, V_history = cache_obj.get(layer_idx)
    K_all = Tensor(np.concatenate([K_history.data, K_heads.data], axis=2))
    V_all = Tensor(np.concatenate([V_history.data, V_heads.data], axis=2))

    # Step 5: Compute attention using new Q with all cached K, V
    # This repeats Module 12's scale/mask/softmax/matmul on raw NumPy rather than
    # calling scaled_dot_product_attention, deliberately: that path builds autograd
    # Function nodes for every step, and a decode step needs no gradients. The
    # masking sentinel is imported from Module 12 rather than written out again.
    K_transposed = np.transpose(K_all.data, (0, 1, 3, 2))
    scores = np.matmul(Q_heads.data, K_transposed) / np.sqrt(head_dim)

    # A single query can still mask cached keys (for example, padding).
    # Match Module 12: binary masks, with at least one allowed key per query.
    if mask is not None:
        allowed = mask.data
        if np.any((allowed != 0) & (allowed != 1)):
            raise ValueError("Attention mask must contain only 0 (blocked) or 1 (allowed)")
        if allowed.ndim == 3:
            allowed = allowed[:, None, :, :]
        allowed = np.broadcast_to(allowed, scores.shape)
        if np.any(~np.any(allowed != 0, axis=-1)):
            raise ValueError("Attention mask must allow at least one key per query")
        scores = np.where(allowed != 0, scores, MASK_VALUE)

    # Stable softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Apply attention to values
    attention_output = np.matmul(attention_weights, V_all.data)

    # Step 6: Merge heads back (Module 12's helper) and project to output
    concat_output = attention._merge_heads(Tensor(attention_output), batch_size, 1)

    return attention.out_proj.forward(concat_output)


# %% [markdown]
r"""
### _create_cache_storage: Validate Model and Allocate Cache

This helper validates that a model conforms to the transformer structural invariants required for KV caching, dynamically sizes head dimensions, and attaches a pre-allocated `KVCache` instance directly onto the model instance.

### Dynamic Cache Allocation Specification

| Model Attribute | Target Type | Derived Cache Specification | Validation Invariant |
|:---|:---|:---|:---|
| `model.embed_dim` | `int` | $d_{\text{model}} = 128$ | Total model hidden dimensionality |
| `model.num_heads` | `int` | $H = 4$ | Number of independent attention heads |
| `head_dim` | Derived | $d_k = d_{\text{model}} // H = 32$ | Invariant: $d_{\text{model}} \pmod H == 0$ |
| `model.num_layers` | `int` | $L = 4$ | Instantiates $L$ dedicated $(K, V)$ tensor tuples |
| `model.max_seq_len` | `int` | $S_{\max} = 64$ | Buffer temporal capacity allocated in DRAM |
| `model.blocks` | `list` | $\text{len}(\text{blocks}) == L$ | Transformer blocks hosting attention layers |

Upon successful validation, the instantiated cache is bound as `model._kv_cache` with lifecycle flag `model._cache_enabled = True`.
"""

# %% nbgrader={"grade": false, "grade_id": "kv-create-cache", "solution": true}
#| exporti
def _create_cache_storage(model: Any) -> Tuple["KVCache", int]:
    """
    Validate model architecture and create a KVCache sized for it.

    TODO: Validate model attributes and create a properly-sized KVCache

    APPROACH:
    1. Check model has required attrs: embed_dim, num_layers, num_heads, max_seq_len, blocks
    2. Calculate head_dim = embed_dim // num_heads (validate divisibility)
    3. Create KVCache with batch_size=1, model dimensions
    4. Attach cache to model as model._kv_cache, set model._cache_enabled = True
    5. Return (cache, head_dim) tuple

    EXAMPLE:
    >>> model = MockGPT()  # embed_dim=128, num_heads=4, etc.
    >>> cache, head_dim = _create_cache_storage(model)
    >>> cache.num_layers  # 4
    >>> head_dim  # 32
    >>> model._cache_enabled  # True

    HINTS:
    - Use hasattr() for duck-typing validation (legitimate for plugin systems)
    - Raise AttributeError with helpful 3-part message if attribute missing
    - Raise ValueError if embed_dim not divisible by num_heads
    """
    ### BEGIN SOLUTION role="scaffold"
    # Validate model has required attributes
    # hasattr() is LEGITIMATE here: plugin system with user-defined models
    required_attrs = ['embed_dim', 'num_layers', 'num_heads', 'max_seq_len', 'blocks']
    for attr in required_attrs:
        if not hasattr(model, attr):
            raise AttributeError(
                f"Model missing required attribute for KV caching\n"
                f"  ❌ Model does not have '{attr}' attribute\n"
                f"  💡 enable_kv_cache() requires a GPT-style transformer with architecture attributes: {', '.join(required_attrs)}\n"
                f"  🔧 Ensure your model class defines: self.{attr} = <value> in __init__()"
            )

    # Calculate head dimension
    if model.num_heads <= 0:
        raise ValueError("num_heads must be positive")
    head_dim = model.embed_dim // model.num_heads
    if model.embed_dim % model.num_heads != 0:
        raise ValueError(
            f"Invalid model architecture for multi-head attention\n"
            f"  ❌ embed_dim={model.embed_dim} is not divisible by num_heads={model.num_heads} (remainder: {model.embed_dim % model.num_heads})\n"
            f"  💡 Each attention head needs equal dimensions. embed_dim must be evenly divisible by num_heads.\n"
            f"  🔧 Use embed_dim={model.num_heads * (model.embed_dim // model.num_heads + 1)} (next valid size) or num_heads={[h for h in [1,2,4,8,12,16] if model.embed_dim % h == 0]}"
        )

    # Create cache for this model
    cache = KVCache(
        batch_size=1,  # Fixed at one sequence: reset() rewinds the cursor, not the batch axis
        max_seq_len=model.max_seq_len,
        num_layers=model.num_layers,
        num_heads=model.num_heads,
        head_dim=head_dim
    )

    # Store cache on model for easy access
    model._kv_cache = cache
    model._cache_enabled = True

    return cache, head_dim
    ### END SOLUTION

# %% [markdown]
r"""
### 🧪 Unit Test: _create_cache_storage

**What we're testing**: Model validation, head_dim calculation, and cache creation
**Why it matters**: Cache must match the model's architecture exactly or attention will produce wrong results
**Expected**: Valid models get caches; invalid models get clear error messages
"""

# %% nbgrader={"grade": true, "grade_id": "test-create-cache", "locked": true, "points": 5}
def test_unit_create_cache_storage():
    """🧪 Test _create_cache_storage validates model and creates cache."""
    print("🧪 Unit Test: _create_cache_storage...")

    # Mock model with valid attributes
    class MockGPT:
        def __init__(self):
            self.embed_dim = 128
            self.num_layers = 4
            self.num_heads = 4
            self.max_seq_len = 64
            self.blocks = [None] * 4  # Placeholder blocks

    # Test 1: Valid model creates cache
    model = MockGPT()
    cache, head_dim = _create_cache_storage(model)
    assert head_dim == 32, f"Expected head_dim=32, got {head_dim}"
    assert cache.num_layers == 4, "Cache layers should match model"
    assert cache.num_heads == 4, "Cache heads should match model"
    assert cache.max_seq_len == 64, "Cache max_seq should match model"
    assert model._cache_enabled == True, "Model should be flagged as cache-enabled"
    assert model._kv_cache is cache, "Cache should be attached to model"

    # Test 2: Missing attribute raises AttributeError
    class IncompleteModel:
        def __init__(self):
            self.embed_dim = 128
            # Missing num_layers, num_heads, etc.

    try:
        _create_cache_storage(IncompleteModel())
        assert False, "Should raise AttributeError for incomplete model"
    except AttributeError as e:
        assert "num_layers" in str(e) or "num_heads" in str(e), "Error should name missing attribute"

    # Test 3: Indivisible embed_dim raises ValueError
    class BadDimModel:
        def __init__(self):
            self.embed_dim = 127  # Not divisible by 4
            self.num_layers = 2
            self.num_heads = 4
            self.max_seq_len = 32
            self.blocks = [None] * 2

    try:
        _create_cache_storage(BadDimModel())
        assert False, "Should raise ValueError for indivisible dimensions"
    except ValueError as e:
        assert "divisible" in str(e).lower(), "Error should mention divisibility"

    print("✅ _create_cache_storage works correctly!")

if __name__ == "__main__":
    test_unit_create_cache_storage()


# %% [markdown]
r"""
### CachedAttention: The Stand-In That Chooses the Path

`CachedAttention` takes the place of a block's attention layer while the cache is enabled. It keeps the original layer as `self.attention` and decides which execution path each forward call takes. Decoupling the **dispatch decision** here from the **numerical computation** in `_cached_generation_step` makes both independently testable.

### Stand-In Dispatch Decision Matrix

| Invocation Scope | Sequence Length (`x.shape[1]`) | Target Route | Executed Implementation | Computational Complexity |
|:---|:---|:---|:---|:---|
| **Outside Scope (`_generation_active=False`)** | Any ($S \ge 1$) | **Original Path** | `self.attention.forward(x, mask)` | Full causal attention, supports training autograd |
| **Inside Scope (`_generation_active=True`)** | Multi-token ($S > 1$) | **Original Path** | `self.attention.forward(x, mask)` | Full-sequence fallback: the cache is NOT populated |
| **Inside Scope (`_generation_active=True`)** | Single token ($S == 1$) | **Cached Path** | `_cached_generation_step(x, ...)` | $\mathcal{O}(1)$ projection write + $\mathcal{O}(t)$ prefix attention |

**Read that middle row carefully, because it is a trap.** In a production serving stack, a multi-token forward inside the generation scope is the *prefill*: the whole prompt goes through in one parallel pass and writes every prompt position into the cache at once. TinyTorch's stand-in does no such thing. It hands a multi-token input straight to the original attention, which knows nothing about the cache, so `seq_pos` stays where it was and not one row is written. The next single-token decode step then attends to whatever prefix the cache happens to hold, which after a skipped prefill is nothing at all, and it returns confident, wrong logits with no error and no warning. Prefill in this module is therefore **sequential**: feed the prompt one token at a time, exactly as `_cached_generate` does, so that every prompt position writes its own $(K, V)$ row.

Inside `cache.generation()`, the prompt tokens take the cached path one at a time for that reason. The initial prompt token encounters an empty cache, attends strictly to itself, and registers its $(K, V)$ representation into slot 0 for subsequent tokens. The exact token coordinate is synchronized via `start_pos=cache.seq_pos`, enabling positional embeddings to preserve temporal coordinates.
"""

# %% nbgrader={"grade": false, "grade_id": "kv-cached-attention", "solution": true}
#| export
class CachedAttention:
    """
    Stand-in for one block's attention layer while a KV cache is enabled.

    Holds the original attention layer and the cache. enable_kv_cache() puts one
    of these at block.attention; disable_kv_cache() puts the original back.
    """

    def __init__(self, attention: Any, cache_obj: KVCache, layer_idx: int) -> None:
        self.attention = attention      # the original MultiHeadAttention layer
        self.cache = cache_obj
        self.layer_idx = layer_idx

    def parameters(self) -> List[Tensor]:
        """The stand-in owns no weights of its own."""
        return self.attention.parameters()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Route one call to the right path.

        TODO: Implement the two-path dispatch

        APPROACH:
        1. Outside cache.generation(), or for a full sequence, return
           self.attention.forward(x, mask) unchanged
        2. Inside that scope, route one new token through
           _cached_generation_step, including its optional attention mask

        Note that path 1 writes nothing to the cache, so a multi-token forward
        inside the generation scope is a fallback, not a parallel prefill.
        Prefill here is sequential: one token per forward.

        EXAMPLE:
        >>> stand_in = CachedAttention(block.attention, cache, layer_idx=0)
        >>> stand_in.forward(x_train)   # (1, 10, D): original attention, cache untouched
        >>> with cache.generation():
        ...     stand_in.forward(x_token)  # (1, 1, D): cached step, cache updated

        HINTS:
        - x.shape[1] is the sequence length
        - The cached step handles an empty cache itself (the first token attends to itself)
        """
        ### BEGIN SOLUTION role="scaffold"
        if not self.cache._generation_active or x.shape[1] > 1:
            return self.attention.forward(x, mask)
        return _cached_generation_step(x, self.attention, self.cache, self.layer_idx, mask)
        ### END SOLUTION

    def __call__(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.forward(x, mask)

# %% [markdown]
r"""
### 🧪 Unit Test: CachedAttention

**What we're testing**: The stand-in routes ordinary forwards to the original attention and explicitly scoped single tokens through the cache, and the cached path reproduces uncached causal attention
**Why it matters**: Wrong routing causes silent correctness bugs (training reads the cache, or generation ignores it), and a cache that changes the numbers is not an optimization
**Expected**: Full-sequence output identical to the original layer with the cache untouched; token-by-token outputs match the causal attention output at every position
"""

# %% nbgrader={"grade": true, "grade_id": "test-cached-attention", "locked": true, "points": 10}
def test_unit_cached_attention():
    """🧪 Test CachedAttention routes calls correctly and matches uncached attention."""
    print("🧪 Unit Test: CachedAttention...")
    from tinytorch.core.attention import MultiHeadAttention

    embed_dim, num_heads, seq_len = 32, 4, 5
    attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    cache = KVCache(batch_size=1, max_seq_len=16, num_layers=1,
                    num_heads=num_heads, head_dim=embed_dim // num_heads)
    stand_in = CachedAttention(attention, cache, layer_idx=0)

    # Path 1: a full sequence goes to the original layer, unchanged
    x = Tensor(rng.standard_normal((1, seq_len, embed_dim)))
    causal = Tensor(np.tril(np.ones((1, seq_len, seq_len))))
    full = attention.forward(x, causal)
    routed = stand_in.forward(x, causal)
    assert np.allclose(routed.data, full.data), "Full sequences must use the original attention"
    assert cache.seq_pos == 0, "A full-sequence call must not touch the cache"

    # Path 2: one token at a time through the cache reproduces the causal result
    with cache.generation():
        for t in range(seq_len):
            out_t = stand_in.forward(x[:, t:t+1, :])
            cache.advance()
            assert out_t.shape == (1, 1, embed_dim), "One token in, one token out"
            assert np.allclose(out_t.data[0, 0], full.data[0, t], atol=1e-5), \
                f"Cached output at position {t} differs from uncached causal attention"
    assert cache.seq_pos == seq_len, "Each advance() should cache one more token"
    # Leaving the scope restores ordinary forwards, even for a single token.
    ordinary = stand_in.forward(x[:, :1, :])
    assert np.allclose(ordinary.data, attention.forward(x[:, :1, :]).data), \
        "An ordinary single-token forward must not consume cached history"

    print("   Full sequence: routed to the original attention, cache untouched")
    print(f"   {seq_len} single tokens: cached path matches causal attention at every position")
    print("✅ CachedAttention routes correctly and matches uncached attention!")

if __name__ == "__main__":
    test_unit_cached_attention()


# %% [markdown]
r"""
### _cached_generate: Generation Loop with KV Cache

This helper coordinates the two-phase autoregressive generation pipeline: sequential prompt prefilling followed by token-by-token generation with static buffer advancement.

### Two-Phase Generation Lifecycle

| Execution Phase | Input Token Tensor | Active Cache Slice | Attention Mechanism | Arithmetic Complexity |
|:---|:---|:---|:---|:---|
| **Phase 1: Prefill** | Prompt tokens $t \in [0, P-1]$ sequentially | Populates slots $0 \dots P-1$ | Single query attending to current prefix | $\mathcal{O}(P)$ projections, $\mathcal{O}(P^2)$ attention |
| **Phase 2: Decode Step 1** | Single token $\mathbf{x}_{P} \in \mathbb{R}^{1 \times 1}$ | Writes to slot $P$, reads $0 \dots P$ | Vector-matrix GEMV ($1 \times P$ scores) | 1 new projection, $P+1$ score pairs |
| **Phase 2: Decode Step $k$** | Single token $\mathbf{x}_{P+k-1} \in \mathbb{R}^{1 \times 1}$ | Writes to slot $P+k-1$, reads $0 \dots P+k-1$ | Vector-matrix GEMV ($1 \times (P+k)$ scores) | 1 new projection, $P+k$ score pairs |

At each decode step, `cache.advance()` moves the write cursor forward by 1. When `temperature == 0.0`, greedy argmax selection is used; for non-zero temperatures, stable categorical sampling with logit scaling is applied.
"""

# %% nbgrader={"grade": false, "grade_id": "kv-cached-generate", "solution": true}
#| exporti
def _cached_generate(model: Any, prompt_tokens: List[int], max_new_tokens: int,
                     temperature: float, cache: KVCache) -> List[int]:
    """
    Run autoregressive generation using the KV cache.

    TODO: Implement the cached generation loop

    APPROACH:
    1. Process prompt tokens one at a time to populate cache (prefill phase)
    2. Get the last token's logits and sample next token
    3. Loop for max_new_tokens steps:
       a. Feed ONLY the new token through the model (seq_len=1)
       b. The CachedAttention stand-ins write each token's K,V into the cache
       c. Advance cache position after each token
       d. Sample next token from logits with temperature scaling
    4. Return list of generated token indices; do not forward the last sampled
       token because no later prediction needs its logits

    EXAMPLE:
    >>> generated = _cached_generate(model, prompt_tokens=[0, 1, 2],
    ...                               max_new_tokens=5, temperature=1.0,
    ...                               cache=cache)
    >>> len(generated)  # 5 new tokens

    HINTS:
    - Prefill: feed each prompt token one at a time via model.forward(token_tensor, start_pos=cache.seq_pos)
      so every token, including the first, writes its K,V into the cache at its true position
    - Generation: model.forward(single_token_tensor, start_pos=cache.seq_pos) processes one token
    - Use temperature scaling: logits / temperature before softmax
    - Use rng.choice with softmax probabilities to sample
    - Advance cache.advance() after each token (both prefill and generation)
    - Stable softmax: subtract max before exp to avoid overflow

    Args:
        model: Transformer model whose attention layers are wrapped by enable_kv_cache
        prompt_tokens: List of integer token IDs for the prompt
        max_new_tokens: Number of new tokens to generate
        temperature: Sampling temperature (higher = more random)
        cache: KVCache instance (already attached to model)

    Returns:
        List of generated token IDs (integers)
    """
    ### BEGIN SOLUTION role="scaffold"
    if not isinstance(max_new_tokens, (int, np.integer)) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if len(prompt_tokens) == 0:
        raise ValueError("prompt_tokens must contain at least one token")
    # The last sampled token is returned, not fed back into the model, so the run
    # needs len(prompt) + max_new_tokens - 1 slots, one fewer than it generates.
    if max_new_tokens > 0 and len(prompt_tokens) + max_new_tokens - 1 > cache.max_seq_len:
        raise ValueError("Prompt and generation exceed cache capacity")
    generated = []
    cache.reset()   # a fresh sequence: the cursor back to slot 0, no rows from the last request

    if max_new_tokens == 0:
        return generated

    with cache.generation():
        # Phase 1: PREFILL - process prompt tokens one at a time to populate cache
        # Each token goes through the CachedAttention stand-ins, which write its
        # K,V into the cache. start_pos=cache.seq_pos gives the token its true
        # position, so a single-token forward computes exactly what the full
        # sequence would have computed for that position.
        for token in prompt_tokens:
            token_tensor = Tensor(np.array([[token]]))  # (1, 1)
            logits = model.forward(token_tensor, start_pos=cache.seq_pos)
            cache.advance()

        # Get logits for last prompt token (predicts next token)
        last_logits = logits.data[0, -1, :]  # (vocab_size,)

        # Phase 2: GENERATE - one token at a time using cache
        for step in range(max_new_tokens):
            # Zero temperature means greedy decoding, including tied logits.
            if temperature == 0:
                next_token = int(np.argmax(last_logits))
            else:
                scaled_logits = last_logits / temperature
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
                probs = exp_logits / np.sum(exp_logits)
                next_token = int(rng.choice(len(probs), p=probs))
            generated.append(next_token)
            if step == max_new_tokens - 1:
                break

            # Feed single token through model (cache handles history)
            token_tensor = Tensor(np.array([[next_token]]))  # (1, 1)
            logits = model.forward(token_tensor, start_pos=cache.seq_pos)  # (1, 1, vocab_size)
            cache.advance()

            last_logits = logits.data[0, -1, :]

    return generated
    ### END SOLUTION

# %% [markdown]
r"""
### 🧪 Unit Test: _cached_generate

**What we're testing**: The autoregressive generation loop with cache advancement
**Why it matters**: The generation loop must correctly advance the cache and produce valid token IDs
**Expected**: Generates the requested number of tokens, all valid indices into the vocabulary
"""

# %% nbgrader={"grade": true, "grade_id": "test-cached-generate", "locked": true, "points": 10}
def test_unit_cached_generate():
    """🧪 Test _cached_generate produces correct number of valid tokens."""
    print("🧪 Unit Test: _cached_generate...")

    vocab_size = 50

    # Create a minimal mock model that returns random logits
    class MockModel:
        def __init__(self):
            self.embed_dim = 64
            self.num_layers = 1
            self.num_heads = 2
            self.max_seq_len = 128
            self.blocks = []

        def forward(self, x, start_pos=0):
            # Return random logits shaped (batch, seq_len, vocab_size)
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            return Tensor(rng.standard_normal((batch_size, seq_len, vocab_size)))

    model = MockModel()

    # Create cache (not attached to blocks since mock has none)
    cache = KVCache(batch_size=1, max_seq_len=128, num_layers=1,
                    num_heads=2, head_dim=32)

    # Test 1: Generate correct number of tokens
    prompt = [0, 1, 2]
    max_new = 5
    generated = _cached_generate(model, prompt, max_new, temperature=1.0, cache=cache)
    assert len(generated) == max_new, f"Expected {max_new} tokens, got {len(generated)}"

    # Test 2: All tokens are valid indices
    for token in generated:
        assert 0 <= token < vocab_size, f"Token {token} out of vocab range [0, {vocab_size})"

    # Test 3: Cache position advanced correctly
    # The final sampled token needs no forward pass: 3 + 5 - 1 = 7 advances.
    expected_pos = len(prompt) + max_new - 1
    assert cache.seq_pos == expected_pos, f"Expected cache pos={expected_pos}, got {cache.seq_pos}"

    # Test 4: Generate with low temperature (more deterministic)
    cache.reset()
    generated_low_temp = _cached_generate(model, [0], 3, temperature=0.01, cache=cache)
    assert len(generated_low_temp) == 3, "Should generate 3 tokens with low temperature"

    print("✅ _cached_generate works correctly!")

if __name__ == "__main__":
    test_unit_cached_generate()


# %% [markdown]
r"""
### enable_kv_cache: Compose the Cache Into the Model

This is the main entry point that composes the subsystem helpers into an integrated runtime. It attaches the pre-allocated cache and wraps each transformer block's attention layer in place.

### Dynamic Model Enhancement Lifecycle

| Sequence Step | Invoked Primitive | Model Modification | Invariant Enforced |
|:---|:---|:---|:---|
| **Step 1: Validate & Allocate** | `_create_cache_storage(model)` | Attaches `model._kv_cache` | Verifies head dimension divisibility and allocates DRAM buffers |
| **Step 2: Unwrap Idempotence** | Check `isinstance(block.attention, CachedAttention)` | Unwraps previous stand-in if already present | Prevents nested wrapper recursion if enabled repeatedly |
| **Step 3: Wrap Attention** | `CachedAttention(block.attention, cache, layer_idx)` | Replaces `block.attention` with stand-in | Preserves original weights, parameters, and duck typing |
| **Step 4: Expose Telemetry** | `cache.get_memory_usage()` | Displays layer and head allocations | Informs operator of buffer footprint in MB |
"""

# %% nbgrader={"grade": false, "grade_id": "kv-enable-cache", "solution": true}
#| export
def enable_kv_cache(model: Any) -> KVCache:
    """
    Enable KV caching for a transformer model without editing its layers.

    TODO: Compose helpers to create the cache and wrap the attention layers

    APPROACH:
    1. Call _create_cache_storage(model) to validate and create the cache
    2. For each block, replace block.attention with a CachedAttention stand-in
       that owns the original layer (unwrap first if one is already there)
    3. Print confirmation with cache statistics
    4. Return the cache object

    This is the wrapping pattern: a stand-in owns the original object and
    forwards to it, and removing the stand-in restores the original. Module 06
    used a different pattern, completing each operation class with @method_of,
    because there the added half is permanent; here it has to be removable.

    Args:
        model: A GPT-style transformer model with:
               - model.embed_dim (int)
               - model.num_layers (int)
               - model.num_heads (int)
               - model.max_seq_len (int)
               - model.blocks (list of TransformerBlock objects)

    Returns:
        cache: KVCache object for this model

    EXAMPLE:
    >>> from tinytorch.core.transformers import GPT
    >>> model = GPT(vocab_size=100, embed_dim=128, num_layers=4, num_heads=4)
    >>> cache = enable_kv_cache(model)
    >>> hasattr(model, '_kv_cache')  # True
    >>> model._cache_enabled  # True
    >>> cache.num_layers  # 4 (matches model)

    HINTS:
    - _create_cache_storage handles validation, KVCache creation, and model attachment
    - isinstance(block.attention, CachedAttention) tells you a stand-in is already in place
    - The stand-in needs the original layer, the cache, and its layer index
    """
    ### BEGIN SOLUTION role="scaffold"
    # Step 1: Validate model and create cache
    cache, head_dim = _create_cache_storage(model)

    # Step 2: Put a stand-in in front of each block's attention layer
    for layer_idx, block in enumerate(model.blocks):
        if isinstance(block.attention, CachedAttention):   # enabled twice: start from the original
            block.attention = block.attention.attention
        block.attention = CachedAttention(block.attention, cache, layer_idx)

    # Step 3: Print confirmation
    print("⚡ KV Cache enabled for model!")
    print(f"   Architecture: {model.num_layers} layers × {model.num_heads} heads × {head_dim}D")
    print(f"   Memory: {cache.get_memory_usage()['total_mb']:.2f} MiB")
    print("   Cache stored in: model._kv_cache")
    print()
    print("💡 To disable: call disable_kv_cache(model)")
    print()

    return cache
    ### END SOLUTION


# %% nbgrader={"grade": false, "grade_id": "kv-disable-cache", "solution": false}
#| export
def disable_kv_cache(model: Any) -> None:
    """
    Disable KV caching and restore original attention behavior.

    Args:
        model: Model with caching enabled

    EXAMPLE:
        ```python
        cache = enable_kv_cache(model)
        # ... do cached generation ...
        disable_kv_cache(model)  # Back to normal
        ```
    """
    if not getattr(model, '_cache_enabled', False):
        print("⚠️  KV cache not enabled on this model")
        return

    # Take the stand-ins out; each one still holds the original layer
    for block in model.blocks:
        if isinstance(block.attention, CachedAttention):
            block.attention = block.attention.attention

    # Clean up
    model._cache_enabled = False
    if hasattr(model, '_kv_cache'):
        delattr(model, '_kv_cache')

    print("✓ KV cache disabled, original attention restored")


# %% [markdown]
r"""
### 🧪 Unit Test: Non-Invasive Cache Integration

This test validates that `enable_kv_cache()` works without breaking the model.

**What we're testing**: Non-invasive cache integration with transformer models
**Why it matters**: Must add caching without modifying existing modules (forward-only learning)
**Expected**: Cache enables/disables cleanly, model forward pass still works
"""

# %% nbgrader={"grade": true, "grade_id": "test-noninvasive", "locked": true, "points": 10}
def test_unit_noninvasive_integration():
    """🧪 Unit Test: Non-Invasive Cache Integration"""
    print("🧪 Unit Test: Non-Invasive Cache Integration...")

    # Create a mock transformer-like object for testing
    class MockTransformerBlock:
        def __init__(self):
            self.attention = self

        def forward(self, x, mask=None):
            # Simple pass-through for testing
            return x

    class MockGPT:
        def __init__(self):
            self.vocab_size = 100
            self.embed_dim = 128
            self.num_layers = 4
            self.num_heads = 4
            self.max_seq_len = 64
            self.blocks = [MockTransformerBlock() for _ in range(self.num_layers)]

    # Test 1: Enable caching
    model = MockGPT()
    print("   Test 1: Enable caching on model")
    cache = enable_kv_cache(model)
    assert hasattr(model, '_kv_cache'), "Model should have _kv_cache attribute"
    assert hasattr(model, '_cache_enabled'), "Model should have _cache_enabled flag"
    assert model._cache_enabled == True, "Cache should be enabled"
    assert cache is model._kv_cache, "Returned cache should match model._kv_cache"

    # Test 2: Attention forward still works
    print("   Test 2: Attention forward pass still works")
    test_input = Tensor(rng.standard_normal((1, 10, 128)))
    for block in model.blocks:
        output = block.attention.forward(test_input)
        assert output.shape == test_input.shape, "Forward pass should preserve shape"

    # Test 3: Disable caching
    print("   Test 3: Disable caching")
    disable_kv_cache(model)
    assert model._cache_enabled == False, "Cache should be disabled"
    assert not hasattr(model, '_kv_cache'), "Cache object should be removed"

    # Test 4: Can re-enable
    print("   Test 4: Re-enable caching")
    _ = enable_kv_cache(model)
    assert model._cache_enabled == True, "Cache should be re-enabled"

    print("✅ Non-invasive cache integration works correctly!")

if __name__ == "__main__":
    test_unit_noninvasive_integration()


# %% [markdown]
r"""
## 📊 Systems Analysis: KV Cache Performance

Let's analyze the performance characteristics and trade-offs of KV caching. Understanding these trade-offs is essential for making informed decisions about when and how to use caching in production systems.
"""

# %% nbgrader={"grade": false, "grade_id": "analyze-memory", "solution": false}
def analyze_kvcache_memory():
    """
    📊 Analyze KV cache memory usage across different configurations.

    Educational Purpose:
        Demonstrates how cache memory scales with model architecture.
        Students discover:
        - Linear scaling with sequence length O(n)
        - Memory overhead as percentage of model parameters
        - Trade-off between cache size and speedup gains

    Analyzes (decimal MB, 10^6 bytes, matching this module's tables):
        - Tiny models (128D): ~0.13 MB
        - Small models (512D): ~2.10 MB
        - Medium models (768D): ~9.44 MB
        - Large models (1024D): ~33.55 MB

    Key Insight:
        Cache size is set by context length, not by parameter count. The
        ratio below compares it to one transformer block's parameters (~12·d²).

    Production Context:
        GPT-3 (96 layers, 96 heads, head_dim 128, 2048 context): ~19.33 GB per
        sequence in FP32, which is why serving systems budget memory per user.
    """
    print("📊 Analyzing KV Cache Memory Usage...")
    print()

    # Test different model configurations
    configs = [
        (128, 4, 32, "Tiny"),
        (512, 8, 64, "Small"),
        (768, 12, 128, "Medium"),
        (1024, 16, 256, "Large"),
    ]

    print("Model Config | Cache Memory | Per Layer | Cache / Block Params")
    print("-" * 60)

    for embed_dim, num_layers, seq_len, name in configs:
        # Memory per layer: 2 tensors (K, V) × batch × seq_len × embed_dim × 4 bytes
        batch_size = 1
        memory_per_layer = 2 * batch_size * seq_len * embed_dim * _BYTES_PER_FLOAT32 / _BYTES_PER_MB
        total_memory = memory_per_layer * num_layers

        # Model parameter memory: a transformer block has about 12·d² parameters
        # (4·d² in attention projections, 8·d² in the MLP)
        params_per_layer = 12 * embed_dim * embed_dim
        model_memory = params_per_layer * num_layers * _BYTES_PER_FLOAT32 / _BYTES_PER_MB

        overhead_pct = (total_memory / model_memory) * 100 if model_memory > 0 else 0

        print(f"{name:12s} | {total_memory:11.2f} MB | {memory_per_layer:8.2f} MB | {overhead_pct:6.1f}%")

    print()
    print("💡 Key Insights:")
    print("   • Cache memory scales linearly with sequence length (O(n))")
    print("   • Longer sequences require proportionally more cache memory")
    print("   • The ratio compares the cache to block parameters (~12·d² each); it grows with context")
    print()
    print("🚀 Production Context:")
    print("   • GPT-3 (96 layers, 96 heads, head_dim 128, 2048 context): ~19.33 GB per sequence in FP32")
    print("   • Trade-off: memory per cached token buys away the O(n²) attention recomputation")
    print("   • Worth it for inference-heavy workloads!")

# %% nbgrader={"grade": false, "grade_id": "analyze-speedup", "solution": false}
def analyze_kvcache_speedup():
    """
    📊 Measure KV cache speedup vs vanilla attention.

    Educational Purpose:
        Shows students WHY caching provides dramatic speedup through
        concrete complexity analysis. Compares O(n²) vs O(n) growth.

    Demonstrates:
        - Naive approach: attention over the whole context for every new token
        - Cached approach: one token of new work per step
        - Measured wall-clock speedup on a tiny GPT, next to the attention score-pair ratio

    Key Insight:
        Speedup GROWS with generation length, but sublinearly: the measured
        gain runs roughly 1.2x at 10 tokens to a little over 2x at 100, so a
        10x longer generation buys under 2x more speedup. The score-pair ratio
        climbs far faster (about 8.8x to 69x) because it counts only the
        arithmetic the cache eliminates. The clock also pays for weight reads,
        the MLP, prefix copies, and Python overhead, which the cache leaves
        untouched, and decoding is bandwidth-bound on those weight reads.

    Production Reality:
        This is why ChatGPT can generate responses in real-time.
        Without caching, conversational AI would be economically impossible.
    """
    print("\n📊 Analyzing KV Cache Speedup...")
    print()

    from tinytorch.core.transformers import GPT

    # A tiny GPT: big enough to show the trend, small enough to run in seconds
    model = GPT(vocab_size=64, embed_dim=64, num_layers=2, num_heads=4, max_seq_len=256)
    prompt = [1, 2, 3, 4]
    prompt_tensor = Tensor(np.array([prompt]))  # GPT.generate takes a (1, seq_len) Tensor

    cache = enable_kv_cache(model)  # _cached_generate resets it for every new sequence

    print("Generation Length | Without Cache | With Cache | Measured | Attention ops ratio")
    print("-" * 80)

    for gen_length in [10, 25, 50, 100]:
        # Without cache: model.generate() re-runs the whole sequence for every token.
        # CachedAttention only changes forwards inside cache.generation(), so
        # the path below is the plain Module 13 attention.
        start = time.perf_counter()
        model.generate(prompt_tensor, max_new_tokens=gen_length, temperature=1.0)
        time_without = (time.perf_counter() - start) * 1000

        # With cache: one token of work per step
        start = time.perf_counter()
        _cached_generate(model, prompt, gen_length, 1.0, cache)
        time_with = (time.perf_counter() - start) * 1000

        measured = time_without / max(time_with, 1e-6)
        contexts = np.arange(len(prompt), len(prompt) + gen_length)
        # Full forwards score t² query/key pairs; cached forwards score t.
        # Token-by-token prefill also scores 1 + ... + (prompt_len - 1) pairs.
        ops_ratio = np.sum(contexts ** 2) / (np.sum(contexts) + sum(range(len(prompt))))

        print(f"{gen_length:17d} | {time_without:10.1f} ms | {time_with:8.1f} ms | {measured:6.1f}× | {ops_ratio:6.1f}×")

    disable_kv_cache(model)
    print()
    print("💡 Key Insights:")
    print("   • Speedup grows with generation length (longer = better ROI)")
    print("   • The score-pair ratio is an operation count, not a wall-clock ceiling;")
    print("     projections, the MLP, prefix copies, and Python overhead also affect timing")
    print()
    print("🚀 Production Reality:")
    print("   • Every production LLM server caches K/V; without it, per-token cost")
    print("     grows with the conversation and long chats become unaffordable")

if __name__ == "__main__":
    analyze_kvcache_memory()
    analyze_kvcache_speedup()


# %% [markdown]
r"""
## 🧪 Module Integration Test

Final validation that everything works together correctly before module completion.
"""

# %% nbgrader={"grade": true, "grade_id": "module-integration", "locked": true, "points": 20}
def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire KV Caching module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)
    print()

    # Run all unit tests
    print("Running unit tests...")
    test_unit_kvcache()
    print()
    test_unit_create_cache_storage()
    print()
    test_unit_cached_attention()
    print()
    test_unit_cached_generate()
    print()
    test_unit_noninvasive_integration()
    print()

    print("Running integration scenarios...")
    print()

    # Integration Test: Complete KV Cache Workflow
    print("🧪 Integration Test: Complete KV Cache Workflow...")
    batch_size, max_seq_len = 1, 128
    num_layers, num_heads, head_dim = 4, 8, 64

    cache = KVCache(batch_size, max_seq_len, num_layers, num_heads, head_dim)

    # Simulate generation loop (processing multiple tokens)
    for _ in range(5):
        for layer_idx in range(num_layers):
            # Simulate new key-value pairs
            new_key = Tensor(rng.standard_normal((batch_size, num_heads, 1, head_dim)))
            new_value = Tensor(rng.standard_normal((batch_size, num_heads, 1, head_dim)))

            # Update cache
            cache.update(layer_idx, new_key, new_value)

        # Advance position after all layers processed
        cache.advance()

    # Verify cache state
    assert cache.seq_pos == 5, f"Expected seq_pos=5, got {cache.seq_pos}"

    # Verify retrieval
    for layer_idx in range(num_layers):
        cached_k, cached_v = cache.get(layer_idx)
        assert cached_k.shape == (batch_size, num_heads, 5, head_dim)
        assert cached_v.shape == (batch_size, num_heads, 5, head_dim)

    print("✅ Complete KV cache workflow validated!")
    print()

    # Integration Test: Memory Tracking
    print("🧪 Integration Test: Memory Tracking...")
    mem_info = cache.get_memory_usage()
    assert mem_info['total_mb'] > 0
    assert mem_info['cache_tensors'] == num_layers * 2
    print(f"✅ Memory tracking: {mem_info['total_mb']:.2f} MiB for {mem_info['cache_tensors']} tensors")
    print()

    print("=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 18")


# %% [markdown]
r"""
## 🤔 ML Systems Reflection Questions

Every question below is answerable from one expression, the cache-size formula from the Production Serving Memory Formulas table:

$$\mathcal{M}_{\text{cache}} = 2 \cdot L \cdot B \cdot H \cdot S_{\max} \cdot d_k \cdot (\text{bytes per element})$$

The leading 2 counts keys and values, and $H$ is the number of **key/value** heads, which matters once grouped-query attention enters in Question 5. Unless a question states otherwise, use this module's **running serving model**, the GPT-2 Small row of that table: $L = 12$ layers, $H = 12$ heads, $d_k = 64$, $S_{\max} = 1{,}024$, `float32` (4 bytes per element), $B = 1$ per sequence. Compute its per-sequence cache once and reuse it:

$$\mathcal{M} = 2 \times 12 \times 1 \times 12 \times 1{,}024 \times 64 \times 4 = 75{,}497{,}472 \text{ bytes} \approx 75.5 \text{ MB}$$

Byte counts here are decimal (MB $= 10^6$, GB $= 10^9$, TB $= 10^{12}$ bytes). Work each question before reading the analysis that follows it.

---

### Question 1: Cache Size Calculation

A 12-layer transformer serves a batch of 8 sequences. It has 12 attention heads per layer, 64 dimensions per head, a maximum sequence length of 2,048, and a `float32` cache.

**Answer these**:
1. How many bytes does the full KV cache occupy across all layers?
2. The model holds 125M parameters in `float32`. Is the cache bigger or smaller than the weights, and by what factor?
3. Naive decoding reprojects every past token at every step. How many key/value projections does one 2,048-token sequence cost naively, how many are strictly necessary, and how many are pure waste?

**Worked Systems Analysis**:
- **One cache tensor shape**:
  $$\text{Shape} = (B, H, S_{\max}, d_k) = (8, 12, 2048, 64)$$
- **Elements per tensor**:
  $$8 \times 12 \times 2048 \times 64 = 12,582,912 \text{ elements}$$
- **Tensors per layer**:
  $$\text{Key Cache } \mathbf{K} + \text{Value Cache } \mathbf{V} = 2 \text{ tensors per layer}$$
- **Total across 12 layers**:
  $$12 \text{ layers} \times 2 \text{ tensors/layer} = 24 \text{ cache tensors}$$
- **Total elements**:
  $$24 \times 12,582,912 = 301,989,888 \text{ elements}$$
- **Memory footprint in float32 (4 bytes per element)**:
  $$\text{Total Bytes} = 301,989,888 \times 4 \text{ bytes} = 1,207,959,552 \text{ bytes}$$
  $$\text{Memory in MiB} = \frac{1,207,959,552}{1024^2} = \mathbf{1,152.0 \text{ MiB}} \quad (\mathbf{1.208 \text{ GB}} \text{ decimal})$$

**Follow-up Analysis (Overhead vs Model Parameters)**:
If this model has 125M parameters ($125 \times 10^6 \times 4 \text{ bytes} = 500 \text{ MB}$), compare like with like, in bytes:
$$\text{Cache-to-Model Ratio} = \frac{1,207,959,552 \text{ B}}{500,000,000 \text{ B}} = \mathbf{241.6\%}$$
The KV cache occupies **more than double** the memory of the neural network weights themselves. (Divide 1,152 MiB by 500 MB and you get 230.4%, but that mixes binary and decimal units and is not a ratio of anything. Keep both sides in bytes.)

**Projection accounting**: naive decoding costs $\frac{2048 \times 2049}{2} = 2{,}098{,}176$ key/value projections per sequence, of which only $2{,}048$ are necessary, so $2{,}096{,}128$ (about 2.1 million) are pure recomputation.

**Is this overhead acceptable?**
Yes, because those 2.1 million redundant projections per sequence would land as inter-token latency that grows with every token generated, breaching serving SLAs long before the sequence finishes. The overhead does explain why modern architectures replace Multi-Head Attention (MHA) with **Grouped-Query Attention (GQA)** or **Multi-Query Attention (MQA)** (Llama 2/3, Mistral). GQA gives each group of query heads one shared key/value head, so a group size of 4 to 8 stores 4 to 8 times fewer KV heads and cuts the cache to a quarter or an eighth. MQA is the limiting case with a single key/value head shared by every query head, which for this 12-head model is a $12\times$ reduction.

---

### Question 2: Speed vs Memory Trade-Off

Your `KVCache` eliminates the $\mathcal{O}(S^2)$ projection recomputation but reserves persistent memory for every active token. Serve the running model (75.5 MB of cache per sequence) to 1,000 concurrent users on one host with 64 GB of RAM.

**Answer these**:
1. Is the trade-off worth making for an interactive chatbot? Argue it from inter-token latency, not from FLOP counts.
2. How much cache memory do 1,000 concurrent users require, and what does that host do when asked for it?
3. Name two production techniques that make high concurrency affordable, and state precisely what each one recovers.

**Systems Serving Analysis (1000 Concurrent Users)**:

| Metric | Without Cache (Stateless Forward) | With Cache (`KVCache`) |
|:---|:---|:---|
| **Inter-Token Latency (ITL)** | Degrades linearly $\mathcal{O}(t)$; requests timeout | Constant $\mathcal{O}(1)$ projection time per token |
| **FLOP Waste Rate** | $1 - \frac{2}{t+1} \to 99.8\%$ wasted recomputation at $t = 1{,}024$ | **0%** redundant projection FLOPs |
| **Memory Footprint (1000 users)** | Minimal ($\approx 0 \text{ MB}$ persistent KV memory) | $1{,}000 \times 75.5 \text{ MB} = \mathbf{75.5 \text{ GB DRAM}}$ |
| **Server Fate (64 GB host)** | High CPU utilization, low memory pressure | Over budget by 11.5 GB: **Out of Memory** |

**Worked Systems Analysis**:
1. **Chatbot Feasibility**: The trade-off is mandatory. Human users perceive delays above $100 \text{ ms}$ as sluggish, and without caching the per-token cost grows with the conversation, so token 1,000 costs roughly $1{,}000\times$ what token 1 cost. Interactivity fails not because the total FLOPs are large but because the *last* token is the slowest.
2. **64 GB Host Failure**: Asking for 75.5 GB on a 64 GB host first drives the OS into swapping, which costs orders of magnitude in latency because a page fault services from disk instead of DRAM, and then invokes the OOM killer on the serving daemon. A capacity overrun of 18% does not degrade gracefully; it takes the service down.
3. **High-Concurrency Architecture**:
   - **PagedAttention (vLLM)**: Hold the cache in fixed-size blocks (16 tokens per block) addressed through a block table, the same indirection an operating system page table provides. This eliminates external fragmentation outright and bounds internal fragmentation to at most one partially filled block per sequence. The vLLM paper measures prior systems wasting 60% to 80% of KV memory on over-reservation and fragmentation, and reports waste under 4% with paging, so what is recovered is wasted *capacity*, not compute.
   - **Tiered Memory Swapping**: Active generating sequences stay in fast GPU HBM. When a request enters human think-time (waiting for the user to type), its cache is asynchronously DMA-transferred to host CPU RAM or local NVMe SSD over PCIe, freeing HBM for other active decode steps. What is recovered here is HBM *residency* during idle turns, paid for with transfer bandwidth and prefetch latency.

---

### Question 3: Batch Inference Scaling

Each sequence in a batch keeps its own $(K, V)$ slices, so the cache grows with $B$. Take the running model and raise the batch from 1 to 8. Assume two measurements as given: a single sequence decodes at 500 tok/s, and inside a batch of 8 each sequence decodes at 420 tok/s.

**Answer these**:
1. What is the cache footprint at $B = 8$, and exactly how does it scale with $B$?
2. Per-sequence decoding slows by only 16% while eight times as much work is done. Why does batching cost so little here, and which hardware limit explains it?
3. What is total throughput at $B = 1$ and at $B = 8$, and when would you deliberately choose the smaller batch anyway?

**Worked Systems Analysis**:
- **Cache Memory at Batch 8**:
  $$\mathcal{M}(8) = 2 \times 12 \times 8 \times 12 \times 1{,}024 \times 64 \times 4 = 603{,}979{,}776 \text{ B} \approx \mathbf{604.0 \text{ MB}}$$
  That is exactly $8 \times 75.5 \text{ MB}$. $B$ enters the formula once, to the first power, so the cache is strictly linear in batch size, $\mathcal{O}(B)$.
- **Per-Sequence Generation Rate**:
  At batch size 1, decoding is **memory-bandwidth bound**, as established in 📐: reading the whole weight set out of HBM to produce one token vector leaves the arithmetic units mostly idle. Batching $B = 8$ sequences reuses each loaded weight matrix across 8 token vectors, turning a bandwidth-bound GEMV into a compute-dense GEMM. The bytes moved barely change, so per-sequence latency rises only marginally, from 500 to $\approx 420 \text{ tok/s}$.
- **Throughput Calculation**:
  $$\text{Throughput}(B=1) = 1 \times 500 = 500 \text{ total tok/s}$$
  $$\text{Throughput}(B=8) = 8 \times 420 = \mathbf{3,360 \text{ total tok/s}} \quad (\mathbf{6.72\times} \text{ throughput increase})$$

**Production Batching Selection**:
- **High Batch Size ($B = 8\text{--}32$)**: Optimal for offline document summarization, batch classification, and synthetic dataset generation where throughput (tokens per dollar) is the economic objective.
- **Low Batch Size ($B = 1\text{--}2$)**: Optimal for interactive voice assistants, live code completion, and streaming customer support where Time to First Token (TTFT) and Inter-Token Latency (ITL) must stay inside human perception bounds ($<50 \text{ ms}$). Batching raises aggregate throughput by holding each individual request slightly longer, which is the wrong trade when one request is all the user can see.

---

### Question 4: Cache Eviction for Long Conversations

A conversation on the running model reaches $S_{\max} = 1{,}024$ tokens and the pre-allocated buffer is full. The obvious move is to evict the oldest tokens and keep going.

**Answer these**:
1. What does FIFO eviction actually cost, and why is token 0 a special case rather than just the oldest one?
2. The cache grows only linearly in $S_{\max}$, so why do production systems cap the context window at all?
3. A clinical assistant must never forget a patient's allergies. Design a policy that keeps the cache bounded without losing that information.

**Worked Systems Analysis**:
1. **Context Loss from Eviction**: Evicting historical tokens drops earlier dialogue, user constraints, and instructions. Dropping the *initial* tokens does something worse than forgetting. Empirical work on StreamingLLM shows that the first tokens act as **attention sinks**, absorbing a large share of the softmax mass regardless of what they mean, because softmax must put its mass somewhere even when no key is a good match. Evict token 0 and the remaining keys have to absorb that mass, which drives perplexity up sharply.
2. **Context Window Limits**: Three separate ceilings, only one of which is the cache. Physical HBM capacity bounds the cache, prefill attention is quadratic in prompt length so a long prompt is expensive before a single token is generated, and rotary positional embeddings degrade when queried past the range they were trained on. The last of these is a quality limit, not a memory limit, which is why raising $S_{\max}$ alone does not buy usable context.
3. **Medical Chatbot Strategy**: A clinical assistant cannot drop patient history or allergies by naive FIFO. Combine three mechanisms:
   - **StreamingLLM Sink Retention**: Keep the first 4 tokens (the attention sinks) pinned in cache permanently.
   - **Context Summarization**: Periodically summarize older conversation turns with an auxiliary model call and inject the compressed clinical summary into the active prompt.
   - **Retrieval-Augmented Generation (RAG)**: Keep full transcripts in an external vector store and fetch the relevant history on demand, which moves unbounded state out of a fixed-size buffer entirely.

---

### Question 5: Multi-User Serving at Production Scale

Serve 10,000 concurrent conversations of a 13B-parameter model in `float16` with grouped-query attention: $L = 40$ layers, 40 query heads sharing $H = 8$ key/value heads, $d_k = 128$, $S_{\max} = 4{,}096$, $B = 1$ per conversation. Weights are `float16` as well.

**Answer these**:
1. What is the total serving memory, cache plus weights?
2. Can a single H100 (80 GB of HBM) host it? How many would it take?
3. Compare Strategy A, everything resident in HBM, against Strategy B, tiered offloading to host DRAM between turns.

**Worked Systems Analysis**:
- **Cache per Conversation** (note that $H$ is the 8 key/value heads, not the 40 query heads):
  $$2 \times 40 \times 1 \times 8 \times 4{,}096 \times 128 \times 2 = 671{,}088{,}640 \text{ B} \approx \mathbf{671.1 \text{ MB}}$$
  With full MHA's 40 key/value heads the same context would cost 3.36 GB, so GQA is already saving $5\times$ before any other technique is applied.
- **Total Cache Memory**:
  $$10{,}000 \times 671.1 \text{ MB} = \mathbf{6.71 \text{ TB}}$$
- **Model Parameters (13B in float16)**:
  $$13 \times 10^9 \times 2 \text{ bytes} = 26 \times 10^9 \text{ bytes} = \mathbf{26 \text{ GB}}$$
- **Total Serving Memory Required**:
  $$6{,}710.9 \text{ GB (Cache)} + 26 \text{ GB (Model)} = \mathbf{6{,}736.9 \text{ GB}} \approx \mathbf{6.74 \text{ TB}}$$
  The weights are 0.4% of the total. At this concurrency the model is a rounding error and the cache *is* the system.

**Systems Architectural Answers**:
1. **Single GPU Feasibility**: **Infeasible.** An NVIDIA H100 provides 80 GB of HBM3, so 6,736.9 GB exceeds one GPU by $84\times$. Resident storage alone needs $\lceil 6{,}736.9 / 80 \rceil = \mathbf{85}$ H100s, which is 11 eight-GPU nodes, and those GPUs are bought for capacity rather than for compute.
2. **Production Cache Management**:
   - **Prefix Caching**: Shared system prompts and tool definitions are hashed and stored once, then shared across thousands of sessions instead of being recomputed and re-stored per user.
   - **Continuous Paged Batching**: Sequences join and leave the running batch at every token step (iteration-level scheduling), so a finished request releases its blocks immediately instead of at the end of a fixed batch.
   - **Quantized KV Caching**: Storing keys and values at FP8 or INT4 instead of FP16 shrinks the cache by $2\times$ to $4\times$, which is the single largest lever available here because the cache dominates the budget.
3. **Memory Residency Trade-Off (A vs B)**:
   - **Strategy A (All in HBM)**: Minimal latency and no transfer overhead, but it reserves all 85 accelerators for residency, which is millions of dollars of hardware standing by to hold bytes rather than to compute.
   - **Strategy B (Tiered Offloading)**: An asynchronous two-tier hierarchy. Generating sequences live in HBM; when a turn completes, a background thread copies that conversation's cache over PCIe to host DRAM (or NVMe). When the user's next query arrives ten seconds later, the cache is prefetched back before generation begins. Since most conversations are idle most of the time, this keeps close to Strategy A's latency at a fraction of Strategy A's cost, which is the whole point of a memory hierarchy.
"""

# %% [markdown]
r"""
## ⭐ Aha Moment: KV Cache Avoids Recomputation

**What you built:** A KV Cache that stores key-value pairs to avoid redundant attention computation.

**Why it matters:** When generating text token-by-token, naive attention recomputes the same
K,V values for all previous tokens at each step. With KV caching, you compute once and reuse!
This is why ChatGPT responds so fast. It is not recomputing everything every token.

At context length n, caching reduces attention work per new token from O(n²) to O(n).
Total attention work across n generated tokens still grows as O(n²).
"""

# %%
def demo_memoization():
    """🎯 See KV cache store and reuse values."""
    print("🎯 AHA MOMENT: KV Cache Avoids Recomputation")
    print("=" * 45)

    # Create a cache for 2-layer transformer
    # (batch=1, max_seq=100, layers=2, heads=4, head_dim=64)
    cache = KVCache(batch_size=1, max_seq_len=100, num_layers=2,
                    num_heads=4, head_dim=64)

    # Simulate generating 5 tokens one at a time
    print("Generating tokens and caching K,V pairs...")
    for token_idx in range(5):
        # For each new token, compute K,V (shape: batch, heads, 1, head_dim)
        new_k = Tensor(rng.standard_normal((1, 4, 1, 64)))
        new_v = Tensor(rng.standard_normal((1, 4, 1, 64)))

        # Update cache for layer 0
        cache.update(0, new_k, new_v)
        cache.advance()  # Move to next position

    print(f"Cached K,V for {cache.seq_pos} tokens")

    # Retrieve all cached values
    k_all, v_all = cache.get(0)
    print(f"Retrieved: K{k_all.shape}, V{v_all.shape}")

    print("\n✨ Compute once, reuse for every later token!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_memoization()

# %% [markdown]
r"""
## 🚀 MODULE SUMMARY: Memoization

Congratulations! You have completed the primary inference optimization that makes production language model serving economically and computationally viable.

### Key Accomplishments

| Architectural Capability | Concrete Implementation | Verification Standard | Systems Impact |
|:---|:---|:---|:---|
| **Static Buffer Allocation** | `KVCache.__init__` | Contiguous float32 DRAM allocation | Zero dynamic memory allocation per token |
| **$\mathcal{O}(1)$ State Update** | `KVCache.update` | Direct slice assignment into DRAM | Eliminates array re-creation overhead |
| **Prefix Retrieval** | `KVCache.get` | Slices prefix up to write cursor | Enables single-query vector-matrix attention, at an $\mathcal{O}(t)$ copy per call |
| **Non-Invasive Stand-In** | `CachedAttention` | Wraps block attention transparently | Original layer restored intact by `disable_kv_cache` |
| **Autoregressive Loop** | `_cached_generate` | Two-phase prefill & decode loop | $\frac{S+1}{2}\times$ projection reduction (50.5$\times$ at $S=100$) |

### Systems Insights Discovered
- **Recomputation Elimination**: Caching $\mathbf{K}$ and $\mathbf{V}$ converts quadratic key/value projections from $\frac{S(S+1)}{2} \in \mathcal{O}(S^2)$ to $S \in \mathcal{O}(S)$, removing $4{,}950$ of $5{,}050$ projections at $S = 100$, which is $98\%$ of projection FLOPs.
- **Memory Capacity Trade-off**: Cache memory scales as $\mathcal{M} = 2 \cdot L \cdot B \cdot H \cdot S_{\max} \cdot d_k \cdot 4 \text{ bytes}$ in float32. At high batch sizes or long contexts the cache dwarfs the weights: a 12-layer model with a 125M-parameter weight set carries a 1.208 GB cache at $B = 8$, $S_{\max} = 2{,}048$, which is $241.6\%$ of its 500 MB of weights.
- **FLOP Counts Are Not Wall-Clock**: The module's own measurement separates the two. The attention score-pair ratio climbs from $8.8\times$ to about $69\times$ across generation lengths 10 to 100, while the measured speedup only moves from $1.2\times$ to a little over $2\times$, because a decode step is bound by the bytes of weights it must read, not by the arithmetic the cache eliminates.
- **Batching Amortization**: Because single-sequence decoding is memory-bandwidth bound, batching reuses each loaded weight matrix across several concurrent token vectors. At the running model's measured rates (500 tok/s at $B = 1$, 420 tok/s per sequence at $B = 8$) aggregate throughput rises $6.72\times$ for an $8\times$ linear rise in cache memory.

### Ready for Next Steps
You now own a working cache, and it is worth being precise about what it does not yet do, because Modules 19 and 20 measure and serve exactly this code:
- **Batch size is pinned at 1.** `_create_cache_storage` allocates with `batch_size=1`, and `reset()` rewinds the cursor without touching the batch axis, so batched serving needs a cache allocated for it up front.
- **Prefill is sequential.** A multi-token forward inside `cache.generation()` falls through to the original attention and writes nothing, so the prompt must be fed one token at a time. Production stacks prefill the whole prompt in one parallel pass.
- **No eviction and no paging.** The buffer is one contiguous allocation per layer for the full context window, so a full cache raises an error rather than evicting, and idle sequences hold their slots. PagedAttention, attention sinks, and tiered offloading all live above this layer.
- **Float32 only, and no head sharing.** Quantized keys and values (FP8, INT4) and grouped-query attention are the two largest levers on cache size, and neither is implemented here.
- **Retrieval copies.** `get()` wraps its slice in a `Tensor`, which copies the prefix, so retrieval costs $\mathcal{O}(t)$ bytes per layer per token.

Export with: `tito module complete 18`

**Next**: In Module 19 (Benchmarking), you will measure and benchmark end-to-end latency, memory bandwidth utilization, and token throughput across TinyTorch models!
"""
