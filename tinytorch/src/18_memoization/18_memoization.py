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

<img src="memoization_blueprint.svg" width="100%" alt="TinyTorch Framework Blueprint: Module 18 Memoization" />

### Architectural Roadmap

| Optimization Stage | Core Technique | Hardware & Algorithmic Focus | Primary Target |
|:---|:---|:---|:---|
| **14. Profiling** | Microsecond Benchmarks & Tracing | Profiler timer loops, Roofline bounds | Identify compute vs memory bottlenecks |
| **15. Quantization** | Symmetric/Asymmetric INT8 | 8-bit scale & zero-point arithmetic | 4× weight footprint & memory bus bandwidth |
| **16. Compression** | Magnitude Pruning & Distillation | Weight sparsity & student distillation | Redundant parameter elimination |
| **17. Acceleration** | SIMD GEMM, Fusion, `im2col` | Memory traffic elimination & systolic arrays | Kernel overhead & hardware utilization |
| **18. Memoization** *(Active)* | **Static KV Cache Buffers** | **$\mathcal{O}(1)$ decode steps & zero recomputation** | **Autoregressive decoding latency** |
| **19–20. Serving & Capstone** | End-to-End Pipeline Integration | End-to-end throughput & TTFT/ITL serving | Production inference deployment |

## 🎯 Learning Objectives
By the end of this module, you will:
1. Formulate memoization as a general systems optimization pattern: trading persistent DRAM memory capacity for $\mathcal{O}(S)$ computational reuse.
2. Mathematically derive the $\mathcal{O}(S^2) \to \mathcal{O}(S)$ reduction in key and value projection operations during autoregressive generation.
3. Construct a production-grade `KVCache` class featuring contiguous buffer pre-allocation, pointer advancement, and zero dynamic heap reallocation.
4. Implement a non-invasive `CachedAttention` wrapper that preserves forward compatibility and clean model encapsulation.
5. Benchmark memory capacity budgets against measured wall-clock speedups across varying batch sizes and context window limits.

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/18_memoization/memoization.ipynb`  
**Building Side:** Code exports to `tinytorch.perf.memoization`

<img src="memoization_source_card.svg" width="100%" alt="Source Code Mapping Card for Module 18 Memoization" />

```python
# How to use this module:
from tinytorch.perf.memoization import KVCache, enable_kv_cache, disable_kv_cache
```

## 📋 Module Dependencies

| Dependency Module | Exported Abstraction | Consumed Functional Role | Memory & Architectural Invariant |
|:---|:---|:---|:---|
| **Module 01 (`01_tensor`)** | `Tensor` | Contiguous N-D array storage & slicing | Pre-allocated float32 buffers without autograd overhead |
| **Module 12 (`12_attention`)** | `MultiHeadAttention` | Attention projections & head transformations | Splits `Q, K, V` into `(B, H, S, D)` and recombines output |
| **Module 13 (`13_transformers`)** | `GPT`, `TransformerBlock` | Autoregressive language model backbone | Non-invasive duck-typing wrapper for block attention |
| **Module 14 (`14_profiling`)** | `Profiler` | High-resolution microsecond timer | Quantifies latency scaling with and without cache |
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": false}
#| default_exp perf.memoization
#| export

import numpy as np
rng = np.random.default_rng(7)
import time
from typing import Tuple, Optional, Dict, List, Iterator
from contextlib import contextmanager

# Import TinyTorch components from previous modules
from tinytorch.core.tensor import Tensor

# Internal constants for memory calculations (not exported)
_BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
_MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion

# %% [markdown]
r"""
## 💡 Introduction: Why Memoization Matters for Transformers

Before implementing KV caching, let us profile naive autoregressive generation to isolate the fundamental computational bottleneck of modern transformer inference.

<img src="memoization_generation_overview.svg" width="100%" alt="Autoregressive KV Cache Generation Overview" />

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
    print("   • Doubling the sequence roughly quadruples the time once the arrays are big")

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

**Systems Takeaway**: DRAM is fast and spacious, while compute and memory bus bandwidth are hard hardware ceilings. Trading $\mathcal{O}(S)$ persistent cache memory capacity for an order-of-magnitude reduction in latency and FLOPs is the standard architecture of LLM serving.
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
5. **Zero-Copy Slicing**: Exposing valid token history $\mathbf{K}[:, :, :t, :]$ via views for immediate vector-matrix GEMV attention.

<img src="kv_cache_state_machine.svg" width="100%" alt="KV Cache Buffer State Machine" />

<img src="kv_cache_buffer_card.svg" width="100%" alt="KV Cache Buffer Card" />

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

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Calculate memory usage of the cache system.

        Returns:
            Dictionary with memory statistics in MB
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
    print(f"   Cache initialized: {mem_usage['total_mb']:.2f} MB")

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
## 🏗️ Cache-Aware Generation

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
2. Process prompt tokens at `start_pos=cache.seq_pos` and populate the cache
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
| **Llama 2 (7B)** | 32 | 32 | 128 | 4,096 | $\approx 2.15 \text{ GB}$ |
| **GPT-3 (175B)** | 96 | 96 | 128 | 2,048 | $\approx 19.33 \text{ GB}$ |
"""

# %% nbgrader={"grade": false, "grade_id": "cached-generation-step", "solution": false}
#| export
def _cached_generation_step(x, attention, cache_obj, layer_idx, mask=None):
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
    # Using .data (numpy) for inference-only operation (no gradients needed)
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
        scores = np.where(allowed != 0, scores, -np.inf)

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
### _create_cache_storage -- Validate Model and Allocate Cache

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
#| export
def _create_cache_storage(model):
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
        batch_size=1,  # Default to single sequence; can be reset for batch inference
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
### CachedAttention -- The Stand-In That Chooses the Path

`CachedAttention` takes the place of a block's attention layer while the cache is enabled. It keeps the original layer as `self.attention` and decides which execution path each forward call takes. Decoupling the **dispatch decision** here from the **numerical computation** in `_cached_generation_step` makes both independently testable.

### Stand-In Dispatch Decision Matrix

| Invocation Scope | Sequence Length (`x.shape[1]`) | Target Route | Executed Implementation | Computational Complexity |
|:---|:---|:---|:---|:---|
| **Outside Scope (`_generation_active=False`)** | Any ($S \ge 1$) | **Original Path** | `self.attention.forward(x, mask)` | Full causal attention, supports training autograd |
| **Inside Scope (`_generation_active=True`)** | Multi-token ($S > 1$) | **Original Path** | `self.attention.forward(x, mask)` | Parallel prompt processing / full-sequence fallback |
| **Inside Scope (`_generation_active=True`)** | Single token ($S == 1$) | **Cached Path** | `_cached_generation_step(x, ...)` | $\mathcal{O}(1)$ projection write + $\mathcal{O}(t)$ prefix attention |

Inside `cache.generation()`, the prompt tokens take the cached path sequentially. The initial prompt token encounters an empty cache, attends strictly to itself, and registers its $(K, V)$ representation into slot 0 for subsequent tokens. The exact token coordinate is synchronized via `start_pos=cache.seq_pos`, enabling positional embeddings to preserve temporal coordinates.
"""

# %% nbgrader={"grade": false, "grade_id": "kv-cached-attention", "solution": true}
#| export
class CachedAttention:
    """
    Stand-in for one block's attention layer while a KV cache is enabled.

    Holds the original attention layer and the cache. enable_kv_cache() puts one
    of these at block.attention; disable_kv_cache() puts the original back.
    """

    def __init__(self, attention, cache_obj, layer_idx):
        self.attention = attention      # the original MultiHeadAttention layer
        self.cache = cache_obj
        self.layer_idx = layer_idx

    def parameters(self):
        """The stand-in owns no weights of its own."""
        return self.attention.parameters()

    def forward(self, x, mask=None):
        """
        Route one call to the right path.

        TODO: Implement the two-path dispatch

        APPROACH:
        1. Outside cache.generation(), or for a full sequence, return
           self.attention.forward(x, mask) unchanged
        2. Inside that scope, route one new token through
           _cached_generation_step, including its optional attention mask

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

    def __call__(self, x, mask=None):
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
### _cached_generate -- Generation Loop with KV Cache

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
#| export
def _cached_generate(model, prompt_tokens, max_new_tokens, temperature, cache):
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
    # The last sampled token is returned, not fed back into the model.
    if len(prompt_tokens) + max_new_tokens > cache.max_seq_len:
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
### enable_kv_cache -- Composition: Wire Cache Into Model

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
def enable_kv_cache(model):
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
    print(f"⚡ KV Cache enabled for model!")
    print(f"   Architecture: {model.num_layers} layers × {model.num_heads} heads × {head_dim}D")
    print(f"   Memory: {cache.get_memory_usage()['total_mb']:.2f} MB")
    print(f"   Cache stored in: model._kv_cache")
    print()
    print(f"💡 To disable: call disable_kv_cache(model)")
    print()

    return cache
    ### END SOLUTION


# %% nbgrader={"grade": false, "grade_id": "kv-disable-cache", "solution": false}
#| export
def disable_kv_cache(model):
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

    Analyzes:
        - Tiny models (128D): ~0.12 MB
        - Small models (512D): ~2 MB
        - Medium models (768D): ~9 MB
        - Large models (1024D): ~32 MB

    Key Insight:
        Cache size is set by context length, not by parameter count. The
        ratio below compares it to one transformer block's parameters (~12·d²).

    Production Context:
        GPT-3 (96 layers, 96 heads, head_dim 128, 2048 context): ~18 GB per
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
        memory_per_layer = 2 * batch_size * seq_len * embed_dim * _BYTES_PER_FLOAT32 / _MB_TO_BYTES
        total_memory = memory_per_layer * num_layers

        # Model parameter memory: a transformer block has about 12·d² parameters
        # (4·d² in attention projections, 8·d² in the MLP)
        params_per_layer = 12 * embed_dim * embed_dim
        model_memory = params_per_layer * num_layers * _BYTES_PER_FLOAT32 / _MB_TO_BYTES

        overhead_pct = (total_memory / model_memory) * 100 if model_memory > 0 else 0

        print(f"{name:12s} | {total_memory:11.2f} MB | {memory_per_layer:8.2f} MB | {overhead_pct:6.1f}%")

    print()
    print("💡 Key Insights:")
    print("   • Cache memory scales linearly with sequence length (O(n))")
    print("   • Longer sequences require proportionally more cache memory")
    print("   • The ratio compares the cache to block parameters (~12·d² each); it grows with context")
    print()
    print("🚀 Production Context:")
    print("   • GPT-3 (96 layers, 96 heads, head_dim 128, 2048 context): ~18 GB per sequence in FP32")
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
        Speedup is SUPER-LINEAR with generation length because:
        - Longer sequences → more redundant computation without cache
        - Cache benefit compounds: saves O(n²) → O(n) at EVERY step

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
    print(f"✅ Memory tracking: {mem_info['total_mb']:.2f} MB for {mem_info['cache_tensors']} tensors")
    print()

    print("=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 18")


# %% [markdown]
r"""
## 🤔 ML Systems Reflection Questions

### Question 1: Cache Size Calculation

A 12-layer transformer has 12 attention heads per layer, 64-dimensional embeddings per head, maximum sequence length of 2048, and batch size of 8.

**Quantitative Derivation**:
- **One cache tensor shape**:
  $$\text{Shape} = (B, H, S_{\max}, d_k) = (8, 12, 2048, 64)$$
- **Elements per tensor**:
  $$8 \times 12 \times 2048 \times 64 = 12,582,912 \text{ elements}$$
- **Tensors per layer**:
  $$\text{Key Cache } \mathbf{K} + \text{Value Cache } \mathbf{V} = 2 \text{ tensors per layer}$$
- **Total across 12 layers**:
  $$12 \text{ layers} \times 2 \text{ tensors/layer} = 24 \text{ cache tensors}$$
- **Total elements**:
  $$24 \times 12,582,912 = 301,990,092 \text{ elements}$$
- **Memory footprint in float32 (4 bytes per element)**:
  $$\text{Total Bytes} = 301,990,092 \times 4 \text{ bytes} = 1,207,960,368 \text{ bytes}$$
  $$\text{Memory in MiB} = \frac{1,207,960,368}{1024^2} \approx \mathbf{1,152.0 \text{ MiB}} \quad (\approx 1.208 \text{ GB decimal})$$

**Follow-up Analysis (Overhead vs Model Parameters)**:
If this model has 125M parameters ($125 \times 10^6 \times 4 \text{ bytes} = 500 \text{ MB}$):
$$\text{Cache-to-Model Ratio} = \frac{1,152 \text{ MB}}{500 \text{ MB}} = \mathbf{230.4\%}$$
The KV cache occupies **more than double** the memory of the neural network weights themselves!

**Is this overhead acceptable?**
Yes, because without caching, decoding 2048 tokens at batch size 8 would require $\approx 2.1$ million redundant projections per sequence, causing severe inter-token latency spikes that breach serving SLAs. However, this massive overhead explains why modern architectures replace Multi-Head Attention (MHA) with **Multi-Query Attention (MQA)** or **Grouped-Query Attention (GQA)** (e.g., Llama 2/3, Mistral), which share key and value heads across $4\times$ to $8\times$ query heads, slashing cache memory footprint by $75\%\text{--}87.5\%$.

---

### Question 2: Speed vs Memory Trade-Off

Your `KVCache` eliminates the $\mathcal{O}(S^2)$ projection recomputation but reserves persistent memory for every active token.

**Systems Serving Analysis (1000 Concurrent Users)**:

| Metric | Without Cache (Stateless Forward) | With Cache (`KVCache`) |
|:---|:---|:---|
| **Inter-Token Latency (ITL)** | Degrades linearly $\mathcal{O}(t)$; requests timeout | Constant $\mathcal{O}(1)$ projection time per token |
| **FLOP Waste Rate** | $1 - \frac{2}{t+1} \to 99.8\%$ wasted recomputation | **0%** redundant projection FLOPs |
| **Memory Footprint (1000 users)** | Minimal ($\approx 0 \text{ MB}$ persistent KV memory) | $1,000 \times 100 \text{ MB} = \mathbf{100 \text{ GB DRAM}}$ |
| **Server Crash Risk (64 GB RAM)** | High CPU utilization, low memory pressure | **Out of Memory (OOM) Kernel Panic** |

**Systems Answers**:
1. **Chatbot Feasibility**: The trade-off is absolutely mandatory. For interactive chat, human users perceive delays $>100 \text{ ms}$ as sluggish. Without caching, generating token 1,000 takes $1000\times$ longer than token 1, violating real-time interactivity.
2. **64 GB RAM Failure**: Attempting to allocate 100 GB on a 64 GB host triggers OS page swapping to disk, dropping throughput by $1,000\times$, before invoking the Linux OOM Killer to terminate the model serving daemon.
3. **High-Concurrency Architecture**:
   - **PagedAttention (vLLM)**: Manage KV cache memory using virtual memory pages (e.g. 16 tokens per block), eliminating internal and external memory fragmentation and reclaiming 20–40% wasted buffer space.
   - **Tiered Memory Swapping**: Active generating sequences remain in fast GPU HBM. When a request enters human think-time (waiting for the user to type their next response), the KV cache is asynchronously DMA-transferred to host CPU RAM or local NVMe SSD over PCIe, freeing GPU HBM for other active decode steps.

---

### Question 3: Batch Inference Scaling

With `KVCache`, each sequence in a batch maintains its own dedicated $(K, V)$ tensor slices.

**Batch Scaling Calculations**:
- **Predicted Cache Memory (Batch 8)**:
  $$\mathcal{M}(8) = 8 \times 50 \text{ MB} = \mathbf{400 \text{ MB}} \quad (\text{Scales strictly linearly } \mathcal{O}(B))$$
- **Per-Sequence Generation Rate**:
  At batch size 1, inference is **memory-bandwidth bound**: reading the multi-gigabyte model weights from HBM to processor registers for a single token vector achieves only a tiny fraction of peak compute. Batching $B=8$ sequences allows the processor to reuse the loaded weight matrices across 8 token vectors simultaneously, transforming memory-bound GEMV into compute-dense GEMM. Consequently, per-sequence latency increases only marginally (e.g. from 500 to $\approx 420 \text{ tok/s}$).
- **Throughput Calculation**:
  $$\text{Throughput}(B=1) = 1 \times 500 = 500 \text{ total tok/s}$$
  $$\text{Throughput}(B=8) = 8 \times 420 = \mathbf{3,360 \text{ total tok/s}} \quad (\mathbf{6.72\times} \text{ throughput increase!})$$

**Production Batching Selection**:
- **High Batch Size ($B = 8\text{--}32$)**: Optimal for offline document summarization, batch classification, and synthetic dataset generation where throughput (tokens/dollar) is the primary economic objective.
- **Low Batch Size ($B = 1\text{--}2$)**: Optimal for interactive voice assistants, live developer code completion, and streaming customer support where Time to First Token (TTFT) and Inter-Token Latency (ITL) must remain under strict human perception bounds ($<50 \text{ ms}$).

---

### Question 4: Cache Eviction for Long Conversations

When a conversation exceeds `max_seq_len = 2048`, the pre-allocated buffer is full.

**Eviction Dynamics & Production Solutions**:
1. **Context Loss from Eviction**: Evicting historical tokens drops earlier dialogue, user constraints, and instructions. Crucially, dropping the initial prompt causes severe attention instability: empirical research (StreamingLLM) demonstrates that initial tokens act as **attention sinks**, absorbing significant softmax mass regardless of language semantics. Evicting token 0 triggers catastrophic perplexity explosion!
2. **Context Window Limits**: Production systems enforce strict token bounds due to physical GPU HBM limits, quadratic self-attention complexity during prefill, and degradation of rotary positional embeddings (RoPE) when queried beyond their pretraining context window.
3. **Medical Chatbot Strategy**: A clinical AI cannot drop patient medical history or allergies via naive FIFO eviction. The required systems solution is:
   - **StreamingLLM Sink Retention**: Keep the first 4 tokens (attention sinks) permanently in cache.
   - **Context Summarization**: Periodically summarize older conversation turns using an auxiliary background LLM call, injecting the compressed clinical summary into the active prompt.
   - **Retrieval-Augmented Generation (RAG)**: Store detailed conversational transcripts in an external vector database, fetching relevant medical history on-the-fly via similarity search.

---

### Question 5: Production Reality: Multi-User Serving

**Scale Calculation for 10,000 Concurrent Conversations**:
- **Total Cache Memory**:
  $$10,000 \times 200 \text{ MB} = 2,000,000 \text{ MB} = \mathbf{2,000 \text{ GB}} = \mathbf{2.0 \text{ TB}}$$
- **Model Parameters (13B float32)**:
  $$13 \times 10^9 \times 4 \text{ bytes} \approx 52 \times 10^9 \text{ bytes} = \mathbf{52 \text{ GB}}$$
- **Total Serving Memory Required**:
  $$2,000 \text{ GB (Cache)} + 52 \text{ GB (Model)} = \mathbf{2,052 \text{ GB}} \approx \mathbf{2.05 \text{ TB}}$$

**Systems Architectural Answers**:
1. **Single GPU Feasibility**: **Completely infeasible.** An NVIDIA H100 GPU provides 80 GB of HBM3. Storing 2,052 GB exceeds a single GPU by over **$25\times$**, requiring a distributed cluster of at least 32 H100 GPUs ($32 \times 80 = 2,560 \text{ GB}$).
2. **Production Cache Management**:
   - **Prefix Caching**: Common system prompts and developer tools are hashed and stored once, shared across thousands of user sessions simultaneously.
   - **Continuous Paged Batching**: Sequences are dynamically grouped and ungrouped at every single token step (iteration-level scheduling).
   - **Quantized KV Caching**: Quantizing keys and values from FP16 to FP8 or INT4 reduces cache volume by $2\times\text{--}4\times$ with negligible quality loss.
3. **Memory Residency Trade-Off (A vs B)**:
   - **Strategy A (All in HBM)**: Minimal latency, zero swapping overhead, but prohibitive hardware cost ($> \$300,000$ in GPUs).
   - **Strategy B (Tiered Offloading)**: Asynchronous two-tier memory hierarchy. Generating tokens reside in GPU HBM. Once a token turn completes, an asynchronous background thread copies the cache over PCIe to host CPU DRAM (or NVMe SSD). When the user submits their next query 10 seconds later, the cache is pre-fetched back into GPU HBM before generation begins. This delivers the speed of Option A at a fraction of the cost of Option B!
"""

# %% [markdown]
r"""
## ⭐ Aha Moment: KV Cache Avoids Recomputation

**What you built:** A KV Cache that stores key-value pairs to avoid redundant attention computation.

**Why it matters:** When generating text token-by-token, naive attention recomputes the same
K,V values for all previous tokens at each step. With KV caching, you compute once and reuse!
This is why ChatGPT responds so fast—it's not recomputing everything every token.

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
## 🚀 MODULE SUMMARY: KV Caching (Memoization)

Congratulations! You have completed the primary inference optimization that makes production language model serving economically and computationally viable.

### Systems Milestone Scorecard

| Architectural Capability | Concrete Implementation | Verification Standard | Systems Impact |
|:---|:---|:---|:---|
| **Static Buffer Allocation** | `KVCache.__init__` | Contiguous float32 DRAM allocation | Zero dynamic memory allocation per token |
| **$\mathcal{O}(1)$ State Update** | `KVCache.update` | Direct slice assignment into DRAM | Eliminates array re-creation overhead |
| **Prefix GEMV Retrieval** | `KVCache.get` | Slices prefix up to write cursor | Enables single-query vector-matrix attention |
| **Non-Invasive Stand-In** | `CachedAttention` | Wraps block attention transparently | Full forward/backward compatibility preserved |
| **Autoregressive Loop** | `_cached_generate` | Two-phase prefill & decode loop | $\frac{S+1}{2}\times$ projection FLOP reduction |

### Quantitative Systems Principles Established
- **Recomputation Elimination**: Caching $\mathbf{K}$ and $\mathbf{V}$ converts quadratic key/value projections from $\frac{S(S+1)}{2} \in \mathcal{O}(S^2)$ to $S \in \mathcal{O}(S)$, saving $98\%$ of projection FLOPs at $S=100$.
- **Memory Capacity Trade-off**: Cache memory scales as $\mathcal{M} = 2 \cdot L \cdot B \cdot H \cdot S_{\max} \cdot d_k \cdot 4 \text{ bytes}$. At high batch sizes or long contexts, KV cache memory footprint dwarfs the neural network weights.
- **Batching Amortization**: Because single-sequence decoding is memory-bandwidth bound, batching amortizes weight loading from HBM across multiple concurrent tokens, boosting aggregate serving throughput by $6\times\text{--}8\times$.

Export with: `tito module complete 18`

**Next**: In Module 19 (Benchmarking), you will measure and benchmark end-to-end latency, memory bandwidth utilization, and token throughput across TinyTorch models!
"""
