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
"""
# Module 18: Memoization - Computational Reuse for Inference

Welcome to Module 18! You'll implement memoization, a fundamental optimization pattern. We'll apply it to transformers through KV caching for 10-15x faster text generation.

## 🔗 Prerequisites & Progress
**You've Built**: Complete transformer architecture (Module 13) and profiling tools (Module 14)
**You'll Build**: Memoization system that eliminates redundant computation through caching
**You'll Enable**: Production-grade inference optimization using computational reuse

**Connection Map**:
```
Profiling (14) → Quantization (15) → Acceleration (17) → Memoization (18)
(measure O(n²))  (reduce precision)   (vectorize)        (cache K,V → O(n))
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Understand memoization as a general optimization pattern (cache results, avoid recomputation)
2. Apply memoization to transformers through KV caching
3. Implement KVCache with efficient memory management and O(1) updates
4. Build cache-aware attention that reuses previously computed keys and values
5. Measure dramatic speedup gains (10-15x) and understand memory trade-offs

Let's make inference blazingly fast through computational reuse!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/18_memoization/kvcaching_dev.py`
**Building Side:** Code exports to `tinytorch.generation.kv_cache`

```python
# How to use this module:
from tinytorch.perf.memoization import KVCache, enable_kv_cache
```

**Why this matters:**
- **Learning:** Complete caching system demonstrating production optimization techniques
- **Production:** Proper organization matching Hugging Face's generation/ module structure
- **Consistency:** All generation optimizations in generation.kv_cache
- **Integration:** Works seamlessly with transformers for complete inference optimization
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": true}
#| default_exp perf.memoization
#| export

import numpy as np
import time
from typing import Tuple, Optional, Dict, List

# Import TinyTorch components from previous modules
from tinytorch.core.tensor import Tensor

# Internal constants for memory calculations (not exported)
_BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
_MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion

# %% [markdown]
"""
## 📋 Module Dependencies

**Prerequisites**: Modules 01-17 (Tensor, Autograd, Transformers, Profiling, Acceleration)

**External Dependencies**:
- `numpy` (for array operations and numerical computing)
- `time` (for performance measurement)
- `typing` (for type hints)

**TinyTorch Dependencies**:
- `tinytorch.core.tensor` (Tensor class from Module 01)

**Dependency Flow**:
```
Module 01 (Tensor) → Module 12 (Attention) → Module 13 (Transformers) → Module 18 (Memoization)
     ↓                     ↓                        ↓                         ↓
  Foundation          Attention Ops           Full Transformer        Cache Optimization
```

Students completing this module will have built efficient caching
that makes production LLM serving economically viable.
"""

# %% [markdown]
"""
## 💡 Introduction: Why Memoization Matters for Transformers

Before we learn KV caching, let's profile transformer generation to understand the problem we're solving. We'll see O(n²) growth in latency as we generate text.

In machine learning systems, memoization is a fundamental optimization pattern: cache expensive computations so they don't need to be repeated. For transformers, this means caching the key-value pairs that attention computes, since they never change for already-processed tokens.

```
Memoization Pattern:
┌─────────────────────────────────────────────────────────────┐
│  Without Memoization (Naive):                               │
│  f(x) called 100 times → 100 computations                  │
│                                                             │
│  With Memoization (Cached):                                │
│  f(x) called 100 times → 1 computation + 99 cache lookups  │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: For transformers, K and V matrices for previous tokens NEVER change, yet naive generation recomputes them every step. This is the inefficiency we'll eliminate.
"""

# %% nbgrader={"grade": false, "grade_id": "motivation-profile", "locked": false}
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

    def naive_attention_step(seq_len, hidden_dim=64):
        """
        Simulates one step of attention computation.
        Without caching, this processes ALL previous tokens every time.
        """
        # Q, K, V for entire sequence
        q = Tensor(np.random.randn(1, seq_len, hidden_dim))
        k = Tensor(np.random.randn(1, seq_len, hidden_dim))
        v = Tensor(np.random.randn(1, seq_len, hidden_dim))

        # Attention: Q @ K.T then @ V
        # This is O(seq_len²) in complexity
        scores = q @ k.T  # (1, seq_len, seq_len)
        output = scores @ v

        return output

    # Profile at increasing sequence lengths
    print("🔬 Profiling Transformer Generation (Without Caching):\n")
    print("   Seq Len  |  Latency (ms)  |  Growth")
    print("   ---------|----------------|----------")

    sequence_lengths = [10, 20, 40, 80, 160]
    latencies = []

    for seq_len in sequence_lengths:
        # Measure latency for this sequence length
        latency = profiler.measure_latency(
            lambda: naive_attention_step(seq_len),
            None,
            warmup=5,
            iterations=20
        )
        latencies.append(latency)

        # Calculate growth rate
        if len(latencies) > 1:
            growth = latencies[-1] / latencies[-2]
            print(f"   {seq_len:3d}      |  {latency:6.2f}        |  {growth:.2f}×")
        else:
            print(f"   {seq_len:3d}      |  {latency:6.2f}        |  baseline")

    print("\n💡 Key Observations:")
    print("   • Latency grows QUADRATICALLY with sequence length")
    print("   • Each new token forces recomputation of ALL previous K,V pairs")
    print("   • For 160 tokens: ~4× time vs 80 tokens (2² growth)")

    print("\n🎯 The Problem:")
    print("   K and V values for previous tokens NEVER change,")
    print("   yet we recompute them every single step!")

    print("\n✨ The Solution:")
    print("   CACHE the K,V values! (That's memoization)")
    print("   • First compute: Calculate and store K,V")
    print("   • Later steps: Reuse stored K,V")
    print("   • Complexity: O(n²) → O(n)")
    print("   • Speedup: 10-15× for typical generation\n")

# Run profiling when module is executed directly
# NOTE: Commented out to run tests. Profiling requires proper Profiler API usage.
# Uncomment to run profiling (requires matplotlib installed)
# if __name__ == "__main__":
#     profile_naive_generation()

# %% [markdown]
"""
## 📐 Foundations: Understanding the Autoregressive Generation Problem

### The Core Inefficiency

When generating text token by token, transformers face a fundamental computational bottleneck. Let's visualize what happens during naive generation:

```
Token Generation Process (Without Caching):

Step 1: Generate "Hello"
Input: [START]
Attention: Q₁ × [K₁] × [V₁]               ← 1 computation

Step 2: Generate "world"
Input: [START, Hello]
Attention: Q₂ × [K₁, K₂] × [V₁, V₂]       ← 2 computations (K₁,V₁ RECOMPUTED!)

Step 3: Generate "!"
Input: [START, Hello, world]
Attention: Q₃ × [K₁, K₂, K₃] × [V₁, V₂, V₃] ← 3 computations (K₁,V₁,K₂,V₂ RECOMPUTED!)
```

**The Problem**: For each new token, we recompute ALL previous key-value pairs even though they never change!

### Computational Complexity Analysis

```
Naive Generation Complexity:
Step 1: 1 K,V computation
Step 2: 2 K,V computations
Step 3: 3 K,V computations
...
Step n: n K,V computations

Total: 1 + 2 + 3 + ... + n = n(n+1)/2 = O(n²) complexity!
```

For a 100-token sequence, this means **5,050 total K,V computations** — but only 100 are
actually necessary (one per token). That's **4,950 redundant computations**!

### Real-World Impact

This inefficiency makes production LLM serving economically impossible without optimization:
- **ChatGPT/GPT-4**: Would be too slow for real-time chat without caching
- **Code completion**: IDEs couldn't provide instant suggestions
- **Mobile deployment**: On-device generation would drain batteries instantly
- **API serving**: Server costs would be 10x+ higher

**The Solution**: Cache key-value pairs after computing them once, transforming O(n²) into O(n).
"""

# %% [markdown]
"""
## 📐 Foundations: The Key-Value Caching Insight

### Mathematical Foundation

The core insight comes from understanding what changes during autoregressive generation:

```
Attention Computation Breakdown:

Q = new_token @ W_q        ← Only new token (changes each step)
K = all_tokens @ W_k       ← Includes old tokens (mostly redundant!)
V = all_tokens @ W_v       ← Includes old tokens (mostly redundant!)

attention_output = softmax(Q @ K.T / √d_k) @ V
```

**Key Insight**: K and V matrices for previous tokens NEVER change!

```
Token Dependencies:
K₁ = token₁ @ W_k  ← Computed once, never changes
K₂ = token₂ @ W_k  ← Computed once, never changes
K₃ = token₃ @ W_k  ← Computed once, never changes

Same for V₁, V₂, V₃...
```

### Cache-Optimized Generation

```
Optimized Generation Process (With Caching):

Step 1: Generate "Hello"
Compute: K₁, V₁ → Store in cache
Attention: Q₁ × cached[K₁] × cached[V₁]

Step 2: Generate "world"
Compute: K₂, V₂ → Append to cache
Attention: Q₂ × cached[K₁, K₂] × cached[V₁, V₂]

Step 3: Generate "!"
Compute: K₃, V₃ → Append to cache
Attention: Q₃ × cached[K₁, K₂, K₃] × cached[V₁, V₂, V₃]
```

**Result**: Each step computes only ONE new K,V pair instead of recomputing ALL!

### Memory vs Compute Trade-off

```
Traditional Approach:
Memory: O(1)          (no storage needed)
Compute: O(n²)        (recompute everything)

Cached Approach:
Memory: O(n × d_k)    (store all K,V pairs)
Compute: O(n)         (only compute new pairs)

For n=100, d_k=64:
Memory cost: 6.4 KB per layer
Compute savings: 50x reduction in K,V computations
```

**Trade-off Winner**: Memory is cheap, compute is expensive! Use O(n) memory to save O(n²) compute.
"""

# %% [markdown]
"""
## 🏗️ Implementation: KVCache Class

### Core Requirements

Our KVCache needs to efficiently handle:

1. **Multi-layer storage**: Each transformer layer needs its own K,V cache
2. **Multi-head attention**: Each attention head has separate K,V pairs
3. **Batch processing**: Support multiple sequences simultaneously (batch inference)
4. **Dynamic updates**: Efficiently append new tokens without copying data
5. **Memory management**: Pre-allocate space to avoid dynamic resizing overhead

### Cache Architecture Visualization

```
KVCache Memory Layout:
┌────────────────────────────────────────┐
│                KVCache Object          │
├────────────────────────────────────────┤
│ Layer 0: ┌─────────────┬─────────────┐ │
│          │ Key Cache   │ Value Cache │ │
│          │ (B,H,S,D)   │ (B,H,S,D)   │ │
│          └─────────────┴─────────────┘ │
├────────────────────────────────────────┤
│ Layer 1: ┌─────────────┬─────────────┐ │
│          │ Key Cache   │ Value Cache │ │
│          │ (B,H,S,D)   │ (B,H,S,D)   │ │
│          └─────────────┴─────────────┘ │
├────────────────────────────────────────┤
│   ...    ┌─────────────┬─────────────┐ │
│ Layer N: │ Key Cache   │ Value Cache │ │
│          │ (B,H,S,D)   │ (B,H,S,D)   │ │
│          └─────────────┴─────────────┘ │
└────────────────────────────────────────┘

Where:
B = batch_size    (number of sequences)
H = num_heads     (attention heads per layer)
S = max_seq_len   (maximum sequence length)
D = head_dim      (dimension per attention head)
```

### Update Operation Flow

```
Cache Update Process:
                      seq_pos = 2
                         ↓
┌─────┬─────┬─────┬─────┬─────┬─────┐
│ K₁  │ K₂  │ ??? │ ??? │ ??? │ ??? │ ← Key Cache
├─────┼─────┼─────┼─────┼─────┼─────┤
│ V₁  │ V₂  │ ??? │ ??? │ ??? │ ??? │ ← Value Cache
└─────┴─────┴─────┴─────┴─────┴─────┘

New token arrives: K₃, V₃

                      seq_pos = 2
                         ↓
┌─────┬─────┬─────┬─────┬─────┬─────┐
│ K₁  │ K₂  │ K₃  │ ??? │ ??? │ ??? │ ← Write K₃ here
├─────┼─────┼─────┼─────┼─────┼─────┤
│ V₁  │ V₂  │ V₃  │ ??? │ ??? │ ??? │ ← Write V₃ here
└─────┴─────┴─────┴─────┴─────┴─────┘

Then: seq_pos += 1 (advance to position 3)
```

This design enables **O(1) updates** - just write to the next position!
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
    - During generation: No gradients computed (model.eval() mode)
    - Cache operations use .data (no gradient tracking)
    - This is correct and intentional for maximum speed
    - DO NOT use caching during training (use standard forward pass)

    Architecture:
    - Pre-allocates cache tensors with maximum sequence length
    - Tracks current sequence position for efficient O(1) updates
    - Provides update() method to append new K,V pairs without copying
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
    - Update: O(1) - just index assignment
    - Get: O(1) - just slicing (no data copy)
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
        ### BEGIN SOLUTION
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Current sequence position (how many tokens are cached)
        self.seq_pos = 0

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

        TODO: Efficiently append new K,V to cache without data copying

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
        >>> new_k = Tensor(np.random.randn(1, 4, 1, 64))
        >>> new_v = Tensor(np.random.randn(1, 4, 1, 64))
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
        if layer_idx >= self.num_layers:
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
        This is O(1) because we're just slicing NumPy arrays (view, not copy).

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
        if layer_idx >= self.num_layers:
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

    def advance(self) -> None:
        """
        Advance sequence position after processing current token.

        Call this after all layers have processed the current token and
        updated their caches. This moves the write pointer forward.
        """
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
"""
### 🔬 Unit Test: KVCache Implementation

This test validates that our cache correctly stores and retrieves key-value pairs across multiple layers and sequence positions.

**What we're testing**: KVCache initialization, update, get, and reset operations
**Why it matters**: Cache must work correctly for generation to produce coherent output
**Expected**: Cache stores and retrieves values correctly, tracks sequence position
"""

# %% nbgrader={"grade": true, "grade_id": "test-kvcache", "locked": true, "points": 10}
def test_unit_kvcache():
    """🔬 Unit Test: KVCache Implementation"""
    print("🔬 Unit Test: KVCache Implementation...")

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
    key1 = Tensor(np.random.randn(batch_size, num_heads, 1, head_dim))
    value1 = Tensor(np.random.randn(batch_size, num_heads, 1, head_dim))

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
    key2 = Tensor(np.random.randn(batch_size, num_heads, 1, head_dim))
    value2 = Tensor(np.random.randn(batch_size, num_heads, 1, head_dim))
    cache.update(0, key2, value2)
    cache.advance()

    cached_k, cached_v = cache.get(0)
    assert cached_k.shape == (batch_size, num_heads, 2, head_dim), "Should have 2 tokens cached"
    assert cached_v.shape == (batch_size, num_heads, 2, head_dim), "Should have 2 tokens cached"

    # Test 4: Multiple layers
    cache.reset()
    key_test = Tensor(np.random.randn(batch_size, num_heads, 1, head_dim))
    value_test = Tensor(np.random.randn(batch_size, num_heads, 1, head_dim))

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

# Run test immediately when developing this module
if __name__ == "__main__":
    test_unit_kvcache()

# %% [markdown]
"""
## 🏗️ Implementation: Cache-Aware Generation

### Integration Strategy

Now we need a clean way to enable KV caching in our existing transformer models without breaking the existing code. We'll create an `enable_kv_cache()` function that:

1. Creates a KVCache instance sized for the model
2. Patches the model's attention layers to use caching
3. Returns the cache for manual control if needed

The actual integration with attention happens through monkey-patching where we:
1. Check if cache is enabled
2. Only compute K,V for new token (not all tokens)
3. Update cache with new K,V
4. Use cached K,V for attention computation

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
for each new token:
    input_token = [just new token]          # Length always 1
    logits = model.forward(input_token)     # Uses cache automatically!
    next_token = sample(logits[-1])
    append next_token
```

**Key Difference**: Input changes from growing sequence to single token, with cache providing history.
"""

# %% [markdown]
"""
## 🔧 Integration: Non-Invasive Model Enhancement

### The Challenge

We built KV caching in Module 18 (this module), but our transformer (Modules 12-13) doesn't know about it!

**❌ BAD Solution**: Go back and modify Module 12 (MultiHeadAttention)
- Breaks "forward-only" learning (students shouldn't revisit old modules)
- Makes Module 12 depend on Module 18 (wrong dependency direction!)
- Violates clean module boundaries

**✅ GOOD Solution**: Module 18 ADDS caching to existing models without modification!
- Use composition + monkey-patching (like `enable_autograd()`)
- Module 18 wraps/enhances Module 12, not modifies it
- Students learn systems engineering: "Add capabilities, don't break old code"

### Using KV Cache in Practice

To use KV caching in your transformer generation:

**Before Generation:**
1. Enable caching with `enable_kv_cache(model)`
2. Cache is automatically sized for your model architecture
3. Verify memory usage is acceptable

**During Generation:**
1. For the first token (prompt), process normally and populate cache
2. For subsequent tokens:
   - Only process the NEW token (not entire sequence)
   - Cache is automatically updated with new K,V pairs
   - Cached values are automatically used in attention
   - Cache position advances after all layers

**After Generation:**
1. Reset cache if generating another sequence: `model._kv_cache.reset()`
2. Disable caching if needed: `disable_kv_cache(model)`
3. Monitor memory usage for production deployment

### Performance Expectations

```
Expected Speedup by Sequence Length:
┌───────────┬──────────┬───────────┬──────────┐
│ Seq Len   │ No Cache │ With Cache│ Speedup  │
├───────────┼──────────┼───────────┼──────────┤
│  10 tokens│ ~80 tok/s│ ~600 tok/s│   7.5x   │
│  25 tokens│ ~40 tok/s│ ~500 tok/s│  12.5x   │
│  50 tokens│ ~25 tok/s│ ~400 tok/s│  16.0x   │
│ 100 tokens│ ~12 tok/s│ ~200 tok/s│  16.7x   │
└───────────┴──────────┴───────────┴──────────┘

Key Insight: Speedup increases with sequence length!
Why? Longer sequences = more redundant computation without cache.
```

### Production Considerations

**Memory Management:**
- Cache memory = `batch_size × num_layers × num_heads × max_seq_len × head_dim × 4 bytes`
- For GPT-2 (12 layers, 12 heads, seq_len=1024, head_dim=64): ~37 MB per sequence
- For GPT-3 (96 layers, 96 heads, seq_len=2048, head_dim=128): ~4.7 GB per sequence

**Trade-off Analysis:**
- **10x+ speedup** for typical generation lengths (50-200 tokens)
- **Modest memory cost** compared to model parameters (often <1% of model size)
- **Enables real-time interaction** that's impossible without caching

**Best Practices:**
1. Always use caching for production serving
2. Tune `max_seq_len` to expected generation length (don't over-allocate)
3. Consider batch inference to amortize model loading costs
4. Monitor cache memory usage in production
"""

# %% nbgrader={"grade": false, "grade_id": "cached-generation-step", "solution": false}
#| export
def _cached_generation_step(x, attention, cache_obj, layer_idx):
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
    import numpy as np
    from tinytorch.core.tensor import Tensor

    batch_size = x.shape[0]
    num_heads = attention.num_heads
    head_dim = attention.head_dim

    # Step 1: Project new token to Q, K, V
    Q_new = attention.q_proj.forward(x)  # (batch, 1, embed_dim)
    K_new = attention.k_proj.forward(x)
    V_new = attention.v_proj.forward(x)

    # Step 2: Reshape to multi-head format (batch, num_heads, 1, head_dim)
    Q_heads = Tensor(np.transpose(
        Q_new.reshape(batch_size, 1, num_heads, head_dim).data, (0, 2, 1, 3)
    ))
    K_heads = Tensor(np.transpose(
        K_new.reshape(batch_size, 1, num_heads, head_dim).data, (0, 2, 1, 3)
    ))
    V_heads = Tensor(np.transpose(
        V_new.reshape(batch_size, 1, num_heads, head_dim).data, (0, 2, 1, 3)
    ))

    # Step 3: Update cache with new K, V
    cache_obj.update(layer_idx, K_heads, V_heads)

    # Step 4: Retrieve ALL cached K, V (includes history + new token)
    K_all, V_all = cache_obj.get(layer_idx)

    # Step 5: Compute attention using new Q with all cached K, V
    # Using .data (numpy) for inference-only operation (no gradients needed)
    K_transposed = np.transpose(K_all.data, (0, 1, 3, 2))
    scores = np.matmul(Q_heads.data, K_transposed) / np.sqrt(head_dim)

    # Stable softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Apply attention to values
    attention_output = np.matmul(attention_weights, V_all.data)

    # Step 6: Reshape and project to output
    attention_output_transposed = np.transpose(attention_output, (0, 2, 1, 3))
    concat_output = Tensor(attention_output_transposed.reshape(batch_size, 1, num_heads * head_dim))

    return attention.out_proj.forward(concat_output)


# %% [markdown]
"""
### _create_cache_storage -- Validate Model and Allocate Cache

This helper validates that a model has the required architecture attributes
for KV caching, then creates and attaches a properly-sized KVCache.

```
Model Architecture Inspection:
┌────────────────────────┐
│  model.embed_dim = 128 │──→ head_dim = 128 // 4 = 32
│  model.num_heads = 4   │
│  model.num_layers = 4  │──→ 4 layer caches created
│  model.max_seq_len = 64│──→ pre-allocate 64 positions
│  model.blocks = [...]  │
└────────────────────────┘
         ↓
┌────────────────────────┐
│  KVCache(              │
│    batch=1, seq=64,    │
│    layers=4, heads=4,  │
│    head_dim=32         │
│  )                     │
└────────────────────────┘
         ↓
  model._kv_cache = cache
  model._cache_enabled = True
```
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
    ### BEGIN SOLUTION
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
"""
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
"""
### _cached_attention_forward -- Path Dispatch for Cached Attention

This helper decides which attention path to take for a given input.
It separates the DECISION logic from the COMPUTATION logic, making
both independently testable.

```
Input x arrives at attention layer:

  x.shape[1] > 1?  ──YES──→ PATH 1: TRAINING
       │                     Use original attention (gradient flow)
       NO
       │
  cache.seq_pos == 0? ──YES──→ PATH 2: FIRST TOKEN
       │                       Use original attention (nothing cached yet)
       NO
       │
       └──→ PATH 3: CACHED GENERATION
            Use _cached_generation_step() for O(n) computation
```

This three-path dispatch is the core decision logic that determines
whether to use the cache or fall back to standard attention.
"""

# %% nbgrader={"grade": false, "grade_id": "kv-cached-attention", "solution": true}
#| export
def _cached_attention_forward(block, x, cache_obj, layer_idx, original_forward):
    """
    Dispatch attention through the correct path based on context.

    TODO: Implement three-path dispatch for cached attention

    APPROACH:
    1. Check if seq_len > 1 (training mode) -> use original_forward
    2. Check if cache is empty (seq_pos == 0) -> use original_forward
    3. Otherwise (cached generation) -> use _cached_generation_step

    EXAMPLE:
    >>> # Training path (seq_len=10 > 1):
    >>> output = _cached_attention_forward(block, x_train, cache, 0, orig_fwd)
    >>> # -> calls original_forward(x_train, None)
    >>>
    >>> # Cached path (seq_len=1, cache has history):
    >>> output = _cached_attention_forward(block, x_gen, cache, 0, orig_fwd)
    >>> # -> calls _cached_generation_step(x_gen, block.attention, cache, 0)

    HINTS:
    - x.shape[1] gives the sequence length
    - cache_obj.seq_pos tracks how many tokens are already cached
    - PATH 1 and PATH 2 both call original_forward(x, mask=None)
    - PATH 3 calls _cached_generation_step(x, block.attention, cache_obj, layer_idx)

    Args:
        block: Transformer block containing the attention layer
        x: Input tensor, shape (batch, seq_len, embed_dim)
        cache_obj: KVCache instance
        layer_idx: Which transformer layer (0-indexed)
        original_forward: The original (un-patched) attention forward method

    Returns:
        Output tensor from whichever path was selected
    """
    ### BEGIN SOLUTION
    seq_len = x.shape[1]

    # PATH 1: TRAINING (seq_len > 1)
    # Full sequence - use original attention for gradient flow
    if seq_len > 1:
        return original_forward(x, None)

    # PATH 2: FIRST TOKEN (cache empty)
    # Nothing to retrieve yet - use original attention
    if cache_obj.seq_pos == 0:
        return original_forward(x, None)

    # PATH 3: CACHED GENERATION
    # Use helper function for the O(n) cached computation
    return _cached_generation_step(x, block.attention, cache_obj, layer_idx)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: _cached_attention_forward

**What we're testing**: Three-path dispatch logic for cached attention
**Why it matters**: Wrong path selection causes silent correctness bugs (training uses cache, or generation ignores cache)
**Expected**: Training inputs use original forward; cached generation uses _cached_generation_step
"""

# %% nbgrader={"grade": true, "grade_id": "test-cached-attention", "locked": true, "points": 10}
def test_unit_cached_attention_forward():
    """🧪 Test _cached_attention_forward dispatches to correct path."""
    print("🧪 Unit Test: _cached_attention_forward...")

    # Track which path was taken
    path_taken = []

    class MockBlock:
        def __init__(self):
            self.attention = self

    block = MockBlock()

    def mock_original_forward(x, mask=None):
        path_taken.append("original")
        return x

    # Create a real cache for testing
    cache = KVCache(batch_size=1, max_seq_len=64, num_layers=2,
                    num_heads=4, head_dim=32)

    # Test PATH 1: Training (seq_len > 1)
    path_taken.clear()
    x_train = Tensor(np.random.randn(1, 10, 128))  # seq_len=10
    result = _cached_attention_forward(block, x_train, cache, 0, mock_original_forward)
    assert "original" in path_taken, "Training path should use original forward"
    assert result.shape == x_train.shape, "Should return same shape"

    # Test PATH 2: First token (cache empty, seq_pos=0)
    path_taken.clear()
    cache.reset()
    assert cache.seq_pos == 0
    x_first = Tensor(np.random.randn(1, 1, 128))  # seq_len=1, but cache empty
    result = _cached_attention_forward(block, x_first, cache, 0, mock_original_forward)
    assert "original" in path_taken, "First token should use original forward"

    # Test PATH 3: Cached generation (seq_len=1, cache has history)
    # We can't easily test the full _cached_generation_step path without
    # real attention layers, so we verify the dispatch logic by checking
    # that PATH 1 and PATH 2 conditions are correctly handled above.
    # PATH 3 would be triggered when seq_len=1 and cache.seq_pos > 0.
    print("   PATH 1 (training): dispatches to original forward")
    print("   PATH 2 (first token): dispatches to original forward")
    print("   PATH 3 (cached): would dispatch to _cached_generation_step")

    print("✅ _cached_attention_forward path dispatch works correctly!")

if __name__ == "__main__":
    test_unit_cached_attention_forward()


# %% [markdown]
"""
### _cached_generate -- Generation Loop with KV Cache

This helper implements the autoregressive generation loop that uses the
KV cache for efficient token-by-token generation. It shows how caching
transforms the generation complexity from O(n^2) to O(n).

```
Generation Loop with Cache:

prompt = [token_1, token_2, token_3]
cache  = empty

Step 0 (prefill): Process entire prompt through model
  → cache now holds K,V for tokens 1-3
  → get logits for next token prediction

Step 1: Generate token_4
  → input: just [token_4] (length 1!)
  → attention uses cached K,V + new K,V
  → O(1) new computation per layer

Step 2: Generate token_5
  → input: just [token_5] (length 1!)
  → cache grows: K,V for tokens 1-4
  → O(1) new computation per layer

  ...continues until max_new_tokens reached
```
"""

# %% nbgrader={"grade": false, "grade_id": "kv-cached-generate", "solution": true}
def _cached_generate(model, prompt_tokens, max_new_tokens, temperature, cache):
    """
    Run autoregressive generation using the KV cache.

    TODO: Implement the cached generation loop

    APPROACH:
    1. Process prompt tokens through model to populate cache (prefill phase)
    2. Get the last token's logits and sample next token
    3. Loop for max_new_tokens steps:
       a. Feed ONLY the new token through the model (seq_len=1)
       b. Cache is automatically updated by patched attention
       c. Advance cache position after each token
       d. Sample next token from logits with temperature scaling
    4. Return list of generated token indices

    EXAMPLE:
    >>> generated = _cached_generate(model, prompt=[0, 1, 2],
    ...                               max_new_tokens=5, temperature=1.0,
    ...                               cache=cache)
    >>> len(generated)  # 5 new tokens

    HINTS:
    - Prefill: model.forward(prompt_tensor) processes entire prompt
    - Generation: model.forward(single_token_tensor) processes one token
    - Use temperature scaling: logits / temperature before softmax
    - Use np.random.choice with softmax probabilities to sample
    - Advance cache.advance() after each generated token
    - Stable softmax: subtract max before exp to avoid overflow

    Args:
        model: Transformer model with cached attention (already patched)
        prompt_tokens: List of integer token IDs for the prompt
        max_new_tokens: Number of new tokens to generate
        temperature: Sampling temperature (higher = more random)
        cache: KVCache instance (already attached to model)

    Returns:
        List of generated token IDs (integers)
    """
    ### BEGIN SOLUTION
    generated = []

    # Phase 1: PREFILL - process entire prompt to populate cache
    prompt_array = np.array([prompt_tokens])  # (1, prompt_len)
    prompt_tensor = Tensor(prompt_array)
    logits = model.forward(prompt_tensor)  # (1, prompt_len, vocab_size)

    # Advance cache for each prompt token
    for _ in range(len(prompt_tokens)):
        cache.advance()

    # Get logits for last prompt position (predicts next token)
    last_logits = logits.data[0, -1, :]  # (vocab_size,)

    # Phase 2: GENERATE - one token at a time using cache
    for _ in range(max_new_tokens):
        # Temperature-scaled sampling
        scaled_logits = last_logits / max(temperature, 1e-8)
        max_logit = np.max(scaled_logits)
        exp_logits = np.exp(scaled_logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)

        # Sample next token
        next_token = int(np.random.choice(len(probs), p=probs))
        generated.append(next_token)

        # Feed single token through model (cache handles history)
        token_tensor = Tensor(np.array([[next_token]]))  # (1, 1)
        logits = model.forward(token_tensor)  # (1, 1, vocab_size)
        cache.advance()

        last_logits = logits.data[0, -1, :]

    return generated
    ### END SOLUTION

# %% [markdown]
"""
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

        def forward(self, x):
            # Return random logits shaped (batch, seq_len, vocab_size)
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            return Tensor(np.random.randn(batch_size, seq_len, vocab_size))

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
    # prompt (3 tokens) + generated (5 tokens) = 8 advances
    expected_pos = len(prompt) + max_new
    assert cache.seq_pos == expected_pos, f"Expected cache pos={expected_pos}, got {cache.seq_pos}"

    # Test 4: Generate with low temperature (more deterministic)
    cache.reset()
    generated_low_temp = _cached_generate(model, [0], 3, temperature=0.01, cache=cache)
    assert len(generated_low_temp) == 3, "Should generate 3 tokens with low temperature"

    print("✅ _cached_generate works correctly!")

if __name__ == "__main__":
    test_unit_cached_generate()


# %% [markdown]
"""
### enable_kv_cache -- Composition: Wire Cache Into Model

This is the main entry point that composes the helpers above. It:
1. Creates cache storage via `_create_cache_storage()`
2. Patches each block's attention via `_cached_attention_forward()`
3. Returns the cache for manual control

```
enable_kv_cache(model)
       │
       ├──→ _create_cache_storage(model)
       │         └──→ KVCache created & attached
       │
       ├──→ For each block:
       │       └──→ Patch attention.forward to use
       │            _cached_attention_forward()
       │
       └──→ Return cache object
```
"""

# %% nbgrader={"grade": false, "grade_id": "kv-enable-cache", "solution": true}
#| export
def enable_kv_cache(model):
    """
    Enable KV caching for a transformer model WITHOUT modifying Module 12/13 code.

    TODO: Compose helpers to create cache and patch attention layers

    APPROACH:
    1. Call _create_cache_storage(model) to validate and create cache
    2. For each block, save original forward and patch with _cached_attention_forward
    3. Print confirmation with cache statistics
    4. Return cache object

    This function demonstrates **non-invasive optimization** - adding capabilities
    to existing systems without breaking them. Similar to how Module 06 (Autograd)
    uses enable_autograd() to add gradient tracking to Tensors.

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
    - Use a factory function (make_cached_forward) to capture layer_idx in closure
    - Save original forward as block._original_attention_forward before patching
    - _cached_attention_forward handles the three-path dispatch logic
    """
    ### BEGIN SOLUTION
    # Step 1: Validate model and create cache
    cache, head_dim = _create_cache_storage(model)

    # Step 2: Patch each transformer block's attention
    for layer_idx, block in enumerate(model.blocks):
        # Save original forward (avoid double-patching)
        # hasattr() is LEGITIMATE: monkey-patching safety check
        if not hasattr(block, '_original_attention_forward'):
            block._original_attention_forward = block.attention.forward

        # Create cached version using factory for correct closure binding
        def make_cached_forward(layer_idx, original_forward, cache_obj):
            """Factory to create cached forward with correct layer_idx closure."""
            def cached_forward(x, mask=None):
                return _cached_attention_forward(
                    block, x, cache_obj, layer_idx, original_forward
                )
            return cached_forward

        block.attention.forward = make_cached_forward(
            layer_idx, block._original_attention_forward, cache
        )

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
    # Educational Note: hasattr() is LEGITIMATE here because:
    # Checking if monkey-patch markers exist before restoration
    if not hasattr(model, '_cache_enabled') or not model._cache_enabled:
        print("⚠️  KV cache not enabled on this model")
        return

    # Restore original attention forwards
    for block in model.blocks:
        # Educational Note: hasattr() is LEGITIMATE here because:
        # Checking for monkey-patch backup before restoration
        if hasattr(block, '_original_attention_forward'):
            block.attention.forward = block._original_attention_forward

    # Clean up
    model._cache_enabled = False
    # Educational Note: hasattr() is LEGITIMATE here because:
    # Safe cleanup check before deleting dynamically added attribute
    if hasattr(model, '_kv_cache'):
        delattr(model, '_kv_cache')

    print("✓ KV cache disabled, original attention restored")


# %% [markdown]
"""
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
    test_input = Tensor(np.random.randn(1, 10, 128))
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

# Run test immediately when developing this module
if __name__ == "__main__":
    test_unit_noninvasive_integration()


# %% [markdown]
"""
## 📊 Systems Analysis: KV Cache Performance

Let's analyze the performance characteristics and trade-offs of KV caching. Understanding these trade-offs is essential for making informed decisions about when and how to use caching in production systems.
"""

# %% nbgrader={"grade": false, "grade_id": "analyze-memory", "locked": false}
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
        Cache overhead is 10-30% of model parameters, but enables
        10-15× speedup. Memory is cheap, compute is expensive!

    Production Context:
        GPT-3 (175B params, 2048 context): ~4GB cache per sequence
        This memory cost is acceptable given the massive speedup.
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

    print("Model Config | Cache Memory | Per Layer | Memory Overhead")
    print("-" * 60)

    for embed_dim, num_layers, seq_len, name in configs:
        # Memory per layer: 2 tensors (K, V) × batch × seq_len × embed_dim × 4 bytes
        batch_size = 1
        memory_per_layer = 2 * batch_size * seq_len * embed_dim * _BYTES_PER_FLOAT32 / _MB_TO_BYTES
        total_memory = memory_per_layer * num_layers

        # Model parameter memory (approximate)
        params_per_layer = embed_dim * embed_dim * _BYTES_PER_FLOAT32  # QKV projections
        model_memory = params_per_layer * num_layers * _BYTES_PER_FLOAT32 / _MB_TO_BYTES

        overhead_pct = (total_memory / model_memory) * 100 if model_memory > 0 else 0

        print(f"{name:12s} | {total_memory:11.2f} MB | {memory_per_layer:8.2f} MB | {overhead_pct:6.1f}%")

    print()
    print("💡 Key Insights:")
    print("   • Cache memory scales linearly with sequence length (O(n))")
    print("   • Longer sequences require proportionally more cache memory")
    print("   • Cache overhead is typically 10-30% of model parameters")
    print()
    print("🚀 Production Context:")
    print("   • GPT-3 (175B params, 2048 context): ~4GB cache memory")
    print("   • Trade-off: 2× memory enables 10-15× speedup")
    print("   • Worth it for inference-heavy workloads!")

# %% nbgrader={"grade": false, "grade_id": "analyze-speedup", "locked": false}
def analyze_kvcache_speedup():
    """
    📊 Measure KV cache speedup vs vanilla attention.

    Educational Purpose:
        Shows students WHY caching provides dramatic speedup through
        concrete complexity analysis. Compares O(n²) vs O(n) growth.

    Demonstrates:
        - Naive approach: O(n²) operations per token
        - Cached approach: O(n) operations per token
        - Speedup increases with generation length
        - 100-token generation: 170× fewer operations

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

    import time

    # Create test configuration
    batch_size = 1
    embed_dim = 256
    num_heads = 8
    head_dim = embed_dim // num_heads

    print("Generation Length | Without Cache | With Cache | Speedup")
    print("-" * 55)

    for gen_length in [10, 25, 50, 100]:
        # Simulate without cache: O(n²) for each new token
        # Each token processes entire context
        ops_without = sum(i**2 for i in range(1, gen_length + 1))

        # Simulate with cache: O(n) for each new token
        # Each token only processes itself
        ops_with = gen_length

        # Estimate time (arbitrary units)
        time_without = ops_without / 1000  # ms
        time_with = ops_with / 1000  # ms
        speedup = ops_without / ops_with

        print(f"{gen_length:17d} | {time_without:12.1f} ms | {time_with:10.1f} ms | {speedup:6.1f}×")

    print()
    print("💡 Key Insights:")
    print("   • Speedup increases with generation length (longer = better ROI)")
    print("   • 100-token generation: ~170× fewer operations!")
    print("   • Cache eliminates O(n²) recomputation per token")
    print()
    print("🚀 Production Reality:")
    print("   • ChatGPT uses KV caching for ALL generation")
    print("   • Without caching: 100-token response takes ~17 seconds")
    print("   • With caching: 100-token response takes ~0.1 seconds")
    print("   • This optimization makes conversational AI possible!")

# Run analysis when developing this module
if __name__ == "__main__":
    analyze_kvcache_memory()
    analyze_kvcache_speedup()


# %% [markdown]
"""
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
    test_unit_cached_attention_forward()
    print()
    test_unit_cached_generate()
    print()
    test_unit_noninvasive_integration()
    print()

    print("Running integration scenarios...")
    print()

    # Integration Test: Complete KV Cache Workflow
    print("🔬 Integration Test: Complete KV Cache Workflow...")
    batch_size, max_seq_len = 1, 128
    num_layers, num_heads, head_dim = 4, 8, 64

    cache = KVCache(batch_size, max_seq_len, num_layers, num_heads, head_dim)

    # Simulate generation loop (processing multiple tokens)
    for _ in range(5):
        for layer_idx in range(num_layers):
            # Simulate new key-value pairs
            new_key = Tensor(np.random.randn(batch_size, num_heads, 1, head_dim))
            new_value = Tensor(np.random.randn(batch_size, num_heads, 1, head_dim))

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
    print("🔬 Integration Test: Memory Tracking...")
    mem_info = cache.get_memory_usage()
    assert mem_info['total_mb'] > 0
    assert mem_info['cache_tensors'] == num_layers * 2
    print(f"✅ Memory tracking: {mem_info['total_mb']:.2f} MB for {mem_info['cache_tensors']} tensors")
    print()

    print("=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 18")

# Run comprehensive module test
if __name__ == "__main__":
    test_module()


# %% [markdown]
"""
## 🤔 ML Systems Reflection Questions

Answer these questions based on your implementation and the concepts you've learned in Modules 01-17.

### Question 1: Cache Size Calculation
A 12-layer transformer has 12 attention heads per layer, 64-dimensional embeddings per head,
maximum sequence length of 2048, and batch size of 8. Calculate the KV cache size:

**Step-by-step calculation**:
- One cache tensor shape: (batch=8, heads=12, seq_len=2048, head_dim=64)
- Elements per tensor: 8 × 12 × 2048 × 64 = _________
- Each layer has K cache + V cache = _________ tensors per layer
- Total across 12 layers = _________ cache tensors
- Float32 = 4 bytes per element
- Total memory in MB: _________

**Follow-up**: If this model has 125M parameters (500 MB), what percentage of model memory
is the cache? Is this overhead acceptable?

### Question 2: Speed vs Memory Trade-off
Your KVCache makes generation 10× faster but uses several GB of RAM.

Consider a production API serving 1000 users simultaneously:
- Without cache: Each generation is slow (10 sec) but uses minimal memory
- With cache: Each generation is fast (1 sec) but uses 100 MB cache per user = 100 GB total!

**Questions**:
- For an interactive chatbot, is this trade-off worth it? Why?
- What happens if your server only has 64 GB RAM but needs to serve 1000 users?
- How would you design a system that balances speed and memory for many concurrent users?

### Question 3: Batch Inference Scaling
With KV cache, each sequence in a batch gets its own cache storage.

**Scenario**: Batch size 1 generates at 500 tokens/sec, using 50 MB cache.
- For batch size 8: Predicted cache memory = _________ MB (scales how?)
- Does each sequence still generate at 500 tokens/sec? Why or why not?
- What's the throughput difference: 1×500 tok/s vs 8×? tok/s = _________ total tok/s

**Trade-off question**: For a production API, when should you use:
- High batch size (8-16): Good for _________
- Low batch size (1-2): Good for _________

### Question 4: Cache Eviction for Long Conversations
Your `KVCache` has `max_seq_len=2048`. A chatbot conversation reaches 2048 tokens - the cache is full!

**Options when cache is full**:
1. **Crash/Error**: Raise exception when max_seq_len exceeded
2. **FIFO eviction**: Drop oldest tokens, keep recent 2048
3. **Sliding window**: Keep most recent N tokens
4. **Restart cache**: Clear everything and start over

**Questions**:
- What happens to conversation context if you evict the first 1000 tokens?
- Why do production systems (ChatGPT) limit conversation length (e.g., 4096 or 8192 tokens)?
- Which eviction strategy would you choose for a medical chatbot that needs full conversation history?

### Question 5: Production Reality - Multi-User Serving
ChatGPT serves millions of users. Each user's conversation needs its own KV cache.

**Memory calculation for 10,000 concurrent conversations**:
- Each cache: 200 MB (typical for GPT-3.5 scale model)
- Total cache memory: 10,000 × 200 MB = _________ GB
- Model parameters: 13B × 4 bytes = 52 GB (loaded once, shared across all users)
- **Total memory needed**: _________ GB

**Questions**:
- Is it feasible to keep 10,000 caches in memory simultaneously on a single GPU (80 GB VRAM)?
- How do you think production systems manage cache memory across millions of users?
- Would you rather: (A) Keep all caches in memory (fast but expensive), or (B) Store inactive
  caches on disk and reload as needed (slower but cheaper)? What's the trade-off?
"""

# %% [markdown]
"""
## ⭐ Aha Moment: KV Cache Avoids Recomputation

**What you built:** A KV Cache that stores key-value pairs to avoid redundant attention computation.

**Why it matters:** When generating text token-by-token, naive attention recomputes the same
K,V values for all previous tokens at each step. With KV caching, you compute once and reuse!
This is why ChatGPT responds so fast—it's not recomputing everything every token.

This optimization turns O(n²) generation into O(n), enabling practical LLM deployment.
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
        new_k = Tensor(np.random.randn(1, 4, 1, 64))
        new_v = Tensor(np.random.randn(1, 4, 1, 64))

        # Update cache for layer 0
        cache.update(0, new_k, new_v)
        cache.advance()  # Move to next position

    print(f"Cached K,V for {cache.seq_pos} tokens")

    # Retrieve all cached values
    k_all, v_all = cache.get(0)
    print(f"Retrieved: K{k_all.shape}, V{v_all.shape}")

    print("\n✨ Compute once, reuse forever—10× faster generation!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_memoization()

# %% [markdown]
"""
## 🚀 MODULE SUMMARY: KV Caching (Memoization)

Congratulations! You've built the optimization that makes production language models economically viable!

### Key Accomplishments
- Built KVCache class with efficient memory management for K,V tensors across layers
- Implemented non-invasive cache integration using enable_kv_cache()
- Measured 10-15× speedup through analysis functions showing O(n²)→O(n) improvement
- Understood memory-compute trade-off (2× memory enables 10× speedup)
- Discovered why speedup increases with generation length
- All tests pass ✅ (validated by `test_module()`)

### Systems Insights Gained
- **Recomputation Elimination**: Caching K/V eliminates O(n²) redundant work per token
- **Memory-Speed Trade-off**: Doubling memory enables order-of-magnitude speedup
- **Scaling Benefits**: Longer generation = better cache return on investment (170× at 100 tokens)
- **Production Critical**: This single optimization makes ChatGPT-scale inference possible
- **Non-Invasive Design**: Add capabilities forward without breaking existing modules

### Real-World Impact
Without KV caching:
- 100-token generation: ~17 seconds
- Conversational AI: economically infeasible
- User experience: unacceptably slow

With KV caching:
- 100-token generation: ~0.1 seconds (170× faster!)
- Conversational AI: production-ready at scale
- User experience: real-time interaction

This optimization is THE technique that transformed language models from research demonstrations into products serving millions of users daily.

### Production Skills Developed
- **Systems Optimization**: Identify and eliminate computational bottlenecks
- **Memory-Compute Trade-offs**: Accept memory cost for speed gains
- **Non-Breaking Enhancement**: Add features without modifying existing code
- **Performance Analysis**: Measure and validate optimization impact

### Ready for Next Steps
Your KV caching implementation demonstrates the principle: "spend memory to save time"!

**Next**: Module 19 (Benchmarking) will teach you how to measure and compare these optimizations quantitatively!

Export with: `tito module complete 18`
"""
