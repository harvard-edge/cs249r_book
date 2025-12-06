---
title: "Memoization - Computational Reuse for Inference"
description: "Transform O(n²) transformer generation into O(n) through KV caching, achieving 10-15x speedup"
difficulty: "⭐⭐⭐ (3/4)"
time_estimate: "4-5 hours"
prerequisites: ["Transformers", "Profiling", "Quantization", "Compression"]
next_steps: ["Acceleration"]
learning_objectives:
  - "Understand memoization as a fundamental optimization pattern that caches computational results"
  - "Implement KVCache data structures for efficient memory management with O(1) updates"
  - "Apply caching to transformers by storing and reusing attention keys and values"
  - "Measure O(n²) to O(n) complexity reduction and 10-15x generation speedup"
  - "Analyze memory-speed trade-offs and understand when caching benefits justify costs"
---

# 17. Memoization - Computational Reuse for Inference

**OPTIMIZATION TIER** | Difficulty: ⭐⭐⭐ (3/4) | Time: 4-5 hours

## Overview

Memoization is a fundamental optimization pattern: cache computational results to avoid redundant work. You'll apply this pattern to transformers through KV (Key-Value) caching, transforming O(n²) autoregressive generation into O(n) complexity and achieving 10-15x speedup. This optimization makes production language model serving economically viable.

This is inference-only optimization - you'll implement caching patterns used in every production LLM from ChatGPT to Claude to GitHub Copilot.

## Learning Objectives

By the end of this module, you will be able to:

- **Understand Memoization Pattern**: Recognize when computational reuse through caching applies to ML problems and understand the memory-speed trade-off
- **Implement KVCache Structure**: Build efficient cache data structures with O(1) updates, proper memory management, and multi-layer support
- **Apply Caching to Transformers**: Integrate KV caching into attention layers without modifying existing transformer code (non-invasive enhancement)
- **Measure Performance Gains**: Profile latency improvements, measure O(n²) → O(n) complexity reduction, and understand speedup characteristics
- **Analyze Production Trade-offs**: Calculate cache memory costs, understand cache invalidation policies, and recognize when caching justifies its overhead

## Build → Use → Optimize

This module follows TinyTorch's **Build → Use → Optimize** framework:

1. **Build**: Implement KVCache data structure with efficient updates, cached attention integration, and multi-layer cache management
2. **Use**: Apply caching to GPT text generation, measure 10-15x speedup over naive generation, and validate output correctness
3. **Optimize**: Profile memory bandwidth bottlenecks, measure cache hit rates, and understand when memory cost exceeds latency benefit

## Why This Matters

### KV Cache Optimization Flow

Caching stores computed keys and values, avoiding recomputation for each new token:

```{mermaid}
graph LR
    A[Token i<br/>Compute K_i, V_i] --> B[Cache<br/>Store K_i, V_i]
    B --> C[Token i+1<br/>New computation]
    C --> D[Reuse<br/>K_i, V_i from cache]
    D --> E[Only compute<br/>K_{i+1}, V_{i+1}]
    E --> F[10-15× speedup]

    style A fill:#e3f2fd
    style C fill:#e3f2fd
    style B fill:#f3e5f5
    style D fill:#fff3e0
    style E fill:#ffe0b2
    style F fill:#f0fdf4
```

**Optimization**: Compute K,V once → Cache → Reuse for all future tokens → O(n²) → O(n) complexity

### The Autoregressive Generation Problem

Without caching, transformer generation has quadratic complexity:

```
Naive Generation (O(n²) complexity):
Step 1: Generate token 1  → Compute attention for [t₀]                (1 computation)
Step 2: Generate token 2  → Compute attention for [t₀, t₁]            (2 computations, t₀ RECOMPUTED!)
Step 3: Generate token 3  → Compute attention for [t₀, t₁, t₂]        (3 computations, t₀,t₁ RECOMPUTED!)
...
Step n: Generate token n  → Compute attention for [t₀, ..., tₙ]       (n computations, ALL RECOMPUTED!)

Total: 1 + 2 + 3 + ... + n = n(n+1)/2 = O(n²) complexity!
For 100 tokens: ~5,050 redundant K,V computations
```

**The Key Insight**: K and V matrices for previous tokens NEVER change, yet we recompute them every step!

### The Caching Solution

```
Cached Generation (O(n) complexity):
Step 1: Compute K₁, V₁ → Cache them → Attention with cached[K₁, V₁]
Step 2: Compute K₂, V₂ → Cache them → Attention with cached[K₁, K₂, V₁, V₂]  (reuse K₁, V₁!)
Step 3: Compute K₃, V₃ → Cache them → Attention with cached[K₁, K₂, K₃, V₁, V₂, V₃]  (reuse all!)

Total: 1 + 1 + 1 + ... + 1 = n computations (50x reduction for n=100!)
```

### Production Impact

KV caching is mandatory for all production LLM serving:

- **ChatGPT/GPT-4**: Would be 50-100x slower without caching, making conversational AI economically infeasible
- **Claude**: Caches up to 100K tokens of context, enabling long document processing
- **GitHub Copilot**: Real-time code completion requires sub-100ms latency - impossible without caching
- **Google Gemini**: Multi-level caching (KV + intermediate layers) serves billions of requests daily

Without KV caching, the computational cost would make these services prohibitively expensive.

### Memory-Speed Trade-off

```
Traditional Approach (No Cache):
Memory:  O(1)          Cost: Negligible
Compute: O(n²)         Cost: Prohibitive for long sequences

Cached Approach (KV Cache):
Memory:  O(n × d_k)    Cost: ~18MB per batch for GPT-2
Compute: O(n)          Cost: 10-15x faster than naive

Trade-off Winner: Memory is cheap, compute is expensive!
Use O(n) memory to save O(n²) compute.
```

## Implementation Guide

### Core Components

You'll implement three main components:

#### 1. KVCache Data Structure

```python
class KVCache:
    """
    Efficient key-value cache for autoregressive generation.

    Memory Layout:
        keys:   (num_layers, batch, num_heads, seq_len, d_k)
        values: (num_layers, batch, num_heads, seq_len, d_v)

    For GPT-2 (12 layers, 12 heads, 1024 seq, 64 dims):
        12 layers × 12 heads × 1024 seq × 64 dims = ~9M values
        At FP32 (4 bytes): ~36MB per batch item
        At FP16 (2 bytes): ~18MB per batch item

    Operations:
        update(layer_idx, key, value) -> None  # O(1) append
        get(layer_idx) -> (cached_k, cached_v) # O(1) retrieval
        advance() -> None                       # Increment position
        reset() -> None                         # Clear for new sequence
    """
```

**Key Design Decisions**:
- Pre-allocate cache tensors to avoid dynamic resizing overhead
- Use position counter for O(1) indexed updates (no copying)
- Store per-layer caches to support multi-layer transformers
- Track sequence position externally for clean separation

#### 2. Non-Invasive Cache Integration

```python
def enable_kv_cache(model):
    """
    Enable KV caching WITHOUT modifying Module 12/13 code.

    This demonstrates non-invasive optimization - adding capabilities
    to existing systems without breaking them. Similar to how Module 05
    uses enable_autograd() to add gradient tracking to Tensors.

    Approach:
    1. Create KVCache sized for model architecture
    2. Store cache on model as model._kv_cache
    3. Wrap each attention layer's forward method with caching logic
    4. Intercept attention calls to manage cache automatically

    This is composition + monkey-patching - a critical ML systems pattern!
    """
```

**Why Non-Invasive?**
- Modules 12-13 (Attention, Transformers) work unchanged
- Module 17 ADDS optimization, doesn't BREAK old code
- Teaches "forward-only" systems engineering: never modify earlier modules
- Matches how production systems layer optimizations (vLLM, HuggingFace)

#### 3. Cached Attention Logic

```python
def cached_forward(x, mask=None):
    """
    Cache-aware attention with three paths:

    PATH 1: Training (seq_len > 1)
        → Use original attention (preserve gradients)
        → O(n²) but needed for backpropagation

    PATH 2: First Token (seq_len == 1, cache empty)
        → Use original attention (initialize cache)
        → O(1) - just one token

    PATH 3: Cached Generation (seq_len == 1, cache populated)
        → Compute K,V for NEW token only
        → Retrieve ALL cached K,V (includes history)
        → Attention with cached context
        → O(n) - only compute new, reuse cache
        → THIS IS WHERE THE SPEEDUP HAPPENS!
    """
```

### Implementation Steps

#### Step 1: Design KVCache Structure
1. Initialize cache storage for all layers
2. Pre-allocate tensors with maximum sequence length
3. Track current sequence position (write pointer)
4. Implement update() for O(1) append operations
5. Implement get() for O(1) retrieval of valid cache portion

#### Step 2: Implement Cache Updates
1. Validate layer index and sequence position
2. Write new K,V to current position (indexed assignment)
3. Advance position counter after all layers processed
4. Handle batch dimension and multi-head structure

#### Step 3: Enable Non-Invasive Integration
1. Validate model has required attributes (embed_dim, num_layers, etc.)
2. Calculate head_dim from embed_dim and num_heads
3. Create KVCache instance sized for model
4. Store cache on model with model._kv_cache flag
5. Wrap each block's attention.forward with caching logic

#### Step 4: Implement Cached Attention Forward
1. Detect path: training (seq_len > 1), first token (cache empty), or cached generation
2. For cached path: Compute Q,K,V projections for new token only
3. Reshape to multi-head format (batch, num_heads, 1, head_dim)
4. Update cache with new K,V pairs
5. Retrieve ALL cached K,V (history + new)
6. Compute attention: softmax(Q @ K^T / √d_k) @ V using NumPy (.data)
7. Apply output projection and return

#### Step 5: Validate Correctness
1. Test cache initialization and memory calculation
2. Verify single-token and multi-token updates
3. Validate multi-layer cache synchronization
4. Test reset functionality
5. Measure speedup vs non-cached generation

### Why .data Instead of Tensor Operations?

In cached attention, we use NumPy via `.data` for three reasons:

1. **Explicit Intent**: Makes it crystal clear this is inference-only
   - Training: Uses Tensor operations → gradients tracked
   - Inference: Uses .data → no gradient overhead

2. **Performance**: Avoids any autograd bookkeeping
   - Even small overhead matters in generation hotpath
   - Production LLMs (vLLM, llama.cpp) use similar patterns

3. **Educational Clarity**: Shows students the distinction
   - "When do I need gradients?" (training)
   - "When can I skip them?" (inference)

We COULD use Tensor operations with requires_grad=False, but .data is more explicit and follows industry patterns.

## Getting Started

### Prerequisites

Ensure you understand transformers and profiling:

```bash
# Activate TinyTorch environment
source scripts/activate-tinytorch

# Verify prerequisite modules
tito test transformers
tito test profiling
```

**Required Understanding**:
- Multi-head attention mechanism (Module 12)
- Transformer architecture (Module 13)
- Latency profiling techniques (Module 14)
- O(n²) complexity of attention computation

### Development Workflow

1. **Open the development file**: `modules/17_memoization/memoization_dev.ipynb`
2. **Profile naive generation**: Measure O(n²) growth in latency as sequence lengthens
3. **Implement KVCache class**: Build data structure with update(), get(), advance(), reset()
4. **Test cache operations**: Verify single-token, multi-token, and multi-layer caching
5. **Implement enable_kv_cache()**: Non-invasively patch model attention layers
6. **Build cached attention forward**: Three-path logic (training, first token, cached generation)
7. **Measure speedup**: Profile cached vs non-cached generation, validate O(n) complexity
8. **Export and verify**: `tito module complete 17 && tito test memoization`

## Testing

### Comprehensive Test Suite

Run the full test suite to verify memoization functionality:

```bash
# TinyTorch CLI (recommended)
tito test memoization

# Direct pytest execution
python -m pytest tests/ -k memoization -v
```

### Test Coverage Areas

- ✅ **KVCache Initialization**: Validate cache creation, memory calculation, and initial state
- ✅ **Cache Updates**: Test single-token append, multi-token sequences, and O(1) update performance
- ✅ **Multi-Layer Synchronization**: Verify independent per-layer caches with correct indexing
- ✅ **Cache Retrieval**: Test get() returns only valid cached portion (up to seq_pos)
- ✅ **Non-Invasive Integration**: Validate enable_kv_cache() works without breaking model
- ✅ **Correctness Validation**: Compare cached vs non-cached outputs (should be identical)
- ✅ **Performance Measurement**: Measure speedup at different sequence lengths
- ✅ **Memory Tracking**: Calculate cache size and validate memory usage

### Inline Testing & Profiling

The module includes comprehensive validation with performance measurement:

```python
# Unit Test: KVCache Implementation
🔬 Unit Test: KVCache Implementation...
   Cache initialized: 0.59 MB
✅ Cache initialization successful
✅ Append and retrieval work correctly
✅ Multi-layer caching validated
✅ Reset functionality verified
📈 Progress: KVCache ✓

# Integration Test: Performance Measurement
🔬 Profiling Transformer Generation (Without Caching):
   Seq Len  |  Latency (ms)  |  Growth
   ---------|----------------|----------
    10      |    2.34        |  baseline
    20      |    4.89        |  2.09×
    40      |   10.12        |  2.07×
    80      |   21.45        |  2.12×
   160      |   45.67        |  2.13×

💡 Key Observations:
   • Latency grows QUADRATICALLY with sequence length
   • Each new token forces recomputation of ALL previous K,V pairs
   • For 160 tokens: ~4× time vs 80 tokens (2² growth)

🎯 The Solution: CACHE the K,V values! (That's memoization)
✅ Speedup: 10-15× for typical generation
```

### Manual Testing Examples

```python
from tinytorch.perf.memoization import KVCache, enable_kv_cache

# Test cache with small transformer
cache = KVCache(
    batch_size=1,
    max_seq_len=128,
    num_layers=4,
    num_heads=8,
    head_dim=64
)

# Simulate generation loop
import numpy as np
from tinytorch.core.tensor import Tensor

for step in range(10):
    for layer_idx in range(4):
        # New key-value pairs for this step
        new_k = Tensor(np.random.randn(1, 8, 1, 64))
        new_v = Tensor(np.random.randn(1, 8, 1, 64))

        # Update cache (O(1) operation)
        cache.update(layer_idx, new_k, new_v)

    # Advance position after all layers
    cache.advance()

# Retrieve cached values
cached_k, cached_v = cache.get(layer_idx=0)
print(f"Cached 10 tokens: {cached_k.shape}")  # (1, 8, 10, 64)

# Calculate memory usage
mem_info = cache.get_memory_usage()
print(f"Cache memory: {mem_info['total_mb']:.2f} MB")
```

## Systems Thinking Questions

### Real-World Production Challenges

**Memory-Speed Trade-off Analysis**:
- KV cache uses ~18MB per batch for GPT-2 (FP16). For batch=32, that's 576MB.
- On an 8GB GPU, how many concurrent users can you serve?
- What's the trade-off between batch size and cache size?
- When does memory bandwidth (cache access) become the bottleneck instead of compute?

**Cache Invalidation Policies**:
- In multi-turn chat, when should you clear the cache?
- What happens when context exceeds max_seq_len?
- How do production systems like ChatGPT handle context window limits?
- Compare eviction policies: LRU, FIFO, sliding window, importance-based

**Distributed Caching for Large Models**:
- For models too large for one GPU, you need tensor parallelism
- How do you partition the KV cache across GPUs?
- Which dimension should you shard: layers, heads, or sequence?
- What's the communication overhead for cache synchronization?

**Quantized Caching**:
- Storing cache in INT8 instead of FP16 saves 50% memory
- What's the accuracy impact of quantized KV cache?
- When is this trade-off worth it?
- How does quantization error accumulate over long sequences?

### Production Optimization Patterns

**Multi-Level Caching**:
- What if you cache not just K,V but intermediate layer activations?
- How does HuggingFace's `DynamicCache` differ from static pre-allocation?
- When should you use persistent caching (save to disk) for very long conversations?

**Speculation and Prefetching**:
- What if you predict the next query and pre-compute KV cache?
- How would speculative caching improve throughput?
- What's the risk if speculation is wrong?
- When does prefetching justify its overhead?

### Mathematical Foundations

**Complexity Reduction**:
- Why does KV caching transform O(n²) into O(n)?
- Calculate total operations for naive vs cached generation (n=100)
- What's the crossover point where caching overhead exceeds savings?

**Memory Layout Optimization**:
- Why pre-allocate cache instead of dynamic appending?
- How does cache contiguity affect memory bandwidth?
- Compare row-major vs column-major cache layouts for performance

**Attention Computation Analysis**:
- Why can we cache K,V but not Q (query)?
- What property of autoregressive generation makes caching valid?
- How would bidirectional attention (BERT) change caching strategy?

### HuggingFace Cache Patterns Comparison

**Static vs Dynamic Cache**:
```python
# TinyTorch (Module 17): Static pre-allocation
cache = KVCache(max_seq_len=1024)  # Fixed size, O(1) updates

# HuggingFace: Dynamic cache (DynamicCache class)
cache = DynamicCache()  # Grows as needed, more flexible but slower
```

**When to Use Each**:
- **Static (TinyTorch)**: Known max length, maximum performance, inference serving
- **Dynamic (HuggingFace)**: Variable lengths, exploration, research

**Production Systems (vLLM, TGI)**:
- Use PagedAttention for virtual memory management of KV cache
- Enables efficient memory sharing across requests
- Reduces memory fragmentation for variable-length sequences

## Performance Characteristics

### Expected Speedup by Sequence Length

```
Speedup Characteristics (GPT-2 on CPU):
┌─────────────┬──────────────┬──────────────┬──────────┐
│ Seq Length  │ No Cache     │ With Cache   │ Speedup  │
├─────────────┼──────────────┼──────────────┼──────────┤
│  10 tokens  │  ~80 tok/s   │  ~600 tok/s  │   7.5x   │
│  25 tokens  │  ~40 tok/s   │  ~500 tok/s  │  12.5x   │
│  50 tokens  │  ~25 tok/s   │  ~400 tok/s  │  16.0x   │
│ 100 tokens  │  ~12 tok/s   │  ~200 tok/s  │  16.7x   │
└─────────────┴──────────────┴──────────────┴──────────┘

Key Insight: Speedup increases with sequence length!
Why? Longer sequences = more redundant computation without cache.
```

### Memory Usage by Model Size

```
Cache Memory Requirements (FP16, batch_size=1):
┌──────────────┬────────┬────────┬─────────┬──────────────┐
│ Model        │ Layers │ Heads  │ Seq Len │ Cache Memory │
├──────────────┼────────┼────────┼─────────┼──────────────┤
│ TinyGPT      │   4    │   4    │   128   │   0.5 MB     │
│ GPT-2 (124M) │  12    │  12    │  1024   │  18.0 MB     │
│ GPT-3 (175B) │  96    │  96    │  2048   │   4.7 GB     │
└──────────────┴────────┴────────┴─────────┴──────────────┘

Formula: memory = num_layers × num_heads × max_seq_len × head_dim × 2 × 2 bytes
(2× for K and V, 2 bytes for FP16)
```

### Throughput Impact

**Single Sequence Generation**:
- Without cache: Throughput decreases as sequence grows (O(n²) bottleneck)
- With cache: Throughput stays relatively constant (O(n) scales well)

**Batch Inference**:
- Cache memory scales linearly with batch size
- Throughput increases with batching (amortize model loading)
- Memory becomes limiting factor before compute

## Where This Code Lives in the Final Package

**Package Export**: Code exports to `tinytorch.generation.kv_cache`

```python
# When students install tinytorch, they import your work like this:
from tinytorch.perf.memoization import KVCache, enable_kv_cache, disable_kv_cache
from tinytorch.nn import MultiHeadAttention  # Base class from Module 12
from tinytorch.core.transformer import GPT  # Architecture from Module 13

# Usage in generation:
model = GPT(vocab_size=1000, embed_dim=128, num_layers=4, num_heads=4)
cache = enable_kv_cache(model)  # Non-invasively add caching

# Generate with caching enabled (10-15x faster!)
output = generate_text(model, prompt="Hello", max_new_tokens=100)

# Disable caching if needed
disable_kv_cache(model)
```

Your KV caching implementation becomes the foundation for efficient inference in the TinyTorch package, used by subsequent modules for text generation, chat applications, and deployment scenarios.

## Common Challenges and Solutions

### Challenge 1: Cache Synchronization Across Layers

**Problem**: Keeping cache consistent when different layers process at different speeds or batch items have variable lengths.

**Solution**:
- Use layer indexing to maintain independent per-layer caches
- Advance sequence position only after ALL layers have processed current token
- Handle variable sequence lengths with padding and attention masks

**Code Pattern**:
```python
# Process all layers before advancing
for layer_idx in range(num_layers):
    cache.update(layer_idx, new_k, new_v)

# Now advance position (all layers synchronized)
cache.advance()
```

### Challenge 2: Memory Overhead for Large Models

**Problem**: Cache memory grows with sequence length and batch size, potentially exceeding GPU memory.

**Solution**:
- Implement cache size limits with eviction policies (LRU, FIFO)
- Use FP16 or INT8 quantization for cache storage (50% memory reduction)
- Consider PagedAttention for virtual memory management
- Tune max_seq_len to expected generation length

**Memory Optimization**:
```python
# FP16 caching (2 bytes per element)
cache = KVCache(...).to(dtype=np.float16)  # 50% memory savings

# INT8 caching (1 byte per element)
cache = KVCache(...).to(dtype=np.int8)  # 75% memory savings, accuracy trade-off
```

### Challenge 3: Correctness Validation

**Problem**: Cached generation must produce identical outputs to non-cached generation.

**Solution**:
- Compare cached vs non-cached outputs token-by-token
- Use deterministic sampling (temperature=0) for testing
- Validate cache retrieval returns correct sequence positions
- Test edge cases: first token, cache full, reset

**Validation Pattern**:
```python
# Generate without cache (ground truth)
output_nocache = generate(model, prompt, max_new_tokens=50)

# Generate with cache (optimized)
cache = enable_kv_cache(model)
output_cached = generate(model, prompt, max_new_tokens=50)

# Validate identical outputs
assert np.allclose(output_nocache, output_cached), "Cached output must match!"
```

### Challenge 4: Integration Without Breaking Existing Code

**Problem**: Adding caching shouldn't require modifying Modules 12-13 (attention, transformer).

**Solution**:
- Use composition + monkey-patching (wrap, don't modify)
- Store original forward methods before patching
- Provide disable_kv_cache() to restore original behavior
- Use feature flags (model._cache_enabled) for path selection

**Non-Invasive Pattern**:
```python
# Save original before patching
block._original_attention_forward = block.attention.forward

# Patch with cached version
block.attention.forward = cached_forward

# Restore later if needed
block.attention.forward = block._original_attention_forward
```

## Ready to Build?

You're about to implement the optimization that makes production language models economically viable! KV caching is THE technique that transformed LLMs from research toys into products used by millions daily.

This is where theory meets practice in ML systems engineering. You'll see firsthand how a simple idea - "don't recompute what never changes" - can deliver 10-15x speedup and make the impossible possible.

**What makes this module special**: Unlike many optimizations that require deep algorithmic changes, KV caching is conceptually simple but profoundly impactful. You'll implement it from scratch, measure the dramatic speedup, and understand the memory-speed trade-offs that guide production deployments.

Understanding this optimization from first principles - implementing it yourself, profiling the speedup, analyzing the trade-offs - will give you deep insight into how production ML systems work. This is the optimization that makes ChatGPT, Claude, and GitHub Copilot possible.

Take your time, measure thoroughly, and enjoy building production-ready ML systems!

Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/17_memoization/memoization_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/17_memoization/memoization_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/17_memoization/memoization_dev.ipynb
:class-header: bg-light

Browse the Jupyter notebook source and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.

```

---

<div class="prev-next-area">
<a class="left-prev" href="../16_compression/ABOUT.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../18_acceleration/ABOUT.html" title="next page">Next Module →</a>
</div>
