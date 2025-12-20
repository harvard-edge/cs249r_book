# Module 18: Memoization

:::{admonition} Module Info
:class: note

**OPTIMIZATION TIER** | Difficulty: ●●○○ | Time: 3-5 hours | Prerequisites: 01-14

**Prerequisites: Modules 01-14** means you should be comfortable with:
- Tensor operations, matrix multiplication, and shape manipulation (Module 01)
- Transformer architectures and attention (Modules 12-13)
- Profiling tools (Module 14) to measure speedup

This module introduces optimization techniques that make production language model inference economically viable. If you understand how transformers compute attention and why it's expensive, you're ready to learn how to make inference dramatically faster.
:::

`````{only} html
````{grid} 1 2 3 3
:gutter: 3

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F18_memoization%2F18_memoization.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/18_memoization/18_memoization.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #6b7280; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
```

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/18_memoization.mp3" type="audio/mpeg">
</audio>
```

````
`````

## Overview

Every time a language model generates a token, it performs the same computations over and over. When ChatGPT writes a 100-word response, it recomputes attention values for earlier words hundreds of times, wasting enormous computational resources. This inefficiency makes real-time conversational AI economically impossible without optimization.

Memoization solves this by caching computation results for reuse. In transformers, this manifests as KV caching: storing the key and value matrices from attention computations. Instead of recomputing these matrices for every token, the model computes them once and retrieves them from cache. This single optimization transforms generation from O(n²) to O(n) complexity, enabling 10-15x speedup.

In this module, you'll implement a production-grade KV cache system that makes transformer inference practical at scale. You'll discover why every deployed language model uses this technique.

## Why Memoization After Acceleration?

The Optimization tier divides runtime optimizations into:

- **Acceleration (17)**: General-purpose speedups that apply to ANY computation
- **Memoization (18)**: Domain-specific optimization for transformer generation

**Why this order?** Pedagogically, you learn general techniques before specialized applications:

1. **Acceleration** teaches universal concepts: vectorization, cache locality, kernel fusion. These apply to matrix multiplication, convolutions, attention—everything.
2. **Memoization (KV-cache)** is specialized: it only helps autoregressive transformer generation, trading O(n) memory for O(n²) → O(n) speedup.

Once you understand how to make any code fast (Module 17), you can appreciate this elegant optimization that makes ChatGPT and Claude economically viable (Module 18).

## Learning Objectives

```{tip} By completing this module, you will:

- **Implement** a KVCache class with efficient memory management and O(1) update operations
- **Master** the memory-compute trade-off: accepting O(n) memory overhead for O(n²) to O(n) speedup
- **Understand** why memoization transforms generation complexity from quadratic to linear
- **Connect** your implementation to production systems like ChatGPT and Claude that rely on KV caching
```

## What You'll Build

```{mermaid}
:align: center
:caption: KV Cache System
flowchart LR
    subgraph "KV Cache System"
        A["Cache Storage<br/>Pre-allocated tensors"]
        B["Update Logic<br/>O(1) append"]
        C["Retrieval<br/>O(1) slice"]
        D["Memory Tracking<br/>Usage analysis"]
    end

    A --> B --> C --> D

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#d4edda
    style D fill:#f8d7da
```

**Implementation roadmap:**

| Step | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `KVCache.__init__()` | Pre-allocated cache storage per layer |
| 2 | `KVCache.update()` | O(1) cache append without copying |
| 3 | `KVCache.get()` | O(1) retrieval of cached values |
| 4 | `enable_kv_cache()` | Non-invasive model enhancement |
| 5 | Performance analysis | Measure speedup and memory usage |

**The pattern you'll enable:**
```python
# Enable caching for dramatic speedup
cache = enable_kv_cache(model)
# Generate with 10-15x faster inference
output = model.generate(prompt, max_length=100)
```

### What You're NOT Building (Yet)

To keep this module focused, you will **not** implement:

- Multi-batch cache management (production systems handle thousands of concurrent sequences)
- Cache eviction strategies (handling sequences longer than max_seq_len)
- GPU memory optimization (production uses memory pools and paging)
- Speculative decoding (advanced technique that builds on KV caching)

**You are building the core memoization mechanism.** Advanced cache management comes in production deployment.

## API Reference

This section provides a quick reference for the KVCache class you'll build. Use this as your guide while implementing and debugging.

### KVCache Constructor

```python
KVCache(batch_size: int, max_seq_len: int, num_layers: int,
        num_heads: int, head_dim: int) -> KVCache
```

Pre-allocates cache storage for all transformer layers. Each layer gets two tensors (keys and values) sized to hold the maximum sequence length.

**Parameters:**
- `batch_size`: Number of sequences to cache simultaneously
- `max_seq_len`: Maximum sequence length to support
- `num_layers`: Number of transformer layers in the model
- `num_heads`: Number of attention heads per layer
- `head_dim`: Dimension of each attention head

### Core Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `update` | `update(layer_idx: int, key: Tensor, value: Tensor) -> None` | Append new K,V to cache for given layer |
| `get` | `get(layer_idx: int) -> Tuple[Tensor, Tensor]` | Retrieve cached K,V for attention computation |
| `advance` | `advance() -> None` | Move sequence position forward after processing token |
| `reset` | `reset() -> None` | Clear cache for new generation sequence |
| `get_memory_usage` | `get_memory_usage() -> Dict[str, float]` | Calculate cache memory consumption |

### Helper Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `enable_kv_cache` | `enable_kv_cache(model) -> KVCache` | Non-invasively add caching to transformer |
| `disable_kv_cache` | `disable_kv_cache(model) -> None` | Restore original attention behavior |

## Core Concepts

This section covers the fundamental ideas you need to understand memoization in transformers. These concepts explain why KV caching is the optimization that makes production language models economically viable.

### Caching Computation

Memoization trades memory for speed by storing computation results for reuse. When a function is called with the same inputs repeatedly, computing the result once and caching it eliminates redundant work. This trade-off makes sense when memory is cheaper than computation, which is almost always true for inference.

In transformers, attention is the perfect target for memoization. During autoregressive generation, each new token requires attention over all previous tokens. The naive approach recomputes key and value projections for every previous token at every step, leading to quadratic complexity. But these projections never change once computed, making them ideal candidates for caching.

Here's the core insight in your implementation:

```python
def update(self, layer_idx: int, key: Tensor, value: Tensor) -> None:
    """Update cache with new key-value pairs for given layer."""
    if layer_idx >= self.num_layers:
        raise ValueError(f"Layer index {layer_idx} >= num_layers {self.num_layers}")

    if self.seq_pos >= self.max_seq_len:
        raise ValueError(f"Sequence position {self.seq_pos} >= max_seq_len {self.max_seq_len}")

    # Get cache for this layer
    key_cache, value_cache = self.caches[layer_idx]

    # Update cache at current position (efficient O(1) write)
    key_cache.data[:, :, self.seq_pos:self.seq_pos+1, :] = key.data
    value_cache.data[:, :, self.seq_pos:self.seq_pos+1, :] = value.data
```

This O(1) update operation writes directly to a pre-allocated position in the cache. No array resizing, no data copying, just an indexed assignment. The use of `.data` accesses the underlying NumPy array directly, avoiding gradient tracking overhead since caching is inference-only.

The computational savings compound across generation steps. For a 100-token sequence:
- Without caching: 1 + 2 + 3 + ... + 100 = 5,050 K,V computations
- With caching: 100 K,V computations (one per token)
- Speedup: 50x reduction in K,V computation alone

### KV Cache in Transformers

Transformer attention computes three projections from the input: query (Q), key (K), and value (V). The attention output is computed as softmax(Q @ K^T / sqrt(d_k)) @ V. During generation, each new token produces a new query, but the keys and values from previous tokens remain constant.

Consider generating the sequence "Hello world!":

```
Step 1: Input = ["Hello"]
  Compute: Q₁, K₁, V₁
  Attention: Q₁ @ [K₁] @ [V₁]

Step 2: Input = ["Hello", "world"]
  Compute: Q₂, K₂, V₂
  Attention: Q₂ @ [K₁, K₂] @ [V₁, V₂]
  Problem: K₁ and V₁ are recomputed unnecessarily!

Step 3: Input = ["Hello", "world", "!"]
  Compute: Q₃, K₃, V₃
  Attention: Q₃ @ [K₁, K₂, K₃] @ [V₁, V₂, V₃]
  Problem: K₁, V₁, K₂, V₂ are all recomputed!
```

The cache eliminates this redundancy:

```
Step 1: Compute K₁, V₁ → Cache them
Step 2: Compute K₂, V₂ → Append to cache
  Attention: Q₂ @ cached[K₁, K₂] @ cached[V₁, V₂]
Step 3: Compute K₃, V₃ → Append to cache
  Attention: Q₃ @ cached[K₁, K₂, K₃] @ cached[V₁, V₂, V₃]
```

Each step now computes only one new K,V pair instead of recomputing all previous pairs.

### Gradient Checkpointing

While KV caching optimizes inference, gradient checkpointing addresses the opposite problem: memory consumption during training. Training requires storing intermediate activations for backpropagation, but for very deep networks, this can exceed available memory. Gradient checkpointing trades compute for memory by not storing all activations.

The technique works by discarding some intermediate activations during the forward pass and recomputing them during backpropagation when needed. Instead of storing activations for all layers (requiring O(n) memory where n is the number of layers), checkpointing only stores activations at regular intervals (checkpoints). Between checkpoints, activations are recomputed from the last checkpoint during the backward pass.

For a transformer with 96 layers:
- Without checkpointing: Store 96 sets of activations
- With checkpointing every 12 layers: Store 8 sets, recompute 11 sets during backward
- Memory reduction: 12x decrease
- Compute increase: ~33% slower training (recomputation overhead)

This is the inverse trade-off from KV caching. KV caching spends memory to save compute during inference. Gradient checkpointing spends compute to save memory during training. Both techniques recognize that memory and compute are fungible resources with different costs in different contexts.

### Cache Invalidation

Cache invalidation is one of the hardest problems in computer science because deciding when cached data is still valid requires careful analysis. For KV caching in transformers, invalidation is straightforward because the cached values have well-defined lifetimes.

During generation, cached K,V pairs remain valid for the entire sequence being generated. The cache is invalidated and reset when starting a new generation sequence. This simplicity comes from the autoregressive property: each token depends only on previous tokens, and those dependencies are frozen once computed.

Here's how your implementation handles cache lifecycle:

```python
def reset(self) -> None:
    """Reset cache for new generation sequence."""
    self.seq_pos = 0

    # Zero out caches for clean state (helps with debugging)
    for layer_idx in range(self.num_layers):
        key_cache, value_cache = self.caches[layer_idx]
        key_cache.data.fill(0.0)
        value_cache.data.fill(0.0)
```

The reset operation returns the sequence position to zero and clears the cache data. This is called when starting to generate a new sequence, ensuring no stale data from previous generations affects the current one.

Production systems handle more complex invalidation scenarios:
- **Max length reached**: When the sequence fills the cache, either error out or implement a sliding window
- **Batch inference**: Each sequence in a batch has independent cache state
- **Multi-turn conversation**: Some systems maintain cache across turns, others reset per turn

### Memory-Compute Trade-offs

Every optimization involves trade-offs. KV caching trades memory for speed, and understanding this exchange quantitatively reveals when the technique makes sense.

For a transformer with L layers, H heads per layer, dimension D per head, and maximum sequence length S, the cache requires:

```
Memory = 2 × L × H × S × D × 4 bytes

Example (GPT-2 Small):
L = 12 layers
H = 12 heads
S = 1024 tokens
D = 64 dimensions
Memory = 2 × 12 × 12 × 1024 × 64 × 4 = 75,497,472 bytes ≈ 75 MB
```

For a model with 125 million parameters (500 MB), the cache adds 15% memory overhead. This seems significant until you consider the computational savings.

Without caching, generating a sequence of length N requires computing K,V for:
- Step 1: 1 token
- Step 2: 2 tokens
- Step 3: 3 tokens
- Step N: N tokens
- Total: 1 + 2 + 3 + ... + N = N(N+1)/2 ≈ N²/2 computations

With caching:
- Step 1: 1 token (compute and cache)
- Step 2: 1 token (compute and append)
- Step 3: 1 token (compute and append)
- Step N: 1 token (compute and append)
- Total: N computations

For N = 100 tokens, caching provides 50x reduction in K,V computation. For N = 1000 tokens, the reduction is 500x. The speedup grows with sequence length, making the memory trade-off increasingly favorable for longer generation.

| Sequence Length | Cache Memory | Compute Reduction | Effective Speedup |
|-----------------|--------------|-------------------|-------------------|
| 10 tokens | 75 MB | 5.5x | 3-5x |
| 50 tokens | 75 MB | 25.5x | 8-12x |
| 100 tokens | 75 MB | 50.5x | 10-15x |
| 500 tokens | 75 MB | 250.5x | 12-20x |

The effective speedup is lower than the theoretical compute reduction because attention includes other operations beyond K,V projection, but the benefit is still dramatic.

## Common Errors

These are the errors you'll encounter most often when implementing KV caching. Understanding why they happen will save hours of debugging.

### Cache Position Out of Bounds

**Error**: `ValueError: Sequence position 128 >= max_seq_len 128`

This happens when you try to append to a full cache. The cache is pre-allocated with a maximum sequence length, and attempting to write beyond that length raises an error.

**Cause**: Generation exceeded the maximum sequence length specified when creating the cache.

**Fix**: Either increase `max_seq_len` when creating the cache, or implement cache eviction logic to handle sequences longer than the maximum.

```python
# Create cache with sufficient capacity
cache = KVCache(batch_size=1, max_seq_len=2048,  # Increased from 128
                num_layers=12, num_heads=12, head_dim=64)
```

### Forgetting to Advance Position

**Error**: Cache retrieval returns the same K,V repeatedly, or update overwrites previous values

**Symptom**: Generated text repeats, or cache doesn't grow as expected

**Cause**: Forgetting to call `cache.advance()` after updating all layers for a token.

**Fix**: Always advance the cache position after processing a complete token through all layers:

```python
for layer_idx in range(num_layers):
    cache.update(layer_idx, new_key, new_value)

cache.advance()  # Move to next position for next token
```

### Shape Mismatches

**Error**: Broadcasting error or shape mismatch when updating cache

**Symptom**: `ValueError: could not broadcast input array from shape (1,8,64,64) into shape (1,8,1,64)`

**Cause**: The key and value tensors passed to `update()` must have shape `(batch, heads, 1, head_dim)` with sequence dimension equal to 1 (single new token).

**Fix**: Ensure new K,V tensors represent a single token:

```python
# Correct: Single token (seq_len = 1)
new_key = Tensor(np.random.randn(batch_size, num_heads, 1, head_dim))
cache.update(layer_idx, new_key, new_value)

# Wrong: Multiple tokens (seq_len = 64)
wrong_key = Tensor(np.random.randn(batch_size, num_heads, 64, head_dim))
cache.update(layer_idx, wrong_key, wrong_value)  # This will fail!
```

### Cache Not Reset Between Sequences

**Error**: Second generation includes tokens from first generation

**Symptom**: Model generates text that seems to continue from a previous unrelated sequence

**Cause**: Forgetting to reset the cache when starting a new generation sequence.

**Fix**: Always reset the cache before generating a new sequence:

```python
# Generate first sequence
output1 = model.generate(prompt1)

# Reset cache before second sequence
model._kv_cache.reset()

# Generate second sequence (independent of first)
output2 = model.generate(prompt2)
```

## Production Context

### Your Implementation vs. PyTorch

Your KVCache implementation uses the same conceptual design as production frameworks. The differences lie in scale, optimization level, and integration depth. PyTorch's KV cache implementation is written in C++ and CUDA for speed, supports dynamic batching for serving multiple users, and includes sophisticated memory management with paging and eviction.

| Feature | Your Implementation | PyTorch (Transformers library) |
|---------|---------------------|--------------------------------|
| **Backend** | NumPy (CPU) | C++/CUDA (GPU) |
| **Pre-allocation** | Fixed max_seq_len | Dynamic growth + paging |
| **Batch support** | Single batch size | Dynamic batching |
| **Memory management** | Simple reset | LRU eviction, memory pools |
| **Update complexity** | O(1) | O(1) with optimized kernels |

### Code Comparison

The following comparison shows how KV caching is used in TinyTorch versus production PyTorch. The API patterns are similar because the underlying concept is identical.

`````{tab-set}
````{tab-item} Your TinyTorch
```python
from tinytorch.perf.memoization import enable_kv_cache

# Enable caching
cache = enable_kv_cache(model)

# Generate with caching (10-15x faster)
for _ in range(100):
    logits = model.forward(input_token)
    next_token = sample(logits)
    # Cache automatically used and updated
    input_token = next_token

# Reset for new sequence
cache.reset()
```
````

````{tab-item} ⚡ PyTorch
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

# KV cache enabled automatically during generate()
outputs = model.generate(
    input_ids,
    max_length=100,
    use_cache=True  # KV caching enabled
)

# Cache managed internally by HuggingFace
# Automatically reset between generate() calls
```
````
`````

Let's examine each approach to understand the similarities and differences:

- **Line 1-2 (Imports)**: TinyTorch uses an explicit `enable_kv_cache()` function to opt-in to caching. PyTorch's Transformers library integrates caching directly into the model architecture.
- **Line 4-5 (Setup)**: TinyTorch requires manually enabling the cache and storing the reference. PyTorch handles this transparently when you call `generate()`.
- **Line 7-12 (Generation)**: TinyTorch's loop explicitly manages token generation with the cache working behind the scenes. PyTorch's `generate()` method encapsulates the entire loop and automatically uses caching when `use_cache=True`.
- **Line 14-15 (Reset)**: TinyTorch requires manual cache reset between sequences. PyTorch automatically resets the cache at the start of each `generate()` call.

The core difference is abstraction level. TinyTorch exposes the cache as an explicit object you control, making the optimization visible for learning. PyTorch hides caching inside `generate()` for ease of use in production. Both implementations use the same O(1) append pattern you built.

```{tip} What's Identical

The fundamental algorithm: compute K,V once, append to cache, retrieve for attention. Production systems add memory management and batching, but the core optimization is exactly what you implemented.
```

### Why Memoization Matters at Scale

To appreciate the production impact of KV caching, consider the economics of language model serving:

- **ChatGPT**: Serves millions of requests per day. Without KV caching, serving costs would be 10x higher, making the service economically unviable at current pricing.
- **GitHub Copilot**: Generates code completions in real-time. Without caching, latency would increase from 100ms to 1-2 seconds, breaking the developer experience.
- **Production API serving**: A single V100 GPU serving GPT-2 can handle 50-100 concurrent users with caching, but only 5-10 without it. This 10x difference determines infrastructure costs.

The memory cost is modest compared to the benefit. For a GPT-2 model:
- Model parameters: 500 MB (loaded once, shared across all users)
- KV cache per user: 75 MB
- 10 concurrent users: 750 MB cache + 500 MB model = 1.25 GB total
- Fits comfortably on a 16 GB GPU while delivering 10x throughput

## Check Your Understanding

Test yourself with these systems thinking questions. They're designed to build intuition for the performance characteristics and trade-offs you'll encounter in production ML systems.

**Q1: Cache Memory Calculation**

A 12-layer transformer has 8 attention heads per layer, each head has 64 dimensions, maximum sequence length is 1024, and batch size is 4. Calculate the KV cache memory requirement.

```{admonition} Answer
:class: dropdown

Shape per cache tensor: (batch=4, heads=8, seq=1024, dim=64)

Elements per tensor: 4 × 8 × 1024 × 64 = 2,097,152

Each layer has 2 tensors (K and V): 2 × 2,097,152 = 4,194,304 elements per layer

Total across 12 layers: 12 × 4,194,304 = 50,331,648 elements

Memory: 50,331,648 × 4 bytes = 201,326,592 bytes ≈ **192 MB**

This is why production systems carefully tune batch size and sequence length!
```

**Q2: Complexity Reduction**

Without caching, generating 200 tokens requires how many K,V computations? With caching?

```{admonition} Answer
:class: dropdown

**Without caching**: 1 + 2 + 3 + ... + 200 = 200 × 201 / 2 = **20,100 computations**

**With caching**: 200 computations (one per token)

**Reduction**: 20,100 / 200 = **100.5x fewer K,V computations**

This is why the speedup grows with sequence length!
```

**Q3: Memory-Compute Trade-off**

A model uses 2 GB for parameters. Adding KV cache uses 300 MB. Is this trade-off worthwhile if it provides 12x speedup?

```{admonition} Answer
:class: dropdown

**Memory overhead**: 300 MB / 2000 MB = 15% increase

**Speedup**: 12x faster generation

**Analysis**:
- Cost: 15% more memory
- Benefit: 12x more throughput (or 12x lower latency)
- Result: You can serve 12x more users with 1.15x the memory

**Verdict**: Absolutely worthwhile! Memory is cheap, compute is expensive.

In production, this enables serving 120 users per GPU instead of 10 users, dramatically reducing infrastructure costs.
```

**Q4: Cache Hit Rate**

During generation, what percentage of K,V retrievals come from cache vs. fresh computation after 50 tokens?

```{admonition} Answer
:class: dropdown

At token position 50:
- Fresh computation: 1 new K,V pair
- Cache retrievals: 49 previous K,V pairs
- Total: 50 K,V pairs needed

**Cache hit rate**: 49/50 = **98%**

As generation continues:
- Token 100: 99/100 = 99% hit rate
- Token 500: 499/500 = 99.8% hit rate

The cache hit rate approaches 100% for long sequences, explaining why speedup increases with length!
```

**Q5: Batch Inference Scaling**

Cache memory for batch_size=1 is 75 MB. What is cache memory for batch_size=8?

```{admonition} Answer
:class: dropdown

Cache memory scales linearly with batch size:

**batch_size=8**: 75 MB × 8 = **600 MB**

This is why production systems carefully manage batch size:
- Larger batches → higher throughput (more sequences per second)
- Larger batches → more memory (may hit GPU limits)

Trade-off example on 16 GB GPU:
- Model: 2 GB
- Available for cache: 14 GB
- Max batch size: 14 GB / 75 MB ≈ 186 sequences

Production systems balance batch size against latency requirements and memory constraints.
```

## Further Reading

For students who want to understand the academic foundations and production implementation of memoization in transformers:

### Seminal Papers

- **Attention Is All You Need** - Vaswani et al. (2017). The original transformer paper that introduced the architecture requiring KV caching for efficient generation. Section 3.2 describes the attention mechanism that benefits from memoization. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

- **Generating Sequences With Recurrent Neural Networks** - Graves (2013). Early work on autoregressive generation patterns, establishing the sequential token generation that creates the redundant computation KV caching eliminates. [arXiv:1308.0850](https://arxiv.org/abs/1308.0850)

- **Training Compute-Optimal Large Language Models** - Hoffmann et al. (2022). Analyzes the computational costs of training and inference, quantifying the importance of inference optimizations like KV caching at scale. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

- **FlashAttention: Fast and Memory-Efficient Exact Attention** - Dao et al. (2022). Modern attention optimization that combines with KV caching in production systems, demonstrating complementary optimization strategies. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

### Additional Resources

- **System**: [vLLM documentation](https://vllm.readthedocs.io/) - Production serving system that uses advanced KV cache management with paging
- **Tutorial**: [Hugging Face Text Generation Guide](https://huggingface.co/docs/transformers/main_classes/text_generation) - See `use_cache` parameter in production API
- **Blog**: "The Illustrated Transformer" by Jay Alammar - Visual explanation of attention mechanisms that benefit from caching

## What's Next

```{seealso} Coming Up: Module 19 - Benchmarking

Implement kernel fusion, operator batching, and CPU/GPU optimization techniques. You'll combine multiple operations to reduce memory bandwidth bottlenecks and maximize hardware utilization.
```

**Preview - How Memoization Combines with Future Optimizations:**

| Module | What It Does | Works with Memoization |
|--------|--------------|------------------------|
| **15: Quantization** | Reduce precision to save memory | `KVCache with int8 keys/values → 4x memory reduction` |
| **17: Acceleration** | Optimize computation kernels | `Fused attention + KV cache → minimal memory traffic` |
| **19: Benchmarking** | Measure end-to-end performance | `Profile cache hit rates and speedup gains` |

## Get Started

```{tip} Interactive Options

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/18_memoization/18_memoization.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/18_memoization/18_memoization.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
