# Module 12: Attention Mechanism

## Overview
Build the attention mechanism that revolutionized deep learning and powers GPT, BERT, and modern transformers.

## Time Estimate
**3-4 hours** - This is a complex module with significant systems analysis

## Difficulty
⭐⭐⭐⭐⭐ **Advanced** - Involves quadratic complexity, multi-head parallel processing, and memory scaling

## Prerequisites
You must complete **Modules 01-11** before starting this module:
- Module 01: Tensor operations
- Module 02: Activations (ReLU, Sigmoid)
- Module 03: Linear layers
- Module 04: Loss functions
- Module 05: Autograd for gradients
- Module 06: Optimizers
- Module 07: Training loops
- Module 08: DataLoader
- Module 09: Spatial operations (Conv2d, Pooling)
- Module 10: Batch Normalization
- Module 11: Tokenization and Embeddings

**Verify prerequisites pass:**
```bash
pytest modules/01_tensor/test_tensor.py
pytest modules/02_activations/test_activations.py
# ... etc for all 11 modules
```

## What You'll Build

### Core Components
1. **Scaled Dot-Product Attention** - The fundamental attention operation with explicit O(n²) complexity
2. **Multi-Head Attention** - Parallel attention heads for diverse relationship learning
3. **Attention Masking** - Causal masks for autoregressive language modeling

### Key Learning Focus
- **O(n²) Complexity**: Experience quadratic memory scaling with explicit nested loops
- **Attention Weights**: Understanding probability distributions over sequence positions
- **Multi-Head Design**: Why multiple smaller heads outperform single large heads
- **Memory Bottlenecks**: Why attention dominates transformer memory usage

## Module Structure

```
12_attention/
├── README.md                    # This file
├── attention_dev.py             # Main implementation (you work here)
└── test_attention.py            # Automated tests
```

## Implementation Highlights

### Part 1: Scaled Dot-Product Attention
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute attention with explicit O(n²) loops:
    1. scores = Q @ K^T (nested loops show quadratic complexity)
    2. scores = scores / √d_k (scaling for stability)
    3. Apply mask (set future positions to -inf)
    4. weights = softmax(scores) (probability distribution)
    5. output = weights @ V (weighted combination)
    """
```

**Educational Philosophy**: We use explicit nested loops (not NumPy vectorization) so you can **see and feel** the O(n²) complexity that makes attention both powerful and expensive.

### Part 2: Multi-Head Attention
```python
class MultiHeadAttention:
    """
    Split attention into multiple parallel heads:
    1. Project input to Q, K, V
    2. Split into num_heads parallel streams
    3. Apply attention to each head independently
    4. Concatenate and project back

    Each head learns different relationships:
    - Head 1: Local syntax patterns
    - Head 2: Long-range dependencies
    - Head 3: Semantic similarity
    - Head 4: Positional patterns
    """
```

## Systems Analysis Focus

### Memory Scaling Crisis
```
Sequence Length | Attention Matrix | Memory per Layer
----------------|------------------|------------------
128 tokens      | 128 × 128        | 64 KB
512 tokens      | 512 × 512        | 1 MB (16× larger!)
2048 tokens     | 2048 × 2048      | 16 MB (256× larger!)

GPT-3 (96 layers, 2048 context):
Total Attention Memory = 96 × 16 MB = 1.5 GB
```

### Why This Matters for Production
- **FlashAttention**: Modern technique to reduce O(n²) memory to O(n)
- **Sparse Attention**: Only compute attention for specific patterns
- **Long-Context Research**: Active frontier because of this quadratic wall
- **GPU Memory Limits**: Why 32K+ context is challenging even with massive GPUs

## Connection to Other Modules

### Leads To
→ **Module 13: Transformers** - Complete transformer blocks with attention + FFN
→ **Module 14: Language Models** - GPT-style autoregressive models
→ **Module 15: Fine-tuning** - Adapting pre-trained transformers

### Dependencies
← **Module 11: Embeddings** - Provides input representations for attention
← **Module 03: Linear Layers** - Used for Q/K/V projections
← **Module 05: Autograd** - Enables gradient computation through attention

## What You'll Experience

### The "Aha!" Moments
1. **Quadratic Complexity**: See why doubling sequence length quadruples computation
2. **Attention Patterns**: Visualize which tokens attend to which
3. **Causal Masking**: Understand autoregressive generation constraints
4. **Multi-Head Specialization**: Why parallel heads outperform single attention

### Real-World Impact
The attention mechanism you'll build is **mathematically identical** to what powers:
- ChatGPT and GPT-4
- BERT and RoBERTa
- Vision Transformers (ViT)
- CLIP and multimodal models

## Testing Strategy

### Unit Tests (Immediate Feedback)
- `test_unit_scaled_dot_product_attention()` - Core attention mechanism
- `test_unit_multihead_attention()` - Multi-head architecture

### Integration Tests
- `test_attention_scenarios()` - Realistic transformer configurations
- `analyze_attention_complexity()` - O(n²) memory/time scaling
- `analyze_attention_timing()` - Actual performance measurements

### Final Validation
```bash
# Run comprehensive module test
python attention_dev.py

# Or run automated test suite
pytest test_attention.py
```

## Common Challenges

### Challenge 1: Understanding O(n²) Complexity
**Problem**: "Why does attention scale quadratically?"
**Solution**: Look at the nested loops in `scaled_dot_product_attention()`:
```python
for i in range(seq_len):      # Each query position
    for j in range(seq_len):  # Attends to each key position
        # This is the O(n²) pattern!
```

### Challenge 2: Multi-Head Dimensions
**Problem**: "Why split embed_dim across heads?"
**Solution**:
- embed_dim=512, num_heads=8 → head_dim=64
- Each head gets 64 dimensions to work with
- Same total parameters, but diverse parallel processing

### Challenge 3: Causal Masking
**Problem**: "Why set future positions to -∞?"
**Solution**: Softmax(-∞) = 0, so future positions get zero attention weight
```python
# Before mask:  scores = [2.1, 3.5, 1.8, 2.9]
# After mask:   scores = [2.1, -∞, -∞, -∞]  (can't see future)
# After softmax: weights = [1.0, 0.0, 0.0, 0.0]
```

### Challenge 4: Memory Errors
**Problem**: "Out of memory with long sequences"
**Solution**: This is expected! Attention's O(n²) memory is the reason:
- seq_len=1024 → 4MB per layer
- seq_len=2048 → 16MB per layer (4× more!)
- This is why FlashAttention research exists

## Debugging Tips

### Print Attention Shapes
```python
print(f"Q shape: {Q.shape}")  # (batch, seq_len, d_model)
print(f"Scores shape: {scores.shape}")  # (batch, seq_len, seq_len)
print(f"Weights sum: {weights.sum(axis=-1)}")  # Should be all 1.0
```

### Visualize Attention Matrix
```python
import matplotlib.pyplot as plt
plt.imshow(weights[0], cmap='viridis')
plt.xlabel("Key positions")
plt.ylabel("Query positions")
plt.colorbar(label="Attention weight")
plt.show()
```

### Check Masking
```python
# Causal mask should be lower triangular
print(mask[0])  # Upper triangle should be 0 (or False)
print(weights[0])  # Upper triangle should be ~0.0 after softmax
```

## Resources for Deep Dive

### Papers
- **"Attention Is All You Need"** (Vaswani et al., 2017) - Original transformer paper
- **"FlashAttention"** (Dao et al., 2022) - Efficient attention with O(n) memory
- **"Reformer"** (Kitaev et al., 2020) - Efficient transformers with locality-sensitive hashing

### Concepts to Explore
- Query-Key-Value architecture philosophy
- Softmax temperature and attention sharpness
- Attention head specialization in trained models
- Sparse attention patterns (local, strided, global)

## Success Criteria

You've mastered this module when you can:
- [ ] Explain why attention scales as O(n²) in memory and computation
- [ ] Implement scaled dot-product attention with explicit loops
- [ ] Build multi-head attention with proper dimension handling
- [ ] Apply causal masking for autoregressive models
- [ ] Visualize and interpret attention weight matrices
- [ ] Understand why attention is the memory bottleneck in transformers
- [ ] All tests pass: `python attention_dev.py` shows ✅

## Next Steps

After completing this module:
1. **Export**: Run `tito module complete 12`
2. **Verify**: Check that attention functions are available in `tinytorch.core.attention`
3. **Advance**: Move to **Module 13: Transformers** to build complete transformer blocks!

---

**Ready to build the mechanism that powers modern AI?** Open `attention_dev.py` and let's implement the attention that changed everything! 🚀
