---
title: "Compression - Pruning and Model Compression"
description: "Implement pruning techniques to reduce model size while preserving accuracy"
difficulty: "⭐⭐⭐"
time_estimate: "5-6 hours"
prerequisites: ["15_quantization"]
next_steps: ["17_memoization"]
learning_objectives:
  - "Understand compression trade-offs: sparsity ratios vs actual speedup vs accuracy retention"
  - "Implement magnitude-based pruning to identify and systematically remove unimportant weights"
  - "Design structured pruning strategies that create hardware-friendly sparsity patterns"
  - "Apply knowledge distillation to transfer teacher model knowledge to smaller student models"
  - "Measure compression ratios and sparsity levels while understanding deployment constraints"
---

# 16. Compression - Pruning and Model Compression

**OPTIMIZATION TIER** | Difficulty: ⭐⭐⭐ (3/4) | Time: 5-6 hours

## Overview

Modern neural networks are massively overparameterized. BERT has 110M parameters but can compress to 40% size with 97% accuracy retention (DistilBERT). GPT-2 can be pruned 90% and retrained to similar performance (Lottery Ticket Hypothesis). Model compression techniques remove unnecessary parameters to enable practical deployment on resource-constrained devices.

This module implements core compression strategies: magnitude-based pruning (removing smallest weights), structured pruning (removing entire channels for hardware efficiency), knowledge distillation (training smaller models from larger teachers), and low-rank approximation (matrix factorization). You'll understand the critical trade-offs between compression ratio, inference speedup, and accuracy retention.

**Important reality check**: The implementations in this module demonstrate compression algorithms using NumPy, focusing on educational understanding of the techniques. Achieving actual inference speedup from sparse models requires specialized hardware support (NVIDIA's 2:4 sparsity, specialized sparse CUDA kernels) or optimized libraries (torch.sparse, cuSPARSE) beyond this module's scope. You'll learn when compression helps versus when it creates overhead without benefits.

## Learning Objectives

By the end of this module, you will be able to:

- **Understand compression fundamentals**: Differentiate between unstructured sparsity (scattered zeros), structured sparsity (removed channels), and architectural compression (distillation)
- **Implement magnitude pruning**: Remove weights below importance thresholds to achieve 50-95% sparsity with minimal accuracy loss
- **Design structured pruning**: Remove entire computational units (channels, neurons) using importance metrics like L2 norm
- **Apply knowledge distillation**: Train student models to match teacher performance using temperature-scaled soft targets
- **Analyze compression trade-offs**: Measure when pruning reduces model size without delivering proportional speedup, and understand hardware constraints

## Build → Use → Reflect

This module follows TinyTorch's **Build → Use → Reflect** framework:

1. **Build**: Implement magnitude pruning, structured pruning, knowledge distillation, and low-rank approximation algorithms
2. **Use**: Apply compression techniques to realistic neural networks and measure sparsity, parameter reduction, and memory savings
3. **Reflect**: Understand why 90% unstructured sparsity rarely accelerates inference, when structured pruning delivers real speedups, and how compression strategies must adapt to hardware constraints

## Implementation Guide

### Sparsity Measurement

Before compression, you need to quantify model density:

```python
def measure_sparsity(model) -> float:
    """Calculate percentage of zero weights in model."""
    total_params = 0
    zero_params = 0

    for param in model.parameters():
        total_params += param.size
        zero_params += np.sum(param.data == 0)

    return (zero_params / total_params) * 100.0
```

**Why this matters**: Sparsity measurement reveals how much redundancy exists. A 90% sparse model has only 10% active weights, but achieving speedup from this sparsity requires specialized hardware or storage formats.

### Magnitude-Based Pruning (Unstructured)

Remove individual weights with smallest absolute values:

```python
def magnitude_prune(model, sparsity=0.9):
    """Remove smallest weights to achieve target sparsity."""
    # Collect all weights (excluding biases)
    all_weights = []
    weight_params = []

    for param in model.parameters():
        if len(param.shape) > 1:  # Skip 1D biases
            all_weights.extend(param.data.flatten())
            weight_params.append(param)

    # Find threshold at desired percentile
    magnitudes = np.abs(all_weights)
    threshold = np.percentile(magnitudes, sparsity * 100)

    # Zero out weights below threshold
    for param in weight_params:
        mask = np.abs(param.data) >= threshold
        param.data = param.data * mask
```

**Characteristics**:
- **Compression**: Can achieve 90%+ sparsity with minimal accuracy loss
- **Speed reality**: Creates scattered zeros that don't accelerate dense matrix operations
- **Storage benefit**: Sparse formats (CSR, COO) reduce memory when combined with specialized storage
- **Hardware requirement**: Needs sparse tensor support for any speedup (torch.sparse, cuSPARSE)

**Critical insight**: High sparsity ratios don't equal speedup. Dense matrix operations (GEMM) are highly optimized; sparse operations require irregular memory access and specialized kernels. Without hardware acceleration, 90% sparse models run at similar speeds to dense models.

### Structured Pruning (Hardware-Friendly)

Remove entire channels or neurons for actual hardware benefits:

```python
def structured_prune(model, prune_ratio=0.5):
    """Remove entire channels based on L2 norm importance."""
    for layer in model.layers:
        if isinstance(layer, Linear) and hasattr(layer, 'weight'):
            weight = layer.weight.data

            # Calculate L2 norm for each output channel
            channel_norms = np.linalg.norm(weight, axis=0)

            # Identify channels to remove (lowest importance)
            num_channels = weight.shape[1]
            num_to_prune = int(num_channels * prune_ratio)

            if num_to_prune > 0:
                # Get indices of weakest channels
                prune_indices = np.argpartition(
                    channel_norms, num_to_prune
                )[:num_to_prune]

                # Zero entire channels
                weight[:, prune_indices] = 0

                if layer.bias is not None:
                    layer.bias.data[prune_indices] = 0
```

**Characteristics**:
- **Compression**: 30-70% typical (coarser granularity than magnitude pruning)
- **Speed benefit**: Smaller dense matrices enable faster computation when architecturally reduced
- **Accuracy trade-off**: Loses more accuracy than unstructured pruning at same sparsity level
- **Hardware friendly**: Regular memory access patterns work well with standard dense operations

**Critical insight**: Structured pruning achieves lower compression ratios but enables real speedup when combined with architectural changes. Simply zeroing channels doesn't help—you need to physically remove them from the model architecture to see benefits.

### Knowledge Distillation

Transfer knowledge from large teacher models to smaller students:

```python
class KnowledgeDistillation:
    """Compress models through teacher-student training."""

    def __init__(self, teacher_model, student_model,
                 temperature=3.0, alpha=0.7):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature  # Soften distributions
        self.alpha = alpha  # Balance soft vs hard targets

    def distillation_loss(self, student_logits,
                         teacher_logits, true_labels):
        """Combined loss: soft targets + hard labels."""
        # Temperature-scaled softmax for soft targets
        student_soft = softmax(student_logits / self.temperature)
        teacher_soft = softmax(teacher_logits / self.temperature)

        # Soft loss: learn from teacher's knowledge
        soft_loss = kl_divergence(student_soft, teacher_soft)

        # Hard loss: learn correct answers
        student_hard = softmax(student_logits)
        hard_loss = cross_entropy(student_hard, true_labels)

        # Weighted combination
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

**Why distillation works**:
- **Soft targets**: Teacher's probability distributions reveal uncertainty and class relationships
- **Temperature scaling**: Higher temperatures (T=3-5) soften sharp predictions, providing richer training signal
- **Architectural freedom**: Student can have completely different architecture, not just pruned weights
- **Accuracy preservation**: Students often match 95-99% of teacher performance with 5-10× fewer parameters

**Production example**: DistilBERT uses distillation to compress BERT from 110M to 66M parameters (40% reduction) while retaining 97% accuracy on GLUE benchmarks.

### Low-Rank Approximation

Compress weight matrices through SVD factorization:

```python
def low_rank_approximate(weight_matrix, rank_ratio=0.5):
    """Factorize matrix using truncated SVD."""
    m, n = weight_matrix.shape

    # Perform singular value decomposition
    U, S, V = np.linalg.svd(weight_matrix, full_matrices=False)

    # Truncate to target rank
    max_rank = min(m, n)
    target_rank = max(1, int(rank_ratio * max_rank))

    U_truncated = U[:, :target_rank]
    S_truncated = S[:target_rank]
    V_truncated = V[:target_rank, :]

    # Reconstruct: W ≈ U @ diag(S) @ V
    return U_truncated, S_truncated, V_truncated
```

**Compression math**:
- Original matrix: m × n parameters
- Factorized: (m × k) + k + (k × n) = k(m + n + 1) parameters
- Compression achieved when: k < mn/(m+n+1)
- Example: (1000×1000) = 1M params → (1000×100 + 100×1000) = 200K params (80% reduction)

**When low-rank works**: Large matrices with redundancy (common in fully-connected layers). **When it fails**: Small matrices or convolutions with less redundancy.

### Complete Compression Pipeline

Combine multiple techniques for maximum compression:

```python
def compress_model(model, compression_config):
    """Apply comprehensive compression strategy."""
    stats = {
        'original_params': sum(p.size for p in model.parameters()),
        'original_sparsity': measure_sparsity(model),
        'applied_techniques': []
    }

    # Apply magnitude pruning
    if 'magnitude_prune' in compression_config:
        sparsity = compression_config['magnitude_prune']
        magnitude_prune(model, sparsity=sparsity)
        stats['applied_techniques'].append(f'magnitude_{sparsity}')

    # Apply structured pruning
    if 'structured_prune' in compression_config:
        ratio = compression_config['structured_prune']
        structured_prune(model, prune_ratio=ratio)
        stats['applied_techniques'].append(f'structured_{ratio}')

    stats['final_sparsity'] = measure_sparsity(model)
    return stats

# Example usage
config = {
    'magnitude_prune': 0.8,   # 80% sparsity
    'structured_prune': 0.3   # Remove 30% of channels
}
stats = compress_model(model, config)
print(f"Achieved {stats['final_sparsity']:.1f}% sparsity")
```

**Multi-stage strategy**: Different techniques target different redundancy types. Magnitude pruning removes unimportant individual weights; structured pruning removes redundant channels; distillation creates fundamentally smaller architectures.

## Getting Started

### Prerequisites

Ensure you understand compression foundations:

```bash
# Activate TinyTorch environment
source scripts/activate-tinytorch

# Verify prerequisite modules
tito test quantization
```

**Required knowledge**:
- Neural network training and fine-tuning (pruned models need retraining)
- Gradient-based optimization (fine-tuning after compression)
- Quantization techniques (often combined with pruning for multiplicative gains)

**From previous modules**:
- **Tensor operations**: Weight manipulation and masking
- **Optimizers**: Fine-tuning compressed models
- **Quantization**: Combining compression techniques (10× pruning + 4× quantization = 40× total)

### Development Workflow

1. **Open the development file**: `modules/16_compression/compression_dev.ipynb`
2. **Implement sparsity measurement**: Calculate percentage of zero weights across model
3. **Build magnitude pruning**: Remove smallest weights using percentile thresholds
4. **Create structured pruning**: Remove entire channels based on L2 norm importance
5. **Implement knowledge distillation**: Build teacher-student training with temperature scaling
6. **Add low-rank approximation**: Factor large matrices using truncated SVD
7. **Build compression pipeline**: Combine techniques sequentially
8. **Export and verify**: `tito module complete 16 && tito test compression`

## Testing

### Comprehensive Test Suite

Run the full test suite to verify compression functionality:

```bash
# TinyTorch CLI (recommended)
tito test compression

# Direct pytest execution
python -m pytest tests/ -k compression -v
```

### Test Coverage Areas

- ✅ **Sparsity measurement**: Correctly counts zero vs total parameters
- ✅ **Magnitude pruning**: Achieves target sparsity with appropriate threshold selection
- ✅ **Structured pruning**: Removes entire channels, creates block sparsity patterns
- ✅ **Knowledge distillation**: Combines soft and hard losses with temperature scaling
- ✅ **Low-rank approximation**: Reduces parameters through SVD factorization
- ✅ **Compression pipeline**: Sequential application preserves functionality

### Inline Testing & Validation

The module includes comprehensive validation:

```python
🔬 Unit Test: Measure Sparsity...
✅ measure_sparsity works correctly!

🔬 Unit Test: Magnitude Prune...
✅ magnitude_prune works correctly!

🔬 Unit Test: Structured Prune...
✅ structured_prune works correctly!

🔬 Integration Test: Complete compression pipeline...
✅ Achieved 82.5% sparsity with 2 techniques

📊 Progress: Compression module ✓
```

### Manual Testing Examples

```python
from compression_dev import (
    magnitude_prune, structured_prune,
    measure_sparsity, KnowledgeDistillation
)

# Test magnitude pruning
model = Sequential(Linear(100, 50), Linear(50, 10))
print(f"Initial sparsity: {measure_sparsity(model):.1f}%")

magnitude_prune(model, sparsity=0.9)
print(f"After pruning: {measure_sparsity(model):.1f}%")

# Test structured pruning
structured_prune(model, prune_ratio=0.3)
print(f"After structured: {measure_sparsity(model):.1f}%")

# Test knowledge distillation
teacher = Sequential(Linear(100, 200), Linear(200, 50))
student = Sequential(Linear(100, 50))  # 3× smaller
kd = KnowledgeDistillation(teacher, student)
```

## Systems Thinking Questions

### Real-World Applications

- **Mobile deployment**: DistilBERT achieves 40% size reduction with 97% accuracy retention, enabling BERT on mobile devices
- **Edge inference**: MobileNetV2/V3 combine structured pruning with depthwise convolutions for <10MB models running real-time on phones
- **Production acceleration**: NVIDIA TensorRT applies automatic pruning + quantization for 3-10× speedup on inference workloads
- **Model democratization**: GPT distillation (DistilGPT-2) creates 40% smaller models approaching full performance on consumer hardware

### Compression Theory Foundations

- **Lottery Ticket Hypothesis**: Pruned networks can retrain to full accuracy from initial weights, suggesting networks contain sparse "winning ticket" subnetworks
- **Overparameterization insights**: Modern networks have excess capacity for easier optimization, not representation—most parameters help training, not inference
- **Information bottleneck**: Compression forces models to distill essential knowledge, sometimes improving generalization by removing noise
- **Hardware-algorithm co-design**: Effective compression requires algorithms designed for hardware constraints (memory bandwidth, cache locality, SIMD width)

### Performance Characteristics and Trade-offs

- **Unstructured sparsity limitations**: 90% sparse models rarely accelerate without specialized hardware—dense GEMM operations are too optimized
- **Structured sparsity benefits**: Removing entire channels enables speedup when architecturally implemented (smaller dense matrices, not just zeros)
- **Compression-accuracy curves**: Accuracy degrades gradually until critical sparsity threshold, then collapses—find the "knee" of the curve
- **Iterative pruning advantage**: Gradual compression with fine-tuning (10 steps × 10% sparsity increase) achieves higher compression with better accuracy than one-shot pruning
- **Multiplicative compression**: Combining techniques multiplies gains—90% pruning (10× reduction) + INT8 quantization (4× reduction) = 40× total compression

## Ready to Build?

You're about to implement compression techniques that transform research models into deployable systems. These optimizations bridge the gap between what's possible in the lab and what's practical in production on resource-constrained devices.

Understanding compression from first principles—implementing pruning algorithms yourself rather than using torch.nn.utils.prune—gives you deep insight into the trade-offs between model size, inference speed, and accuracy. You'll discover why most sparsity doesn't accelerate inference, when structured pruning actually helps, and how to design compression strategies for different deployment scenarios (mobile apps need aggressive compression; cloud services need balanced approaches).

This module emphasizes honest engineering: you'll see that achieving 90% sparsity is straightforward but getting speedup from that sparsity requires specialized hardware or libraries beyond these NumPy implementations. Production compression combines multiple techniques sequentially, carefully measuring accuracy after each stage and stopping when degradation exceeds acceptable thresholds.

Take your time with this module. Compression is where theory meets deployment constraints, where algorithmic elegance confronts hardware reality. The techniques you implement here enable real-world ML deployment at scale!

Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/16_compression/compression_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/16_compression/compression_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/16_compression/compression_dev.py
:class-header: bg-light

Browse the Python source code and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.
```

---

<div class="prev-next-area">
<a class="left-prev" href="../modules/15_quantization/ABOUT.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../modules/17_memoization/ABOUT.html" title="next page">Next Module →</a>
</div>
