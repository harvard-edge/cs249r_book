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
# Module 16: Compression - Pruning and Model Optimization

Welcome to Module 16! You are about to build model compression techniques that reduce parameter count and compute footprint: magnitude and structured pruning, low-rank SVD factorization, and knowledge distillation.

<div align="center">
  <div align="center">
  <img src="compression_blueprint.svg" alt="Compression Blueprint" width="380px">
</div>
</div>

## 🔗 Prerequisites & Progress

**You've Built**: End-to-end optimization pipeline with profiling (`14_profiling`) and simulated INT8 quantization (`15_quantization`).
**You'll Build**: Unstructured magnitude pruning, structured channel pruning, truncated SVD low-rank factorization, and teacher-student knowledge distillation with temperature scaling.
**You'll Enable**: Cascaded model compression workflows, measuring parameter sparsity, FLOP reductions, and output drift across compressed architectures.

### Architectural Roadmap

| Tier | Subsystem | Primitives & Capabilities | Status |
| :--- | :--- | :--- | :--- |
| **Modules 01–08** | Foundation Tier | `Tensor`, `Function`, `Linear`, `GELU`, `SGD`, `Adam`, `Trainer` | Completed |
| **Modules 09–13** | Architecture Tier | `Conv2d`, `BPETokenizer`, `EmbeddingLayer`, `MultiHeadAttention`, `GPT` | Completed |
| **Module 14** | Systems Diagnostics | `Profiler`, `count_flops`, `measure_memory`, `measure_latency` | Completed |
| **Module 15** | Precision Reduction | `quantize_int8`, `dequantize_int8`, `QuantizedLinear`, `quantize_model` | Completed |
| **Module 16** | **Model Compression** | `magnitude_prune`, `structured_prune`, `low_rank_approximate`, `KnowledgeDistillation` | **Active Subsystem** |
| **Modules 17–20** | Advanced Acceleration | `TritonKernels`, `KVCache`, `BenchmarkingSuite`, `TinyGPT` | Downstream Consumers |

## 🎯 Learning Objectives

By the end of this module, you will:
1. Implement global magnitude-based pruning and track sparsity ratios.
2. Build structured channel pruning that produces dense, hardware-accelerated submatrices.
3. Compute truncated SVD low-rank factorizations and verify parameter break-even thresholds ($r^*$).
4. Implement teacher-student knowledge distillation with temperature-scaled soft targets and KL divergence.
5. Compose multi-stage compression cascades targeting mobile and edge deployment budgets.

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/16_compression/compression.ipynb`
**Building Side:** Code exports to `tinytorch.perf.compression`

```python
from tinytorch.perf.compression import (
    measure_sparsity, magnitude_prune, structured_prune,
    low_rank_approximate, KnowledgeDistillation, compress_model, Compressor,
)
```
"""

# %% [markdown]
r"""
## 📋 Module Dependencies

### Dependency Inventory

| Module Dependency | Component Imported | Purpose in Compression Engine |
| :--- | :--- | :--- |
| **Module 01: Tensors** | `Tensor` | Multi-dimensional array container storing weights and gradients |
| **Module 02: Activations** | `ReLU` | Non-linear activations traversed in model graph |
| **Module 03: Layers** | `Linear`, `Sequential` | Neural layer abstraction for weight extraction and replacement |
| **Module 04: Losses** | `log_softmax` | Numerically stable log-probabilities for distillation loss |
| **Module 06: Autograd** | `autograd` | Automatic differentiation tracking gradients to student parameters |
| **Module 07: Optimizers** | `SGD` | Parameter update verification during student training |
| **Module 14: Profiling** | `Profiler` | Memory and parameter baseline accounting |
| **Standard Library / NumPy** | `copy`, `numpy` | Model cloning and vectorized linear algebra operations |
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": false}
#| default_exp perf.compression
#| export

import numpy as np
rng = np.random.default_rng(7)
import copy
from typing import Dict, Any

# Import from TinyTorch package (previous modules must be completed and exported)
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear, Sequential
from tinytorch.core.activations import ReLU
from tinytorch.core.losses import log_softmax
import tinytorch.core.autograd  # Module 06: record gradients for student training

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion

# %% [markdown]
"""
## 💡 Introduction: Why Compression Matters

Before we learn compression, let's profile a layer's weights with the Module 14
Profiler. The layer below is freshly initialized, so its weights follow the
initializer's distribution, not anything learned. That is exactly the point of
the exercise: a fresh layer already has a wide spread of magnitudes, and the
question every pruning method asks is how much of the small end can go. The
evidence that the answer is "most of it" comes from trained networks, which is
why Milestone 06 measures accuracy before and after pruning on real digits.
"""

# %%
# Profile weight distribution to discover pruning opportunities
# Module 14 (Profiling) must be completed before Module 16
from tinytorch.perf.profiling import Profiler

def show_weight_distribution_motivation():
    """Display weight distribution analysis - motivates compression techniques."""
    profiler = Profiler()

    # Create a model and analyze its weights
    model = Linear(512, 512)
    param_count = profiler.count_parameters(model)

    print("🧪 Profiling Parameter Distribution (freshly initialized Linear(512, 512)):\n")
    print(f"   Total parameters: {param_count:,}")
    print(f"   Model memory: {param_count * BYTES_PER_FLOAT32 / MB_TO_BYTES:.1f} MB (FP32)")

    # Analyze weight distribution
    weights = model.weight.data.flatten()
    abs_weights = np.abs(weights)
    largest = np.max(abs_weights)

    print("\n   |Weight| Statistics:")
    print(f"   Mean: {np.mean(abs_weights):.4f}")
    print(f"   Std:  {np.std(abs_weights):.4f}")
    print(f"   Min:  {np.min(abs_weights):.4f}")
    print(f"   Max:  {largest:.4f}")

    # Thresholds relative to the largest weight, so the table means the same
    # thing whatever scale the initializer (or a trained network) uses
    fractions = [0.01, 0.05, 0.10, 0.25]
    below = {}
    print("\n   Weights Below a Fraction of the Largest |Weight|:")
    print("   Threshold        |  Percentage")
    print("   -----------------|--------------")
    for fraction in fractions:
        below[fraction] = np.sum(abs_weights < fraction * largest) / len(weights) * 100
        print(f"   < {fraction:4.0%} of max   |  {below[fraction]:5.1f}%")

    print("\n💡 Key Observations (computed from this layer):")
    print(f"   • {below[0.10]:.0f}% of the weights are under a tenth of the largest one")
    print(f"   • {below[0.25]:.0f}% are under a quarter of it; the spread is wide even before training")
    print("   • A fresh layer proves only that magnitudes vary; trained networks are the evidence")
    print("     that the small end can go (Han et al., 2015, removed 50-90% of the weights of")
    print("     trained networks with little accuracy loss)")

    print("\n🎯 The Problem:")
    print("   Every weight, however small, costs the same to store and multiply.")

    print("\n✨ The Solution:")
    print("   Prune (remove) small weights:")
    print("   • Magnitude pruning: Set small weights to zero")
    print("   • Structured pruning: Zero entire neurons/channels so they can later be deleted")
    print("   • Then measure what it cost: this module measures output drift,")
    print("     Milestone 06 measures accuracy\n")

if __name__ == "__main__":
    show_weight_distribution_motivation()

# %% [markdown]
r"""
### Model Compression Concepts

Deep neural networks are typically over-parameterized: their parameter matrices contain extensive statistical redundancy. While over-parameterization facilitates convex optimization dynamics during gradient descent training, inference requires only a fraction of the trained weight volume.

Model compression bridges the gap between research models and production edge constraints by eliminating redundant parameters, factorizing low-rank manifolds, and distilling dark knowledge into compact student topologies.

<div align="center">
  <div align="center">
  <img src="compression_methods_overview.svg" alt="Compression Methods Overview" width="680px">
</div>
</div>

### The Model Compression Taxonomy

| Category | Compression Paradigm | Operational Mechanism | Hardware Acceleration Profile | Accuracy Trade-Off |
| :--- | :--- | :--- | :--- | :--- |
| **Weight-Based** | **Magnitude Pruning** | Zero out individual scalar weights $\lvert W_{i, j} \rvert < \tau$ | Requires sparse BLAS (CSR/COO); dense kernels see $0\times$ speedup | Minimal loss at $\le 70\%$ sparsity |
| **Weight-Based** | **Structured Pruning** | Zero out entire columns/channels $\lVert W_{:, c} \rVert_2 < \tau$ | Dense GEMM dimension reduction ($M \times N \to M \times N'$); immediate speedup | Higher risk of capacity collapse |
| **Weight-Based** | **Low-Rank SVD** | Truncate singular spectrum $W \approx A B$ | Replaces 1 large GEMM with 2 small GEMMs ($r(M+N)$ FLOPs) | Exact rank-dependent error bound |
| **Architecture-Based** | **Knowledge Distillation** | Train smaller student on softened teacher logits | Native execution on any hardware target at full student throughput | Often matches or exceeds raw student |
"""

# %% [markdown]
r"""
## 📐 Foundations: Mathematical Formulations

Understanding the mathematical principles behind compression enables disciplined trade-offs between parameter volume, arithmetic intensity, and representation fidelity.

### 1. Magnitude-Based Pruning (Unstructured)

Magnitude pruning hypothesizes that weights with the smallest absolute values contribute least to output activations. For a weight matrix $W \in \mathbb{R}^{M \times N}$, a binary pruning mask $M \in \{0, 1\}^{M \times N}$ is evaluated via a magnitude cutoff threshold $\tau$:

$$M_{i, j} = \begin{cases} 1 & \text{if } |W_{i, j}| \ge \tau \\ 0 & \text{otherwise} \end{cases}, \quad W_{\text{pruned}} = W \odot M$$

Given target sparsity $S \in [0, 1]$, the scalar threshold $\tau$ satisfies:

$$\tau = \text{Quantile}\left( \{|W_{i, j}|\}, \, S \right)$$

$$\text{Global Sparsity} = \frac{\sum_{i, j} \mathbf{1}(W_{i, j} = 0)}{M \times N} = 1 - \frac{\|W_{\text{pruned}}\|_0}{\|W\|_0}$$

### 2. Structured Channel Pruning

Unstructured pruning leaves scattered scalar zeros in dense storage arrays, requiring sparse BLAS representations (e.g. Compressed Sparse Row/Column, COO) that rarely yield wall-clock speedups on modern GPU tensor cores unless sparsity exceeds $80\%\text{--}90\%$.

Structured pruning eliminates contiguous parameter slices (such as entire output channels or column vectors in linear projections), directly shrinking the matrix dimensions:

$$\|W_{:, c}\|_2 = \sqrt{\sum_{r=1}^M W_{r, c}^2}, \quad c \in \{1, \dots, N\}$$

Channels with the lowest $\ell_2$-norm are zeroed out or excised, reducing dense matrix multiply dimensions from $(M \times N)$ to $(M \times N')$, delivering guaranteed FLOP reductions on dense hardware.

### 3. Knowledge Distillation & Dark Knowledge

Knowledge distillation trains a compact student model $\mathcal{S}$ with parameters $\theta_s$ to mimic the continuous predictive probability distribution of an over-parameterized teacher $\mathcal{T}$ with frozen parameters $\theta_t$.

The total objective function balances soft target mimicry with ground-truth supervised cross-entropy:

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{soft}} + (1 - \alpha) \cdot \mathcal{L}_{\text{hard}}$$

$$\mathcal{L}_{\text{soft}} = D_{\text{KL}}\left( \sigma\left(\frac{z_t}{T}\right) \,\Big\|\, \sigma\left(\frac{z_s}{T}\right) \right) = \sum_{k=1}^C p_k^t \log \left(\frac{p_k^t}{p_k^s}\right)$$

where temperature parameter $T > 1$ softens probability distributions over output classes $C$:

$$p_k = \sigma\left(\frac{z}{T}\right)_k = \frac{\exp(z_k / T)}{\sum_{j=1}^C \exp(z_j / T)}$$

<div align="center">
  <div align="center">
  <img src="distillation_temperature_dark_knowledge.svg" alt="Distillation Temperature Softening" width="320px">
</div>
</div>

At $T=1$, the softmax distribution is peaked, suppressing inter-class correlations ("dark knowledge"). At $T=3\text{--}5$, minor class probabilities rise by orders of magnitude, exposing structural similarities between related semantic categories (e.g., distinguishing between a sedan and an SUV versus a truck).

### 4. Low-Rank Approximation (Truncated SVD)

By the Eckart-Young-Mirsky Theorem, the optimal rank-$r$ approximation of a weight matrix $W \in \mathbb{R}^{M \times N}$ in Frobenius norm is obtained via truncated Singular Value Decomposition:

$$W = U \Sigma V^T \approx U_r \Sigma_r V_r^T$$

where $U_r \in \mathbb{R}^{M \times r}$, $\Sigma_r = \text{diag}(\sigma_1, \dots, \sigma_r) \in \mathbb{R}^{r \times r}$, and $V_r^T \in \mathbb{R}^{r \times N}$.

<div align="center">
  <div align="center">
  <img src="svd_rank_breakeven.svg" alt="SVD Rank Break-Even Curve" width="680px">
</div>
</div>

By absorbing $\Sigma_r$ into $U_r$ or $V_r^T$, the linear projection $y = x W^T$ is factorized into two cascaded low-rank operations:

$$A = U_r \sqrt{\Sigma_r} \in \mathbb{R}^{M \times r}, \quad B = \sqrt{\Sigma_r} V_r^T \in \mathbb{R}^{r \times N} \implies W \approx A B$$

$$\text{Parameter Volume}: \quad P_{\text{dense}} = M \times N \quad \longrightarrow \quad P_{\text{factorized}} = r \cdot (M + N + 1)$$

$$\text{Break-Even Rank Threshold } r^*: \quad r^* = \frac{M \cdot N}{M + N + 1} \quad \left(\text{When } M = N, \quad r^* \approx \frac{N}{2}\right)$$

If the chosen rank $r$ exceeds $r^*$, the factorized representation contains *more* parameters and requires *more* FLOPs than the original dense layer!
"""

# %% [markdown]
r"""
## 🏗️ Implementation: Sparsity Measurement

Sparsity quantifies the proportion of non-contributing (zero-valued) parameters within a network graph:

$$\text{Sparsity} = \frac{N_{\text{zeros}}}{N_{\text{total}}} = \frac{\sum_{p \in \Theta} \sum_{i} \mathbf{1}(p_i = 0)}{\sum_{p \in \Theta} \text{size}(p)}$$

| Matrix State | Active Non-Zero Elements | Memory Footprint (FP32) | Hardware Execution Behavior |
| :--- | :--- | :--- | :--- |
| **Dense Matrix ($0\%$ Sparsity)** | $M \times N$ ($100\%$ active) | $M \times N \times 4$ bytes | Peak memory bandwidth and dense GEMM throughput |
| **Unstructured Sparse ($75\%$ Sparsity)** | $0.25 \times M \times N$ active | $M \times N \times 4$ bytes (Dense) / $0.5 \times M \times N \times 4$ bytes (COO) | No dense speedup without sparse library; memory access divergence |
| **Structured Sparse ($50\%$ Channels Zeroed)** | $0.50 \times M \times N$ active | $0.5 \times M \times N \times 4$ bytes (after reshape) | Immediate $2\times$ reduction in dense GEMM FLOPs and latency |
"""

# %% nbgrader={"grade": false, "grade_id": "measure-sparsity", "solution": true}
#| export
def measure_sparsity(model) -> float:
    """
    Calculate the percentage of zero weights in a model.

    TODO: Count zero weights and total weights across all layers

    APPROACH:
    1. Iterate through all model parameters
    2. Count zeros using np.sum(weights == 0)
    3. Count total parameters
    4. Return percentage: zeros / total * 100

    Args:
        model: Model with .parameters() method

    Returns:
        Sparsity percentage (0.0-100.0)

    EXAMPLE:
    >>> # Create test model with explicit composition
    >>> layer1 = Linear(10, 5)
    >>> layer2 = Linear(5, 2)
    >>> model = Sequential(layer1, layer2)
    >>> sparsity = measure_sparsity(model)
    >>> print(f"Model sparsity: {sparsity:.1f}%")
    Model sparsity: 0.0%  # Before pruning

    HINT: Use np.sum() to count zeros efficiently
    """
    ### BEGIN SOLUTION role="scaffold"
    total_params = 0
    zero_params = 0

    for param in model.parameters():
        # Only count weight matrices (2D), not biases (1D)
        # Biases are often initialized to zero, which would skew sparsity
        if len(param.shape) > 1:
            total_params += param.size
            zero_params += np.sum(param.data == 0)

    if total_params == 0:
        return 0.0

    return (zero_params / total_params) * 100.0
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Sparsity Measurement

This test validates our sparsity measurement function works correctly.

**What we're testing**: Zero weight counting and percentage calculation
**Why it matters**: Accurate sparsity measurement is essential for compression evaluation
**Expected**: Correct sparsity percentages for dense and sparse models
"""

# %% nbgrader={"grade": true, "grade_id": "test-measure-sparsity", "locked": true, "points": 5}
def test_unit_measure_sparsity():
    """🧪 Test sparsity measurement functionality."""
    print("🧪 Unit Test: Measure Sparsity...")

    # Test with dense model - explicit composition shows structure
    layer1 = Linear(4, 3)
    layer2 = Linear(3, 2)
    model = Sequential(layer1, layer2)  # Test helper for parameter collection

    initial_sparsity = measure_sparsity(model)
    assert initial_sparsity < 1.0, f"Expected <1% sparsity (dense model), got {initial_sparsity}%"

    # Test with manually sparse model - students see which weights are zeroed
    layer1.weight.data[0, 0] = 0  # Zero out specific weight
    layer1.weight.data[1, 1] = 0  # Zero out another weight
    sparse_sparsity = measure_sparsity(model)
    assert sparse_sparsity > 0, f"Expected >0% sparsity, got {sparse_sparsity}%"

    print("✅ measure_sparsity works correctly!")

if __name__ == "__main__":
    test_unit_measure_sparsity()

# %% [markdown]
r"""
## 🏗️ Magnitude-Based Pruning

Magnitude pruning is the foundational weight-level compression technique. It operates on the hypothesis that weights with near-zero absolute values contribute minimally to downstream activation norms and gradient flow.

### How Magnitude Pruning Works

In an unstructured setting, pruning computes an empirical magnitude threshold across tensor weights and masks values below it:

$$\tau = \text{Quantile}\left(\{|W_{i, j}|\}, \, S\right), \quad M_{i, j} = \mathbf{1}(|W_{i, j}| \ge \tau), \quad W_{\text{pruned}} = W \odot M$$

| Pipeline Step | Operational Description | Concrete Example Values | State Transformation |
| :--- | :--- | :--- | :--- |
| **1. Collect Weights** | Flatten all layer parameter tensors into a single array | $L_1: [2.1, 0.08, -1.8, 0.04, 3.2, \dots]$, $L_2: [0.7, 2.4, -0.05, \dots]$ | 20 FP32 parameters |
| **2. Rank Magnitudes** | Compute absolute values $\lvert W \rvert$ and sort stably | $[0.01, 0.02, 0.03, 0.04, 0.05, \dots, 2.4, 2.8, 3.2]$ | Ranked magnitude list |
| **3. Compute Threshold** | Find quantile cutoff corresponding to target sparsity $S$ | For $S = 50\%$, median value between 10th and 11th rank | Threshold $\tau \approx 0.40$ |
| **4. Apply Binary Mask** | Zero out elements where $\lvert W_{i, j} \rvert < \tau$ in-place | $L_1: [2.1, 0.0, -1.8, 0.0, 3.2, \dots]$, $L_2: [0.7, 2.4, 0.0, \dots]$ | $50\%$ sparse weight matrix |

### Memory Impact: Dense vs. Coordinate Sparse (COO) Formats

| Format | Storage Schema | Footprint at $0\%$ Sparsity | Footprint at $50\%$ Sparsity | Footprint at $90\%$ Sparsity | Hardware Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dense Tensor** | Flat buffer $N \times 4\text{ B}$ | $20 \times 4 = 80\text{ B}$ | $20 \times 4 = 80\text{ B}$ ($0\times$ saving) | $20 \times 4 = 80\text{ B}$ ($0\times$ saving) | Baseline ($1.0\times$) |
| **Coordinate (COO)** | $(\text{val}, \text{idx}) \to 4\text{ B} + 4\text{ B}$ | $20 \times 8 = 160\text{ B}$ ($2\times$ overhead) | $10 \times 8 = 80\text{ B}$ ($0\times$ saving) | $2 \times 8 = 16\text{ B}$ ($80\%$ saving) | Requires sparse BLAS; slow $<80\%$ |
| **Compressed Row (CSR)** | $\text{val}[4\text{B}] + \text{col}[2\text{B}] + \text{ptr}$ | $20 \times 6 + 12 = 132\text{ B}$ | $10 \times 6 + 12 = 72\text{ B}$ ($10\%$ saving) | $2 \times 6 + 12 = 24\text{ B}$ ($70\%$ saving) | Speedup only at $\ge 85\%\text{--}90\%$ sparsity |

### Global vs. Layer-Wise Pruning

Global thresholding treats all parameters across the entire network as a unified distribution, selecting a single threshold $\tau$ for the whole model:

- **Preserved Network Capacity**: Layers with higher dynamic range and larger gradient impact automatically retain more parameters.
- **Layer-Sensitivity Awareness**: Automatically allocates higher sparsity to redundant layers (e.g. wide intermediate feed-forward projections) while protecting sensitive bottleneck layers.
- **Boundary Risk**: Extremely deep architectures can experience layer collapse if an entire layer's weights fall below $\tau$; layer-wise guardrails or minimum density constraints are applied in production pipelines.
"""

# %% nbgrader={"grade": false, "grade_id": "magnitude-prune", "solution": true}
#| export
def magnitude_prune(model, sparsity=0.9):
    """
    Remove weights with smallest magnitudes to achieve target sparsity.

    TODO: Implement global magnitude-based pruning

    APPROACH:
    1. Collect all weights from the model
    2. Calculate absolute values to get magnitudes
    3. Rank magnitudes and select floor(sparsity * weight_count) entries
    4. Zero those entries in-place; stable ordering breaks magnitude ties

    EXAMPLE:
    >>> # Create model with explicit layer composition
    >>> layer1 = Linear(100, 50)
    >>> layer2 = Linear(50, 10)
    >>> model = Sequential(layer1, layer2)
    >>> original_params = sum(p.size for p in model.parameters())
    >>> magnitude_prune(model, sparsity=0.8)
    >>> final_sparsity = measure_sparsity(model)
    >>> print(f"Achieved {final_sparsity:.1f}% sparsity")
    Achieved 80.0% sparsity

    HINTS:
    - Use np.argsort(..., kind="stable") to rank even equal magnitudes
    - Modify model parameters in-place
    - Consider only weight matrices, not biases
    """
    ### BEGIN SOLUTION
    if not np.isfinite(sparsity) or not 0 <= sparsity <= 1:
        raise ValueError("sparsity must be between 0 and 1")
    weight_params = [p for p in model.parameters() if p.ndim > 1]
    if not weight_params:
        return model

    # Rank globally, including existing zeros. Selecting by index also handles
    # tied magnitudes and the endpoints 0% and 100% exactly.
    magnitudes = np.concatenate([np.abs(p.data).ravel() for p in weight_params])
    prune_count = int(sparsity * magnitudes.size)
    prune_mask = np.zeros(magnitudes.size, dtype=bool)
    prune_mask[np.argsort(magnitudes, kind="stable")[:prune_count]] = True
    offset = 0
    for param in weight_params:
        mask = prune_mask[offset:offset + param.size].reshape(param.shape)
        param.data[mask] = 0
        offset += param.size
    return model
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Magnitude Pruning

This test validates magnitude-based pruning works correctly with threshold selection.

**What we're testing**: Weight removal based on magnitude threshold
**Why it matters**: Core technique for creating sparse neural networks
**Expected**: Achieves target sparsity with smallest weights removed
"""

# %% nbgrader={"grade": true, "grade_id": "test-magnitude-prune", "locked": true, "points": 10}
def test_unit_magnitude_prune():
    """🧪 Test magnitude-based pruning functionality."""
    print("🧪 Unit Test: Magnitude Prune...")

    # Create test model with explicit composition - students see structure
    layer1 = Linear(4, 3)
    layer2 = Linear(3, 2)
    model = Sequential(layer1, layer2)

    # Set specific weight values for predictable testing
    # Students can see exactly which weights we're testing
    layer1.weight.data = np.array([
        [1.0, 2.0, 3.0],    # Large weights - should survive pruning
        [0.1, 0.2, 0.3],    # Medium weights
        [4.0, 5.0, 6.0],    # Large weights - should survive pruning
        [0.01, 0.02, 0.03]  # Tiny weights - will be pruned
    ])

    initial_sparsity = measure_sparsity(model)
    assert initial_sparsity < 1.0, "Model should start with minimal sparsity (<1%)"

    # Apply 50% pruning - removes smallest 50% of weights
    magnitude_prune(model, sparsity=0.5)
    final_sparsity = measure_sparsity(model)

    # Should achieve approximately 50% sparsity
    assert 40 <= final_sparsity <= 60, f"Expected ~50% sparsity, got {final_sparsity}%"

    # Verify largest weights survived - students understand pruning criteria
    remaining_weights = layer1.weight.data[layer1.weight.data != 0]
    assert len(remaining_weights) > 0, "Some weights should remain"
    assert np.all(np.abs(remaining_weights) >= 0.1), "Large weights should survive"

    print("✅ magnitude_prune works correctly!")

if __name__ == "__main__":
    test_unit_magnitude_prune()

# %% [markdown]
r"""
## 🏗️ Structured Pruning

While unstructured magnitude pruning introduces fine-grained, scattered zeros throughout parameter arrays, structured pruning removes entire architectural units—such as output channels, projection columns, or attention heads. This directly alters tensor dimensions, producing compact dense matrices that standard dense BLAS libraries (cuBLAS, OpenBLAS, Apple Accelerate) can execute with immediate wall-clock speedup.

### Unstructured vs. Structured Sparsity Comparison

| Property | Unstructured Sparsity (Scalar Pruning) | Structured Sparsity (Channel Pruning) |
| :--- | :--- | :--- |
| **Pruning Granularity** | Individual weights $W_{i, j} = 0$ | Entire slice $W_{:, c} = \mathbf{0}$ or sliced out |
| **Tensor Dimensions** | Preserved $(M \times N)$ | Reduced $(M \times N')$, where $N' = (1 - P) N$ |
| **Memory Access Pattern** | Non-contiguous, index-divergent, irregular | Fully contiguous, cache-line aligned, coalesceable |
| **Hardware Acceleration** | Requires custom sparse GEMM kernels ($\ge 80\%$ sparsity threshold) | Immediate linear speedup on standard dense GEMM hardware |
| **Accuracy Retention** | Exceptional (minimizes loss at $\le 75\%$ sparsity) | Moderate (removing full channels risks capacity drop) |
| **Index Overhead** | $4\text{--}8\text{ bytes}$ per non-zero value (COO/CSR) | $0\text{ bytes}$ (tensor is simply smaller) |

### Channel Importance Ranking Formulations

To prune $k$ channels with minimal loss of representational fidelity, each channel $c \in \{1, \dots, N\}$ is assigned an importance score $\mathcal{I}_c$:

$$\begin{aligned}
\text{Method 1: } \ell_2\text{-Norm (Standard)} \quad & \mathcal{I}_c = \|W_{:, c}\|_2 = \sqrt{\sum_{r=1}^M W_{r, c}^2} \\
\text{Method 2: Activation Magnitude} \quad & \mathcal{I}_c = \mathbb{E}_{x \sim \mathcal{D}}\left[ \frac{1}{B} \sum_{b=1}^B |y_{b, c}| \right] \\
\text{Method 3: First-Order Taylor Expansion} \quad & \mathcal{I}_c = \left| \sum_{r=1}^M \frac{\partial \mathcal{L}}{\partial W_{r, c}} \cdot W_{r, c} \right|
\end{aligned}$$

| Channel Ranking Step | Operational Procedure | Systems Implication |
| :--- | :--- | :--- |
| **1. Compute Channel Norms** | Calculate $\mathcal{I}_c = \lVert W_{:, c} \rVert_2$ for all $c \in \{0, \dots, N-1\}$ | Vectorized reduction across columns (BLAS-1 `nrm2`) |
| **2. Rank Channel Importance** | Sort indices $c$ by $\mathcal{I}_c$ in ascending order | Minimal sort overhead over $N$ scalar norms |
| **3. Identify Prune Set** | Select bottom $\lfloor \text{prune\_ratio} \times N \rfloor$ channels | Preserves highest-energy representations |
| **4. Mask / Excise Channels** | Zero out entire columns $W_{:, c} = 0$ (or physically slice $W$) | Slices dense GEMM memory traffic from $M \times N$ to $M \times N'$ |

### Hardware Benefits of Structured Sparsity

1. **Memory Coalescing**: DRAM burst transactions fetch 32-byte or 64-byte contiguous cache lines. Structured pruning preserves dense contiguous row/column traversal, achieving near-100% bus utilization.
2. **SIMD / Tensor Core Vectorization**: Hardware vector units (AVX-512, NEON, Tensor Cores) require contiguous vectors of 8, 16, or 32 elements. Excising full channels maintains full vector lane occupancy without zero-mask predication.
3. **Zero Indexing Overhead**: Avoids pointer arrays and index indirection (`ptr[i]`, `col_idx[k]`), eliminating pointer chase stalls and TLB misses.
4. **Cache Residency**: Reducing tensor dimensions decreases L1/L2 cache footprint, fitting working sets entirely within SRAM.
"""

# %% nbgrader={"grade": false, "grade_id": "structured-prune", "solution": true}
#| export
def structured_prune(model, prune_ratio=0.5):
    """
    Remove entire channels/neurons based on L2 norm importance.

    TODO: Implement structured pruning for Linear layers

    APPROACH:
    1. For each hidden Linear layer (every Linear but the last, whose output
       channels are the classes), calculate the L2 norm of each output channel
    2. Rank channels by importance (L2 norm)
    3. Remove lowest importance channels by setting to zero
    4. This creates block sparsity that's hardware-friendly

    EXAMPLE:
    >>> # Create model with explicit layers
    >>> layer1 = Linear(100, 50)
    >>> layer2 = Linear(50, 10)
    >>> model = Sequential(layer1, layer2)
    >>> original_shape = layer1.weight.shape
    >>> structured_prune(model, prune_ratio=0.3)
    >>> # 30% of layer1's channels are now completely zero; the head (layer2) is untouched
    >>> final_sparsity = measure_sparsity(model)
    >>> print(f"Structured sparsity: {final_sparsity:.1f}%")
    Structured sparsity: 27.3%

    HINTS:
    - Calculate L2 norm for all channels at once: np.linalg.norm(weight, axis=0)
    - Find the lowest-norm channels: np.argpartition(norms, k)[:k] or np.argsort(norms)[:k]
    - Set entire channels to zero: weight[:, prune_indices] = 0
    """
    ### BEGIN SOLUTION role="scaffold"
    if not np.isfinite(prune_ratio) or not 0 <= prune_ratio <= 1:
        raise ValueError("prune_ratio must be between 0 and 1")
    # Prune the hidden Linear layers. The last Linear is the head: its output
    # channels are the classes, so zeroing them removes classes, not neurons.
    # A model with a single Linear has nothing else to prune and is pruned as is.
    def collect_linears(layer):
        # Sequential containers may contain other Sequential containers. Walk
        # their execution order so the terminal classifier is identified once.
        if isinstance(layer, Linear):
            return [layer]
        return [linear for child in getattr(layer, 'layers', [])
                for linear in collect_linears(child)]

    linears = collect_linears(model)
    hidden = linears[:-1] if len(linears) > 1 else linears

    for layer in hidden:
        weight = layer.weight.data

        # Calculate L2 norm for each output channel (column)
        channel_norms = np.linalg.norm(weight, axis=0)

        # Find channels to prune (lowest importance)
        num_channels = weight.shape[1]
        num_to_prune = int(num_channels * prune_ratio)

        if num_to_prune > 0:
            # Get indices of channels to prune (smallest norms)
            prune_indices = np.argsort(channel_norms, kind="stable")[:num_to_prune]

            # Zero out entire channels
            weight[:, prune_indices] = 0

            # Also zero corresponding bias elements if bias exists
            if layer.bias is not None:
                layer.bias.data[prune_indices] = 0

    return model
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Structured Pruning

This test validates structured pruning removes entire channels correctly.

**What we're testing**: Channel-wise pruning based on L2 norm importance
**Why it matters**: Creates hardware-friendly sparsity patterns
**Expected**: Entire channels zeroed, not scattered weights
"""

# %% nbgrader={"grade": true, "grade_id": "test-structured-prune", "locked": true, "points": 10}
def test_unit_structured_prune():
    """🧪 Test structured pruning functionality."""
    print("🧪 Unit Test: Structured Prune...")

    # Create test model with explicit layers - students see the architecture
    layer1 = Linear(4, 6)
    layer2 = Linear(6, 2)
    model = Sequential(layer1, layer2)

    # Set predictable weights for testing
    # Students can see channel importance: col 0,2,4 = large, col 1,3,5 = small
    layer1.weight.data = np.array([
        [1.0, 0.1, 2.0, 0.05, 3.0, 0.01],  # Channels with varying importance
        [1.1, 0.11, 2.1, 0.06, 3.1, 0.02],  # Large values in columns 0,2,4
        [1.2, 0.12, 2.2, 0.07, 3.2, 0.03],  # Small values in columns 1,3,5
        [1.3, 0.13, 2.3, 0.08, 3.3, 0.04]   # Pruning removes small channels
    ])

    initial_sparsity = measure_sparsity(model)
    assert initial_sparsity < 1.0, "Model should start with minimal sparsity (<1%)"

    # Apply 33% structured pruning (2 out of 6 channels)
    # This removes entire channels, not scattered weights
    structured_prune(model, prune_ratio=0.33)
    final_sparsity = measure_sparsity(model)

    # Check that some channels are completely zero
    weight = layer1.weight.data
    zero_channels = np.sum(np.all(weight == 0, axis=0))
    assert zero_channels >= 1, f"Expected at least 1 zero channel, got {zero_channels}"

    # Check that non-zero channels are completely preserved
    # This is structured pruning - entire channels are zero or non-zero
    for col in range(weight.shape[1]):
        channel = weight[:, col]
        assert np.all(channel == 0) or np.all(channel != 0), "Channels should be fully zero or fully non-zero"

    print("✅ structured_prune works correctly!")

if __name__ == "__main__":
    test_unit_structured_prune()

# %% [markdown]
"""
## 🏗️ Low-Rank Approximation

Low-rank approximation discovers that large weight matrices often contain redundant information that can be captured with much smaller matrices through mathematical decomposition.

### The Intuition Behind Low-Rank Approximation

Imagine you're storing a massive spreadsheet where many columns are highly correlated. Instead of storing all columns separately, you could store a few "basis" columns and coefficients for how to combine them to recreate the original data.

```
Low-Rank Decomposition Visualization:

Three stored factors for rank k = 2:
    W (4×5) ≈ U (4×2) @ diag(S (2,)) @ V^T (2×5)

Parameter Reduction:
- Original: 4 × 5 = 20 parameters
- Compressed: (4 × 2) + 2 + (2 × 5) = 20 parameters
- Size ratio: 20/20 = 1.0 (no savings for this small example)

For larger matrices, savings become dramatic:
- W (1000×1000): 1M parameters
- U (1000×100) + S (100,) + V^T (100×1000): 200,100 parameters
- Size ratio: 0.2001 (79.99% savings)
```

### SVD: The Mathematical Foundation

Singular Value Decomposition (SVD) finds the optimal low-rank approximation by identifying the most important "directions" in the data:

```
SVD Decomposition:
    W = U × Σ × V^T

Where:
    U: Left singular vectors (input patterns)
    Σ: Singular values (importance weights)
    V^T: Right singular vectors (output patterns)

Truncated SVD (Rank-k approximation):
    W ≈ U[:,:k] × Σ[:k] × V^T[:k,:]

Quality vs Compression Trade-off:
    Higher k → Better approximation, less compression
    Lower k → More compression, worse approximation

Choosing Optimal Rank:
    Method 1: Fixed ratio (k = ratio × min(m,n))
    Method 2: Energy threshold (keep 90% of singular value energy)
    Method 3: Error threshold (reconstruction error < threshold)
```

### When Low-Rank Works Best

Low-rank approximation works well when:
- **Matrices are large**: Compression benefits scale with size
- **Data has structure**: Correlated patterns enable compression
- **Moderate accuracy loss acceptable**: Some precision traded for efficiency

It works poorly when:
- **Matrices are already small**: Overhead exceeds benefits
- **Data is random**: No patterns to exploit
- **High precision required**: SVD introduces approximation error
"""

# %% nbgrader={"grade": false, "grade_id": "low-rank-approx", "solution": true}
#| export

def low_rank_approximate(weight_matrix, rank_ratio=0.5):
    """
    Approximate weight matrix using low-rank decomposition (SVD).

    TODO: Implement SVD-based low-rank approximation

    APPROACH:
    1. Perform SVD: W = U @ S @ V^T
    2. Keep only top k singular values where k = rank_ratio * min(dimensions)
    3. Reconstruct: W_approx = U[:,:k] @ diag(S[:k]) @ V[:k,:]
    4. Return decomposed matrices for memory savings

    EXAMPLE:
    >>> weight = rng.standard_normal((100, 50))
    >>> U, S, V = low_rank_approximate(weight, rank_ratio=0.3)
    >>> # Original: 100*50 = 5000 params
    >>> # Compressed: 100*15 + 15 + 15*50 = 2265 params (54.7% reduction)

    HINTS:
    - Use np.linalg.svd() for decomposition
    - Choose k = int(rank_ratio * min(m, n))
    - Return copies of U[:,:k], S[:k], V[:k,:] for reconstruction
    - Copies free the full SVD buffers after this function returns
    """
    ### BEGIN SOLUTION role="scaffold"
    if not np.isfinite(rank_ratio) or not 0 < rank_ratio <= 1:
        raise ValueError("rank_ratio must be in (0, 1]")
    m, n = weight_matrix.shape

    # Perform SVD
    U, S, V = np.linalg.svd(weight_matrix, full_matrices=False)

    # Determine target rank
    max_rank = min(m, n)
    target_rank = max(1, int(rank_ratio * max_rank))

    # Copy the slices so they release the full SVD backing arrays. Views
    # would keep the uncompressed factors alive despite their smaller shapes.
    U_truncated = U[:, :target_rank].copy()
    S_truncated = S[:target_rank].copy()
    V_truncated = V[:target_rank, :].copy()

    return U_truncated, S_truncated, V_truncated
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Low-Rank Approximation

This test validates SVD-based matrix factorization for compression.

**What we're testing**: Truncated SVD decomposition and reconstruction
**Why it matters**: Enables significant parameter reduction for large matrices
**Expected**: Correct factorization with acceptable reconstruction error
"""

# %% nbgrader={"grade": true, "grade_id": "test-low-rank", "locked": true, "points": 10}
def test_unit_low_rank_approximate():
    """🧪 Test low-rank approximation functionality."""
    print("🧪 Unit Test: Low-Rank Approximate...")

    # Create test weight matrix
    original_weight = rng.standard_normal((20, 15))
    original_params = original_weight.size

    # Apply low-rank approximation
    U, S, V = low_rank_approximate(original_weight, rank_ratio=0.4)

    # Check dimensions
    target_rank = int(0.4 * min(20, 15))  # min(20,15) = 15, so 0.4*15 = 6
    assert U.shape == (20, target_rank), f"Expected U shape (20, {target_rank}), got {U.shape}"
    assert S.shape == (target_rank,), f"Expected S shape ({target_rank},), got {S.shape}"
    assert V.shape == (target_rank, 15), f"Expected V shape ({target_rank}, 15), got {V.shape}"

    # Check parameter reduction
    compressed_params = U.size + S.size + V.size
    compression_ratio = compressed_params / original_params
    assert compression_ratio < 1.0, f"Should compress, but ratio is {compression_ratio}"

    # Check reconstruction quality
    reconstructed = U @ np.diag(S) @ V
    reconstruction_error = np.linalg.norm(original_weight - reconstructed)
    relative_error = reconstruction_error / np.linalg.norm(original_weight)
    # Low-rank approximation trades accuracy for compression - error is expected
    assert relative_error < 0.7, f"Reconstruction error too high: {relative_error}"

    print("✅ low_rank_approximate works correctly!")

if __name__ == "__main__":
    test_unit_low_rank_approximate()

# %% [markdown]
r"""
## 🏗️ Knowledge Distillation

Knowledge distillation transfers the dark knowledge embedded in a wide, highly accurate teacher model $\mathcal{T}$ into an architecturally compact student model $\mathcal{S}$. While traditional supervised learning forces models to fit sparse, one-hot ground-truth labels, distillation exposes continuous relative probabilities across all negative classes.

<div align="center">
  <div align="center">
  <img src="distillation_temperature_dark_knowledge.svg" alt="Distillation Temperature and Dark Knowledge" width="320px">
</div>
</div>

### Teacher vs. Student Systems Profile

| Model Role | Parameters | FP32 Footprint | Batch Latency | Relative Accuracy | Deployment Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Teacher ($\mathcal{T}$)** | $100\text{M}$ | $400\text{ MB}$ | $500\text{ ms}$ | $95.0\%$ (Baseline) | Cloud Training Server (Multi-GPU) |
| **Student ($\mathcal{S}$)** | $10\text{M}$ ($10\times$ smaller) | $40\text{ MB}$ ($10\times$ smaller) | $50\text{ ms}$ ($10\times$ faster) | $93.2\%$ ($1.8\%$ gap) | Edge Device / Mobile SoC |

### Temperature Scaling: Revealing Dark Knowledge

At standard temperature $T = 1$, standard softmax exponentiates dominant logits, crushing sub-dominant probabilities to near zero. Temperature scaling softens this distribution, revealing fine-grained cross-class relationships:

$$p_i(z, T) = \frac{\exp(z_i / T)}{\sum_{j=1}^C \exp(z_j / T)}$$

| Distribution Parameter | Class 1 (Cat) | Class 2 (Dog) | Class 3 (Car) | Distribution Entropy | Gradient Signal Dynamics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Teacher Raw Logits $z$** | $1.0$ | $2.0$ | $0.5$ | — | Unbounded linear pre-activations |
| **Standard Softmax ($T = 1$)** | $0.231$ | $0.628$ | $0.140$ | $0.94\text{ nats}$ | Hardened decisions; negligible gradient on negative classes |
| **Softened Softmax ($T = 3$)** | $0.312$ | $0.432$ | $0.256$ | $1.07\text{ nats}$ | Reveals dog/cat visual overlap ("dark knowledge"); rich gradients |
| **Extreme Softmax ($T = 10$)** | $0.332$ | $0.367$ | $0.301$ | $1.10\text{ nats}$ | High entropy; approaches uniform distribution ($1/C = 0.333$) |

### Distillation Loss Formulation

The complete training objective balances continuous distribution matching against discrete ground-truth cross-entropy:

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{soft}} + (1 - \alpha) \cdot \mathcal{L}_{\text{hard}}$$

$$\mathcal{L}_{\text{soft}} = D_{\text{KL}}\left( p(z_T, T) \,\|\, q(z_S, T) \right) = \sum_{i=1}^C p_i(z_T, T) \log \left( \frac{p_i(z_T, T)}{q_i(z_S, T)} \right)$$

$$\mathcal{L}_{\text{hard}} = \text{CrossEntropy}\left(q(z_S, 1), \, y_{\text{true}}\right) = -\sum_{i=1}^C y_{\text{true}, i} \log q_i(z_S, 1)$$

| Hyperparameter | Canonical Range | Optimization Objective | Boundary Effect |
| :--- | :--- | :--- | :--- |
| **Distillation Weight $\alpha$** | $0.5\text{--}0.8$ | Regulates gradient contribution of teacher vs. ground-truth | $\alpha = 1.0$: Pure imitation; $\alpha = 0.0$: Regular training |
| **Distillation Temperature $T$** | $2.0\text{--}5.0$ | Expands logit entropy to transmit non-argmax correlations | $T \to \infty$: Uniform probabilities; $T = 1.0$: Standard argmax |

Both loss components average across mini-batch samples, ensuring scale-invariant optimization regardless of batch dimension. The teacher parameters remain strictly frozen; autograd gradients flow only through the student's log-probability computational graph.
"""

# %% nbgrader={"grade": false, "grade_id": "distillation", "solution": true}
#| export
class KnowledgeDistillation:
    """
    Knowledge distillation for model compression.

    Train a smaller student model to mimic a larger teacher model.
    """

    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.7):
        """
        Initialize knowledge distillation.

        TODO: Set up teacher and student models with distillation parameters

        APPROACH:
        1. Store teacher and student models
        2. Set temperature for softening probability distributions
        3. Set alpha for balancing hard vs soft targets

        EXAMPLE:
        >>> # Create teacher with more capacity (explicit layers)
        >>> teacher_l1 = Linear(100, 200)
        >>> teacher_l2 = Linear(200, 50)
        >>> teacher = Sequential(teacher_l1, teacher_l2)
        >>>
        >>> # Create smaller student (explicit layer)
        >>> student = Sequential(Linear(100, 50))
        >>>
        >>> kd = KnowledgeDistillation(teacher, student, temperature=4.0, alpha=0.8)
        >>> print(f"Temperature: {kd.temperature}, Alpha: {kd.alpha}")
        Temperature: 4.0, Alpha: 0.8

        HINTS:
        - Simply assign the parameters to instance variables
        - Temperature typically ranges from 3-5 for effective softening
        - Alpha of 0.7 means 70% soft targets, 30% hard targets

        Args:
            teacher_model: Large, pre-trained model
            student_model: Smaller model to train
            temperature: Softening parameter for distributions
            alpha: Weight for soft target loss (1-alpha for hard targets)
        """
        ### BEGIN SOLUTION role="scaffold"
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        ### END SOLUTION

    def distillation_loss(self, student_logits: Tensor, teacher_logits: Tensor, true_labels) -> Tensor:
        """
        Calculate a differentiable, batch-mean distillation loss.

        TODO: Implement knowledge distillation loss function

        APPROACH:
        1. Validate class IDs or target probabilities for each sample
        2. Compute student log-probabilities with log_softmax, preserving autograd
        3. Compute detached teacher probabilities and batch-mean KL(teacher || student)
        4. Compute the batch-mean hard loss and combine using alpha

        EXAMPLE:
        >>> kd = KnowledgeDistillation(teacher, student)
        >>> loss = kd.distillation_loss(student_out, teacher_out, labels)
        >>> loss.backward()  # Gradients flow to the student, not the teacher
        >>> print(f"Distillation loss: {loss.data:.4f}")

        HINTS:
        - Use temperature to soften distributions: logits/temperature
        - Wrap teacher_logits.data in a fresh Tensor to treat it as a constant
        - Sum over classes, then mean over samples for both loss terms
        - Class IDs may be NumPy integers or integer-valued Tensor data
        """
        ### BEGIN SOLUTION role="scaffold"
        if student_logits.ndim != 2 or student_logits.shape != teacher_logits.shape:
            raise ValueError("Student and teacher logits must have matching (batch, classes) shapes")
        batch_size, num_classes = student_logits.shape
        if batch_size == 0 or num_classes == 0:
            raise ValueError("Distillation needs at least one sample and one class")
        labels = true_labels.data if isinstance(true_labels, Tensor) else np.asarray(true_labels)
        if labels.shape == (batch_size,):
            if (not np.all(np.isfinite(labels)) or np.any(labels != np.floor(labels))
                    or np.any(labels < 0) or np.any(labels >= num_classes)):
                raise ValueError("Class labels must be integer IDs in [0, num_classes)")
            labels = labels.astype(np.intp)
        elif labels.shape == student_logits.shape:
            if (not np.all(np.isfinite(labels)) or np.any(labels < 0)
                    or not np.allclose(labels.sum(axis=1), 1.0)):
                raise ValueError("Target probabilities must be nonnegative and sum to one per sample")
        else:
            raise ValueError("Labels must have shape (batch,) or (batch, classes)")

        # Only the student side is differentiable. The teacher's log-probabilities
        # are constants even when its forward pass was recorded by autograd.
        student_log_probs = log_softmax(student_logits / self.temperature)
        teacher_log_probs = log_softmax(Tensor(teacher_logits.data) / self.temperature)
        teacher_probs = Tensor(np.exp(teacher_log_probs.data))
        soft_loss = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(axis=-1).mean()

        hard_log_probs = log_softmax(student_logits)
        if labels.ndim == 1:
            hard_loss = hard_log_probs[np.arange(batch_size), labels].mean() * -1.0
        else:
            hard_loss = (Tensor(labels) * hard_log_probs).sum(axis=-1).mean() * -1.0

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        ### END SOLUTION

    def _softmax(self, logits):
        """Compute softmax with numerical stability."""
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def _kl_divergence(self, p, q):
        """Compute KL divergence between distributions."""
        return np.mean(np.sum(p * np.log((p + 1e-8) / (q + 1e-8)), axis=-1))

    def _cross_entropy(self, predictions, labels):
        """Compute a NumPy batch-mean cross-entropy for the analysis table."""
        # Simple implementation for integer labels
        if labels.ndim == 1:
            return -np.mean(np.log(predictions[np.arange(len(labels)), labels] + 1e-8))
        else:
            return -np.mean(np.sum(labels * np.log(predictions + 1e-8), axis=1))

# %% [markdown]
"""
### 🧪 Unit Test: Knowledge Distillation

This test validates teacher-student knowledge transfer with temperature scaling.

**What we're testing**: A differentiable loss combining soft and hard targets
**Why it matters**: Enables training small models with teacher knowledge
**Expected**: Student parameters receive gradients; teacher parameters do not
"""

# %% nbgrader={"grade": true, "grade_id": "test-distillation", "locked": true, "points": 15}
def test_unit_knowledge_distillation():
    """🧪 Test knowledge distillation functionality."""
    print("🧪 Unit Test: Knowledge Distillation...")

    # Create teacher model with more capacity - explicit composition
    teacher_l1 = Linear(10, 20)
    teacher_l2 = Linear(20, 5)
    teacher = Sequential(teacher_l1, teacher_l2)

    # Create smaller student model - explicit composition shows size difference
    student_l1 = Linear(10, 5)
    student = Sequential(student_l1)  # Direct connection, no hidden layer

    # As in Module 07, construct the optimizer before the first forward pass:
    # registering parameters marks them requires_grad=True.
    from tinytorch.core.optimizers import SGD
    optimizer = SGD(student.parameters(), lr=0.1)

    # Initialize knowledge distillation with temperature scaling
    kd = KnowledgeDistillation(teacher, student, temperature=3.0, alpha=0.7)

    # Create dummy data for testing
    input_data = Tensor(rng.standard_normal((8, 10)))  # Batch of 8 samples
    true_labels = np.array([0, 1, 2, 3, 4, 0, 1, 2])  # Class labels

    # Forward passes - students see explicit data flow through each model
    teacher_output = teacher.forward(input_data)  # Large model predictions
    student_output = student.forward(input_data)  # Small model predictions

    # Calculate distillation loss - combines soft and hard targets
    loss = kd.distillation_loss(student_output, teacher_output, true_labels)

    # Verify loss is reasonable
    assert isinstance(loss, Tensor), f"Loss must preserve autograd, got {type(loss)}"
    assert np.isfinite(loss.data), "Loss should be finite"
    loss.backward()
    assert all(p.grad is not None for p in student.parameters()), "Student needs gradients"
    assert all(p.grad is None for p in teacher.parameters()), "Teacher must remain fixed"

    # The loss must drive learning, not merely print a plausible scalar.
    initial = float(loss.data)
    for _ in range(10):
        optimizer.zero_grad()
        loss = kd.distillation_loss(student(input_data), teacher_output, Tensor(true_labels))
        loss.backward()
        optimizer.step()
    final = kd.distillation_loss(student(input_data), teacher_output, true_labels)
    assert float(final.data) < initial, "Training should reduce the fixed-batch distillation loss"

    print("✅ knowledge_distillation works correctly!")

if __name__ == "__main__":
    test_unit_knowledge_distillation()

# %% [markdown]
r"""
## 🔧 Integration: Complete Compression Pipeline

Industrial model compression cascades multiple complementary techniques in sequence, each attacking a distinct orthogonal dimension of parameter redundancy:

### Multi-Stage Compression Pipeline Cascade

| Pipeline Stage | Applied Technique | Model State | Footprint | Cumulative Size Reduction | Relative Accuracy | Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 0** | Baseline Dense Model | Full FP32 Dense Weights | $100\text{ MB}$ | $1.0\times$ (Baseline) | $100.0\%$ | $500\text{ ms}$ |
| **Stage 1** | Magnitude Pruning ($80\%$ scalar sparsity) | Coordinate / CSR Format | $20\text{ MB}$ | $5.0\times$ | $98.1\%$ | $480\text{ ms}$ (Dense) / $250\text{ ms}$ (Sparse) |
| **Stage 2** | Structured Pruning ($30\%$ channels excised) | Reduced Dense Dimensions | $14\text{ MB}$ | $7.1\times$ | $96.4\%$ | $175\text{ ms}$ |
| **Stage 3** | Low-Rank SVD (50% rank reduction on large GEMMs) | Factorized $A B$ Matrices | $10\text{ MB}$ | $10.0\times$ | $95.2\%$ | $120\text{ ms}$ |
| **Stage 4** | Knowledge Distillation + Retraining | Compact Student Topology | $5\text{ MB}$ | $20.0\times$ | $93.5\%$ | $50\text{ ms}$ |

$$\text{Total Compression} = \frac{\text{Initial Size}}{\text{Final Size}} = \frac{100\text{ MB}}{5\text{ MB}} = 20\times, \quad \text{Speedup} = \frac{500\text{ ms}}{50\text{ ms}} = 10\times, \quad \Delta \text{Acc} = -6.5\%$$

### Target Deployment Configurations and System Budgets

| Target Platform | Memory & Latency SLA | Recommended Strategy Cascade | Hardware Constraints & Rationale |
| :--- | :--- | :--- | :--- |
| **Mobile Smartphone** | Footprint $< 10\text{ MB}$<br>Latency $< 100\text{ ms}$ | 1. Knowledge Distillation ($10\times$ student)<br>2. Structured Pruning ($50\%$ channels)<br>3. Post-Training Quantization (INT8) | Limited DRAM bandwidth ($30\text{--}50\text{ GB/s}$); NEON/NPU tensor accelerators favor contiguous dense buffers over irregular sparse formats. |
| **Embedded Edge IoT** | Footprint $< 50\text{ MB}$<br>Latency $< 200\text{ ms}$ | 1. Structured Pruning ($30\%$ channels)<br>2. Low-Rank Factorization ($50\%$ rank)<br>3. INT8 / FP16 Mixed Precision | Tight SRAM cache limits ($256\text{ KB}\text{--}2\text{ MB}$); factorized GEMMs keep intermediate working sets resident in SRAM. |
| **Cloud Multi-Tenant** | Target: Max Throughput<br>Cost / Query SLA | 1. Conservative Magnitude Pruning ($50\%$ sparsity)<br>2. 2:4 Structured Sparsity (Ampere+ Tensor Cores)<br>3. Dynamic Batching + FP8 / BF16 | High compute density; leverages native NVIDIA 2:4 sparse tensor core acceleration for $2\times$ throughput without capacity loss. |
"""

# %% nbgrader={"grade": false, "grade_id": "compress-model-comprehensive", "solution": true}
#| export
def compress_model(model, compression_config):
    """
    Apply comprehensive model compression based on configuration.

    TODO: Implement complete compression pipeline

    APPROACH:
    1. Apply magnitude pruning if specified
    2. Apply structured pruning if specified
    3. Return compression statistics
    Low-rank factors require rebuilding layers; use low_rank_approximate separately.

    EXAMPLE:
    >>> config = {
    ...     'magnitude_prune': 0.8,
    ...     'structured_prune': 0.3
    ... }
    >>> stats = compress_model(model, config)
    >>> print(f"Final sparsity: {stats['final_sparsity']:.1f}%")
    Final sparsity: 85.0%

    HINT: Apply techniques sequentially and measure results
    """
    ### BEGIN SOLUTION role="scaffold"
    # Validate before modifying any weights; unsupported work must not be
    # recorded as successfully applied.
    if 'low_rank' in compression_config:
        raise ValueError("Use low_rank_approximate() and rebuild the layer from its factors")
    unknown = set(compression_config) - {'magnitude_prune', 'structured_prune'}
    if unknown:
        raise ValueError(f"Unknown compression techniques: {sorted(unknown)}")
    for ratio in compression_config.values():
        if not np.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError("Pruning ratios must be between 0 and 1")
    original_params = sum(p.size for p in model.parameters())
    original_sparsity = measure_sparsity(model)

    stats = {
        'original_params': original_params,
        'original_sparsity': original_sparsity,
        'applied_techniques': []
    }

    # Apply magnitude pruning
    if 'magnitude_prune' in compression_config:
        sparsity = compression_config['magnitude_prune']
        magnitude_prune(model, sparsity=sparsity)
        stats['applied_techniques'].append(f'magnitude_prune_{sparsity}')

    # Apply structured pruning
    if 'structured_prune' in compression_config:
        ratio = compression_config['structured_prune']
        structured_prune(model, prune_ratio=ratio)
        stats['applied_techniques'].append(f'structured_prune_{ratio}')

    # Final measurements
    final_sparsity = measure_sparsity(model)
    stats['final_sparsity'] = final_sparsity
    stats['sparsity_increase'] = final_sparsity - original_sparsity

    return stats
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Comprehensive Model Compression

This test validates the complete compression pipeline with multiple techniques.

**What we're testing**: Sequential application of compression techniques
**Why it matters**: Real deployments combine multiple compression methods
**Expected**: Cumulative sparsity increase and tracking of applied techniques
"""

# %% nbgrader={"grade": true, "grade_id": "test-compression-integration", "locked": true, "points": 20}
def test_unit_compress_model():
    """🧪 Test comprehensive model compression."""
    print("🧪 Unit Test: Compress Model...")

    # Create test model with explicit layers - students see the full architecture
    layer1 = Linear(20, 15)
    layer2 = Linear(15, 10)
    layer3 = Linear(10, 5)
    model = Sequential(layer1, layer2, layer3)

    # Define compression configuration
    # Students understand what each technique does
    config = {
        'magnitude_prune': 0.7,    # Remove 70% of smallest weights
        'structured_prune': 0.2     # Remove 20% of least important channels
    }

    # Apply compression pipeline - multiple techniques sequentially
    stats = compress_model(model, config)

    # Verify statistics - students understand what was measured
    assert 'original_params' in stats, "Should track original parameter count"
    assert 'final_sparsity' in stats, "Should track final sparsity"
    assert 'applied_techniques' in stats, "Should track applied techniques"

    # Verify compression was applied successfully
    assert stats['final_sparsity'] > stats['original_sparsity'], "Sparsity should increase"
    assert len(stats['applied_techniques']) == 2, "Should apply both techniques"

    # Verify model still has reasonable structure after compression
    remaining_params = sum(np.count_nonzero(p.data) for p in model.parameters())
    assert remaining_params > 0, "Model should retain some parameters"

    print("✅ compress_model works correctly!")

if __name__ == "__main__":
    test_unit_compress_model()

# %% [markdown]
"""
### The Compressor Class: Consolidated for Export

Now that we've implemented all compression techniques, let's create a consolidated class
for export to the tinytorch package. This allows milestones to use the complete compression system.
"""

# %% nbgrader={"grade": false, "grade_id": "compression_export", "solution": false}
#| export
class Compressor:
    """
    Complete compression system for milestone use.

    Provides pruning, distillation, and low-rank approximation techniques.

    This class delegates to the standalone functions (measure_sparsity, magnitude_prune, etc.)
    that students implement, providing a clean OOP interface for milestones.

    Note: Compressor methods return fractions (0-1) for consistency with benchmarking,
    while standalone functions return percentages (0-100) for educational clarity.
    """

    @staticmethod
    def measure_sparsity(model) -> float:
        """Measure the sparsity of a model (returns fraction 0-1)."""
        # Delegate to standalone function and convert percentage to fraction
        return measure_sparsity(model) / 100.0

    @staticmethod
    def magnitude_prune(model, sparsity=0.5):
        """Prune model weights by magnitude. Delegates to standalone function."""
        return magnitude_prune(model, sparsity)

    @staticmethod
    def structured_prune(model, prune_ratio=0.5):
        """Prune entire neurons/channels. Delegates to standalone function."""
        return structured_prune(model, prune_ratio)

    @staticmethod
    def compress_model(model, compression_config: Dict[str, Any]):
        """
        Apply complete compression pipeline to a model.

        Args:
            model: Model to compress
            compression_config: Dictionary with compression settings
                - 'magnitude_sparsity': float (0-1)
                - 'structured_prune_ratio': float (0-1)

        Returns:
            Compressed model with sparsity stats (fractions 0-1)
        """
        key_map = {'magnitude_sparsity': 'magnitude_prune',
                   'structured_prune_ratio': 'structured_prune'}
        unknown = set(compression_config) - key_map.keys()
        if unknown:
            raise ValueError(f"Unknown compression settings: {sorted(unknown)}")
        report = compress_model(model, {key_map[k]: v for k, v in compression_config.items()})
        stats = {'original_sparsity': report['original_sparsity'] / 100.0,
                 'final_sparsity': report['final_sparsity'] / 100.0}
        # An ideal nonzero-weight ratio, not actual dense-array byte savings.
        stats['compression_ratio'] = (1.0 / (1.0 - stats['final_sparsity'])
                                      if stats['final_sparsity'] < 1.0 else float('inf'))

        return model, stats

# Note: measure_sparsity, magnitude_prune, structured_prune are defined earlier in this module.
# The Compressor class above delegates to those functions, providing an OOP interface for milestones.

# %% [markdown]
"""
## 📊 Systems Analysis: Compression Trade-offs

Understanding the real-world effectiveness of different compression techniques through systematic measurement and comparison.

The fundamental challenge in model compression is balancing three competing objectives: model size, inference speed, and prediction accuracy.
"""

# %% [markdown]
"""
### Measuring Compression Impact with Profiler

Now let's use the **Profiler** tool from Module 14 to measure the actual parameter reduction from pruning. This demonstrates the complete workflow: profile baseline (M14) → apply compression (M16) → measure impact (M14+M16).

This is the production workflow: measure → prune → validate → deploy.
"""

# %% nbgrader={"grade": false, "grade_id": "demo-profiler-compression", "solution": false}
# Import Profiler from Module 14 (already imported above)

def explore_compression_with_profiler():
    """📊 Demonstrate parameter reduction using Profiler from Module 14."""
    print("📊 Measuring Compression Impact with Profiler")
    print("=" * 70)

    profiler = Profiler()

    # Create a simple model (Linear already imported above)
    model = Linear(512, 256)

    print("\n🏋️  BEFORE: Dense Model")
    print("-" * 70)

    # Measure baseline
    param_count_before = profiler.count_parameters(model)
    sparsity_before = measure_sparsity(model)
    input_shape = (32, 512)
    memory_before = profiler.measure_memory(model, input_shape)

    print(f"   Parameters: {param_count_before:,}")
    print(f"   Sparsity: {sparsity_before:.1f}% (zeros)")
    print(f"   Memory: {memory_before['parameter_memory_mb']:.2f} MB")
    print(f"   Active parameters: {int(param_count_before * (1 - sparsity_before / 100)):,}")

    # Apply magnitude pruning
    target_sparsity = 0.7  # Remove 70% of parameters
    print(f"\n✂️  Applying {target_sparsity*100:.0f}% Magnitude Pruning...")
    pruned_model = magnitude_prune(model, sparsity=target_sparsity)  # prunes in place, returns the same model

    print("\n🪶 AFTER: Pruned Model")
    print("-" * 70)

    # Measure after pruning
    param_count_after = profiler.count_parameters(pruned_model)
    sparsity_after = measure_sparsity(pruned_model)
    memory_after = profiler.measure_memory(pruned_model, input_shape)

    print(f"   Parameters: {param_count_after:,} (same, but many are zero)")
    print(f"   Sparsity: {sparsity_after:.1f}% (zeros)")
    print(f"   Memory: {memory_after['parameter_memory_mb']:.2f} MB (same storage)")
    print(f"   Active parameters: {int(param_count_after * (1 - sparsity_after / 100)):,}")

    print("\n📈 COMPRESSION RESULTS")
    print("=" * 70)
    sparsity_gain = sparsity_after - sparsity_before
    active_before = int(param_count_before * (1 - sparsity_before / 100))
    active_after = int(param_count_after * (1 - sparsity_after / 100))
    reduction_ratio = active_before / active_after if active_after > 0 else 1
    params_removed = active_before - active_after

    print(f"   Sparsity increased: {sparsity_before:.1f}% → {sparsity_after:.1f}%")
    print(f"   Active params reduced: {active_before:,} → {active_after:,}")
    print(f"   Parameters removed: {params_removed:,} ({sparsity_gain:.1f}% of total)")
    print(f"   Compression ratio: {reduction_ratio:.1f}x fewer active parameters")

    print("\n💡 Key Insight:")
    print(f"   Magnitude pruning removes {sparsity_gain:.0f}% of parameters")
    print(f"   Ideal value-only storage ratio: {reduction_ratio:.1f}x; sparse indices add overhead.")
    print(f"   Critical for: edge devices, mobile apps, energy efficiency")
    print("\n✅ This is the power of compression: remove what doesn't matter!")

if __name__ == "__main__":
    explore_compression_with_profiler()

# %% [markdown]
"""
#### Comparing Compression Techniques

Let's analyze compression ratios across different techniques systematically.
"""

# %%
def analyze_compression_techniques():
    """📊 Compare compression ratios across different techniques."""
    print("📊 Analyzing Compression Techniques")
    print("=" * 60)

    # Create baseline model (Linear already imported above)
    model_configs = [
        ("Small MLP", [Linear(128, 64), Linear(64, 32)]),
        ("Medium MLP", [Linear(512, 256), Linear(256, 128)]),
        ("Large MLP", [Linear(1024, 512), Linear(512, 256)])
    ]

    print(f"\n{'Model':<15} {'Technique':<20} {'Sparsity':<12} {'Compression':<12}")
    print("-" * 60)

    for model_name, layers in model_configs:
        # Create model with explicit composition
        model = Sequential(*layers)
        baseline_params = sum(p.size for p in model.parameters())

        # Prune a deep copy so each technique starts from the same weights
        mag_model = Sequential(*copy.deepcopy(layers))
        magnitude_prune(mag_model, sparsity=0.8)
        mag_sparsity = measure_sparsity(mag_model)
        mag_ratio = 1.0 / (1.0 - mag_sparsity / 100) if mag_sparsity < 100 else float('inf')

        print(f"{model_name:<15} {'Magnitude (80%)':<20} {mag_sparsity:>10.1f}% {mag_ratio:>10.1f}x")

        struct_model = Sequential(*copy.deepcopy(layers))
        structured_prune(struct_model, prune_ratio=0.5)
        struct_sparsity = measure_sparsity(struct_model)
        struct_ratio = 1.0 / (1.0 - struct_sparsity / 100) if struct_sparsity < 100 else float('inf')

        print(f"{'':<15} {'Structured (50%)':<20} {struct_sparsity:>10.1f}% {struct_ratio:>10.1f}x")
        print()

    print("💡 Key Insights:")
    print("   • Magnitude pruning achieves higher sparsity (80%+)")
    print("   • Structured pruning creates hardware-friendly patterns")
    print("   • Larger models compress more effectively")
    print("   • Compression ratio = 1 / (1 - sparsity)")

if __name__ == "__main__":
    analyze_compression_techniques()

# %% [markdown]
"""
#### Knowledge Distillation Analysis

Now let's analyze how knowledge distillation compares to other compression techniques for different compression ratios and accuracy preservation goals.
"""

# %%
def analyze_distillation_effectiveness():
    """📊 Analyze knowledge distillation compression and accuracy trade-offs."""
    print("\n📊 Analyzing Knowledge Distillation Effectiveness")
    print("=" * 60)

    # A big teacher and a small student, both untrained: measure the loss itself
    teacher = Sequential(Linear(32, 128), ReLU(), Linear(128, 10))
    student = Sequential(Linear(32, 16), ReLU(), Linear(16, 10))
    teacher_params = sum(p.size for p in teacher.parameters())
    student_params = sum(p.size for p in student.parameters())
    print(f"\nTeacher: {teacher_params:,} params   Student: {student_params:,} params   "
          f"({teacher_params / student_params:.1f}x smaller)")

    x = Tensor(rng.standard_normal((16, 32)))
    labels = rng.integers(0, 10, size=16)
    teacher_logits = teacher.forward(x)
    student_logits = student.forward(x)

    print(f"\n{'Temperature':<12} {'Soft (KL)':<12} {'Hard (CE)':<12} {'Combined'}")
    print("-" * 50)

    for temperature in [1.0, 2.0, 5.0, 10.0]:
        kd = KnowledgeDistillation(teacher, student, temperature=temperature, alpha=0.7)
        soft = kd._kl_divergence(kd._softmax(teacher_logits.data / temperature),
                                 kd._softmax(student_logits.data / temperature))
        hard = kd._cross_entropy(kd._softmax(student_logits.data), labels)
        combined = kd.distillation_loss(student_logits, teacher_logits, labels)
        print(f"{temperature:<12.1f} {soft:<12.4f} {hard:<12.4f} {combined.data:.4f}")

    print("\n💡 Knowledge Distillation Insights:")
    print("   • Higher temperature flattens both distributions, so the KL term shrinks;")
    print("     Hinton et al. scale it by T² to compensate (omitted here for clarity)")
    print("   • The soft term carries the teacher's ranking of wrong answers, not just the right one")
    print("   • More effective than naive pruning for large reductions")
    print("   • Requires retraining (unlike pruning/quantization)")
    print("\n🚀 Best Use Case:")
    print("   Deploy small student models on edge devices")
    print("   Train expensive teacher once, distill many students")

if __name__ == "__main__":
    analyze_distillation_effectiveness()

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly before module completion.
"""

# %% nbgrader={"grade": true, "grade_id": "module-integration", "locked": true, "points": 20}
def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire compression module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_measure_sparsity()
    test_unit_magnitude_prune()
    test_unit_structured_prune()
    test_unit_low_rank_approximate()
    test_unit_knowledge_distillation()
    test_unit_compress_model()

    print("\nRunning integration scenarios...")

    # Test 1: Complete compression pipeline
    print("🧪 Integration Test: Complete compression pipeline...")

    # Create a realistic model with explicit layers - students see the architecture
    input_layer = Linear(784, 512)    # Input layer (like MNIST)
    hidden1 = Linear(512, 256)         # Hidden layer 1
    hidden2 = Linear(256, 128)         # Hidden layer 2
    output_layer = Linear(128, 10)     # Output layer
    model = Sequential(input_layer, hidden1, hidden2, output_layer)

    original_params = sum(p.size for p in model.parameters())
    print(f"Original model: {original_params:,} parameters")

    # Apply comprehensive compression - students see each technique
    compression_config = {
        'magnitude_prune': 0.8,    # Remove 80% of smallest weights
        'structured_prune': 0.3     # Remove 30% of channels
    }

    stats = compress_model(model, compression_config)
    final_sparsity = measure_sparsity(model)

    # Validate compression results
    assert final_sparsity > 70, f"Expected >70% sparsity, got {final_sparsity:.1f}%"
    assert stats['sparsity_increase'] > 70, "Should achieve significant compression"
    assert len(stats['applied_techniques']) == 2, "Should apply both techniques"

    print(f"✅ Achieved {final_sparsity:.1f}% sparsity with {len(stats['applied_techniques'])} techniques")

    # Test 2: Knowledge distillation setup
    print("🧪 Integration Test: Knowledge distillation...")

    # Create teacher with more capacity - explicit layers show architecture
    teacher_l1 = Linear(100, 200)
    teacher_l2 = Linear(200, 50)
    teacher = Sequential(teacher_l1, teacher_l2)

    # Create smaller student - explicit shows size difference
    student_l1 = Linear(100, 50)
    student = Sequential(student_l1)  # 3x fewer parameters

    kd = KnowledgeDistillation(teacher, student, temperature=4.0, alpha=0.8)

    # Verify setup
    teacher_params = sum(p.size for p in teacher.parameters())
    student_params = sum(p.size for p in student.parameters())
    compression_ratio = student_params / teacher_params

    assert compression_ratio < 0.5, f"Student should be <50% of teacher size, got {compression_ratio:.2f}"
    assert kd.temperature == 4.0, "Temperature should be set correctly"
    assert kd.alpha == 0.8, "Alpha should be set correctly"

    print(f"✅ Knowledge distillation: {compression_ratio:.2f}x size reduction")

    # Test 3: Low-rank approximation
    print("🧪 Integration Test: Low-rank approximation...")

    large_matrix = rng.standard_normal((200, 150))
    U, S, V = low_rank_approximate(large_matrix, rank_ratio=0.3)

    original_size = large_matrix.size
    compressed_size = U.size + S.size + V.size
    compression_ratio = compressed_size / original_size

    assert compression_ratio < 0.7, f"Should achieve compression, got ratio {compression_ratio:.2f}"

    # Test reconstruction
    reconstructed = U @ np.diag(S) @ V
    error = np.linalg.norm(large_matrix - reconstructed) / np.linalg.norm(large_matrix)
    # Low-rank approximation trades accuracy for compression - some error is expected
    assert error < 0.7, f"Reconstruction error too high: {error:.3f}"

    print(f"✅ Low-rank: {compression_ratio:.2f}x compression, {error:.3f} error")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 16")

# %% [markdown]
r"""
## 🤔 ML Systems Reflection Questions

### Question 1: Compression Trade-offs

**Question**: You implemented magnitude pruning that removes $90\%$ of weights from a $10\text{M}$ parameter model.

- **Active Parameters**:
  $$N_{\text{active}} = N_{\text{total}} \times (1 - \text{Sparsity}) = 10\text{M} \times (1 - 0.90) = \mathbf{1.0\text{M}}\text{ parameters}$$

- **Theoretical Minimum Storage**:
  $$S_{\text{raw}} = 1.0\text{M} \times 4\text{ bytes} = \mathbf{4.0\text{ MB}}$$
  *(Note on sparse format overheads)*:
  - In Coordinate format (COO: value $+$ 32-bit row $+$ 32-bit column index), storage is $1.0\text{M} \times (4 + 4 + 4) = 12\text{ MB}$.
  - In Compressed Sparse Row (CSR: FP32 values $+$ 16-bit column indices $+$ row pointers), storage is $1.0\text{M} \times (4 + 2) + (\text{rows} + 1) \times 4 \approx \mathbf{6.0\text{ MB}}$.

- **Why Actual Speedup Is Less Than $10\times$**:
  1. **Sparse Overhead on Dense Tensor Cores**: Standard systolic arrays (NVIDIA Tensor Cores, Apple Neural Engine) require aligned contiguous blocks. Unstructured zeroing yields $0\times$ speedup under standard dense GEMM kernels.
  2. **Memory Access Divergence & Irregularity**: When sparse BLAS kernels traverse non-zero elements via pointer indirection (`col_indices[k]`), cache-line utilization drops drastically due to non-coalesced DRAM accesses.
  3. **Amdahl's Law & Non-Pruned Operations**: Layer norms, activations, residual additions, and memory bandwidth bounds remain unpruned, setting a strict roofline limit on achievable speedup.

---

### Question 2: Structured vs. Unstructured Sparsity

**Question**: Your structured pruning removes entire channels, while magnitude pruning creates scattered zeros.

| Evaluation Criterion | Winner | Technical Analysis & Systems Rationale |
| :--- | :--- | :--- |
| **Hardware Acceleration** | **Structured Pruning** | Directly decreases physical matrix dimensions $(M \times N \to M \times N')$. Standard BLAS libraries execute dense GEMMs on contiguous arrays at peak hardware FLOP/s without specialized hardware. |
| **Accuracy Retention** | **Unstructured Pruning** | Allows the optimization manifold to retain isolated high-magnitude synaptic connections across all channels. Preserves critical feature representations at $\ge 80\%$ sparsity where structured pruning suffers catastrophic capacity collapse. |
| **Memory Predictability** | **Structured Pruning** | Preserves linear strided memory access without index indirection tables, ensuring full cache-line utilization ($64\text{ bytes} = 16\text{ FP32 values}$) and prefetcher effectiveness. |

**Deployment Selection Strategy**:
- **Mobile / Edge / CPU**: Choose **Structured Pruning** (or 2:4 structured sparsity on Ampere+ GPUs) because edge accelerators lack high-efficiency sparse BLAS runtimes.
- **Offline Storage / Bandwidth-Constrained Distribution**: Choose **Unstructured Pruning** with Huffman/entropy coding for over-the-air (OTA) binary delivery, followed by sparse decompression or on-device retraining.

---

### Question 3: Knowledge Distillation Efficiency

**Question**: A teacher model has $100\text{M}$ parameters, student has $10\text{M}$ parameters, both achieve $85\%$ accuracy. Teacher latency is $100\text{ ms}$, student latency is $15\text{ ms}$.

- **Compression Ratio**:
  $$\text{Compression Ratio} = \frac{N_{\text{teacher}}}{N_{\text{student}}} = \frac{100\text{M}}{10\text{M}} = \mathbf{10.0\times}$$

- **Latency Speedup**:
  $$\text{Speedup} = \frac{\text{Latency}_{\text{teacher}}}{\text{Latency}_{\text{student}}} = \frac{100\text{ ms}}{15\text{ ms}} \approx \mathbf{6.67\times}$$

- **Why Speedup Is Smaller Than the Compression Ratio ($6.67\times < 10.0\times$)**:
  1. **Memory Bandwidth vs. Compute Roofline**: Small models often become memory-bandwidth bound or kernel-launch overhead dominated at batch size 1.
  2. **Fixed Layer Overheads**: Both models execute identical sequence length tokenization, positional encodings, layer norms, and softmax operations that scale with sequence length $S$ rather than parameter count $P$.
  3. **Hardware Latency Floor**: Kernel invocation overheads, CUDA stream synchronization, and PCIe transfers impose constant additive latency independent of model parameter volume.

---

### Question 4: Low-Rank Decomposition

**Question**: Approximate a $(512, 256)$ weight matrix with rank $r = 64$ using SVD.

- **Original Parameter Count**:
  $$P_{\text{orig}} = 512 \times 256 = \mathbf{131{,}072}\text{ parameters}$$

- **Decomposed Parameter Count**:
  $$P_{\text{decomposed}} = (512 \times 64) + 64 + (64 \times 256) = 32{,}768 + 64 + 16{,}384 = \mathbf{49{,}216}\text{ parameters}$$
  *(If absorbing $\Sigma$ into factors $A = U\sqrt{\Sigma}, B = \sqrt{\Sigma}V^T$, parameter count is $32{,}768 + 16{,}384 = 49{,}152$)*.

- **Compression Ratio**:
  $$\text{Compression Ratio} = \frac{131{,}072}{49{,}216} \approx \mathbf{2.66\times} \quad \left(\text{or } \frac{131{,}072}{49{,}152} \approx 2.67\times\right)$$

- **Break-Even Ineffectiveness Rank ($r^*$)**:
  $$r \cdot (M + N + 1) \ge M \cdot N \implies r^* = \frac{M \cdot N}{M + N + 1} = \frac{512 \times 256}{512 + 256 + 1} = \frac{131{,}072}{769} \approx 170.44$$
  Compression becomes ineffective when **$\text{rank } r > 170$**. At $r = 171$, factorized parameter count exceeds dense storage!

---

### Question 5: Pruning Strategy Selection

**Question**: Deploying on a mobile device with a $50\text{ MB}$ model limit and $100\text{ ms}$ latency requirement.

- **Memory Optimization Strategy**: **Magnitude Pruning + Weight Quantization** (achieves extreme compression ratios up to $10\text{--}20\times$ by zeroing non-essential weights and packing non-zeros into INT8/INT4).
- **Speed Optimization Strategy**: **Structured Pruning + Knowledge Distillation** (physically reduces channel dimensions and depth, directly shrinking dense GEMM computation and memory fetch cycles).
- **Optimal Sequential Compression Cascade**:
  1. **Knowledge Distillation**: First train a smaller student architecture designed specifically to fit mobile latency constraints.
  2. **Structured Pruning**: Prune redundant channels in the student model by $20\text{--}30\%$ to optimize cache residency.
  3. **Fine-Tuning / Retraining**: Recover accuracy lost during structural pruning via student fine-tuning.
  4. **Post-Training Quantization (INT8)**: Quantize remaining weights and activations from FP32 to INT8, delivering a final $4\times$ memory reduction and activating high-throughput INT8 mobile DSP/NPU vector engines.
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Pruning Creates Sparsity

**What you built:** Pruning that zeros out small weights, creating sparse models.

**Why it matters:** At 50% sparsity, half the weights are zero. This demo measures
that change on a freshly initialized layer; its dense storage and computation
shape stay the same. Memory savings require a sparse representation, and speed
gains require suitable kernels or rebuilding smaller layers.

Accuracy preservation depends on the trained model and data and may require
fine-tuning. Milestone 06 measures accuracy before and after pruning on real digits.
"""

# %%
def demo_compression():
    """🎯 See pruning create sparsity."""
    print("🎯 AHA MOMENT: Pruning Removes Weights")
    print("=" * 45)

    # Create a model
    layer = Linear(128, 64)

    original_nonzero = np.count_nonzero(layer.weight.data)
    original_total = layer.weight.data.size
    original_bytes = layer.weight.data.nbytes

    # Apply 50% pruning
    magnitude_prune(layer, sparsity=0.5)

    pruned_nonzero = np.count_nonzero(layer.weight.data)
    sparsity = 1 - (pruned_nonzero / original_total)

    print(f"Original: {original_nonzero:,} non-zero weights")
    print(f"After 50% pruning: {pruned_nonzero:,} non-zero weights")
    print(f"\nActual sparsity: {sparsity:.1%}")
    print(f"Half the weights are now zero!")

    print(f"Dense weight storage: {original_bytes:,} → {layer.weight.data.nbytes:,} bytes")
    print("\n✨ Sparsity increased; accuracy and latency still need measurement.")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_compression()

# %% [markdown]
"""
## 🚀 MODULE SUMMARY: Compression

Congratulations! You've built a comprehensive model compression system that can dramatically reduce model size while preserving intelligence!

### Key Accomplishments
- Built magnitude-based and structured pruning techniques with clear sparsity patterns
- Implemented knowledge distillation for teacher-student compression with temperature scaling
- Created low-rank approximation using SVD decomposition for matrix factorization
- Developed sparsity measurement and comprehensive compression pipeline
- All tests pass (validated by `test_module()`)

### Systems Insights Discovered
- **Structured vs Unstructured**: Hardware-friendly patterns vs maximum compression
- **Compression Cascading**: Multiple techniques compound but need careful sequencing
- **Memory vs Speed**: Parameter reduction needs sparse libraries for speedup
- **Deployment Strategy**: Different scenarios require different compression approaches

### Ready for Next Steps
Your compression implementation enables efficient model deployment across diverse hardware constraints!
Export with: `tito module complete 16`

**Next**: Module 17 will add acceleration techniques including vectorization and kernel fusion!
"""
