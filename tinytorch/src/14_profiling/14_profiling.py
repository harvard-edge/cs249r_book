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
# Module 14: Profiling - Measuring What Matters in ML Systems

Welcome to Module 14! You'll build professional profiling tools to measure model performance and uncover optimization opportunities.

## 🔗 Prerequisites & Progress
**You've Built**: Complete ML stack from tensors to transformers
**You'll Build**: Comprehensive profiling system for parameters, FLOPs, memory, and latency
**You'll Enable**: Data-driven optimization decisions and performance analysis

**Connection Map**:
```
All Modules (01-13) → Profiling (14) → Optimization Techniques (15-18)
(implementations)     (measurement)     (targeted fixes)
```

**Before starting this module, verify:**
- [ ] Module 01 (Tensor): Core tensor operations
- [ ] Module 03 (Layers): Linear layer implementation
- [ ] Module 09 (Spatial): Convolutional operations

This module can work standalone with minimal Tensor implementation, but
full functionality requires previous modules for realistic profiling scenarios

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement a complete Profiler class for model analysis
2. Count parameters and FLOPs accurately for different architectures
3. Measure memory usage and latency with statistical rigor
4. Create production-quality performance analysis tools

Let's build the measurement foundation for ML systems optimization!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/14_profiling/profiling_dev.py`
**Building Side:** Code exports to `tinytorch.profiling.profiler`

```python
# How to use this module:
from tinytorch.perf.profiling import Profiler, profile_forward_pass, profile_backward_pass
```

**Why this matters:**
- **Learning:** Complete profiling system for understanding model performance characteristics
- **Production:** Professional measurement tools like those used in PyTorch, TensorFlow
- **Consistency:** All profiling and measurement tools in profiling.profiler
- **Integration:** Works with any model built using TinyTorch components
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": true}
#| default_exp perf.profiling
#| export

import sys
import os
import time
import numpy as np
import tracemalloc
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import gc

# Import from TinyTorch package (previous modules must be completed and exported)
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.spatial import Conv2d

# Constants for memory and performance measurement
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
KB_TO_BYTES = 1024  # Kilobytes to bytes conversion
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion

# %% [markdown]
"""
## 💡 Introduction: Why Profiling Matters in ML Systems

Imagine you're a detective investigating a performance crime. Your model is running slowly, using too much memory, or burning through compute budgets. Without profiling, you're flying blind - making guesses about what to optimize. With profiling, you have evidence.

**The Performance Investigation Process:**
```
Suspect Model → Profile Evidence → Identify Bottleneck → Target Optimization
     ↓               ↓                    ↓                    ↓
   "Too slow"    "200 GFLOP/s"      "Memory bound"      "Reduce transfers"
```

**Questions Profiling Answers:**
- **How many parameters?** (Memory footprint, model size)
- **How many FLOPs?** (Computational cost, energy usage)
- **Where are bottlenecks?** (Memory vs compute bound)
- **What's actual latency?** (Real-world performance)

**Production Importance:**
In production ML systems, profiling isn't optional - it's survival. A model that's 10% more accurate but 100× slower often can't be deployed. Teams use profiling daily to make data-driven optimization decisions, not guesses.

### The Profiling Workflow Visualization
```
Model → Profiler → Measurements → Analysis → Optimization Decision
  ↓        ↓           ↓           ↓            ↓
 GPT   Parameter   125M params   Memory      Use quantization
       Counter     2.5B FLOPs    bound       Reduce precision
```
"""

# %% [markdown]
"""
### 🔗 From Implementation to Optimization: The Profiling Foundation

**In this module (14)**, you'll build the measurement tools to discover optimization opportunities.
**In later modules (15+)**, you'll use these profiling insights to implement optimizations like KV caching.

**The Real ML Engineering Workflow**:
```
Step 1: Measure (This Module!)          Step 2: Analyze
  ↓                                       ↓
Profile baseline → Find bottleneck → Understand cause
40 tok/s          80% in attention    O(n²) recomputation
                                       ↓
Step 4: Validate                      Step 3: Optimize (Future Modules)
  ↓                                       ↓
Profile optimized ← Verify speedup ← Implement optimization
500 tok/s (12.5x)   Measure impact    Design solution
```

**Without profiling**: You'd never know WHERE to optimize!
**Without measurement**: You couldn't verify improvements!

This module teaches the measurement and analysis skills that enable
optimization breakthroughs. You'll profile real models and discover
bottlenecks just like production ML teams do.
"""

# %% [markdown]
"""
## 📐 Foundations: Performance Measurement Principles

Before we build our profiler, let's understand what we're measuring and why each metric matters.

### Parameter Counting - Model Size Detective Work

Parameters determine your model's memory footprint and storage requirements. Every parameter is typically a 32-bit float (4 bytes), so counting them precisely predicts memory usage.

**Parameter Counting Formula:**
```
Linear Layer: (input_features × output_features) + output_features
               ↑              ↑                    ↑
            Weight matrix   Bias vector      Total parameters

Example: Linear(768, 3072) → (768 × 3072) + 3072 = 2,362,368 parameters
Memory: 2,362,368 × 4 bytes = 9.45 MB
```

### FLOP Counting - Computational Cost Analysis

FLOPs (Floating Point Operations) measure computational work. Unlike wall-clock time, FLOPs are hardware-independent and predict compute costs across different systems.

**FLOP Formulas for Key Operations:**
```
Matrix Multiplication (M,K) @ (K,N):
   FLOPs = M × N × K × 2
           ↑   ↑   ↑   ↑
        Rows Cols Inner Multiply+Add

Linear Layer Forward:
   FLOPs = batch_size × input_features × output_features × 2
                      ↑                  ↑                 ↑
                  Matmul cost        Bias add        Operations

Convolution (simplified):
   FLOPs = output_H × output_W × kernel_H × kernel_W × in_channels × out_channels × 2
```

### Memory Profiling - The Three Types of Memory

ML models use memory in three distinct ways, each with different optimization strategies:

**Memory Type Breakdown:**
```
Total Training Memory = Parameters + Activations + Gradients + Optimizer State
                           ↓            ↓           ↓            ↓
                        Model         Forward     Backward     Adam: 2×params
                        weights       pass cache  gradients    SGD: 0×params

Example for 125M parameter model:
Parameters:    500 MB (125M × 4 bytes)
Activations:   200 MB (depends on batch size)
Gradients:     500 MB (same as parameters)
Adam state:  1,000 MB (momentum + velocity)
Total:      2,200 MB (4.4× parameter memory!)
```

### Latency Measurement - Dealing with Reality

Latency measurement is tricky because systems have variance, warmup effects, and measurement overhead. Professional profiling requires statistical rigor.

**Latency Measurement Best Practices:**
```
Measurement Protocol:
1. Warmup runs (10+) → CPU/GPU caches warm up
2. Timed runs (100+) → Statistical significance
3. Outlier handling → Use median, not mean
4. Memory cleanup → Prevent contamination

Timeline:
Warmup: [run][run][run]...[run] ← Don't time these
Timing: [⏱run⏱][⏱run⏱]...[⏱run⏱] ← Time these
Result: median(all_times) ← Robust to outliers
```
"""

# %% [markdown]
"""
## 🏗️ Implementation: Building the Core Profiler Class

Now let's implement our profiler step by step. We'll start with the foundation and build up to comprehensive analysis.

### The Profiler Architecture
```
Profiler Class
├── count_parameters() → Model size analysis
├── count_flops() → Computational cost estimation
├── measure_memory() → Memory usage tracking
├── measure_latency() → Performance timing
├── profile_layer() → Layer-wise analysis
├── profile_forward_pass() → Complete forward analysis
└── profile_backward_pass() → Training analysis

Integration:
All methods work together to provide comprehensive performance insights
```
"""

# %% nbgrader={"grade": false, "grade_id": "profiler_class", "solution": true}
#| export
class Profiler:
    """
    Professional-grade ML model profiler for performance analysis.

    Measures parameters, FLOPs, memory usage, and latency with statistical rigor.
    Used for optimization guidance and deployment planning.
    """

    def __init__(self):
        """
        Initialize profiler with measurement state.

        TODO: Set up profiler tracking structures

        APPROACH:
        1. Create empty measurements dictionary
        2. Initialize operation counters
        3. Set up memory tracking state

        EXAMPLE:
        >>> profiler = Profiler()
        >>> profiler.measurements
        {}

        HINTS:
        - Use defaultdict(int) for operation counters
        - measurements dict will store timing results
        """
        ### BEGIN SOLUTION
        self.measurements = {}
        self.operation_counts = defaultdict(int)
        self.memory_tracker = None
        ### END SOLUTION

    def count_parameters(self, model) -> int:
        """
        Count total trainable parameters in a model.

        TODO: Implement parameter counting for any model with parameters() method

        APPROACH:
        1. Get all parameters from model.parameters() if available
        2. For single layers, count weight and bias directly
        3. Sum total element count across all parameter tensors

        EXAMPLE:
        >>> linear = Linear(128, 64)  # 128*64 + 64 = 8256 parameters
        >>> profiler = Profiler()
        >>> count = profiler.count_parameters(linear)
        >>> print(count)
        8256

        HINTS:
        - Use parameter.data.size for tensor element count
        - Handle models with and without parameters() method
        - Don't forget bias terms when present
        """
        ### BEGIN SOLUTION
        total_params = 0

        # Handle SimpleModel pattern (has .layers attribute)
        if hasattr(model, 'layers'):
            # SimpleModel: iterate through layers
            for layer in model.layers:
                for param in layer.parameters():
                    total_params += param.data.size
        elif hasattr(model, 'parameters'):
            # Model with direct parameters() method
            for param in model.parameters():
                total_params += param.data.size
        elif hasattr(model, 'weight'):
            # Single layer (Linear, Conv2d) - all have .weight
            total_params += model.weight.data.size
            # Check for bias (may be None)
            if hasattr(model, 'bias') and model.bias is not None:
                total_params += model.bias.data.size
        else:
            # No parameters (activations, etc.)
            total_params = 0

        return total_params
        ### END SOLUTION

    def count_flops(self, model, input_shape: Tuple[int, ...]) -> int:
        """
        Count FLOPs (Floating Point Operations) for one forward pass.

        TODO: Implement FLOP counting for different layer types

        APPROACH:
        1. Create dummy input with given shape
        2. Calculate FLOPs based on layer type and dimensions
        3. Handle different model architectures (Linear, Conv2d, Sequential)

        LAYER-SPECIFIC FLOP FORMULAS:
        - Linear: input_features × output_features × 2 (matmul + bias)
        - Conv2d: output_h × output_w × kernel_h × kernel_w × in_channels × out_channels × 2
        - Activation: Usually 1 FLOP per element (ReLU, Sigmoid)

        EXAMPLE:
        >>> linear = Linear(128, 64)
        >>> profiler = Profiler()
        >>> flops = profiler.count_flops(linear, (1, 128))
        >>> print(flops)  # 128 * 64 * 2 = 16384
        16384

        HINTS:
        - Batch dimension doesn't affect per-sample FLOPs
        - Focus on major operations (matmul, conv) first
        - For Sequential models, sum FLOPs of all layers
        """
        ### BEGIN SOLUTION
        # Create dummy input (unused but kept for interface consistency)
        _dummy_input = Tensor(np.random.randn(*input_shape))
        total_flops = 0

        # Handle different model types
        if hasattr(model, '__class__'):
            model_name = model.__class__.__name__

            if model_name == 'Linear':
                # Linear layer: input_features × output_features × 2
                in_features = input_shape[-1]
                out_features = model.weight.shape[1] if hasattr(model, 'weight') else 1
                total_flops = in_features * out_features * 2

            elif model_name == 'Conv2d':
                # Conv2d layer: complex calculation based on output size
                # Simplified: assume we know the output dimensions
                if hasattr(model, 'kernel_size') and hasattr(model, 'in_channels'):
                    _batch_size = input_shape[0] if len(input_shape) > 3 else 1
                    in_channels = model.in_channels
                    out_channels = model.out_channels
                    kernel_h = kernel_w = model.kernel_size

                    # Estimate output size (simplified)
                    input_h, input_w = input_shape[-2], input_shape[-1]
                    output_h = input_h // (model.stride if hasattr(model, 'stride') else 1)
                    output_w = input_w // (model.stride if hasattr(model, 'stride') else 1)

                    total_flops = (output_h * output_w * kernel_h * kernel_w *
                                 in_channels * out_channels * 2)

            elif model_name == 'Sequential' or hasattr(model, 'layers'):
                # Sequential model or model with layers: sum FLOPs of all layers
                current_shape = input_shape
                for layer in model.layers:
                    layer_flops = self.count_flops(layer, current_shape)
                    total_flops += layer_flops
                    # Update shape for next layer (simplified)
                    if hasattr(layer, 'weight'):
                        current_shape = current_shape[:-1] + (layer.weight.shape[1],)

            else:
                # Activation or other: assume 1 FLOP per element
                total_flops = np.prod(input_shape)

        return total_flops
        ### END SOLUTION

    def measure_memory(self, model, input_shape: Tuple[int, ...]) -> Dict[str, float]:
        """
        Measure memory usage during forward pass.

        TODO: Implement memory tracking for model execution

        APPROACH:
        1. Use tracemalloc to track memory allocation
        2. Measure baseline memory before model execution
        3. Run forward pass and track peak usage
        4. Calculate different memory components

        RETURN DICTIONARY:
        - 'parameter_memory_mb': Memory for model parameters
        - 'activation_memory_mb': Memory for activations
        - 'peak_memory_mb': Maximum memory usage
        - 'memory_efficiency': Ratio of useful to total memory

        EXAMPLE:
        >>> linear = Linear(1024, 512)
        >>> profiler = Profiler()
        >>> memory = profiler.measure_memory(linear, (32, 1024))
        >>> print(f"Parameters: {memory['parameter_memory_mb']:.1f} MB")
        Parameters: 2.1 MB

        HINTS:
        - Use tracemalloc.start() and tracemalloc.get_traced_memory()
        - Account for float32 = 4 bytes per parameter
        - Activation memory scales with batch size
        """
        ### BEGIN SOLUTION
        # Start memory tracking
        tracemalloc.start()

        # Measure baseline memory (unused but kept for completeness)
        _baseline_memory = tracemalloc.get_traced_memory()[0]

        # Calculate parameter memory
        param_count = self.count_parameters(model)
        parameter_memory_bytes = param_count * BYTES_PER_FLOAT32
        parameter_memory_mb = parameter_memory_bytes / MB_TO_BYTES

        # Create input and measure activation memory
        dummy_input = Tensor(np.random.randn(*input_shape))
        input_memory_bytes = dummy_input.data.nbytes

        # Estimate activation memory (simplified)
        activation_memory_bytes = input_memory_bytes * 2  # Rough estimate
        activation_memory_mb = activation_memory_bytes / MB_TO_BYTES

        # Run forward pass to measure peak memory usage
        _ = model.forward(dummy_input)

        # Get peak memory
        _current_memory, peak_memory = tracemalloc.get_traced_memory()
        peak_memory_mb = (peak_memory - _baseline_memory) / MB_TO_BYTES

        tracemalloc.stop()

        # Calculate efficiency
        useful_memory = parameter_memory_mb + activation_memory_mb
        memory_efficiency = useful_memory / max(peak_memory_mb, 0.001)  # Avoid division by zero

        return {
            'parameter_memory_mb': parameter_memory_mb,
            'activation_memory_mb': activation_memory_mb,
            'peak_memory_mb': max(peak_memory_mb, useful_memory),
            'memory_efficiency': min(memory_efficiency, 1.0)
        }
        ### END SOLUTION

    def measure_latency(self, model, input_tensor, warmup: int = 10, iterations: int = 100) -> float:
        """
        Measure model inference latency with statistical rigor.

        TODO: Implement accurate latency measurement

        APPROACH:
        1. Run warmup iterations to stabilize performance
        2. Measure multiple iterations for statistical accuracy
        3. Calculate median latency to handle outliers
        4. Return latency in milliseconds

        PARAMETERS:
        - warmup: Number of warmup runs (default 10)
        - iterations: Number of measurement runs (default 100)

        EXAMPLE:
        >>> linear = Linear(128, 64)
        >>> input_tensor = Tensor(np.random.randn(1, 128))
        >>> profiler = Profiler()
        >>> latency = profiler.measure_latency(linear, input_tensor)
        >>> print(f"Latency: {latency:.2f} ms")
        Latency: 0.15 ms

        HINTS:
        - Use time.perf_counter() for high precision
        - Use median instead of mean for robustness against outliers
        - Handle different model interfaces (forward, __call__)
        """
        ### BEGIN SOLUTION
        # Warmup runs to stabilize performance
        for _ in range(warmup):
            _ = model.forward(input_tensor)

        # Measurement runs
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            _ = model.forward(input_tensor)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds

        # Calculate statistics - use median for robustness
        times = np.array(times)
        median_latency = np.median(times)

        return float(median_latency)
        ### END SOLUTION

    def profile_layer(self, layer, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Profile a single layer comprehensively.

        TODO: Implement layer-wise profiling

        APPROACH:
        1. Count parameters for this layer
        2. Count FLOPs for this layer
        3. Measure memory usage
        4. Measure latency
        5. Return comprehensive layer profile

        EXAMPLE:
        >>> linear = Linear(256, 128)
        >>> profiler = Profiler()
        >>> profile = profiler.profile_layer(linear, (32, 256))
        >>> print(f"Layer uses {profile['parameters']} parameters")
        Layer uses 32896 parameters

        HINTS:
        - Use existing profiler methods (count_parameters, count_flops, etc.)
        - Create dummy input for latency measurement
        - Include layer type information in profile
        """
        ### BEGIN SOLUTION
        # Create dummy input for latency measurement
        dummy_input = Tensor(np.random.randn(*input_shape))

        # Gather all measurements
        params = self.count_parameters(layer)
        flops = self.count_flops(layer, input_shape)
        memory = self.measure_memory(layer, input_shape)
        latency = self.measure_latency(layer, dummy_input, warmup=3, iterations=10)

        # Compute derived metrics
        gflops_per_second = (flops / 1e9) / max(latency / 1000, 1e-6)

        return {
            'layer_type': layer.__class__.__name__,
            'parameters': params,
            'flops': flops,
            'latency_ms': latency,
            'gflops_per_second': gflops_per_second,
            **memory
        }
        ### END SOLUTION

    def profile_forward_pass(self, model, input_tensor) -> Dict[str, Any]:
        """
        Comprehensive profiling of a model's forward pass.

        TODO: Implement complete forward pass analysis

        APPROACH:
        1. Use Profiler class to gather all measurements
        2. Create comprehensive performance profile
        3. Add derived metrics and insights
        4. Return structured analysis results

        RETURN METRICS:
        - All basic profiler measurements
        - FLOPs per second (computational efficiency)
        - Memory bandwidth utilization
        - Performance bottleneck identification

        EXAMPLE:
        >>> model = Linear(256, 128)
        >>> input_data = Tensor(np.random.randn(32, 256))
        >>> profiler = Profiler()
        >>> profile = profiler.profile_forward_pass(model, input_data)
        >>> print(f"Throughput: {profile['gflops_per_second']:.2f} GFLOP/s")
        Throughput: 2.45 GFLOP/s

        HINTS:
        - GFLOP/s = (FLOPs / 1e9) / (latency_ms / 1000)
        - Memory bandwidth = memory_mb / (latency_ms / 1000)
        - Consider realistic hardware limits for efficiency calculations
        """
        ### BEGIN SOLUTION
        # Basic measurements
        param_count = self.count_parameters(model)
        flops = self.count_flops(model, input_tensor.shape)
        memory_stats = self.measure_memory(model, input_tensor.shape)
        latency_ms = self.measure_latency(model, input_tensor, warmup=5, iterations=20)

        # Derived metrics
        latency_seconds = latency_ms / 1000.0
        gflops_per_second = (flops / 1e9) / max(latency_seconds, 1e-6)

        # Memory bandwidth (MB/s)
        memory_bandwidth = memory_stats['peak_memory_mb'] / max(latency_seconds, 1e-6)

        # Efficiency metrics
        theoretical_peak_gflops = 100.0  # Assume 100 GFLOP/s theoretical peak for CPU
        computational_efficiency = min(gflops_per_second / theoretical_peak_gflops, 1.0)

        # Bottleneck analysis
        is_memory_bound = memory_bandwidth > gflops_per_second * 100  # Rough heuristic
        is_compute_bound = not is_memory_bound

        return {
            # Basic measurements
            'parameters': param_count,
            'flops': flops,
            'latency_ms': latency_ms,
            **memory_stats,

            # Derived metrics
            'gflops_per_second': gflops_per_second,
            'memory_bandwidth_mbs': memory_bandwidth,
            'computational_efficiency': computational_efficiency,

            # Bottleneck analysis
            'is_memory_bound': is_memory_bound,
            'is_compute_bound': is_compute_bound,
            'bottleneck': 'memory' if is_memory_bound else 'compute'
        }
        ### END SOLUTION

    def profile_backward_pass(self, model, input_tensor, _loss_fn=None) -> Dict[str, Any]:
        """
        Profile both forward and backward passes for training analysis.

        TODO: Implement training-focused profiling

        APPROACH:
        1. Profile forward pass first
        2. Estimate backward pass costs (typically 2× forward)
        3. Calculate total training iteration metrics
        4. Analyze memory requirements for gradients and optimizers

        BACKWARD PASS ESTIMATES:
        - FLOPs: ~2× forward pass (gradient computation)
        - Memory: +1× parameters (gradient storage)
        - Latency: ~2× forward pass (more complex operations)

        EXAMPLE:
        >>> model = Linear(128, 64)
        >>> input_data = Tensor(np.random.randn(16, 128))
        >>> profiler = Profiler()
        >>> profile = profiler.profile_backward_pass(model, input_data)
        >>> print(f"Training iteration: {profile['total_latency_ms']:.2f} ms")
        Training iteration: 0.45 ms

        HINTS:
        - Total memory = parameters + activations + gradients
        - Optimizer memory depends on algorithm (SGD: 0×, Adam: 2×)
        - Consider gradient accumulation effects
        """
        ### BEGIN SOLUTION
        # Get forward pass profile
        forward_profile = self.profile_forward_pass(model, input_tensor)

        # Estimate backward pass (typically 2× forward)
        backward_flops = forward_profile['flops'] * 2
        backward_latency_ms = forward_profile['latency_ms'] * 2

        # Gradient memory (equal to parameter memory)
        gradient_memory_mb = forward_profile['parameter_memory_mb']

        # Total training iteration
        total_flops = forward_profile['flops'] + backward_flops
        total_latency_ms = forward_profile['latency_ms'] + backward_latency_ms
        total_memory_mb = (forward_profile['parameter_memory_mb'] +
                          forward_profile['activation_memory_mb'] +
                          gradient_memory_mb)

        # Training efficiency
        total_gflops_per_second = (total_flops / 1e9) / (total_latency_ms / 1000.0)

        # Optimizer memory estimates
        optimizer_memory_estimates = {
            'sgd': 0,  # No extra memory
            'adam': gradient_memory_mb * 2,  # Momentum + velocity
            'adamw': gradient_memory_mb * 2,  # Same as Adam
        }

        return {
            # Forward pass
            'forward_flops': forward_profile['flops'],
            'forward_latency_ms': forward_profile['latency_ms'],
            'forward_memory_mb': forward_profile['peak_memory_mb'],

            # Backward pass estimates
            'backward_flops': backward_flops,
            'backward_latency_ms': backward_latency_ms,
            'gradient_memory_mb': gradient_memory_mb,

            # Total training iteration
            'total_flops': total_flops,
            'total_latency_ms': total_latency_ms,
            'total_memory_mb': total_memory_mb,
            'total_gflops_per_second': total_gflops_per_second,

            # Optimizer memory requirements
            'optimizer_memory_estimates': optimizer_memory_estimates,

            # Training insights
            'memory_efficiency': forward_profile['memory_efficiency'],
            'bottleneck': forward_profile['bottleneck']
        }
        ### END SOLUTION

# %% [markdown]
"""
## 🏗️ Helper Functions - Quick Profiling Utilities

These helper functions provide simplified interfaces for common profiling tasks.
They make it easy to quickly profile models and analyze characteristics without
manually calling multiple profiler methods.

### Why Helper Functions Matter

In production ML engineering, you often need quick insights without setting up
full profiling workflows. These utilities provide:
- **Quick profiling**: One-line model analysis with formatted output
- **Weight analysis**: Understanding parameter distributions for compression
- **Student-friendly output**: Clear, formatted results for learning

These functions wrap our core Profiler class with convenience interfaces used
in real ML workflows for rapid iteration and debugging.
"""

# %% nbgrader={"grade": false, "grade_id": "helper_quick_profile", "solution": true}
#| export
def quick_profile(model, input_tensor, profiler=None):
    """
    Quick profiling function for immediate insights.

    Provides a simplified interface for profiling that displays key metrics
    in a student-friendly format.

    Args:
        model: Model to profile
        input_tensor: Input data for profiling
        profiler: Optional Profiler instance (creates new one if None)

    Returns:
        dict: Profile results with key metrics

    Example:
        >>> model = Linear(128, 64)
        >>> input_data = Tensor(np.random.randn(16, 128))
        >>> results = quick_profile(model, input_data)
        >>> # Displays formatted output automatically
    """
    if profiler is None:
        profiler = Profiler()

    profile = profiler.profile_forward_pass(model, input_tensor)

    # Display formatted results
    print("🔬 Quick Profile Results:")
    print(f"   Parameters: {profile['parameters']:,}")
    print(f"   FLOPs: {profile['flops']:,}")
    print(f"   Latency: {profile['latency_ms']:.2f} ms")
    print(f"   Memory: {profile['peak_memory_mb']:.2f} MB")
    print(f"   Bottleneck: {profile['bottleneck']}")
    print(f"   Efficiency: {profile['computational_efficiency']*100:.1f}%")

    return profile

# %% nbgrader={"grade": false, "grade_id": "helper_weight_distribution", "solution": true}
#| export
def analyze_weight_distribution(model, percentiles=[10, 25, 50, 75, 90]):
    """
    Analyze weight distribution for compression insights.

    Helps understand which weights are small and might be prunable.
    Used by Module 17 (Compression) to motivate pruning.

    Args:
        model: Model to analyze
        percentiles: List of percentiles to compute

    Returns:
        dict: Weight distribution statistics

    Example:
        >>> model = Linear(512, 512)
        >>> stats = analyze_weight_distribution(model)
        >>> print(f"Weights < 0.01: {stats['below_threshold_001']:.1f}%")
    """
    # Collect all weights
    weights = []
    if hasattr(model, 'parameters'):
        for param in model.parameters():
            weights.extend(param.data.flatten().tolist())
    elif hasattr(model, 'weight'):
        weights.extend(model.weight.data.flatten().tolist())
    else:
        return {'error': 'No weights found'}

    weights = np.array(weights)
    abs_weights = np.abs(weights)

    # Calculate statistics
    stats = {
        'total_weights': len(weights),
        'mean': float(np.mean(abs_weights)),
        'std': float(np.std(abs_weights)),
        'min': float(np.min(abs_weights)),
        'max': float(np.max(abs_weights)),
    }

    # Percentile analysis
    for p in percentiles:
        stats[f'percentile_{p}'] = float(np.percentile(abs_weights, p))

    # Threshold analysis (useful for pruning)
    for threshold in [0.001, 0.01, 0.1]:
        below = np.sum(abs_weights < threshold) / len(weights) * 100
        stats[f'below_threshold_{str(threshold).replace(".", "")}'] = below

    return stats

# %% [markdown]
"""
### 🧪 Unit Test: Helper Functions
This test validates our helper utilities work correctly and provide useful output.
**What we're testing**: Quick profiling and weight distribution analysis
**Why it matters**: These utilities are used daily in production ML workflows
**Expected**: Correct profiles with formatted output
"""

# %% nbgrader={"grade": true, "grade_id": "test_helper_functions", "locked": true, "points": 5}
def test_unit_helper_functions():
    """🔬 Test helper function implementations."""
    print("🔬 Unit Test: Helper Functions...")

    # Test 1: Quick profile function
    from tinytorch.core.layers import Linear
    test_model = Linear(16, 8)
    test_input = Tensor(np.random.randn(8, 16))
    profile = quick_profile(test_model, test_input, profiler=Profiler())

    # Validate profile contains expected keys
    assert 'parameters' in profile, "Quick profile should include parameters"
    assert 'flops' in profile, "Quick profile should include FLOPs"
    assert 'latency_ms' in profile, "Quick profile should include latency"
    print("✅ Quick profile provides comprehensive metrics")

    # Test 2: Weight distribution analysis
    class SimpleModel:
        def __init__(self):
            self.weight = Tensor(np.random.randn(10, 5) * 0.1)  # Small weights

    model = SimpleModel()
    stats = analyze_weight_distribution(model)

    # Validate statistics structure
    assert 'total_weights' in stats, "Should count total weights"
    assert 'mean' in stats, "Should compute mean"
    assert 'std' in stats, "Should compute standard deviation"
    assert stats['total_weights'] == 50, f"Expected 50 weights, got {stats['total_weights']}"
    print(f"✅ Weight distribution analysis: {stats['total_weights']} weights analyzed")

    # Test 3: Weight distribution with no weights
    class NoWeightModel:
        pass

    no_weight_model = NoWeightModel()
    stats = analyze_weight_distribution(no_weight_model)
    assert 'error' in stats, "Should handle models without weights"
    print("✅ Handles models without weights gracefully")

    print("✅ Helper functions work correctly!")

if __name__ == "__main__":
    test_unit_helper_functions()

# %% [markdown]
"""
## 🏗️ Parameter Counting - Model Size Analysis

Parameter counting is the foundation of model profiling. Every parameter contributes to memory usage, training time, and model complexity. Let's validate our implementation.

### Why Parameter Counting Matters
```
Model Deployment Pipeline:
Parameters → Memory → Hardware → Cost
    ↓         ↓         ↓        ↓
  125M    500MB     8GB GPU   $200/month

Parameter Growth Examples:
Small:   GPT-2 Small (124M parameters)   → 500MB memory
Medium:  GPT-2 Medium (350M parameters) → 1.4GB memory
Large:   GPT-2 Large (774M parameters)  → 3.1GB memory
XL:      GPT-2 XL (1.5B parameters)     → 6.0GB memory
```
"""

# %% [markdown]
"""
### 🧪 Unit Test: Parameter Counting
This test validates our parameter counting works correctly for different model types.
**What we're testing**: Parameter counting accuracy for various architectures
**Why it matters**: Accurate parameter counts predict memory usage and model complexity
**Expected**: Correct counts for known model configurations
"""

# %% nbgrader={"grade": true, "grade_id": "test_parameter_counting", "locked": true, "points": 10}
def test_unit_parameter_counting():
    """🔬 Test parameter counting implementation."""
    print("🔬 Unit Test: Parameter Counting...")

    profiler = Profiler()

    # Test 1: Simple model with known parameters
    class SimpleModel:
        def __init__(self):
            self.weight = Tensor(np.random.randn(10, 5))
            self.bias = Tensor(np.random.randn(5))

        def parameters(self):
            return [self.weight, self.bias]

    simple_model = SimpleModel()
    param_count = profiler.count_parameters(simple_model)
    expected_count = 10 * 5 + 5  # weight + bias
    assert param_count == expected_count, f"Expected {expected_count} parameters, got {param_count}"
    print(f"✅ Simple model: {param_count} parameters")

    # Test 2: Model without parameters
    class NoParamModel:
        def __init__(self):
            pass

    no_param_model = NoParamModel()
    param_count = profiler.count_parameters(no_param_model)
    assert param_count == 0, f"Expected 0 parameters, got {param_count}"
    print(f"✅ No parameter model: {param_count} parameters")

    # Test 3: Direct tensor (no parameters)
    test_tensor = Tensor(np.random.randn(2, 3))
    param_count = profiler.count_parameters(test_tensor)
    assert param_count == 0, f"Expected 0 parameters for tensor, got {param_count}"
    print(f"✅ Direct tensor: {param_count} parameters")

    print("✅ Parameter counting works correctly!")

if __name__ == "__main__":
    test_unit_parameter_counting()

# %% [markdown]
"""
## 🏗️ FLOP Counting - Computational Cost Estimation

FLOPs measure the computational work required for model operations. Unlike latency, FLOPs are hardware-independent and help predict compute costs across different systems.

### FLOP Counting Visualization
```
Linear Layer FLOP Breakdown:
Input (batch=32, features=768) × Weight (768, 3072) + Bias (3072)
                    ↓
Matrix Multiplication: 32 × 768 × 3072 × 2 = 150,994,944 FLOPs
Bias Addition:         32 × 3072 × 1      =     98,304 FLOPs
                    ↓
Total FLOPs:                               151,093,248 FLOPs

Convolution FLOP Breakdown:
Input (batch=1, channels=3, H=224, W=224)
Kernel (out=64, in=3, kH=7, kW=7)
                    ↓
Output size: (224×224) → (112×112) with stride=2
FLOPs = 112 × 112 × 7 × 7 × 3 × 64 × 2 = 235,012,096 FLOPs
```

### FLOP Counting Strategy
Different operations require different FLOP calculations:
- **Matrix operations**: M × N × K × 2 (multiply + add)
- **Convolutions**: Output spatial × kernel spatial × channels
- **Activations**: Usually 1 FLOP per element
"""

# %% [markdown]
"""
### 🧪 Unit Test: FLOP Counting
This test validates our FLOP counting for different operations and architectures.
**What we're testing**: FLOP calculation accuracy for various layer types
**Why it matters**: FLOPs predict computational cost and energy usage
**Expected**: Correct FLOP counts for known operation types
"""

# %% nbgrader={"grade": true, "grade_id": "test_flop_counting", "locked": true, "points": 10}
def test_unit_flop_counting():
    """🔬 Test FLOP counting implementation."""
    print("🔬 Unit Test: FLOP Counting...")

    profiler = Profiler()

    # Test 1: Simple tensor operations
    test_tensor = Tensor(np.random.randn(4, 8))
    flops = profiler.count_flops(test_tensor, (4, 8))
    expected_flops = 4 * 8  # 1 FLOP per element for generic operation
    assert flops == expected_flops, f"Expected {expected_flops} FLOPs, got {flops}"
    print(f"✅ Tensor operation: {flops} FLOPs")

    # Test 2: Simulated Linear layer
    class MockLinear:
        def __init__(self, in_features, out_features):
            self.weight = Tensor(np.random.randn(in_features, out_features))
            self.__class__.__name__ = 'Linear'

    mock_linear = MockLinear(128, 64)
    flops = profiler.count_flops(mock_linear, (1, 128))
    expected_flops = 128 * 64 * 2  # matmul FLOPs
    assert flops == expected_flops, f"Expected {expected_flops} FLOPs, got {flops}"
    print(f"✅ Linear layer: {flops} FLOPs")

    # Test 3: Batch size independence
    flops_batch1 = profiler.count_flops(mock_linear, (1, 128))
    flops_batch32 = profiler.count_flops(mock_linear, (32, 128))
    assert flops_batch1 == flops_batch32, "FLOPs should be independent of batch size"
    print(f"✅ Batch independence: {flops_batch1} FLOPs (same for batch 1 and 32)")

    print("✅ FLOP counting works correctly!")

if __name__ == "__main__":
    test_unit_flop_counting()

# %% [markdown]
"""
## 🏗️ Memory Profiling - Understanding Memory Usage Patterns

Memory profiling reveals how much RAM your model consumes during training and inference. This is critical for deployment planning and perf.

### Memory Usage Breakdown
```
ML Model Memory Components:
┌───────────────────────────────────────────────────┐
│                 Total Memory                      │
├─────────────────┬─────────────────┬───────────────┤
│   Parameters    │   Activations   │  Gradients    │
│   (persistent)  │  (per forward)  │ (per backward)│
├─────────────────┼─────────────────┼───────────────┤
│ Linear weights  │ Hidden states   │ ∂L/∂W         │
│ Conv filters    │ Attention maps  │ ∂L/∂b         │
│ Embeddings      │ Residual cache  │ Optimizer     │
└─────────────────┴─────────────────┴───────────────┘

Memory Scaling:
Batch Size → Activation Memory (linear scaling)
Model Size → Parameter + Gradient Memory (linear scaling)
Sequence Length → Attention Memory (quadratic scaling!)
```

### Memory Measurement Strategy
We use Python's `tracemalloc` to track memory allocations during model execution. This gives us precise measurements of memory usage patterns.
"""

# %% [markdown]
"""
### 🧪 Unit Test: Memory Measurement
This test validates our memory tracking works correctly and provides useful metrics.
**What we're testing**: Memory usage measurement and calculation accuracy
**Why it matters**: Memory constraints often limit model deployment
**Expected**: Reasonable memory measurements with proper components
"""

# %% nbgrader={"grade": true, "grade_id": "test_memory_measurement", "locked": true, "points": 10}
def test_unit_memory_measurement():
    """🔬 Test memory measurement implementation."""
    print("🔬 Unit Test: Memory Measurement...")

    profiler = Profiler()

    # Test 1: Basic memory measurement
    test_tensor = Tensor(np.random.randn(10, 20))
    from tinytorch.core.layers import Linear
    test_model = Linear(20, 10)
    memory_stats = profiler.measure_memory(test_model, (10, 20))

    # Validate dictionary structure
    required_keys = ['parameter_memory_mb', 'activation_memory_mb', 'peak_memory_mb', 'memory_efficiency']
    for key in required_keys:
        assert key in memory_stats, f"Missing key: {key}"

    # Validate non-negative values
    for key in required_keys:
        assert memory_stats[key] >= 0, f"{key} should be non-negative, got {memory_stats[key]}"

    print(f"✅ Basic measurement: {memory_stats['peak_memory_mb']:.3f} MB peak")

    # Test 2: Memory scaling with size
    from tinytorch.core.layers import Linear
    small_model = Linear(5, 5)
    large_model = Linear(50, 50)

    small_memory = profiler.measure_memory(small_model, (5, 5))
    large_memory = profiler.measure_memory(large_model, (50, 50))

    # Larger tensor should use more activation memory
    assert large_memory['activation_memory_mb'] >= small_memory['activation_memory_mb'], \
        "Larger tensor should use more activation memory"

    print(f"✅ Scaling: Small {small_memory['activation_memory_mb']:.3f} MB → Large {large_memory['activation_memory_mb']:.3f} MB")

    # Test 3: Efficiency bounds
    assert 0 <= memory_stats['memory_efficiency'] <= 1.0, \
        f"Memory efficiency should be between 0 and 1, got {memory_stats['memory_efficiency']}"

    print(f"✅ Efficiency: {memory_stats['memory_efficiency']:.3f} (0-1 range)")

    print("✅ Memory measurement works correctly!")

if __name__ == "__main__":
    test_unit_memory_measurement()

# %% [markdown]
"""
## 🏗️ Latency Measurement - Accurate Performance Timing

Latency measurement is the most challenging part of profiling because it's affected by system state, caching, and measurement overhead. We need statistical rigor to get reliable results.

### Latency Measurement Challenges
```
Timing Challenges:
┌─────────────────────────────────────────────────┐
│                 Time Variance                   │
├─────────────────┬─────────────────┬─────────────┤
│  System Noise   │   Cache Effects │   Thermal   │
│                 │                 │  Throttling │
├─────────────────┼─────────────────┼─────────────┤
│ Background      │ Cold start vs   │ CPU slows   │
│ processes       │ warm caches     │ when hot    │
│ OS scheduling   │ Memory locality │ GPU thermal │
│ Network I/O     │ Branch predict  │ limits      │
└─────────────────┴─────────────────┴─────────────┘

Solution: Statistical Approach
Warmup → Multiple measurements → Robust statistics (median)
```

### Measurement Protocol
Our latency measurement follows professional benchmarking practices:
1. **Warmup runs** to stabilize system state
2. **Multiple measurements** for statistical significance
3. **Median calculation** to handle outliers
4. **Memory cleanup** to prevent contamination
"""

# %% [markdown]
"""
### 🧪 Unit Test: Latency Measurement
This test validates our latency measurement provides consistent and reasonable results.
**What we're testing**: Timing accuracy and statistical robustness
**Why it matters**: Latency determines real-world deployment feasibility
**Expected**: Consistent timing measurements with proper statistical handling
"""

# %% nbgrader={"grade": true, "grade_id": "test_latency_measurement", "locked": true, "points": 10}
def test_unit_latency_measurement():
    """🔬 Test latency measurement implementation."""
    print("🔬 Unit Test: Latency Measurement...")

    profiler = Profiler()

    # Test 1: Basic latency measurement
    from tinytorch.core.layers import Linear
    test_model = Linear(8, 4)
    test_input = Tensor(np.random.randn(4, 8))
    latency = profiler.measure_latency(test_model, test_input, warmup=2, iterations=5)

    assert latency >= 0, f"Latency should be non-negative, got {latency}"
    assert latency < 1000, f"Latency seems too high for simple operation: {latency} ms"
    print(f"✅ Basic latency: {latency:.3f} ms")

    # Test 2: Measurement consistency
    latencies = []
    for _ in range(3):
        lat = profiler.measure_latency(test_model, test_input, warmup=1, iterations=3)
        latencies.append(lat)

    # Measurements should be in reasonable range
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    assert std_latency < avg_latency, "Standard deviation shouldn't exceed mean for simple operations"
    print(f"✅ Consistency: {avg_latency:.3f} ± {std_latency:.3f} ms")

    # Test 3: Size scaling
    small_model = Linear(2, 2)
    large_model = Linear(20, 20)
    small_input = Tensor(np.random.randn(2, 2))
    large_input = Tensor(np.random.randn(20, 20))

    small_latency = profiler.measure_latency(small_model, small_input, warmup=1, iterations=3)
    large_latency = profiler.measure_latency(large_model, large_input, warmup=1, iterations=3)

    # Larger operations might take longer (though not guaranteed for simple operations)
    print(f"✅ Scaling: Small {small_latency:.3f} ms, Large {large_latency:.3f} ms")

    print("✅ Latency measurement works correctly!")

if __name__ == "__main__":
    test_unit_latency_measurement()

# %% [markdown]
"""
## 🔧 Integration: Advanced Profiling Functions

Now let's validate our higher-level profiling functions that combine core measurements into comprehensive analysis tools.

### Advanced Profiling Architecture
```
Core Profiler Methods → Advanced Analysis Functions → Optimization Insights
        ↓                         ↓                         ↓
count_parameters()      profile_forward_pass()      "Memory-bound workload"
count_flops()          profile_backward_pass()      "Optimize data movement"
measure_memory()       profile_layer()              "Focus on bandwidth"
measure_latency()      benchmark_efficiency()       "Use quantization"
```

### Forward Pass Profiling - Complete Performance Picture

A forward pass profile combines all our measurements to understand model behavior comprehensively. This is essential for optimization decisions.
"""

# %% [markdown]
"""
### Backward Pass Profiling - Training Analysis

Training requires both forward and backward passes. The backward pass typically uses 2× the compute and adds gradient memory. Understanding this is crucial for training perf.

### Training Memory Visualization
```
Training Memory Timeline:
Forward Pass:   [Parameters] + [Activations]
                     ↓
Backward Pass:  [Parameters] + [Activations] + [Gradients]
                     ↓
Optimizer:      [Parameters] + [Gradients] + [Optimizer State]

Memory Examples:
Model: 125M parameters (500MB)
Forward:  500MB params + 100MB activations = 600MB
Backward: 500MB params + 100MB activations + 500MB gradients = 1,100MB
Adam:     500MB params + 500MB gradients + 1,000MB momentum/velocity = 2,000MB

Total Training Memory: 4× parameter memory!
```
"""

# %% [markdown]
"""
### 🧪 Unit Test: Advanced Profiling Functions
This test validates our advanced profiling functions provide comprehensive analysis.
**What we're testing**: Forward and backward pass profiling completeness
**Why it matters**: Training optimization requires understanding both passes
**Expected**: Complete profiles with all required metrics and relationships
"""

# %% nbgrader={"grade": true, "grade_id": "test_advanced_profiling", "locked": true, "points": 15}
def test_unit_advanced_profiling():
    """🔬 Test advanced profiling functions."""
    print("🔬 Unit Test: Advanced Profiling Functions...")

    # Create profiler and test model
    profiler = Profiler()
    from tinytorch.core.layers import Linear
    test_model = Linear(8, 4)
    test_input = Tensor(np.random.randn(4, 8))

    # Test forward pass profiling
    forward_profile = profiler.profile_forward_pass(test_model, test_input)

    # Validate forward profile structure
    required_forward_keys = [
        'parameters', 'flops', 'latency_ms', 'gflops_per_second',
        'memory_bandwidth_mbs', 'bottleneck'
    ]

    for key in required_forward_keys:
        assert key in forward_profile, f"Missing key: {key}"

    assert forward_profile['parameters'] >= 0
    assert forward_profile['flops'] >= 0
    assert forward_profile['latency_ms'] >= 0
    assert forward_profile['gflops_per_second'] >= 0

    print(f"✅ Forward profiling: {forward_profile['gflops_per_second']:.2f} GFLOP/s")

    # Test backward pass profiling
    backward_profile = profiler.profile_backward_pass(test_model, test_input)

    # Validate backward profile structure
    required_backward_keys = [
        'forward_flops', 'backward_flops', 'total_flops',
        'total_latency_ms', 'total_memory_mb', 'optimizer_memory_estimates'
    ]

    for key in required_backward_keys:
        assert key in backward_profile, f"Missing key: {key}"

    # Validate relationships
    assert backward_profile['total_flops'] >= backward_profile['forward_flops']
    assert backward_profile['total_latency_ms'] >= backward_profile['forward_latency_ms']
    assert 'sgd' in backward_profile['optimizer_memory_estimates']
    assert 'adam' in backward_profile['optimizer_memory_estimates']

    # Check backward pass estimates are reasonable
    assert backward_profile['backward_flops'] >= backward_profile['forward_flops'], \
        "Backward pass should have at least as many FLOPs as forward"
    assert backward_profile['gradient_memory_mb'] >= 0, \
        "Gradient memory should be non-negative"

    print(f"✅ Backward profiling: {backward_profile['total_latency_ms']:.2f} ms total")
    print(f"✅ Memory breakdown: {backward_profile['total_memory_mb']:.2f} MB training")
    print("✅ Advanced profiling functions work correctly!")

if __name__ == "__main__":
    test_unit_advanced_profiling()

# %% [markdown]
"""
## 📊 Systems Analysis: Understanding Performance Characteristics

Let's analyze how different model characteristics affect performance. This analysis guides optimization decisions and helps identify bottlenecks.

### Performance Analysis Workflow
```
Model Scaling Analysis:
Size → Memory → Latency → Throughput → Bottleneck Identification
 ↓      ↓        ↓         ↓            ↓
64    1MB     0.1ms    10K ops/s    Memory bound
128   4MB     0.2ms    8K ops/s     Memory bound
256   16MB    0.5ms    4K ops/s     Memory bound
512   64MB    2.0ms    1K ops/s     Memory bound

Insight: This workload is memory-bound → Optimize data movement, not compute!
```
"""

# %% nbgrader={"grade": false, "grade_id": "performance_analysis", "solution": true}
def analyze_model_scaling():
    """📊 Analyze how model performance scales with size."""
    print("📊 Analyzing Model Scaling Characteristics...")

    profiler = Profiler()
    results = []

    # Test different model sizes
    sizes = [64, 128, 256, 512]

    print("\nModel Scaling Analysis:")
    print("Size\tParams\t\tFLOPs\t\tLatency(ms)\tMemory(MB)\tGFLOP/s")
    print("-" * 80)

    for size in sizes:
        # Create models of different sizes for comparison
        from tinytorch.core.layers import Linear
        test_model = Linear(size, size)
        input_shape = (32, size)  # Batch of 32
        dummy_input = Tensor(np.random.randn(*input_shape))

        # Simulate linear layer characteristics
        linear_params = size * size + size  # W + b
        linear_flops = size * size * 2  # matmul

        # Measure actual performance
        latency = profiler.measure_latency(test_model, dummy_input, warmup=3, iterations=10)
        memory = profiler.measure_memory(test_model, input_shape)

        gflops_per_second = (linear_flops / 1e9) / (latency / 1000)

        results.append({
            'size': size,
            'parameters': linear_params,
            'flops': linear_flops,
            'latency_ms': latency,
            'memory_mb': memory['peak_memory_mb'],
            'gflops_per_second': gflops_per_second
        })

        print(f"{size}\t{linear_params:,}\t\t{linear_flops:,}\t\t"
              f"{latency:.2f}\t\t{memory['peak_memory_mb']:.2f}\t\t"
              f"{gflops_per_second:.2f}")

    # Analysis insights
    print("\n💡 Scaling Analysis Insights:")

    # Memory scaling
    memory_growth = results[-1]['memory_mb'] / max(results[0]['memory_mb'], 0.001)
    print(f"Memory grows {memory_growth:.1f}× from {sizes[0]} to {sizes[-1]} size")

    # Compute scaling
    compute_growth = results[-1]['gflops_per_second'] / max(results[0]['gflops_per_second'], 0.001)
    print(f"Compute efficiency changes {compute_growth:.1f}× with size")

    # Performance characteristics
    avg_efficiency = np.mean([r['gflops_per_second'] for r in results])
    if avg_efficiency < 10:  # Arbitrary threshold for "low" efficiency
        print("🚀 Low compute efficiency suggests memory-bound workload")
    else:
        print("🚀 High compute efficiency suggests compute-bound workload")

def analyze_batch_size_effects():
    """📊 Analyze how batch size affects performance and efficiency."""
    print("\n📊 Analyzing Batch Size Effects...")

    profiler = Profiler()
    batch_sizes = [1, 8, 32, 128]
    feature_size = 256

    print("\nBatch Size Effects Analysis:")
    print("Batch\tLatency(ms)\tThroughput(samples/s)\tMemory(MB)\tMemory Efficiency")
    print("-" * 85)

    for batch_size in batch_sizes:
        from tinytorch.core.layers import Linear
        test_model = Linear(feature_size, feature_size)
        input_shape = (batch_size, feature_size)
        dummy_input = Tensor(np.random.randn(*input_shape))

        # Measure performance
        latency = profiler.measure_latency(test_model, dummy_input, warmup=3, iterations=10)
        memory = profiler.measure_memory(test_model, input_shape)

        # Calculate throughput
        samples_per_second = (batch_size * 1000) / latency  # samples/second

        # Calculate efficiency (samples per unit memory)
        efficiency = samples_per_second / max(memory['peak_memory_mb'], 0.001)

        print(f"{batch_size}\t{latency:.2f}\t\t{samples_per_second:.0f}\t\t\t"
              f"{memory['peak_memory_mb']:.2f}\t\t{efficiency:.1f}")

    print("\n💡 Batch Size Insights:")
    print("Larger batches typically improve throughput but increase memory usage")

# Run the analysis
if __name__ == "__main__":
    analyze_model_scaling()
    analyze_batch_size_effects()

# %% [markdown]
"""
## 📊 Optimization Insights: Production Performance Patterns

Understanding profiling results helps guide optimization decisions. Let's analyze different operation types and measurement overhead.

### Operation Efficiency Analysis
```
Operation Types and Their Characteristics:
┌─────────────────┬──────────────────┬──────────────────┬─────────────────┐
│   Operation     │   Compute/Memory │   Optimization   │   Priority      │
├─────────────────┼──────────────────┼──────────────────┼─────────────────┤
│ Matrix Multiply │   Compute-bound  │   BLAS libraries │   High          │
│ Elementwise     │   Memory-bound   │   Data locality  │   Medium        │
│ Reductions      │   Memory-bound   │   Parallelization│   Medium        │
│ Attention       │   Memory-bound   │   FlashAttention │   High          │
└─────────────────┴──────────────────┴──────────────────┴─────────────────┘

Optimization Strategy:
1. Profile first → Identify bottlenecks
2. Focus on compute-bound ops → Algorithmic improvements
3. Focus on memory-bound ops → Data movement optimization
4. Measure again → Verify improvements
```
"""

# %% nbgrader={"grade": false, "grade_id": "optimization_insights", "solution": true}
def benchmark_operation_efficiency():
    """📊 Compare efficiency of different operations for optimization guidance."""
    print("📊 Benchmarking Operation Efficiency...")

    profiler = Profiler()
    operations = []

    # Test different operation types
    size = 256
    input_tensor = Tensor(np.random.randn(32, size))

    # Elementwise operations (memory-bound)
    # Create a simple model wrapper for elementwise operations
    class ElementwiseModel:
        def forward(self, x):
            return x + x  # Simple elementwise operation

    elementwise_model = ElementwiseModel()
    elementwise_latency = profiler.measure_latency(elementwise_model, input_tensor, iterations=20)
    elementwise_flops = size * 32  # One operation per element

    operations.append({
        'operation': 'Elementwise',
        'latency_ms': elementwise_latency,
        'flops': elementwise_flops,
        'gflops_per_second': (elementwise_flops / 1e9) / (elementwise_latency / 1000),
        'efficiency_class': 'memory-bound',
        'optimization_focus': 'data_locality'
    })

    # Matrix operations (compute-bound)
    from tinytorch.core.layers import Linear
    matrix_model = Linear(size, size)
    matrix_latency = profiler.measure_latency(matrix_model, input_tensor, iterations=10)
    matrix_flops = size * size * 2  # Matrix multiplication

    operations.append({
        'operation': 'Matrix Multiply',
        'latency_ms': matrix_latency,
        'flops': matrix_flops,
        'gflops_per_second': (matrix_flops / 1e9) / (matrix_latency / 1000),
        'efficiency_class': 'compute-bound',
        'optimization_focus': 'algorithms'
    })

    # Reduction operations (memory-bound)
    class ReductionModel:
        def forward(self, x):
            return x.sum()  # Sum reduction operation

    reduction_model = ReductionModel()
    reduction_latency = profiler.measure_latency(reduction_model, input_tensor, iterations=20)
    reduction_flops = size * 32  # Sum reduction

    operations.append({
        'operation': 'Reduction',
        'latency_ms': reduction_latency,
        'flops': reduction_flops,
        'gflops_per_second': (reduction_flops / 1e9) / (reduction_latency / 1000),
        'efficiency_class': 'memory-bound',
        'optimization_focus': 'parallelization'
    })

    print("\nOperation Efficiency Comparison:")
    print("Operation\t\tLatency(ms)\tGFLOP/s\t\tEfficiency Class\tOptimization Focus")
    print("-" * 95)

    for op in operations:
        print(f"{op['operation']:<15}\t{op['latency_ms']:.3f}\t\t"
              f"{op['gflops_per_second']:.2f}\t\t{op['efficiency_class']:<15}\t{op['optimization_focus']}")

    print("\n💡 Operation Optimization Insights:")

    # Find most and least efficient
    best_op = max(operations, key=lambda x: x['gflops_per_second'])
    worst_op = min(operations, key=lambda x: x['gflops_per_second'])

    print(f"Most efficient: {best_op['operation']} ({best_op['gflops_per_second']:.2f} GFLOP/s)")
    print(f"Least efficient: {worst_op['operation']} ({worst_op['gflops_per_second']:.2f} GFLOP/s)")

    # Count operation types
    memory_bound_ops = [op for op in operations if op['efficiency_class'] == 'memory-bound']
    compute_bound_ops = [op for op in operations if op['efficiency_class'] == 'compute-bound']

    print(f"\n🚀 Optimization Priority:")
    if len(memory_bound_ops) > len(compute_bound_ops):
        print("Focus on memory optimization: data locality, bandwidth, caching")
    else:
        print("Focus on compute optimization: better algorithms, vectorization")

def analyze_profiling_overhead():
    """📊 Measure the overhead of profiling itself."""
    print("\n📊 Analyzing Profiling Overhead...")

    # Test with and without profiling
    test_tensor = Tensor(np.random.randn(100, 100))
    iterations = 50

    # Without profiling - baseline measurement
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = test_tensor.data.copy()  # Simple operation
    end_time = time.perf_counter()
    baseline_ms = (end_time - start_time) * 1000

    # With profiling - includes measurement overhead
    profiler = Profiler()
    # Create a simple model for profiling overhead measurement
    class TestModel:
        def forward(self, x):
            return x + 1.0

    test_model = TestModel()
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = profiler.measure_latency(test_model, test_tensor, warmup=1, iterations=1)
    end_time = time.perf_counter()
    profiled_ms = (end_time - start_time) * 1000

    overhead_factor = profiled_ms / max(baseline_ms, 0.001)

    print(f"\nProfiling Overhead Analysis:")
    print(f"Baseline execution: {baseline_ms:.2f} ms")
    print(f"With profiling: {profiled_ms:.2f} ms")
    print(f"Profiling overhead: {overhead_factor:.1f}× slower")

    print(f"\n💡 Profiling Overhead Insights:")
    if overhead_factor < 2:
        print("Low overhead - suitable for frequent profiling")
    elif overhead_factor < 10:
        print("Moderate overhead - use for development and debugging")
    else:
        print("High overhead - use sparingly in production")

# Run optimization analysis
if __name__ == "__main__":
    benchmark_operation_efficiency()
    analyze_profiling_overhead()

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "test_module", "locked": true, "points": 20}
def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire profiling module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_helper_functions()
    test_unit_parameter_counting()
    test_unit_flop_counting()
    test_unit_memory_measurement()
    test_unit_latency_measurement()
    test_unit_advanced_profiling()

    print("\nRunning integration scenarios...")

    # Test realistic usage patterns
    print("🔬 Integration Test: Complete Profiling Workflow...")

    # Create profiler
    profiler = Profiler()

    # Create test model and data
    from tinytorch.core.layers import Linear
    test_model = Linear(16, 32)
    test_input = Tensor(np.random.randn(8, 16))

    # Run complete profiling workflow
    print("1. Measuring model characteristics...")
    params = profiler.count_parameters(test_model)
    flops = profiler.count_flops(test_model, test_input.shape)
    memory = profiler.measure_memory(test_model, test_input.shape)
    latency = profiler.measure_latency(test_model, test_input, warmup=2, iterations=5)

    print(f"   Parameters: {params}")
    print(f"   FLOPs: {flops}")
    print(f"   Memory: {memory['peak_memory_mb']:.2f} MB")
    print(f"   Latency: {latency:.2f} ms")

    # Test advanced profiling
    print("2. Running advanced profiling...")
    forward_profile = profiler.profile_forward_pass(test_model, test_input)
    backward_profile = profiler.profile_backward_pass(test_model, test_input)

    assert 'gflops_per_second' in forward_profile
    assert 'total_latency_ms' in backward_profile
    print(f"   Forward GFLOP/s: {forward_profile['gflops_per_second']:.2f}")
    print(f"   Training latency: {backward_profile['total_latency_ms']:.2f} ms")

    # Test bottleneck analysis
    print("3. Analyzing performance bottlenecks...")
    bottleneck = forward_profile['bottleneck']
    efficiency = forward_profile['computational_efficiency']
    print(f"   Bottleneck: {bottleneck}")
    print(f"   Compute efficiency: {efficiency:.3f}")

    # Validate end-to-end workflow
    assert params >= 0, "Parameter count should be non-negative"
    assert flops >= 0, "FLOP count should be non-negative"
    assert memory['peak_memory_mb'] >= 0, "Memory usage should be non-negative"
    assert latency >= 0, "Latency should be non-negative"
    assert forward_profile['gflops_per_second'] >= 0, "GFLOP/s should be non-negative"
    assert backward_profile['total_latency_ms'] >= 0, "Total latency should be non-negative"
    assert bottleneck in ['memory', 'compute'], "Bottleneck should be memory or compute"
    assert 0 <= efficiency <= 1, "Efficiency should be between 0 and 1"

    print("✅ End-to-end profiling workflow works!")

    # Test production-like scenario
    print("4. Testing production profiling scenario...")

    # Simulate larger model analysis
    from tinytorch.core.layers import Linear
    large_model = Linear(512, 256)
    large_input = Tensor(np.random.randn(32, 512))  # Larger model input
    large_profile = profiler.profile_forward_pass(large_model, large_input)

    # Verify profile contains optimization insights
    assert 'bottleneck' in large_profile, "Profile should identify bottlenecks"
    assert 'memory_bandwidth_mbs' in large_profile, "Profile should measure memory bandwidth"

    print(f"   Large model analysis: {large_profile['bottleneck']} bottleneck")
    print(f"   Memory bandwidth: {large_profile['memory_bandwidth_mbs']:.1f} MB/s")

    print("✅ Production profiling scenario works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 14")

# Call before module summary
if __name__ == "__main__":
    test_module()

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Performance Measurement

### Question 1: FLOP Analysis
You implemented a profiler that counts FLOPs for different operations.
For a Linear layer with 1000 input features and 500 output features:
- How many FLOPs are required for one forward pass? _____ FLOPs
- If you process a batch of 32 samples, how does this change the per-sample FLOPs? _____

### Question 2: Memory Scaling
Your profiler measures memory usage for models and activations.
A transformer model has 125M parameters (500MB at FP32).
During training with batch size 16:
- What's the minimum memory for gradients? _____ MB
- With Adam optimizer, what's the total memory requirement? _____ MB

### Question 3: Performance Bottlenecks
You built tools to identify compute vs memory bottlenecks.
A model achieves 10 GFLOP/s on hardware with 100 GFLOP/s peak:
- What's the computational efficiency? _____%
- If doubling batch size doesn't improve GFLOP/s, the bottleneck is likely _____

### Question 4: Profiling Trade-offs
Your profiler adds measurement overhead to understand performance.
If profiling adds 5× overhead but reveals a 50% speedup opportunity:
- Is the profiling cost justified for development? _____
- When should you disable profiling in production? _____
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Know Your Model

**What you built:** A profiler that measures parameters, FLOPs, memory, and latency.

**Why it matters:** You can't optimize what you can't measure! Before making a model faster
or smaller, you need to know where the time and memory go. Your profiler reveals these secrets,
telling you exactly what your model costs in compute and memory.

In the next modules, you'll use profiling to guide quantization and compression decisions.
"""

# %%
def demo_profiling():
    """🎯 See your profiler reveal model secrets."""
    print("🎯 AHA MOMENT: Know Your Model")
    print("=" * 45)

    # Create a simple model
    layer = Linear(784, 128)

    # Profile it
    profiler = Profiler()
    params = profiler.count_parameters(layer)
    flops = profiler.count_flops(layer, input_shape=(1, 784))

    print(f"Model: Linear(784 → 128)")
    print(f"\nParameters: {params:,}")
    print(f"  = 784 × 128 weights + 128 biases")

    print(f"\nFLOPs: {flops:,}")
    print(f"  = 784 × 128 × 2 (multiply-add per output)")

    print(f"\nMemory: {params * 4 / 1024:.1f} KB (at FP32)")

    print("\n✨ Profiling reveals optimization opportunities!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_profiling()

# %% [markdown]
"""
## 🚀 MODULE SUMMARY: Profiling

Congratulations! You've built a comprehensive profiling system for ML performance analysis!

### Key Accomplishments
- Built complete Profiler class with parameter, FLOP, memory, and latency measurement
- Implemented advanced profiling functions for forward and backward pass analysis
- Discovered performance characteristics through scaling and efficiency analysis
- Created production-quality measurement tools for optimization guidance
- All tests pass ✅ (validated by `test_module()`)

### Systems Insights Gained
- **FLOPs vs Reality**: Theoretical operations don't always predict actual performance
- **Memory Bottlenecks**: Many ML operations are limited by memory bandwidth, not compute
- **Batch Size Effects**: Larger batches improve throughput but increase memory requirements
- **Profiling Overhead**: Measurement tools have costs but enable data-driven optimization

### Production Skills Developed
- **Performance Detective Work**: Use data, not guesses, to identify bottlenecks
- **Optimization Prioritization**: Focus efforts on actual bottlenecks, not assumptions
- **Resource Planning**: Predict memory and compute requirements for deployment
- **Statistical Rigor**: Handle measurement variance with proper methodology

### Ready for Next Steps
Your profiling implementation enables optimization modules (15-18) to make data-driven optimization decisions.
Export with: `tito module complete 14`

**Next**: Module 15 (Memoization) will use profiling to discover transformer bottlenecks and fix them!
"""
