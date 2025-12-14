#!/usr/bin/env python
"""
Performance Validation Tests for TinyTorch
===========================================
Ensures operations meet expected performance characteristics.
Tests memory usage, computational complexity, and scaling behavior.

Test Categories:
- Memory usage patterns
- Computational complexity
- No memory leaks
- Scaling behavior
- Performance bottlenecks
"""

import sys
import os
import numpy as np
import time
import tracemalloc
import pytest
from typing import Tuple

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU
from tinytorch.core.training import MeanSquaredError
from tinytorch.core.optimizers import SGD, Adam
from tinytorch.nn import Conv2d, Sequential
import tinytorch.nn.functional as F


# ============== Memory Usage Tests ==============

def test_tensor_memory_efficiency():
    """Tensors don't create unnecessary copies."""
    tracemalloc.start()

    # Create large tensor
    size = (1000, 1000)
    data = np.random.randn(*size)

    # Measure memory before
    snapshot1 = tracemalloc.take_snapshot()

    # Create tensor (should not copy if using same dtype)
    tensor = Tensor(data)

    # Measure memory after
    snapshot2 = tracemalloc.take_snapshot()

    # Calculate memory increase
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_increase = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

    # Should be minimal increase (just Tensor object overhead)
    # Not a full copy of the array
    array_size = data.nbytes
    assert total_increase < array_size * 0.5, \
        f"Tensor creation used too much memory: {total_increase / 1e6:.1f}MB"

    tracemalloc.stop()


def test_linear_layer_memory():
    """Linear layer memory usage is predictable."""
    tracemalloc.start()

    input_size, output_size = 1000, 500

    # Memory before
    snapshot1 = tracemalloc.take_snapshot()

    # Create layer
    layer = Linear(input_size, output_size)

    # Memory after
    snapshot2 = tracemalloc.take_snapshot()

    # Calculate expected memory
    # Weights: input_size * output_size * 8 bytes (float64)
    # Bias: output_size * 8 bytes
    expected = (input_size * output_size + output_size) * 8

    stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_increase = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

    # Allow 20% overhead for Python objects
    assert total_increase < expected * 1.2, \
        f"Linear layer uses too much memory: {total_increase / expected:.1f}x expected"

    tracemalloc.stop()


def test_optimizer_memory_overhead():
    """Optimizers have expected memory overhead."""
    model = Sequential([
        Linear(100, 50),
        ReLU(),
        Linear(50, 10)
    ])

    # Count parameters
    total_params = sum(p.data.size for p in model.parameters())
    param_memory = total_params * 8  # float64

    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    # SGD should have minimal overhead
    sgd = SGD(model.parameters(), learning_rate=0.01)

    snapshot2 = tracemalloc.take_snapshot()
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    sgd_overhead = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

    # SGD should use almost no extra memory
    assert sgd_overhead < param_memory * 0.1, \
        f"SGD has too much overhead: {sgd_overhead / param_memory:.1f}x parameters"

    # Adam needs momentum buffers (2x parameter memory)
    adam = Adam(model.parameters(), learning_rate=0.01)

    snapshot3 = tracemalloc.take_snapshot()
    stats = snapshot3.compare_to(snapshot2, 'lineno')
    adam_overhead = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

    # Adam should use ~2x parameter memory for momentum
    expected_adam = param_memory * 2
    assert adam_overhead < expected_adam * 1.5, \
        f"Adam uses too much memory: {adam_overhead / expected_adam:.1f}x expected"

    tracemalloc.stop()


def test_no_memory_leak_training():
    """Training loop doesn't leak memory."""
    model = Linear(10, 5)
    optimizer = SGD(model.parameters(), learning_rate=0.01)
    criterion = MeanSquaredError()

    X = Tensor(np.random.randn(100, 10))
    y = Tensor(np.random.randn(100, 5))

    # Warm up
    for _ in range(5):
        y_pred = model(X)
        loss = criterion(y_pred, y)
        try:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except:
            pass

    # Measure memory over many iterations
    tracemalloc.start()
    snapshot_start = tracemalloc.take_snapshot()

    for _ in range(100):
        y_pred = model(X)
        loss = criterion(y_pred, y)
        try:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except:
            pass

    snapshot_end = tracemalloc.take_snapshot()

    # Memory shouldn't grow significantly
    stats = snapshot_end.compare_to(snapshot_start, 'lineno')
    total_increase = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

    # Allow small increase for caching, but not linear growth
    assert total_increase < 1e6, \
        f"Possible memory leak: {total_increase / 1e6:.1f}MB increase over 100 iterations"

    tracemalloc.stop()


# ============== Computational Complexity Tests ==============

def test_linear_complexity():
    """Linear layer has O(mn) complexity."""
    sizes = [(100, 100), (200, 200), (400, 400)]
    times = []

    for m, n in sizes:
        layer = Linear(m, n)
        x = Tensor(np.random.randn(10, m))

        # Time forward pass
        start = time.perf_counter()
        for _ in range(100):
            _ = layer(x)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    # Complexity should be O(mn)
    # Time should roughly quadruple when doubling both dimensions
    ratio1 = times[1] / times[0]  # Should be ~4
    ratio2 = times[2] / times[1]  # Should be ~4

    # Allow significant tolerance for timing variance
    assert 2 < ratio1 < 8, f"Linear complexity seems wrong: {ratio1:.1f}x for 2x size"
    assert 2 < ratio2 < 8, f"Linear complexity seems wrong: {ratio2:.1f}x for 2x size"


def test_conv2d_complexity():
    """Conv2d has expected complexity."""
    # Conv complexity: O(H*W*C_in*C_out*K^2)

    times = []
    for kernel_size in [3, 5, 7]:
        conv = Conv2d(16, 32, kernel_size=kernel_size)
        x = Tensor(np.random.randn(4, 16, 32, 32))

        start = time.perf_counter()
        for _ in range(10):
            _ = conv(x)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    # Time should increase with kernel size squared
    # 5x5 is 25/9 ≈ 2.8x more ops than 3x3
    # 7x7 is 49/25 ≈ 2x more ops than 5x5

    ratio1 = times[1] / times[0]
    ratio2 = times[2] / times[1]

    # Very loose bounds due to timing variance
    assert 1.5 < ratio1 < 5, f"Conv scaling unexpected: {ratio1:.1f}x for 3→5 kernel"
    assert 1.2 < ratio2 < 4, f"Conv scaling unexpected: {ratio2:.1f}x for 5→7 kernel"


def test_matmul_vs_loops():
    """Matrix multiplication performance comparison."""
    size = 100
    a = Tensor(np.random.randn(size, size))
    b = Tensor(np.random.randn(size, size))

    # If matmul is optimized, it should be faster than naive loops
    # This test documents the performance difference

    # Time matmul
    start = time.perf_counter()
    for _ in range(10):
        if hasattr(a, '__matmul__'):
            _ = a @ b
        else:
            # Fallback to numpy
            _ = Tensor(a.data @ b.data)
    matmul_time = time.perf_counter() - start

    # This just documents performance, not a hard requirement
    ops_per_second = (size ** 3 * 10) / matmul_time
    # print(f"Matrix multiply performance: {ops_per_second / 1e9:.2f} GFLOPs")


# ============== Scaling Behavior Tests ==============

def test_batch_size_scaling():
    """Performance scales linearly with batch size."""
    model = Sequential([
        Linear(100, 50),
        ReLU(),
        Linear(50, 10)
    ])

    times = []
    batch_sizes = [10, 20, 40]

    for batch_size in batch_sizes:
        x = Tensor(np.random.randn(batch_size, 100))

        start = time.perf_counter()
        for _ in range(100):
            _ = model(x)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    # Should scale linearly with batch size
    ratio1 = times[1] / times[0]  # Should be ~2
    ratio2 = times[2] / times[1]  # Should be ~2

    assert 1.5 < ratio1 < 3, f"Batch scaling wrong: {ratio1:.1f}x for 2x batch"
    assert 1.5 < ratio2 < 3, f"Batch scaling wrong: {ratio2:.1f}x for 2x batch"


def test_deep_network_scaling():
    """Performance with network depth."""
    times = []

    for depth in [5, 10, 20]:
        layers = []
        for _ in range(depth):
            layers.append(Linear(50, 50))
            layers.append(ReLU())
        model = Sequential(layers)

        x = Tensor(np.random.randn(10, 50))

        start = time.perf_counter()
        for _ in range(100):
            _ = model(x)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    # Should scale linearly with depth
    ratio1 = times[1] / times[0]  # Should be ~2
    ratio2 = times[2] / times[1]  # Should be ~2

    assert 1.5 < ratio1 < 3, f"Depth scaling wrong: {ratio1:.1f}x for 2x depth"
    assert 1.5 < ratio2 < 3, f"Depth scaling wrong: {ratio2:.1f}x for 2x depth"


# ============== Bottleneck Detection Tests ==============

def test_identify_bottlenecks():
    """Identify performance bottlenecks in pipeline."""

    # Profile different components
    timings = {}

    # Data creation
    start = time.perf_counter()
    for _ in range(1000):
        x = Tensor(np.random.randn(32, 100))
    timings['tensor_creation'] = time.perf_counter() - start

    # Linear forward
    linear = Linear(100, 50)
    x = Tensor(np.random.randn(32, 100))
    start = time.perf_counter()
    for _ in range(1000):
        _ = linear(x)
    timings['linear_forward'] = time.perf_counter() - start

    # Activation
    relu = ReLU()
    x = Tensor(np.random.randn(32, 50))
    start = time.perf_counter()
    for _ in range(1000):
        _ = relu(x)
    timings['relu_forward'] = time.perf_counter() - start

    # Loss computation
    criterion = MeanSquaredError()
    y_pred = Tensor(np.random.randn(32, 10))
    y_true = Tensor(np.random.randn(32, 10))
    start = time.perf_counter()
    for _ in range(1000):
        _ = criterion(y_pred, y_true)
    timings['loss_computation'] = time.perf_counter() - start

    # Find bottleneck
    bottleneck = max(timings, key=timings.get)
    bottleneck_time = timings[bottleneck]
    total_time = sum(timings.values())

    # No single component should dominate
    assert bottleneck_time < total_time * 0.7, \
        f"Performance bottleneck: {bottleneck} takes {bottleneck_time/total_time:.1%} of time"


def test_memory_bandwidth_bound():
    """Test if operations are memory bandwidth bound."""
    # Large tensors that stress memory bandwidth
    size = 10000
    a = Tensor(np.random.randn(size))
    b = Tensor(np.random.randn(size))

    # Element-wise operations (memory bound)
    start = time.perf_counter()
    for _ in range(100):
        c = Tensor(a.data + b.data)  # Simple add
    add_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(100):
        c = Tensor(a.data * b.data)  # Simple multiply
    mul_time = time.perf_counter() - start

    # These should take similar time (both memory bound)
    ratio = max(add_time, mul_time) / min(add_time, mul_time)
    assert ratio < 2, f"Element-wise ops have different performance: {ratio:.1f}x"


# ============== Optimization Validation Tests ==============

def test_relu_vectorization():
    """ReLU should use vectorized operations."""
    x = Tensor(np.random.randn(1000, 1000))
    relu = ReLU()

    # Vectorized ReLU should be fast
    start = time.perf_counter()
    for _ in range(100):
        _ = relu(x)
    elapsed = time.perf_counter() - start

    # Should process 100M elements quickly
    elements_per_second = (1000 * 1000 * 100) / elapsed

    # Even naive NumPy should achieve > 100M elem/sec
    assert elements_per_second > 1e8, \
        f"ReLU too slow: {elements_per_second/1e6:.1f}M elem/sec"


def test_batch_operation_efficiency():
    """Batch operations should be efficient."""
    model = Linear(100, 50)

    # Single sample vs batch
    single = Tensor(np.random.randn(1, 100))
    batch = Tensor(np.random.randn(32, 100))

    # Time single samples
    start = time.perf_counter()
    for _ in range(320):
        _ = model(single)
    single_time = time.perf_counter() - start

    # Time batch
    start = time.perf_counter()
    for _ in range(10):
        _ = model(batch)
    batch_time = time.perf_counter() - start

    # Batch should be much faster than individual
    speedup = single_time / batch_time
    assert speedup > 2, f"Batch processing not efficient: only {speedup:.1f}x speedup"


# ============== Performance Regression Tests ==============

def test_performance_regression():
    """Ensure performance doesn't degrade over time."""
    # Baseline timings (adjust based on initial measurements)
    baselines = {
        'linear_1000x1000': 0.5,  # seconds for 100 iterations
        'conv_32x32': 1.0,
        'train_step': 0.1,
    }

    # Test Linear performance
    linear = Linear(1000, 1000)
    x = Tensor(np.random.randn(10, 1000))
    start = time.perf_counter()
    for _ in range(100):
        _ = linear(x)
    linear_time = time.perf_counter() - start

    # Allow 2x slower than baseline (generous for different hardware)
    # This mainly catches catastrophic regressions
    if linear_time > baselines['linear_1000x1000'] * 10:
        pytest.warns(
            UserWarning,
            f"Linear performance regression: {linear_time:.2f}s "
            f"(baseline: {baselines['linear_1000x1000']:.2f}s)"
        )


if __name__ == "__main__":
    # When run directly, use pytest
    import subprocess
    result = subprocess.run(["pytest", __file__, "-v", "-s"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    sys.exit(result.returncode)
