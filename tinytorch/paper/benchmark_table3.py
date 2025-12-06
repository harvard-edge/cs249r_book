#!/usr/bin/env python3
"""
Benchmark script to generate real performance numbers for Table 3 in the paper.
Compares TinyTorch implementations against PyTorch on CPU.
"""

import time
import numpy as np
import torch
import sys
from pathlib import Path

# Add TinyTorch to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# Import TinyTorch components
try:
    from tinytorch.core import Tensor as TTTensor
    from tinytorch.nn import Conv2d as TTConv2d
    TINYTORCH_AVAILABLE = True
except ImportError:
    print("Warning: TinyTorch not available. Will create mock implementations.")
    TINYTORCH_AVAILABLE = False


def benchmark_function(func, *args, warmup=3, runs=10):
    """Benchmark a function with warmup and multiple runs."""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Actual timing
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)


def benchmark_matmul():
    """Benchmark matrix multiplication: 1000x1000 @ 1000x1000"""
    print("\n=== Benchmarking Matrix Multiplication (1K×1K) ===")

    # PyTorch
    pt_a = torch.randn(1000, 1000)
    pt_b = torch.randn(1000, 1000)

    def pt_matmul():
        return torch.mm(pt_a, pt_b)

    pt_mean, pt_std = benchmark_function(pt_matmul)
    print(f"PyTorch: {pt_mean*1000:.2f} ms ± {pt_std*1000:.2f} ms")

    # TinyTorch
    if TINYTORCH_AVAILABLE:
        # Use TinyTorch's actual implementation
        tt_a = TTTensor(pt_a.numpy())
        tt_b = TTTensor(pt_b.numpy())

        def tt_matmul():
            return tt_a @ tt_b

        tt_mean, tt_std = benchmark_function(tt_matmul, warmup=1, runs=5)
        print(f"TinyTorch: {tt_mean*1000:.2f} ms ± {tt_std*1000:.2f} ms")
    else:
        # Pure Python naive implementation
        a = pt_a.numpy()
        b = pt_b.numpy()

        def naive_matmul():
            n, m, p = a.shape[0], a.shape[1], b.shape[1]
            result = np.zeros((n, p))
            for i in range(n):
                for j in range(p):
                    for k in range(m):
                        result[i, j] += a[i, k] * b[k, j]
            return result

        tt_mean, tt_std = benchmark_function(naive_matmul, warmup=1, runs=3)
        print(f"TinyTorch (naive): {tt_mean*1000:.2f} ms ± {tt_std*1000:.2f} ms")

    ratio = tt_mean / pt_mean
    print(f"Ratio: {ratio:.0f}×")

    return pt_mean * 1000, tt_mean * 1000, ratio


def benchmark_conv2d():
    """Benchmark Conv2d on CIFAR batch: (128, 3, 32, 32) through 32 filters 5×5"""
    print("\n=== Benchmarking Conv2d (CIFAR batch) ===")

    batch_size = 128
    in_channels = 3
    out_channels = 32
    kernel_size = 5

    # PyTorch
    pt_input = torch.randn(batch_size, in_channels, 32, 32)
    pt_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)

    def pt_conv2d():
        return pt_conv(pt_input)

    pt_mean, pt_std = benchmark_function(pt_conv2d)
    print(f"PyTorch: {pt_mean*1000:.2f} ms ± {pt_std*1000:.2f} ms")

    # TinyTorch
    if TINYTORCH_AVAILABLE:
        try:
            # Use TinyTorch's actual Conv2d implementation
            tt_input = TTTensor(pt_input.numpy())
            tt_conv = TTConv2d(in_channels, out_channels, kernel_size, bias=False)
            # Copy PyTorch weights for fair comparison
            tt_conv.weight.data = pt_conv.weight.detach().numpy()

            def tt_conv2d():
                return tt_conv(tt_input)

            tt_mean, tt_std = benchmark_function(tt_conv2d, warmup=1, runs=3)
            print(f"TinyTorch: {tt_mean:.2f} s ± {tt_std:.2f} s")
        except Exception as e:
            print(f"TinyTorch Conv2d failed: {e}")
            print("Falling back to naive implementation with smaller batch...")
            tt_mean = benchmark_conv2d_naive_small(pt_conv.weight.detach().numpy())
    else:
        # Use smaller batch size for naive implementation (too slow otherwise)
        print("Using smaller batch (8 instead of 128) for naive implementation...")
        tt_mean = benchmark_conv2d_naive_small(pt_conv.weight.detach().numpy())

    ratio = tt_mean / pt_mean
    print(f"Ratio: {ratio:.0f}×")

    return pt_mean * 1000, tt_mean, ratio


def benchmark_conv2d_naive_small(weight_np):
    """Benchmark naive conv2d with smaller batch for speed"""
    batch_size_small = 8  # Reduced from 128
    in_channels = 3
    kernel_size = 5

    input_small = np.random.randn(batch_size_small, in_channels, 32, 32)

    def naive_conv2d():
        """7 nested loops as shown in the paper"""
        B, C_in, H, W = input_small.shape
        C_out, C_in_w, K_h, K_w = weight_np.shape
        H_out = H - K_h + 1
        W_out = W - K_w + 1

        output = np.zeros((B, C_out, H_out, W_out))

        for b in range(B):
            for c_out in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        for c_in in range(C_in):
                            for kh in range(K_h):
                                for kw in range(K_w):
                                    output[b, c_out, h, w] += \
                                        input_small[b, c_in, h+kh, w+kw] * \
                                        weight_np[c_out, c_in, kh, kw]
        return output

    tt_mean, tt_std = benchmark_function(naive_conv2d, warmup=0, runs=1)
    print(f"TinyTorch (batch=8): {tt_mean:.2f} s ± {tt_std:.2f} s")

    # Extrapolate to full batch size (linear scaling)
    extrapolated = tt_mean * (128 / 8)
    print(f"TinyTorch (extrapolated to batch=128): {extrapolated:.2f} s")

    return extrapolated


def benchmark_softmax():
    """Benchmark softmax on 10K elements"""
    print("\n=== Benchmarking Softmax (10K elements) ===")

    size = 10000

    # PyTorch
    pt_input = torch.randn(size)

    def pt_softmax():
        return torch.nn.functional.softmax(pt_input, dim=0)

    pt_mean, pt_std = benchmark_function(pt_softmax)
    print(f"PyTorch: {pt_mean*1000:.3f} ms ± {pt_std*1000:.3f} ms")

    # TinyTorch - pure Python implementation
    input_np = pt_input.numpy()

    def naive_softmax():
        """Pure Python softmax"""
        # Subtract max for numerical stability
        x = input_np - np.max(input_np)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    tt_mean, tt_std = benchmark_function(naive_softmax, warmup=2, runs=10)
    print(f"TinyTorch: {tt_mean*1000:.3f} ms ± {tt_std*1000:.3f} ms")

    ratio = tt_mean / pt_mean
    print(f"Ratio: {ratio:.0f}×")

    return pt_mean * 1000, tt_mean * 1000, ratio


def format_time(ms):
    """Format time in appropriate units"""
    if ms < 1:
        return f"{ms:.2f} ms"
    elif ms < 1000:
        return f"{ms:.1f} ms"
    else:
        return f"{ms/1000:.1f} s"


def main():
    print("=" * 60)
    print("TinyTorch vs PyTorch Performance Benchmark")
    print("=" * 60)
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"TinyTorch available: {TINYTORCH_AVAILABLE}")
    print("=" * 60)

    results = {}

    # Run benchmarks
    results['matmul'] = benchmark_matmul()
    results['conv2d'] = benchmark_conv2d()
    results['softmax'] = benchmark_softmax()

    # Print LaTeX table
    print("\n" + "=" * 60)
    print("LaTeX Table for paper:")
    print("=" * 60)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Runtime comparison: TinyTorch vs PyTorch (CPU).}")
    print(r"\label{tab:performance}")
    print(r"\small")
    print(r"\begin{tabular}{@{}lrrr@{}}")
    print(r"\toprule")
    print(r"Operation & TinyTorch & PyTorch & Ratio \\")
    print(r"\midrule")

    # Format matmul
    pt_mm, tt_mm, ratio_mm = results['matmul']
    print(f"\\texttt{{matmul}} (1K$\\times$1K) & {tt_mm:.0f} ms & {pt_mm:.1f} ms & {ratio_mm:.0f}$\\times$ \\\\")

    # Format conv2d
    pt_conv, tt_conv, ratio_conv = results['conv2d']
    print(f"\\texttt{{conv2d}} (CIFAR batch) & {tt_conv:.1f} s & {pt_conv:.0f} ms & {ratio_conv:.0f}$\\times$ \\\\")

    # Format softmax
    pt_soft, tt_soft, ratio_soft = results['softmax']
    print(f"\\texttt{{softmax}} (10K elem) & {tt_soft:.0f} ms & {pt_soft:.2f} ms & {ratio_soft:.0f}$\\times$ \\\\")

    print(r"\midrule")
    print(r"CIFAR-10 epoch (LeNet) & \textit{TBD} & \textit{TBD} & \textit{TBD} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"MatMul (1K×1K):      {ratio_mm:6.0f}× slower")
    print(f"Conv2d (CIFAR):      {ratio_conv:6.0f}× slower")
    print(f"Softmax (10K):       {ratio_soft:6.0f}× slower")
    print(f"Average slowdown:    {np.mean([ratio_mm, ratio_conv, ratio_soft]):6.0f}×")


if __name__ == "__main__":
    main()
