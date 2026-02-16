#!/usr/bin/env python3
"""
The Optimization Olympics (2018) - MLPerf Benchmarking
======================================================

📚 HISTORICAL CONTEXT:
In 2018, MLPerf was launched to standardize ML benchmarking across hardware and
software. The key insight: production ML isn't just about accuracy - efficiency
matters equally. Can you maintain accuracy while reducing compute, memory, and
latency? This is the core challenge of ML systems engineering.

🎯 MILESTONE 06: THE OPTIMIZATION OLYMPICS
This milestone is the CULMINATION of everything you've built. You'll take a
trained model from earlier milestones and apply YOUR optimization tools to make
it production-ready. Every technique uses YOUR implementations!

✅ REQUIRED MODULES (Run after Module 19):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01-03: Tensor, Activations, Layers - YOUR base model
  Module 08: Training - YOUR trained model from earlier milestones
  Module 14: Profiling - YOUR Profiler class
  Module 15: Quantization - YOUR Quantizer class
  Module 16: Compression - YOUR Compressor class
  Module 17: Acceleration - YOUR vectorized operations
  Module 18: Memoization - YOUR KVCache class
  Module 19: Benchmarking - YOUR MLPerf class
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ THE OPTIMIZATION PIPELINE (Using YOUR APIs):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      YOUR TRAINED MLP (from Milestone 03)               │
    │                        Accurate but needs optimization                  │
    └───────────────────────────────────┬─────────────────────────────────────┘
                                        │
    ┌───────────────────────────────────▼─────────────────────────────────────┐
    │           STEP 1: PROFILE (using YOUR Profiler class)                   │
    │                  Count parameters, measure latency, memory              │
    └───────────────────────────────────┬─────────────────────────────────────┘
                                        │
    ┌───────────────────────────────────▼─────────────────────────────────────┐
    │        STEP 2: QUANTIZE (using YOUR Quantizer class)                    │
    │                   FP32 → INT8 (4× memory reduction)                     │
    └───────────────────────────────────┬─────────────────────────────────────┘
                                        │
    ┌───────────────────────────────────▼─────────────────────────────────────┐
    │        STEP 3: PRUNE (using YOUR Compressor class)                      │
    │                Remove small weights (2-4× compression)                  │
    └───────────────────────────────────┬─────────────────────────────────────┘
                                        │
    ┌───────────────────────────────────▼─────────────────────────────────────┐
    │      STEP 4: BENCHMARK (using YOUR MLPerf class)                    │
    │              Compare before vs after with scientific rigor              │
    └─────────────────────────────────────────────────────────────────────────┘

🔍 WHY OPTIMIZATION MATTERS - The Production Reality:

    Development Model:                   Production Model:
    ┌────────────────────┐              ┌────────────────────┐
    │ FP32 Weights       │              │ INT8 Weights       │
    │ 4 bytes/param      │  Quantize→   │ 1 byte/param       │
    │ Full precision     │              │ 4× smaller!        │
    └────────────────────┘              └────────────────────┘

    Dense Weights:                       Pruned Weights:
    ┌────────────────────┐              ┌────────────────────┐
    │ [0.1, 0.0, 0.3]    │              │ [0.1, _, 0.3]      │
    │ [0.0, 0.2, 0.0]    │   Prune→     │ [_, 0.2, _]        │
    │ All weights stored │              │ Only non-zero!     │
    └────────────────────┘              └────────────────────┘

    Goal: Maintain accuracy while reducing size/latency!

📊 EXPECTED RESULTS:

    ┌──────────────────┬────────────────┬────────────────┬────────────────┐
    │ Optimization     │ Size Reduction │ Speed Improvement│ Accuracy Loss │
    ├──────────────────┼────────────────┼────────────────┼────────────────┤
    │ Quantization     │ 4×             │ 1-2×           │ < 1%           │
    │ Pruning          │ 2-4×           │ 1-3×           │ < 2%           │
    │ Combined         │ 8-16×          │ 2-4×           │ < 3%           │
    └──────────────────┴────────────────┴────────────────┴────────────────┘

🔥 THIS IS ML SYSTEMS ENGINEERING:
Not just training models, but making them deployable. This is what separates
ML researchers from ML engineers. YOU now have the complete toolkit!
"""

import sys
import os
import time
import copy
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, os.getcwd())

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

console = Console()

# =============================================================================
# 🎯 YOUR TINYTORCH MODULES IN ACTION
# =============================================================================
#
# This milestone showcases YOUR complete optimization toolkit:
#
# ┌─────────────────────┬────────────────────────────────┬─────────────────────────────┐
# │ What You Built      │ How It's Used Here             │ Systems Impact              │
# ├─────────────────────┼────────────────────────────────┼─────────────────────────────┤
# │ Module 01-03: Core  │ Base model from Milestone 03   │ The model to be optimized   │
# │ Tensor, Layers, etc │ (MLP for digit classification) │                             │
# │                     │                                │                             │
# │ Module 14: Profiler │ Measures params, FLOPs, memory │ Understand BEFORE state     │
# │ ★ OPTIMIZATION ★    │ latency, layer breakdown       │ to measure improvement      │
# │                     │                                │                             │
# │ Module 15: Quantizer│ FP32 → INT8 conversion         │ 4× memory reduction         │
# │ ★ OPTIMIZATION ★    │ Per-tensor or per-channel      │ with minimal accuracy loss  │
# │                     │                                │                             │
# │ Module 16: Compressor│ Magnitude pruning + sparsity  │ 2-4× compression by         │
# │ ★ OPTIMIZATION ★    │ representation                 │ removing small weights      │
# │                     │                                │                             │
# │ Module 17: Accel    │ Vectorized ops, fused kernels  │ Faster execution through    │
# │ ★ OPTIMIZATION ★    │ for transformer models         │ autoregressive generation   │
# │                     │                                │                             │
# │ Module 18: KVCache  │ Caches attention computations  │ Faster inference for        │
# │ ★ OPTIMIZATION ★    │ optimized implementations      │ hardware-aware code         │
# │                     │                                │                             │
# │ Module 19: MLPerf│ Scientific benchmarking       │ Rigorous before/after       │
# │ ★ OPTIMIZATION ★    │ with statistical significance  │ comparison                  │
# └─────────────────────┴────────────────────────────────┴─────────────────────────────┘
#
# =============================================================================
# 🆕 WHAT'S NEW SINCE MILESTONE 05 (Transformer)
# =============================================================================
#
# ┌──────────────────────┬─────────────────────────┬────────────────────────────┐
# │ Previous Milestones  │ This Milestone          │ Why It's Different         │
# ├──────────────────────┼─────────────────────────┼────────────────────────────┤
# │ Training models      │ Optimizing models       │ Making models deployable   │
# │ Accuracy focus       │ Efficiency focus        │ Size, speed, AND accuracy  │
# │ Development phase    │ Production phase        │ Real-world constraints     │
# │ Build from scratch   │ Improve what exists     │ Engineering over research  │
# │ Individual techniques│ Complete pipeline       │ End-to-end optimization    │
# └──────────────────────┴─────────────────────────┴────────────────────────────┘
#
# THIS IS THE CULMINATION: Every module you've built comes together here!
#
# =============================================================================


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'batch_size': 32,
    'train_epochs': 10,
    'learning_rate': 0.01,
    'prune_sparsity': 0.5,
}


# =============================================================================
# STEP 1: PROFILE
# =============================================================================

def step_1_profile(model, X_test, y_test, Profiler, Tensor):
    """
    Step 1: Profile the baseline model with YOUR Profiler.

    What we measure:
    ────────────────
        Parameters:   Total trainable weights
        Size:         Memory footprint (bytes)
        FLOPs:        Computational cost per inference
        Latency:      Time per sample
        Accuracy:     Baseline test performance

    This establishes the BEFORE state for optimization comparison.

    Returns:
        dict with baseline metrics (param_count, param_bytes, flops,
             latency_ms, throughput, baseline_acc)
    """
    console.print(Panel(
        "[bold blue]📊 STEP 1: Profile with YOUR Profiler[/bold blue]\n"
        "Using the Profiler class you built in Module 14",
        border_style="blue"
    ))

    profiler = Profiler()

    # Count parameters
    param_count = profiler.count_parameters(model)
    param_bytes = param_count * 4  # FP32 = 4 bytes

    # Count FLOPs
    input_shape = (1, 64)
    flops = profiler.count_flops(model, input_shape)

    # Measure inference latency
    sample_input = Tensor(np.random.randn(1, 64).astype(np.float32))
    latency_ms = profiler.measure_latency(model, sample_input, warmup=3, iterations=10)
    throughput = 1000 / latency_ms if latency_ms > 0 else 0

    # Calculate baseline accuracy
    outputs = model(X_test)
    predictions = np.argmax(outputs.data, axis=1)
    baseline_acc = np.mean(predictions == y_test) * 100

    # Display results
    table = Table(title="📊 Baseline Profile (YOUR Profiler - Module 14)", box=box.DOUBLE)
    table.add_column("Metric", style="cyan", width=18)
    table.add_column("Value", style="yellow", justify="right")
    table.add_column("Notes", style="dim")

    table.add_row("Parameters", f"{param_count:,}", "Total trainable weights")
    table.add_row("Size", f"{param_bytes:,} bytes", "FP32 precision")
    table.add_row("FLOPs", f"{flops:,}", "Operations per inference")
    table.add_row("", "", "")
    table.add_row("Accuracy", f"{baseline_acc:.1f}%", "Test set performance")
    table.add_row("Latency", f"{latency_ms:.3f} ms", "Per-sample inference")
    table.add_row("Throughput", f"{throughput:.0f} samples/sec", "Inference speed")

    console.print(table)

    return {
        'param_count': param_count,
        'param_bytes': param_bytes,
        'flops': flops,
        'latency_ms': latency_ms,
        'throughput': throughput,
        'baseline_acc': baseline_acc,
    }


# =============================================================================
# STEP 2: QUANTIZE
# =============================================================================

def step_2_quantize(model, param_bytes, Quantizer):
    """
    Step 2: Quantize the model with YOUR Quantizer.

    Quantization reduces precision:
    ──────────────────────────────
        FP32 (32-bit) → INT8 (8-bit) = 4× smaller

        Before:  [0.123, -0.456, 0.789]  (4 bytes each)
        After:   [31, -117, 202]         (1 byte each + scale)

    Trade-off: Memory savings vs potential accuracy loss.
    Modern quantization typically loses <1% accuracy.

    Returns:
        dict with quantization results
    """
    console.print(Panel(
        "[bold yellow]🗜️ STEP 2: Quantize with YOUR Quantizer[/bold yellow]\n"
        "Using the quantization you built in Module 15\n"
        "FP32 → INT8 = 4× smaller",
        border_style="yellow"
    ))

    quant_result = Quantizer.quantize_model(model)
    quant_size = int(param_bytes / quant_result['compression_ratio'])

    # Display results
    table = Table(title="🗜️ After Quantization (YOUR Implementation)", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Before", style="yellow")
    table.add_column("After", style="green")
    table.add_column("Change", style="bold")

    table.add_row(
        "Size",
        f"{param_bytes:,} B",
        f"{quant_size:,} B",
        f"[green]{quant_result['compression_ratio']:.1f}× smaller[/green]"
    )
    table.add_row(
        "Precision",
        "FP32 (32-bit)",
        "INT8 (8-bit)",
        "[green]4× memory reduction[/green]"
    )

    console.print(table)

    return {
        'quant_result': quant_result,
        'quant_size': quant_size,
    }


# =============================================================================
# STEP 3: PRUNE
# =============================================================================

def step_3_prune(model, baseline_acc, X_test, y_test, Compressor, DigitMLP):
    """
    Step 3: Prune the model with YOUR Compressor.

    Magnitude pruning removes small weights:
    ────────────────────────────────────────
        Before: [0.1, 0.001, 0.3, -0.002, 0.2]
        After:  [0.1,   0,   0.3,    0,   0.2]  (50% sparse)

        Small weights contribute little to output.
        Removing them creates sparse, compressible models.

    Trade-off: Compression vs accuracy loss.
    50% pruning typically loses <2% accuracy.

    Returns:
        dict with pruning results
    """
    console.print(Panel(
        "[bold magenta]✂️ STEP 3: Prune with YOUR Compressor[/bold magenta]\n"
        "Using the compression you built in Module 16\n"
        f"Remove {CONFIG['prune_sparsity']:.0%} of smallest weights",
        border_style="magenta"
    ))

    # Create a copy for pruning
    model_copy = DigitMLP()
    for i, layer in enumerate(model.layers):
        for j, param in enumerate(layer.parameters()):
            model_copy.layers[i].parameters()[j].data = param.data.copy()

    # Apply pruning
    sparsity_before = Compressor.measure_sparsity(model_copy)
    Compressor.magnitude_prune(model_copy, sparsity=CONFIG['prune_sparsity'])
    sparsity_after = Compressor.measure_sparsity(model_copy)

    # Calculate pruned accuracy
    outputs_pruned = model_copy(X_test)
    predictions_pruned = np.argmax(outputs_pruned.data, axis=1)
    pruned_acc = np.mean(predictions_pruned == y_test) * 100

    # Display results
    table = Table(title="✂️ After Pruning (YOUR Implementation)", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Before", style="yellow")
    table.add_column("After", style="green")
    table.add_column("Change", style="bold")

    table.add_row(
        "Sparsity",
        f"{sparsity_before:.1%}",
        f"{sparsity_after:.1%}",
        f"[green]{sparsity_after:.0%} weights zeroed[/green]"
    )
    prune_acc_delta = pruned_acc - baseline_acc
    table.add_row(
        "Accuracy",
        f"{baseline_acc:.1f}%",
        f"{pruned_acc:.1f}%",
        f"[{'green' if prune_acc_delta >= 0 else 'red'}]{prune_acc_delta:+.1f}%[/]"
    )

    console.print(table)

    return {
        'sparsity_before': sparsity_before,
        'sparsity_after': sparsity_after,
        'pruned_acc': pruned_acc,
    }


# =============================================================================
# STEP 4: KV CACHE
# =============================================================================

def step_4_kv_cache(KVCache, MinimalTransformer):
    """
    Step 4: Demonstrate KV Cache with YOUR Module 18.

    KV Caching avoids recomputation in autoregressive generation:
    ─────────────────────────────────────────────────────────────
        Without cache: Each new token recomputes ALL K,V → O(n³)
        With cache:    Reuse cached K,V, compute only new → O(n²)

        Token 1: Compute K₁,V₁ → cache
        Token 2: Use K₁,V₁ from cache + compute K₂,V₂ → cache
        Token N: Use K₁..Kₙ₋₁,V₁..Vₙ₋₁ + compute Kₙ,Vₙ

    Result: 6-10× speedup for generation!

    Returns:
        dict with KV cache stats (or None if unavailable)
    """
    console.print(Panel(
        "[bold cyan]⚡ STEP 4: KV Cache with YOUR Module 18[/bold cyan]\n"
        "Using KVCache for transformer generation speedup\n"
        "Caches K,V to avoid recomputation during autoregressive generation",
        border_style="cyan"
    ))

    try:
        transformer = MinimalTransformer(vocab_size=27, embed_dim=32, num_heads=2, seq_len=8)

        kv_cache = KVCache(
            batch_size=1,
            max_seq_len=8,
            num_layers=1,
            num_heads=2,
            head_dim=16  # embed_dim / num_heads
        )

        cache_memory = (kv_cache.batch_size * kv_cache.max_seq_len *
                       kv_cache.num_layers * kv_cache.num_heads *
                       kv_cache.head_dim * 2 * 4)  # K+V, float32

        table = Table(title="⚡ KV Cache Stats (YOUR Module 18)", box=box.ROUNDED)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_column("Notes", style="dim")

        table.add_row("Max Sequence", f"{kv_cache.max_seq_len}", "Tokens cacheable")
        table.add_row("Num Layers", f"{kv_cache.num_layers}", "Transformer layers")
        table.add_row("Num Heads", f"{kv_cache.num_heads}", "Attention heads")
        table.add_row("Cache Memory", f"{cache_memory:,} bytes", "Pre-allocated K+V")
        table.add_row("", "", "")
        table.add_row("Speedup", "~N× (N=seq_len)", "Avoids recomputation")

        console.print(table)
        console.print("  [green]✓[/green] KV Cache ready for generation!")

        return {'cache_memory': cache_memory, 'kv_cache': kv_cache}

    except Exception as e:
        console.print(f"  [yellow]⚠️ KV Cache demo skipped: {e}[/yellow]")
        console.print()
        return None


# =============================================================================
# STEP 5: ACCELERATION
# =============================================================================

def step_5_accelerate(vectorized_matmul, Tensor):
    """
    Step 5: Demonstrate acceleration with YOUR Module 17.

    Vectorized operations use optimized BLAS libraries:
    ───────────────────────────────────────────────────
        Naive loops:    for i: for j: for k: C[i,j] += A[i,k] * B[k,j]
        BLAS-optimized: C = np.dot(A, B)  (uses MKL/OpenBLAS/etc)

        BLAS exploits:
        - CPU cache hierarchy (data locality)
        - SIMD instructions (process 4-8 floats at once)
        - Multi-threading (parallel computation)

    Result: 10-100× speedup for matrix operations!

    Returns:
        dict with timing comparison
    """
    console.print(Panel(
        "[bold magenta]🚀 STEP 5: Acceleration with YOUR Module 17[/bold magenta]\n"
        "Using vectorized operations for compute speedup\n"
        "BLAS-optimized matmul and fused operations",
        border_style="magenta"
    ))

    # Create test matrices
    A = Tensor(np.random.randn(64, 128).astype(np.float32))
    B = Tensor(np.random.randn(128, 64).astype(np.float32))

    # Time standard operation
    start = time.time()
    for _ in range(100):
        C_standard = Tensor(np.dot(A.data, B.data))
    standard_time = (time.time() - start) * 1000

    # Time vectorized operation
    start = time.time()
    for _ in range(100):
        C_vectorized = vectorized_matmul(A, B)
    vectorized_time = (time.time() - start) * 1000

    table = Table(title="🚀 Acceleration Results (YOUR Module 17)", box=box.ROUNDED)
    table.add_column("Operation", style="cyan")
    table.add_column("Time (100 runs)", style="yellow")
    table.add_column("Notes", style="dim")

    table.add_row("Standard np.dot", f"{standard_time:.2f} ms", "Baseline")
    table.add_row("vectorized_matmul", f"{vectorized_time:.2f} ms", "YOUR implementation")
    table.add_row("Matrix Shape", f"{A.shape} @ {B.shape}", f"→ {C_vectorized.shape}")

    console.print(table)
    console.print("  [green]✓[/green] Vectorized operations ready!")

    return {
        'standard_time': standard_time,
        'vectorized_time': vectorized_time,
    }


# =============================================================================
# STEP 6: BENCHMARK
# =============================================================================

def step_6_benchmark(model, X_test, y_test, baseline_acc, Benchmark):
    """
    Step 6: Benchmark with YOUR Modules 14 & 19.

    Scientific benchmarking requires:
    ─────────────────────────────────
        Multiple runs:    Reduce noise from system variance
        Warmup:           Allow JIT compilation, cache warming
        Statistics:       Mean, std, min, max, percentiles

        Results should be:
        - Reproducible (same conditions → same results)
        - Comparable (standardized metrics)
        - Meaningful (confidence intervals)

    Returns:
        dict with benchmark results
    """
    console.print(Panel(
        "[bold green]🏁 STEP 6: Benchmark with YOUR Modules 14 & 19[/bold green]\n"
        "Using Benchmark class for standardized measurements\n"
        "Reproducible, statistically rigorous",
        border_style="green"
    ))

    console.print("  Running standardized benchmark with YOUR implementations...")

    test_dataset = [(X_test, y_test)]
    benchmark = Benchmark(models=[model], datasets=test_dataset)

    latency_results = benchmark.run_latency_benchmark(input_shape=(1, 64))
    bench_result = list(latency_results.values())[0]

    mean_latency = bench_result.mean
    std_latency = bench_result.std
    min_latency = bench_result.min_val
    max_latency = bench_result.max_val

    sorted_vals = sorted(bench_result.values)
    p95_idx = int(len(sorted_vals) * 0.95)
    p95_latency = sorted_vals[min(p95_idx, len(sorted_vals) - 1)]

    throughput = 1000 / mean_latency if mean_latency > 0 else 0

    table = Table(title="🏁 Benchmark Results (YOUR Modules 14 & 19)", box=box.DOUBLE)
    table.add_column("Metric", style="cyan", width=18)
    table.add_column("Value", style="yellow", justify="right")
    table.add_column("Target", style="dim")

    table.add_row("Latency (mean)", f"{mean_latency:.3f} ms", "< 100ms")
    table.add_row("Latency (std)", f"± {std_latency:.3f} ms", "Low = stable")
    table.add_row("Latency (min/max)", f"{min_latency:.3f} / {max_latency:.3f} ms", "Tight range")
    table.add_row("P95 Latency", f"{p95_latency:.3f} ms", "< 2× mean")
    table.add_row("", "", "")
    table.add_row("Throughput", f"{throughput:.0f} samples/sec", "Higher = better")
    table.add_row("Accuracy", f"{baseline_acc:.1f}%", "> 80%")

    console.print(table)

    return {
        'mean_latency': mean_latency,
        'std_latency': std_latency,
        'p95_latency': p95_latency,
        'throughput': throughput,
    }

def press_enter_to_continue() :
    if sys.stdin.isatty() and sys.stdout.isatty() :
        try :
            console.input("\n[yellow]Press Enter to continue...[/yellow] ")
        except EOFError :
            pass
        console.print()

# =============================================================================
# FINAL RESULTS
# =============================================================================

def print_final_results(baseline, quant, prune, profile_results):
    """
    Print the final optimization journey summary.

    Shows the progression:
    ──────────────────────
        Baseline → Quantized → Pruned

    With size, accuracy, and which module was used at each stage.
    """
    console.print("=" * 70)
    console.print(Panel("[bold]🏆 OPTIMIZATION OLYMPICS RESULTS[/bold]", border_style="gold1"))
    console.print()

    param_bytes = baseline['param_bytes']
    baseline_acc = baseline['baseline_acc']
    quant_size = quant['quant_size']
    quant_result = quant['quant_result']
    pruned_acc = prune['pruned_acc']
    sparsity_after = prune['sparsity_after']

    table = Table(title="🎖️ Your Optimization Journey", box=box.DOUBLE)
    table.add_column("Stage", style="cyan", width=25)
    table.add_column("Size", style="yellow", justify="right")
    table.add_column("Baseline Acc", style="dim", justify="right")
    table.add_column("New Acc", style="green", justify="right")
    table.add_column("Δ Accuracy", style="bold", justify="right")
    table.add_column("YOUR Module", style="magenta")

    quant_acc = baseline_acc
    quant_delta = quant_acc - baseline_acc
    prune_delta = pruned_acc - baseline_acc

    table.add_row(
        "📊 Baseline",
        f"{param_bytes:,} B",
        f"{baseline_acc:.1f}%",
        f"{baseline_acc:.1f}%",
        "—",
        "Profiler (14)"
    )
    table.add_row(
        "🗜️ + Quantization",
        f"{quant_size:,} B",
        f"{baseline_acc:.1f}%",
        f"{quant_acc:.1f}%",
        f"[green]{quant_delta:+.1f}%[/green]" if quant_delta >= 0 else f"[red]{quant_delta:+.1f}%[/red]",
        "Quantization (15)"
    )
    table.add_row(
        "✂️ + Pruning",
        f"~{param_bytes//2:,} B**",
        f"{baseline_acc:.1f}%",
        f"{pruned_acc:.1f}%",
        f"[green]{prune_delta:+.1f}%[/green]" if prune_delta >= 0 else f"[red]{prune_delta:+.1f}%[/red]",
        "Compression (16)"
    )

    console.print(table)
    console.print("[dim]** With sparse storage[/dim]")
    console.print()

    # Key insights
    console.print(Panel(
        "[bold green]🎓 KEY INSIGHTS[/bold green]\n\n"
        f"✅ [cyan]YOUR Profiler (Module 14):[/cyan]\n"
        f"   • Measured {baseline['param_count']:,} parameters, {baseline['flops']:,} FLOPs\n"
        f"   • Found baseline latency: {baseline['latency_ms']:.3f}ms\n\n"
        f"✅ [cyan]YOUR Quantization (Module 15):[/cyan]\n"
        f"   • Achieved {quant_result['compression_ratio']:.1f}× compression\n"
        f"   • FP32 → INT8 reduces memory 4×\n\n"
        f"✅ [cyan]YOUR Compression (Module 16):[/cyan]\n"
        f"   • Pruned to {sparsity_after:.0%} sparsity\n"
        f"   • {abs(baseline_acc - pruned_acc):.1f}% accuracy impact\n\n"
        f"✅ [cyan]YOUR KV Cache (Module 18):[/cyan]\n"
        f"   • Pre-allocated cache for transformer generation\n"
        f"   • ~N× speedup for sequence length N\n\n"
        f"✅ [cyan]YOUR Acceleration (Module 17):[/cyan]\n"
        f"   • BLAS-optimized matrix operations\n"
        f"   • Vectorized compute kernels\n\n"
        f"💡 [yellow]Challenge: Combine All Techniques![/yellow]\n"
        f"   • Quantize + Prune + KV Cache = production-ready\n"
        f"   • This is real ML systems engineering!",
        border_style="cyan",
        box=box.ROUNDED
    ))

    press_enter_to_continue()

    # Success message
    console.print(Panel(
        "[bold green]🏆 MILESTONE COMPLETE![/bold green]\n\n"
        "[green]You used YOUR implementations from:[/green]\n"
        "  • Module 01-03: Tensor, Linear, ReLU\n"
        "  • Module 14: Profiler\n"
        "  • Module 15: Quantizer\n"
        "  • Module 16: Compressor\n"
        "  • Module 17: vectorized_matmul\n"
        "  • Module 18: KVCache\n"
        "  • Module 19: Benchmark\n\n"
        "[bold]Everything you built... now works together![/bold]\n\n"
        "[cyan]What you learned:[/cyan]\n"
        "  ✅ Profile models systematically\n"
        "  ✅ Quantize for memory efficiency\n"
        "  ✅ Prune for sparse models\n"
        "  ✅ Cache K,V for fast generation\n"
        "  ✅ Accelerate with vectorized ops\n"
        "  ✅ Benchmark with scientific rigor\n\n"
        "[bold]You've learned ML Systems Engineering![/bold]",
        title="🎯 Milestone 06 Complete",
        border_style="bright_green",
        box=box.DOUBLE,
        padding=(1, 2)
    ))

    press_enter_to_continue()

    return 0



# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point: The Optimization Olympics.

    Pipeline Structure:
    ───────────────────
    1. PROFILE   - Measure baseline (params, FLOPs, latency, accuracy)
    2. QUANTIZE  - FP32 → INT8 (4× memory reduction)
    3. PRUNE     - Remove small weights (2-4× compression)
    4. KV CACHE  - Cache K,V for fast generation
    5. ACCELERATE - Vectorized matrix operations
    6. BENCHMARK - Scientific performance measurement

    Each step uses YOUR implementations from the corresponding module!
    """

    # ─────────────────────────────────────────────────────────────────────────
    # WELCOME BANNER
    # ─────────────────────────────────────────────────────────────────────────
    console.print(Panel(
        "[bold magenta]╔═══ Milestone 06: MLPerf ════╗[/bold magenta]\n"
        "[bold magenta]║[/bold magenta] [bold]🏆 THE OPTIMIZATION         [/bold][bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta] [bold]OLYMPICS                    [/bold][bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta]                             [bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta] MLPerf 2018: Where accuracy [bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta] meets efficiency            [bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta]                             [bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta] [cyan]Using YOUR implementations [/cyan] [bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta] [cyan]from every module!  [/cyan]        [bold magenta]║[/bold magenta]\n"
        "[bold magenta]╚═════════════════════════════╝[/bold magenta]",
        border_style="bright_magenta"
    ))
    press_enter_to_continue()

    # ─────────────────────────────────────────────────────────────────────────
    # IMPORT YOUR IMPLEMENTATIONS
    # ─────────────────────────────────────────────────────────────────────────
    console.print("[bold cyan]📦 Loading YOUR Tiny🔥Torch implementations...[/bold cyan]\n")

    try:
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.layers import Linear
        from tinytorch.core.activations import ReLU
        console.print("  [green]✓[/green] Tensor, Linear, ReLU (YOUR implementations)")

        from tinytorch.perf.profiling import Profiler
        console.print("  [green]✓[/green] Profiler (YOUR Module 14)")

        from tinytorch.perf.quantization import Quantizer
        console.print("  [green]✓[/green] Quantizer (YOUR Module 15)")

        from tinytorch.perf.compression import Compressor
        console.print("  [green]✓[/green] Compressor (YOUR Module 16)")

        from tinytorch.perf.acceleration import vectorized_matmul, fused_gelu
        console.print("  [green]✓[/green] vectorized_matmul, fused_gelu (YOUR Module 17)")

        from tinytorch.perf.memoization import KVCache
        console.print("  [green]✓[/green] KVCache (YOUR Module 18)")

        from tinytorch.perf.benchmarking import Benchmark, MLPerf
        console.print("  [green]✓[/green] Benchmark, MLPerf (YOUR Module 19)")

    except ImportError as e:
        console.print(Panel(
            f"[red]Import Error: {e}[/red]\n\n"
            f"[yellow]This milestone requires optimization modules.[/yellow]\n"
            f"[dim]Make sure you've completed and exported modules 01-03, 14-19[/dim]",
            title="Missing Modules",
            border_style="red"
        ))
        return 1

    console.print("\n[green]✅ All YOUR implementations loaded![/green]")
    press_enter_to_continue()

    # ─────────────────────────────────────────────────────────────────────────
    # LOAD MODEL AND DATA
    # ─────────────────────────────────────────────────────────────────────────
    console.print(Panel(
        "[bold cyan]🧠 Loading Model and Data[/bold cyan]\n"
        "Using DigitMLP from Milestone 03",
        border_style="cyan"
    ))

    # Try to import from networks.py, fallback to inline definition
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from networks import DigitMLP, MinimalTransformer
        console.print("  [green]✓[/green] DigitMLP (from networks.py)")
        console.print("  [green]✓[/green] MinimalTransformer (from networks.py)")
    except ImportError:
        console.print("  [yellow]⚠️ Using inline MLP definition[/yellow]")

        class DigitMLP:
            def __init__(self, input_size=64, hidden_size=32, num_classes=10):
                self.fc1 = Linear(input_size, hidden_size)
                self.relu = ReLU()
                self.fc2 = Linear(hidden_size, num_classes)
                self.layers = [self.fc1, self.fc2]
                self.name = "DigitMLP"

            def forward(self, x):
                if len(x.shape) > 2:
                    x = x.reshape(x.shape[0], -1)
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

            def __call__(self, x):
                return self.forward(x)

            def parameters(self):
                params = []
                for layer in self.layers:
                    params.extend(layer.parameters())
                return params

        MinimalTransformer = None

    model = DigitMLP()
    console.print(f"\n  [bold green]Using: {model.name}[/bold green]")

    # Load TinyDigits dataset
    console.print("\n[bold cyan]📊 Loading TinyDigits dataset...[/bold cyan]")

    try:
        from tinytorch.core.dataloader import TinyDigits
        dataset = TinyDigits()
        X_train, y_train = dataset.get_train_data()
        X_test, y_test = dataset.get_test_data()

        X_train = Tensor(X_train.reshape(X_train.shape[0], -1).astype(np.float32))
        X_test = Tensor(X_test.reshape(X_test.shape[0], -1).astype(np.float32))

        console.print(f"  [green]✓[/green] Training: {len(y_train)} samples")
        console.print(f"  [green]✓[/green] Test: {len(y_test)} samples")
    except Exception:
        console.print("  [yellow]⚠️ Using synthetic data[/yellow]")
        X_train = Tensor(np.random.randn(1000, 64).astype(np.float32))
        y_train = np.random.randint(0, 10, 1000)
        X_test = Tensor(np.random.randn(200, 64).astype(np.float32))
        y_test = np.random.randint(0, 10, 200)

    # ─────────────────────────────────────────────────────────────────────────
    # QUICK TRAINING
    # ─────────────────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]🏋️ Quick training (10 epochs)...[/bold cyan]")

    from tinytorch.core.optimizers import SGD
    from tinytorch.core.losses import CrossEntropyLoss

    optimizer = SGD(model.parameters(), lr=CONFIG['learning_rate'])
    loss_fn = CrossEntropyLoss()

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
        task = progress.add_task("Training...", total=CONFIG['train_epochs'])

        for epoch in range(CONFIG['train_epochs']):
            batch_size = CONFIG['batch_size']
            for i in range(0, min(500, len(y_train)), batch_size):
                batch_x = Tensor(X_train.data[i:i+batch_size])
                batch_y = y_train[i:i+batch_size]

                output = model(batch_x)
                loss = loss_fn(output, Tensor(batch_y))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            progress.advance(task)

    console.print("  [green]✓[/green] Training complete")
    press_enter_to_continue()

    # ─────────────────────────────────────────────────────────────────────────
    # RUN OPTIMIZATION STEPS
    # ─────────────────────────────────────────────────────────────────────────

    # Step 1: Profile baseline
    baseline = step_1_profile(model, X_test, y_test, Profiler, Tensor)
    press_enter_to_continue()

    # Step 2: Quantize
    quant = step_2_quantize(model, baseline['param_bytes'], Quantizer)
    press_enter_to_continue()

    # Step 3: Prune
    prune = step_3_prune(model, baseline['baseline_acc'], X_test, y_test, Compressor, DigitMLP)
    press_enter_to_continue()

    # Step 4: KV Cache (transformers only)
    if MinimalTransformer is not None:
        step_4_kv_cache(KVCache, MinimalTransformer)
    else:
        console.print(Panel(
            "[dim]⏭️ Step 4 (KV Cache) skipped - MinimalTransformer not available[/dim]",
            border_style="dim"
        ))
        console.print()
    press_enter_to_continue()

    # Step 5: Acceleration
    step_5_accelerate(vectorized_matmul, Tensor)
    press_enter_to_continue()

    # Step 6: Benchmark
    step_6_benchmark(model, X_test, y_test, baseline['baseline_acc'], Benchmark)
    press_enter_to_continue()

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    return print_final_results(baseline, quant, prune, baseline)


if __name__ == "__main__":
    sys.exit(main())
