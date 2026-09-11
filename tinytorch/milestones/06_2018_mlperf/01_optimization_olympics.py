#!/usr/bin/env python3
"""Milestone 06.1: compare optimization candidates against one trained baseline.

Train a small DigitMLP, then independently round and prune copies of its weights.
Measure each candidate's accuracy and latency on the same workload. TinyTorch
executes both candidates as dense float32 arrays: INT8 code storage is a modeled
artifact size, and zero weights alone do not reduce allocated model memory.

The cache lifecycle and vectorized-matmul examples are separate mechanisms, not
additional optimizations applied to this MLP. Part 06.2 measures equivalent GPT
inference with and without a cache. These are classroom experiments inspired by
MLPerf's measurement discipline, not official MLPerf submissions or guarantees
of speed, compression, or preserved accuracy.
"""

import sys
import os
import time
import copy
import pickle
import numpy as np
rng = np.random.default_rng(7)
from pathlib import Path

# Add project root
sys.path.insert(0, os.getcwd())

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

console = Console()


def load_tinydigits_arrays(project_root=None):
    """Load TinyDigits arrays shipped with TinyTorch."""
    root = Path(project_root) if project_root is not None else Path(__file__).parent.parent.parent
    data_dir = root / "datasets" / "tinydigits"
    train_path = data_dir / "train.pkl"
    test_path = data_dir / "test.pkl"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"TinyDigits dataset not found in {data_dir}. "
            "Run: python3 datasets/tinydigits/create_tinydigits.py"
        )

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)

    return (
        train_data["images"],
        train_data["labels"],
        test_data["images"],
        test_data["labels"],
    )

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'batch_size': 32,
    # 40 epochs over the full training set reaches roughly 87% on TinyDigits in
    # about a second. The previous 10 epochs over the first 500 samples reached
    # 36%, which made every accuracy delta in this script indistinguishable from
    # noise: the whole point of the Olympics is what optimization costs a model
    # that actually works.
    'train_epochs': 40,
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
    param_bytes = sum(param.data.nbytes for param in model.parameters())

    # Count FLOPs
    input_shape = (1, 64)
    flops = profiler.count_flops(model, input_shape)

    # Measure inference latency
    sample_input = Tensor(X_test.data[:1])
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
    table.add_row("Serial rate", f"{throughput:.0f} samples/sec", "Reciprocal of single-sample latency")

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

def step_2_quantize(model, param_bytes, baseline_acc, X_test, y_test, Quantizer, DigitMLP):
    """
    Step 2: Quantize the model with YOUR Quantizer.

    Quantization reduces precision:
    ──────────────────────────────
        FP32 (32-bit) → INT8 (8-bit) = 4× smaller

        Before:  [0.123, -0.456, 0.789]  (4 bytes each)
        After:   [31, -117, 127]         (1 byte each + scale)

    Trade-off: Memory savings vs potential accuracy loss.
    The measured accuracy change depends on the model and dataset.

    Returns:
        dict with quantization results
    """
    console.print(Panel(
        "[bold yellow]🗜️ STEP 2: Quantize with YOUR Quantizer[/bold yellow]\n"
        "Using the quantization you built in Module 15\n"
        "Measure rounded-weight accuracy; packed INT8 storage is a separate artifact",
        border_style="yellow"
    ))

    quant_result = Quantizer.quantize_model(model)
    quant_size = int(param_bytes / quant_result['compression_ratio'])

    # Measure what INT8 actually costs in accuracy. Quantizer.quantize_model
    # returns the INT8 tensors and their scales but leaves the model untouched,
    # so rebuild a copy from the dequantized weights and run the same test set
    # through it. Reporting the baseline accuracy here instead, as this script
    # used to, prints a number that was never measured.
    quant_model = copy.deepcopy(model)
    quant_params = [prm for lyr in quant_model.layers for prm in lyr.parameters()]
    for idx, prm in enumerate(quant_params):
        entry = quant_result['quantized_layers'][f'param_{idx}']
        restored = Quantizer.dequantize_tensor(
            entry['quantized'], entry['scale'], entry['zero_point']
        )
        prm.data = restored.data.reshape(entry['original_shape'])

    outputs_quant = quant_model(X_test)
    quant_acc = np.mean(np.argmax(outputs_quant.data, axis=1) == y_test) * 100

    # Display results
    table = Table(title="🗜️ After Quantization (YOUR Implementation)", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Before", style="yellow")
    table.add_column("After", style="green")
    table.add_column("Change", style="bold")

    table.add_row(
        "Modeled packed size",
        f"{param_bytes:,} B",
        f"{quant_size:,} B",
        f"[green]{quant_result['compression_ratio']:.1f}× smaller[/green]"
    )
    table.add_row(
        "Precision",
        "FP32 (32-bit)",
        "INT8 (8-bit)",
        "[dim]Execution remains float32[/dim]"
    )
    quant_acc_delta = quant_acc - baseline_acc
    table.add_row(
        "Accuracy",
        f"{baseline_acc:.1f}%",
        f"{quant_acc:.1f}%",
        f"[{'green' if quant_acc_delta >= 0 else 'red'}]{quant_acc_delta:+.1f}%[/]"
    )

    console.print(table)

    return {
        'quant_result': quant_result,
        'quant_size': quant_size,
        'quant_acc': quant_acc,
        'model': quant_model,
        'actual_bytes': sum(p.data.nbytes for p in quant_model.parameters()),
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
    Measure the accuracy change rather than assuming a fixed tolerance.

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
    model_copy = copy.deepcopy(model)

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
        'model': model_copy,
        'actual_bytes': sum(p.data.nbytes for p in model_copy.parameters()),
    }


# =============================================================================
# STEP 4: KV CACHE
# =============================================================================

def step_4_kv_cache(KVCache, MinimalTransformer):
    """Exercise cache writes, cursor advancement, reads, and reset (not timing)."""
    from tinytorch.core.tensor import Tensor

    cache = KVCache(batch_size=1, max_seq_len=8, num_layers=1,
                    num_heads=2, head_dim=16)
    key = Tensor(rng.standard_normal((1, 2, 1, 16)))
    value = Tensor(rng.standard_normal((1, 2, 1, 16)))
    cache.update(0, key, value)
    cache.advance()
    stored_key, stored_value = cache.get(0)
    np.testing.assert_array_equal(stored_key.data, key.data)
    np.testing.assert_array_equal(stored_value.data, value.data)
    cache_bytes = int(round(cache.get_memory_usage()['total_mb'] * 1024 * 1024))
    cache.reset()
    assert cache.seq_pos == 0
    console.print(f"KV cache write/read/reset passed; allocated {cache_bytes:,} bytes.")
    console.print("Part 06.2 checks cached logits and measures inference speed.")
    return {'cache_memory': cache_bytes, 'kv_cache': cache}


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

    Both paths call NumPy BLAS; this checks numerical equivalence and overhead.

    Returns:
        dict with timing comparison
    """
    console.print(Panel(
        "[bold magenta]🚀 STEP 5: Acceleration with YOUR Module 17[/bold magenta]\n"
        "Verify the vectorized operation and measure its overhead\n"
        "Compare the wrapper against the same NumPy matrix multiplication",
        border_style="magenta"
    ))

    # Create test matrices
    A = Tensor(rng.standard_normal((64, 128)).astype(np.float32))
    B = Tensor(rng.standard_normal((128, 64)).astype(np.float32))

    # Time standard operation
    start = time.perf_counter()
    for _ in range(100):
        C_standard = Tensor(np.dot(A.data, B.data))
    standard_time = (time.perf_counter() - start) * 1000

    # Time vectorized operation
    start = time.perf_counter()
    for _ in range(100):
        C_vectorized = vectorized_matmul(A, B)
    vectorized_time = (time.perf_counter() - start) * 1000

    np.testing.assert_allclose(C_vectorized.data, C_standard.data, rtol=1e-5, atol=1e-5)

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

def step_6_benchmark(model, X_test, y_test, baseline_acc, Benchmark, name="Baseline"):
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

    accuracy = float(np.mean(np.argmax(model(X_test).data, axis=1) == y_test) * 100)
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

    table = Table(title=f"🏁 {name}: measured candidate results", box=box.DOUBLE)
    table.add_column("Metric", style="cyan", width=18)
    table.add_column("Value", style="yellow", justify="right")
    table.add_column("Target", style="dim")

    table.add_row("Latency (mean)", f"{mean_latency:.3f} ms", "< 100ms")
    table.add_row("Latency (std)", f"± {std_latency:.3f} ms", "Low = stable")
    table.add_row("Latency (min/max)", f"{min_latency:.3f} / {max_latency:.3f} ms", "Tight range")
    table.add_row("P95 Latency", f"{p95_latency:.3f} ms", "< 2× mean")
    table.add_row("", "", "")
    table.add_row("Serial rate", f"{throughput:.0f} samples/sec", "Reciprocal of mean latency")
    table.add_row("Accuracy", f"{accuracy:.1f}%", "Same held-out test set")

    console.print(table)

    return {
        'accuracy': accuracy,
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
    """Compare independent candidates; never imply they were combined."""
    table = Table(title="Optimization Olympics: independent candidate measurements")
    for heading in ['Candidate', 'Dense bytes', 'Accuracy', 'Latency (ms)']:
        table.add_column(heading)
    for name, size in [('Baseline', baseline['param_bytes']),
                       ('Rounded weights', quant['actual_bytes']),
                       ('Pruned weights', prune['actual_bytes'])]:
        result = profile_results[name]
        table.add_row(name, f"{size:,}", f"{result['accuracy']:.1f}%",
                      f"{result['mean_latency']:.3f}")
    console.print(table)
    console.print(f"Modeled INT8 codes (excluding metadata): {quant['quant_size']:,} bytes "
                  f"({quant['quant_result']['compression_ratio']:.2f}× ratio).")
    console.print(f"Pruned zero fraction: {prune['sparsity_after']:.1%}.")
    console.print('Both candidate models still execute dense float32 operations. '
                  'Packed storage and sparse execution require additional implementations.')
    console.print('[bold green]MILESTONE 06 COMPLETE: candidates measured against the baseline.[/bold green]')
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
    2. QUANTIZE  - FP32 → rounded weights (modeled INT8 storage)
    3. PRUNE     - Zero small weights (dense storage is unchanged)
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
        "[bold magenta]║[/bold magenta] [cyan]from the optimization modules  [/cyan]        [bold magenta]║[/bold magenta]\n"
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

        from tinytorch.perf.acceleration import vectorized_matmul
        console.print("  [green]✓[/green] vectorized_matmul (YOUR Module 17)")

        from tinytorch.perf.memoization import KVCache
        console.print("  [green]✓[/green] KVCache (YOUR Module 18)")

        from tinytorch.perf.benchmarking import Benchmark
        console.print("  [green]✓[/green] Benchmark (YOUR Module 19)")

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

    # Reuse the milestone network; a broken import must fail visibly.
    sys.path.insert(0, str(Path(__file__).parent))
    from networks import DigitMLP, MinimalTransformer

    model = DigitMLP()
    console.print(f"\n  [bold green]Using: {model.name}[/bold green]")

    # Load TinyDigits dataset
    console.print("\n[bold cyan]📊 Loading TinyDigits dataset...[/bold cyan]")

    try:
        train_images_np, y_train, test_images_np, y_test = load_tinydigits_arrays()

        X_train = Tensor(train_images_np.reshape(train_images_np.shape[0], -1).astype(np.float32))
        X_test = Tensor(test_images_np.reshape(test_images_np.shape[0], -1).astype(np.float32))
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)

        console.print(f"  [green]✓[/green] Training: {len(y_train)} samples")
        console.print(f"  [green]✓[/green] Test: {len(y_test)} samples")
    except FileNotFoundError as e:
        console.print(Panel(
            f"[red]{e}[/red]\n\n"
            "[yellow]Milestone 06 uses the TinyDigits dataset shipped with TinyTorch.[/yellow]",
            title="TinyDigits Missing",
            border_style="red"
        ))
        return 1

    # ─────────────────────────────────────────────────────────────────────────
    # QUICK TRAINING
    # ─────────────────────────────────────────────────────────────────────────
    console.print(f"\n[bold cyan]🏋️ Quick training ({CONFIG['train_epochs']} epochs)...[/bold cyan]")

    from tinytorch.core.optimizers import SGD
    from tinytorch.core.losses import CrossEntropyLoss

    optimizer = SGD(model.parameters(), lr=CONFIG['learning_rate'])
    loss_fn = CrossEntropyLoss()

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
        task = progress.add_task("Training...", total=CONFIG['train_epochs'])

        for epoch in range(CONFIG['train_epochs']):
            batch_size = CONFIG['batch_size']
            for i in range(0, len(y_train), batch_size):
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

    # Training registered gradients; inference measurement should not build tapes.
    for parameter in model.parameters():
        parameter.requires_grad = False
        parameter.grad = None

    # Step 1: Profile baseline
    baseline = step_1_profile(model, X_test, y_test, Profiler, Tensor)
    press_enter_to_continue()

    # Step 2: Quantize
    quant = step_2_quantize(model, baseline['param_bytes'], baseline['baseline_acc'],
                            X_test, y_test, Quantizer, DigitMLP)
    press_enter_to_continue()

    # Step 3: Prune
    prune = step_3_prune(model, baseline['baseline_acc'], X_test, y_test, Compressor, DigitMLP)
    press_enter_to_continue()

    # Step 4: Cache lifecycle (a separate mechanism from the MLP candidates).
    step_4_kv_cache(KVCache, MinimalTransformer)
    press_enter_to_continue()

    # Step 5: Acceleration
    step_5_accelerate(vectorized_matmul, Tensor)
    press_enter_to_continue()

    # Step 6: Benchmark
    candidates = {'Baseline': model, 'Rounded weights': quant['model'],
                  'Pruned weights': prune['model']}
    measurements = {name: step_6_benchmark(candidate, X_test, y_test,
                    baseline['baseline_acc'], Benchmark, name)
                    for name, candidate in candidates.items()}
    press_enter_to_continue()

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    return print_final_results(baseline, quant, prune, measurements)


if __name__ == "__main__":
    sys.exit(main())
