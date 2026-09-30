#!/usr/bin/env python3
"""
The MLPerf Era (2018-Present): The Hardware Optimization Olympics
=============================================================================

📚 HISTORICAL CONTEXT:
In 2018, a consortium of 30+ leading AI companies and academic institutions (including
Google, Harvard, Stanford, and Intel) founded MLPerf (now MLCommons). Their core
insight was transformative: accuracy alone is meaningless without measuring efficiency,
latency, and hardware throughput under strict statistical measurement discipline.

In the post-Moore era, deploying deep learning models in production demands a
principled optimization cascade:
1. Profiling (Module 14): Pinpointing latency, FLOPs, and memory bottlenecks.
2. Quantization (Module 15): Compressing 32-bit floats to 8-bit integers (4x memory drop).
3. Compression (Module 16): Pruning insignificant weights to exploit parameter sparsity.
4. Acceleration (Module 17): Fusing memory operations and vectorizing GEMM arithmetic.
5. Memoization (Module 18): Reusing prior key-value state to slash autoregressive decode.
6. Benchmarking (Module 19): Computing the Pareto frontier across accuracy vs latency.

🎯 MILESTONE 06: THE OPTIMIZATION OLYMPICS
Using YOUR Tiny🔥Torch systems implementations, you will optimize trained neural
networks and evaluate them against rigorous MLPerf-inspired benchmarks. You will
measure real wall-clock latency, parameter memory reduction, and Pareto trade-offs!

✅ REQUIRED MODULES (Run after Module 20):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR strided tensor data structure
  Module 03 (Layers)        : YOUR Linear projection layers
  Module 13 (Transformers)  : YOUR MinimalTransformer architecture
  Module 14 (Profiling)     : YOUR Profiler for FLOPs, memory, and latency
  Module 15 (Quantization)  : YOUR INT8 Quantizer (symmetric and asymmetric)
  Module 16 (Compression)   : YOUR Magnitude Pruning and Sparsity Compressor
  Module 17 (Acceleration)  : YOUR im2col and Vectorized GEMM Kernels
  Module 18 (Memoization)   : YOUR KV Cache state reuse engine
  Module 19 (Benchmarking)  : YOUR Statistical MLPerf Benchmarking Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (The Systems Optimization Cascade):
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  Baseline   │───▶│  Profile    │───▶│  Quantize   │───▶│    Prune    │
    │ FP32 Model  │    │  YOUR M14   │    │  YOUR M15   │    │  YOUR M16   │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                 │                  │
                                            4x Smaller!        50% Sparse!
                                                 ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌────────────────────────────────┐
    │   Pareto    │◀───│  Benchmark  │◀───│ Accelerate & Cache (M17 & M18) │
    │  Frontier   │    │  YOUR M19   │    │ im2col, SIMD GEMM, KV Cache    │
    └─────────────┘    └─────────────┘    └────────────────────────────────┘
"""

from contextlib import nullcontext, redirect_stdout
import copy
import io
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

rng = np.random.default_rng(7)

# Add project root
repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Independent references for the gates (no student module is called there).
from mlperf_gates import (  # noqa: E402
    CAUSAL_LEAK_MAX, CONTEXT_DEPENDENCE_MIN, LOGIT_SPREAD_MIN, POSITION_SPREAD_MIN,
    position_spread, check_benchmark_stats, check_pareto, context_probe, latency_stats_consistent,
    linear_reference_counts, logit_spread, logits_nondegenerate,
    loss_matches_reference, reference_cross_entropy, LOSS_ATOL, LOSS_RTOL,
)

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
# 🎓 ZONE 1: STUDENT CORE LEGO BRICKS (Model Architecture & Optimization Targets)
# =============================================================================
#
# This milestone optimizes the neural network architectures built in earlier
# milestones (Perceptron, DigitMLP, SimpleCNN, MinimalTransformer from networks.py)
# using the systems optimization modules YOU authored:
#   Module 14: Profiler (FLOPs, latency, and memory counting)
#   Module 15: Quantizer (INT8 symmetric and asymmetric affine mapping)
#   Module 16: Compressor (Magnitude pruning and sparsity masking)
#   Module 17: Accelerator (Vectorized tiled GEMM)
#   Module 18: Memoization (KV Cache serving speedup)
#   Module 19: Benchmarker (Statistical MLPerf measurement discipline)
#
# =============================================================================
# 📊 ZONE 2: MILESTONE HARNESS & MLPERF BENCHMARK UX
# =============================================================================

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
# PASS GATES: the milestone passes only when YOUR modules produce these results
# =============================================================================
#
# 2026-09-29: this script used to print "MILESTONE 06 COMPLETE" and exit 0
# whatever the modules did. A sabotage audit passed it with an optimizer that
# never stepped (baseline 16%), a dequantizer that returned zeros (10%), a
# pruner that pruned nothing (0% sparse), and a KV cache that returned zero
# keys. Every threshold below is calibrated against correct runs and against
# those sabotages; the measurements are recorded next to each one.
#
# Calibration (2026-09-29), correct code, 16 runs: the default init (seed 7)
# plus Linear's module RNG rebound to seeds 0-6 and 8-14; 40 epochs; the
# TinyDigits test set has 200 images, so one image is 0.5 points:
#   baseline accuracy 83.0-88.5% (default seed 86.5%)
#   INT8 accuracy - baseline: -1.5 to +0.5 points; roundtrip error 0.50 steps
#   50% prune zero fraction: exactly 50.0% on every seed
#   cached vs recompute logits: max |diff| 0.95e-6 to 1.67e-6
# Sabotaged (default seed unless noted): no-op optimizer.step -> 9.5-16.0%
# baseline (seeds 0, 2, 7, 14); zeroed dequantizer -> roundtrip error 157.6
# steps (10% INT8 accuracy in the audit); no-op pruner -> 0.0% zero;
# KVCache.get returning zero keys -> storage gate fails, logits off by 2.47.
#
# 2026-09-29 (knockout study): the script still passed with YOUR Profiler
# (Module 14) reporting zeros, YOUR BenchmarkResult/pareto_frontier (Module 19)
# returning zero statistics and admitting every candidate, and YOUR embedding
# lookup (Module 11) or LayerNorm/MLP (Module 13) returning zeros inside the
# GPT. Their outputs were displayed but never checked. The Profiler is now
# checked against counts derived from the layer shapes, Module 19 against
# hand-worked fixtures, and the GPT against black-box language-model probes
# (see mlperf_gates.py for the references and their calibration).
BASELINE_ACC_FLOOR = 80.0      # %: 3 points under the worst correct seed, 64 over the best sabotage
QUANT_ACC_MAX_DROP = 3.0       # points: 2x the worst correct drop (-1.5 = three images)
PRUNE_SPARSITY_TOL = 0.02      # absolute fraction around CONFIG['prune_sparsity']
CACHE_LOGIT_TOL = 1e-4         # max |cached - recomputed| logit
INT8_MIN, INT8_MAX = -128, 127
INT8_METADATA_BYTES = 8        # one float32 scale + one int32 zero point per tensor

GATE_RESULTS = []


class MilestoneGateFailure(Exception):
    """A pass gate failed: YOUR module did not produce a correct result."""

    def __init__(self, gate, measured, required, lesson):
        super().__init__(f"{gate}: measured {measured}, required {required}")
        self.gate, self.measured, self.required, self.lesson = gate, measured, required, lesson


def gate(name, passed, measured, required, lesson):
    """Record one pass gate; stop the milestone with a teaching message if it failed."""
    GATE_RESULTS.append((name, measured, required, bool(passed)))
    if not passed:
        raise MilestoneGateFailure(name, measured, required, lesson)


def speed_label(ratio, meaningful=2.0):
    """Describe a baseline/candidate time ratio honestly: faster, slower, or a wash."""
    # 2026-09-29: every ratio used to print "N× FASTER ⚡", including 1.1×
    # measurements and ratios below 1 (a slowdown).
    if ratio >= meaningful:
        return f"[bold bright_green]{ratio:.1f}× faster ⚡[/bold bright_green]"
    if ratio >= 1.05:
        return f"[green]{ratio:.2f}× faster[/green]"
    if ratio > 1 / 1.05:
        return f"[yellow]about the same ({ratio:.2f}×)[/yellow]"
    return f"[red]{1 / ratio:.2f}× slower[/red]"


def int8_storage_bytes(quant_result):
    """Bytes of the INT8 artifact, counted from the stored code arrays.

    One byte per code plus one scale and one zero point per tensor. This reads
    the arrays YOUR quantizer returned rather than trusting its reported
    compression_ratio, which is just another number the quantizer computed.
    """
    entries = quant_result['quantized_layers'].values()
    codes = sum(np.asarray(entry['quantized'].data).size for entry in entries)
    return codes + len(quant_result['quantized_layers']) * INT8_METADATA_BYTES


def check_int8_artifact(params, quant_result, Quantizer, record=True):
    """Verify every parameter became real INT8 codes that decode back to itself.

    Stops at the first bad parameter with a teaching message. When every
    parameter passes, records one summary line per check (record=False skips
    the summary, for the extra CNN and GPT artifacts in Step 7). Returns the
    worst roundtrip error in quantization steps (scale units).
    """
    def must(name, ok, measured, required, lesson):
        if not ok:
            gate(name, False, measured, required, lesson)

    entries = quant_result.get('quantized_layers', {})
    worst_steps, n_codes, lo, hi = 0.0, 0, INT8_MAX, INT8_MIN
    for idx, param in enumerate(params):
        entry = entries.get(f'param_{idx}')
        where = f"parameter {idx} (shape {param.data.shape})"
        must("INT8 artifact covers every parameter", entry is not None,
             f"no entry for {where}", "one entry per parameter",
             "Quantizer.quantize_model must return a 'param_<i>' entry for every model parameter.")
        codes = np.asarray(entry['quantized'].data, dtype=np.float64)
        scale, zero_point = entry['scale'], entry['zero_point']
        must("INT8 codes are integers in [-128, 127]",
             codes.size == param.data.size and np.all(np.isfinite(codes))
             and np.all(codes == np.round(codes))
             and codes.min() >= INT8_MIN and codes.max() <= INT8_MAX,
             f"{where}: {codes.size} codes, range [{codes.min():.3g}, {codes.max():.3g}]",
             f"{param.data.size} integer codes in [{INT8_MIN}, {INT8_MAX}]",
             "quantize_int8 must round value/scale + zero_point to the nearest integer and clamp "
             "it to the INT8 range. Codes outside it, or fractional codes, are not INT8.")
        must("INT8 scale and zero point are valid",
             np.isfinite(scale) and scale > 0 and float(zero_point) == round(float(zero_point))
             and INT8_MIN <= zero_point <= INT8_MAX,
             f"{where}: scale={scale!r}, zero_point={zero_point!r}",
             f"scale > 0, integer zero_point in [{INT8_MIN}, {INT8_MAX}]",
             "scale is (max - min) / 255 and zero_point is the integer code that 0.0 maps to.")
        restored = np.asarray(Quantizer.dequantize_tensor(entry['quantized'], scale, zero_point).data)
        error = float(np.max(np.abs(restored.reshape(param.data.shape) - param.data))) \
            if restored.size == param.data.size else float('inf')
        steps = error / scale
        # Round-to-nearest leaves at most half a step of error inside the range.
        must("dequantize(quantize(w)) recovers w", steps <= 1.0 + 1e-3,
             f"{where}: max error {error:.3g} = {steps:.1f} steps",
             "at most 1 quantization step",
             "dequantize_int8 must invert the mapping: (q - zero_point) * scale. An error of "
             "many steps means the decoded weights are not the weights you quantized.")
        worst_steps = max(worst_steps, steps)
        n_codes += codes.size
        lo, hi = min(lo, int(codes.min())), max(hi, int(codes.max()))
    if record:
        gate("INT8 codes are integers in [-128, 127]", True,
             f"{n_codes:,} codes in {len(params)} tensors, range [{lo}, {hi}]",
             "every parameter, integer codes", "")
        gate("dequantize(quantize(w)) recovers w", True,
             f"worst error {worst_steps:.2f} steps", "<= 1 quantization step", "")
    return worst_steps


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

    # Independent reference: count from the Linear layers' shapes by hand.
    ref_params, ref_flops = linear_reference_counts(model.layers)

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
    console.print(f"  [dim]Reference from the layer shapes: {ref_params:,} parameters "
                  f"(in*out + out per Linear), {ref_flops:,} FLOPs (2*in*out per Linear: one "
                  f"multiply and one add per weight; bias adds and ReLU not counted).[/dim]")

    gate("Profiler counts every parameter", param_count == ref_params,
         f"{param_count:,}", f"{ref_params:,} (from the layer shapes)",
         "count_parameters must add up the element counts of every weight and bias the "
         "model owns (Module 14). Linear(in, out) owns in*out weights plus out biases.")
    gate("Profiler FLOPs match the layer shapes", flops == ref_flops,
         f"{flops:,}", f"{ref_flops:,} (2*in*out per Linear)",
         "count_flops must charge each Linear 2 * in_features * out_features per sample "
         "(one multiply and one add per weight) and sum over the layers (Module 14).")
    gate("Profiler latency is a real measurement",
         np.isfinite(latency_ms) and latency_ms > 0,
         f"{latency_ms:.4f} ms", "> 0 ms and finite",
         "measure_latency must time real forward passes with time.perf_counter() and "
         "return the median in milliseconds (Module 14). Zero means nothing was timed.")

    gate("Baseline accuracy clears the floor", baseline_acc >= BASELINE_ACC_FLOOR,
         f"{baseline_acc:.1f}%", f">= {BASELINE_ACC_FLOOR:.0f}%",
         "The baseline DigitMLP did not learn. Every optimization below is measured against "
         "this model, so a broken baseline makes every comparison meaningless. Check YOUR "
         "optimizer's step() (Module 07), zero_grad(), and loss.backward() (Modules 04 and 06).")

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
    worst_steps = check_int8_artifact(list(model.parameters()), quant_result, Quantizer)
    # Counted from the stored code arrays, not param_bytes / reported ratio.
    quant_size = int8_storage_bytes(quant_result)
    measured_ratio = param_bytes / quant_size

    # Measure what INT8 actually costs in accuracy. Quantizer.quantize_model
    # returns the INT8 tensors and their scales but leaves the model untouched,
    # so rebuild a copy from the dequantized weights and run the same test set
    # through it to measure the candidate's accuracy.
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
        f"[green]{measured_ratio:.2f}× smaller[/green]"
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
    console.print(f"  [dim]INT8 bytes counted from YOUR code arrays: one byte per code plus "
                  f"{INT8_METADATA_BYTES} bytes of scale and zero point per tensor. "
                  f"Worst roundtrip error: {worst_steps:.2f} quantization steps.[/dim]")

    gate("INT8 accuracy stays near baseline", quant_acc >= baseline_acc - QUANT_ACC_MAX_DROP,
         f"{quant_acc:.1f}% ({quant_acc_delta:+.1f} points)",
         f">= baseline - {QUANT_ACC_MAX_DROP:.0f} points",
         "Rounding every weight to 256 levels should barely move a trained MLP. A large drop "
         "means YOUR quantize/dequantize pair loses the weights' information.")

    console.print(Panel(
        "[bold yellow]⚠️  MLSys Reality Check: Storage Compression ≠ Compute Speedup[/bold yellow]\n\n"
        f"• [bold green]What INT8 achieves:[/bold green] {measured_ratio:.2f}× smaller weight storage here (at most 4×: 32-bit → 8-bit codes, less the scale metadata).\n"
        "• [bold yellow]Why latency is flat:[/bold yellow] In pure Python/NumPy, we perform [dim]simulated quantization[/dim]: weights\n"
        "  are stored as 8-bit integers but dequantized back to float32 at runtime to execute standard BLAS GEMM.\n"
        "• [bold cyan]Hardware reality:[/bold cyan] Without hardware-native INT8 GEMM tensor cores (e.g., NVIDIA DP4A,\n"
        "  Apple Neural Engine, or ARM NEON/dotprod), quantization yields massive memory savings but zero CPU speedup.",
        border_style="yellow",
        box=box.ROUNDED,
    ))

    return {
        'quant_result': quant_result,
        'quant_size': quant_size,
        'measured_ratio': measured_ratio,
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

    # Count zeros ourselves from the stored weight matrices (the same 2D
    # parameters measure_sparsity counts) instead of trusting the report.
    weights = [p.data for p in model_copy.parameters() if p.data.ndim > 1]
    zero_fraction = float(sum(np.sum(w == 0) for w in weights) / sum(w.size for w in weights))
    target = CONFIG['prune_sparsity']

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
        f"{zero_fraction:.1%}",
        f"[green]{zero_fraction:.1%} of weights are zero[/green]"
    )
    prune_acc_delta = pruned_acc - baseline_acc
    table.add_row(
        "Accuracy",
        f"{baseline_acc:.1f}%",
        f"{pruned_acc:.1f}%",
        f"[{'green' if prune_acc_delta >= 0 else 'red'}]{prune_acc_delta:+.1f}%[/]"
    )

    console.print(table)

    gate("Pruning reaches the requested sparsity", abs(zero_fraction - target) <= PRUNE_SPARSITY_TOL,
         f"{zero_fraction:.1%} zero (YOUR measure_sparsity reported {sparsity_after:.1%})",
         f"{target:.0%} ± {PRUNE_SPARSITY_TOL:.0%}",
         "magnitude_prune(model, sparsity) must zero that fraction of the weight matrices, "
         "choosing the smallest magnitudes, in place on the model it is given.")

    return {
        'sparsity_before': sparsity_before,
        'sparsity_after': zero_fraction,
        'pruned_acc': pruned_acc,
        'model': model_copy,
        'actual_bytes': sum(p.data.nbytes for p in model_copy.parameters()),
    }


# =============================================================================
# STEP 4: KV CACHE
# =============================================================================

def step_4_kv_cache(KVCache, MinimalTransformer, GPT=None,
                    enable_kv_cache=None, disable_kv_cache=None):
    """Check cache storage, then that cached decoding reproduces recompute logits."""
    from tinytorch.core.tensor import Tensor

    console.print(Panel(
        "[bold green]💾 STEP 4: Verify YOUR KV Cache (Module 18)[/bold green]\n"
        "A cache is an optimization only if it changes nothing but the cost",
        border_style="green"
    ))

    cache = KVCache(batch_size=1, max_seq_len=8, num_layers=1,
                    num_heads=2, head_dim=16)
    key = Tensor(rng.standard_normal((1, 2, 1, 16)))
    value = Tensor(rng.standard_normal((1, 2, 1, 16)))
    cache.update(0, key, value)
    cache.advance()
    stored_key, stored_value = cache.get(0)
    stored_ok = (np.shape(stored_key.data) == key.data.shape
                 and np.shape(stored_value.data) == value.data.shape
                 and np.array_equal(stored_key.data, key.data)
                 and np.array_equal(stored_value.data, value.data))
    gate("KV cache returns what was stored", stored_ok,
         "cache.get(0) differs from the update" if not stored_ok else "exact",
         "exact K and V after update + advance",
         "KVCache.update must write K and V at the current position, and get() must return "
         "every position written so far, unchanged.")
    cache_bytes = int(round(cache.get_memory_usage()['total_mb'] * 1024 * 1024))
    cache.reset()
    gate("KV cache reset rewinds the cursor", cache.seq_pos == 0,
         f"seq_pos={cache.seq_pos}", "seq_pos == 0",
         "reset() must return the cache to position 0 so the next request starts fresh.")

    # The equivalence that makes caching legal: token-by-token cached decoding
    # must produce the same next-token logits as recomputing every prefix.
    if GPT is None:
        from tinytorch.core.transformers import GPT
    if enable_kv_cache is None or disable_kv_cache is None:
        from tinytorch.perf.memoization import enable_kv_cache, disable_kv_cache
    gpt = GPT(vocab_size=28, embed_dim=32, num_layers=2, num_heads=2, max_seq_len=32)
    tokens = np.random.default_rng(11).integers(0, 28, (1, 16))
    # An all-zero model caches perfectly (0 == 0), so first check the GPT
    # built from YOUR Modules 11-13 computes a language model at all.
    check_gpt_logits(gpt, tokens, Tensor)
    with redirect_stdout(io.StringIO()):
        expected = replay_prefixes(gpt, tokens, Tensor)
        kv = enable_kv_cache(gpt)
        try:
            actual = replay_prefixes(gpt, tokens, Tensor, cache=kv)
        finally:
            disable_kv_cache(gpt)
    diff = max_abs_diff(actual, expected) if actual.shape == expected.shape else float('inf')
    gate("Cached logits match recomputed logits", diff <= CACHE_LOGIT_TOL,
         f"max |Δ| = {diff:.2e} over {tokens.shape[1]} tokens", f"<= {CACHE_LOGIT_TOL:.0e}",
         "Decoding one token at a time with YOUR KV cache must give the same logits as running "
         "the full prefix. A mismatch means the cache stores, returns, or positions K and V "
         "wrongly (check update/get indexing and that advance() runs once per token).")

    console.print(f"  [green]✓[/green] Storage, reset, and cached-vs-recomputed logits agree "
                  f"(max |Δ| = {diff:.1e}); cache allocated {cache_bytes:,} bytes.")
    console.print("  [dim]Part 06.2 measures what the cache buys in speed.[/dim]")
    return {'cache_memory': cache_bytes, 'kv_cache': cache, 'cache_max_abs_diff': diff}


def check_gpt_logits(gpt, tokens, Tensor):
    """
    Gate the GPT that Steps 4, 5 and 7 cache, time, and quantize.

    The references are properties, not numbers: an untrained GPT's logits are
    random, but any working one varies its predictions, never lets a position
    read later tokens, and always lets a position read earlier ones. Three
    forward passes per cut make this far cheaper than rebuilding the transformer
    in NumPy, and it still stops a zeroed embedding lookup (Module 11) or a
    zeroed LayerNorm/MLP (Module 13), both of which the cache check misses.
    """
    with redirect_stdout(io.StringIO()):
        spread = logit_spread(gpt(Tensor(tokens)).data)
        probe = context_probe(gpt, tokens[0], gpt.vocab_size, Tensor)
        positions = position_spread(gpt, int(tokens[0][0]), len(tokens[0]), Tensor)
    gate("GPT logits vary across tokens and positions", logits_nondegenerate(spread),
         f"std {spread['vocab']:.2e} across vocab, {spread['positions']:.2e} across positions",
         f">= {LOGIT_SPREAD_MIN:.0e} each",
         "YOUR GPT gives the same (often all-zero) scores everywhere. Check that "
         "LayerNorm and the MLP (Module 13), Linear (Module 03), and Tensor matmul and add "
         "(Module 01) return their computed values.")
    gate("GPT never reads future tokens", probe['leak'] <= CAUSAL_LEAK_MAX,
         f"changing later tokens moved earlier logits by {probe['leak']:.2e}",
         f"<= {CAUSAL_LEAK_MAX:.0e}",
         "Position i may only read tokens 0..i. Check create_causal_mask and that "
         "attention applies it before the softmax (Modules 12 and 13).")
    gate("GPT reads its context", probe['dependence'] >= CONTEXT_DEPENDENCE_MIN,
         f"changing earlier tokens moved the logits by at most {probe['dependence']:.2e}",
         f">= {CONTEXT_DEPENDENCE_MIN:.0e}",
         "The prediction at a position ignores the tokens before it. Check that the "
         "embedding lookup returns each token's row (Module 11) and that attention mixes "
         "positions (Module 12).")
    gate("GPT tells positions apart", positions >= POSITION_SPREAD_MIN,
         f"one repeated token gave logits within {positions:.2e} at every position",
         f">= {POSITION_SPREAD_MIN:.0e}",
         "Without positional encoding, attention sees a set, not a sequence. Check that "
         "YOUR PositionalEncoding (Module 11) adds the position table instead of "
         "returning its input.")
    return spread, probe


# =============================================================================
# STEP 5: ACCELERATION
# =============================================================================

def step_5_accelerate(vectorized_matmul, Tensor, im2col_conv2d=None, Conv2d=None,
                      enable_kv_cache=None, disable_kv_cache=None, GPT=None):
    """
    Step 5: Demonstrate acceleration with YOUR Modules 17 & 18.

    Evaluates three concrete systems accelerations:
    1. Vectorized Matrix Multiply (Module 17): Hardware SIMD replacing interpreter loops
    2. Spatial Convolution Lowering (Module 17): im2col + one GEMM vs YOUR Module 09 Conv2d
    3. Autoregressive Memoization (Module 18): KV-Cache eliminating quadratic recomputation

    Returns:
        dict with timing comparison
    """
    console.print(Panel(
        "[bold magenta]🚀 STEP 5: Kernel Acceleration with YOUR Modules 17 & 18[/bold magenta]\n"
        "Benchmark three concrete kernel optimizations you implemented:\n"
        "• Kernel 1: Vectorized BLAS GEMM vs 3 nested interpreter loops\n"
        "• Kernel 2: Spatial Convolution Lowering (im2col + GEMM) vs YOUR Module 09 Conv2d\n"
        "• Kernel 3: Autoregressive KV-Cache Attention Memoization vs O(N²) prefix recomputation",
        border_style="magenta"
    ))

    # -------------------------------------------------------------------------
    # KERNEL 1: Dense Matrix Multiply (GEMM)
    # -------------------------------------------------------------------------
    A = rng.standard_normal((32, 32)).astype(np.float32)
    B = rng.standard_normal((32, 32)).astype(np.float32)

    start = time.perf_counter()
    C_loop = np.zeros((32, 32), dtype=np.float32)
    for i in range(32):
        for j in range(32):
            for k in range(32):
                C_loop[i, j] += A[i, k] * B[k, j]
    gemm_loop_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    C_vec = vectorized_matmul(Tensor(A), Tensor(B))
    gemm_vec_ms = (time.perf_counter() - start) * 1000
    gemm_speedup = gemm_loop_ms / gemm_vec_ms if gemm_vec_ms > 0 else 1.0

    gemm_err = float(np.max(np.abs(np.asarray(C_vec.data) - C_loop))) \
        if np.shape(C_vec.data) == C_loop.shape else float('inf')
    gate("vectorized_matmul matches the loop reference", gemm_err <= 1e-3,
         f"max |Δ| = {gemm_err:.2e}", "<= 1e-3 (float32 accumulation order)",
         "A faster matmul that computes a different answer is not an optimization. "
         "Check vectorized_matmul (Module 17).")

    box1 = Panel(
        f"[bold cyan]Baseline (3 Nested Interpreter Loops):[/bold cyan]  {gemm_loop_ms:.2f} ms\n"
        f"[bold green]Vectorized (NumPy BLAS call):[/bold green]           {gemm_vec_ms:.2f} ms\n"
        f"[bold yellow]Measured ratio:[/bold yellow]                         {speed_label(gemm_speedup)}\n\n"
        f"[dim]• Workload: 32×32 @ 32×32 Matrix Multiply\n"
        f"• Mechanism: Module 17 vectorized_matmul replaces interpreter loops with hardware SIMD[/dim]",
        title="[bold cyan]🏎️  Kernel 1: Dense Matrix Multiply (Module 17 Vectorization)[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
    )
    console.print(box1)

    # -------------------------------------------------------------------------
    # KERNEL 2: Spatial Convolution Lowering (im2col)
    # -------------------------------------------------------------------------
    if im2col_conv2d is None:
        from tinytorch.perf.acceleration import im2col_conv2d as _im2col
        im2col_conv2d = _im2col
    if Conv2d is None:
        from tinytorch.core.spatial import Conv2d as _Conv2d
        Conv2d = _Conv2d

    conv_layer = Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
    x_conv = Tensor(rng.standard_normal((4, 1, 8, 8)).astype(np.float32))

    # Warmup
    _ = conv_layer(x_conv)
    _ = im2col_conv2d(x_conv, conv_layer.weight, conv_layer.bias, padding=1)

    start = time.perf_counter()
    for _ in range(10):
        out_loop = conv_layer(x_conv)
    conv_loop_ms = (time.perf_counter() - start) * 1000 / 10

    start = time.perf_counter()
    for _ in range(10):
        out_im2col = im2col_conv2d(x_conv, conv_layer.weight, conv_layer.bias, padding=1)
    conv_im2col_ms = (time.perf_counter() - start) * 1000 / 10

    conv_err = float(np.max(np.abs(np.asarray(out_im2col.data) - out_loop.data))) \
        if np.shape(out_im2col.data) == out_loop.data.shape else float('inf')
    gate("im2col_conv2d matches Conv2d", conv_err <= 1e-3,
         f"max |Δ| = {conv_err:.2e}", "<= 1e-3",
         "im2col + GEMM must compute exactly the convolution Conv2d computes. Check the patch "
         "order in im2col and the weight reshape in im2col_conv2d (Module 17).")
    conv_speedup = conv_loop_ms / conv_im2col_ms if conv_im2col_ms > 0 else 1.0

    box2 = Panel(
        f"[bold cyan]Baseline (YOUR Conv2d forward):[/bold cyan]         {conv_loop_ms:.2f} ms\n"
        f"[bold green]Lowered (im2col Patch GEMM):[/bold green]            {conv_im2col_ms:.2f} ms\n"
        f"[bold yellow]Measured ratio:[/bold yellow]                         {speed_label(conv_speedup)}\n\n"
        f"[dim]• Workload: 4-Channel 3×3 Conv2d on 8×8 Spatial Patches (Batch 4)\n"
        f"• Baseline is whatever YOUR Module 09 Conv2d does, including its autograd bookkeeping\n"
        f"• Mechanism: Module 17 im2col unrolls patches so the whole convolution is one BLAS GEMM[/dim]",
        title="[bold magenta]⚡ Kernel 2: Spatial Convolution Lowering (Module 17 im2col)[/bold magenta]",
        border_style="magenta",
        box=box.ROUNDED,
    )
    console.print(box2)

    # -------------------------------------------------------------------------
    # KERNEL 3: Autoregressive Memoization (KV-Cache)
    # -------------------------------------------------------------------------
    if enable_kv_cache is None or disable_kv_cache is None:
        from tinytorch.perf.memoization import enable_kv_cache as _ekv, disable_kv_cache as _dkv
        enable_kv_cache, disable_kv_cache = _ekv, _dkv
    if GPT is None:
        from tinytorch.core.transformers import GPT as _GPT
        GPT = _GPT

    gpt = GPT(vocab_size=28, embed_dim=32, num_layers=2, num_heads=2, max_seq_len=32)
    tokens = rng.integers(0, 28, (1, 16))

    with redirect_stdout(io.StringIO()):
        start = time.perf_counter()
        for _ in range(2):
            for pos in range(tokens.shape[1]):
                _ = gpt(Tensor(tokens[:, :pos+1]))
        uncached_ms = (time.perf_counter() - start) * 1000 / 2

        cache = enable_kv_cache(gpt)
        start = time.perf_counter()
        for _ in range(2):
            cache.reset()
            with cache.generation():
                for pos in range(tokens.shape[1]):
                    _ = gpt(Tensor(tokens[:, pos:pos+1]), start_pos=cache.seq_pos)
                    cache.advance()
        cached_ms = (time.perf_counter() - start) * 1000 / 2
        disable_kv_cache(gpt)

    cache_speedup = uncached_ms / cached_ms if cached_ms > 0 else 1.0

    box3 = Panel(
        f"[bold cyan]Baseline (O(N²) Prefix Recomputation):[/bold cyan] {uncached_ms:.2f} ms\n"
        f"[bold green]KV-Cached (one new token per step):[/bold green]   {cached_ms:.2f} ms\n"
        f"[bold yellow]Measured ratio:[/bold yellow]                     {speed_label(cache_speedup)}\n\n"
        f"[dim]• Workload: 16-Token Autoregressive Generation (TinyGPT)\n"
        f"• Mechanism: Module 18 KVCache memoizes past Key/Value tensors instead of recomputing them\n"
        f"• At 16 tokens Python overhead dominates, so the measured gain is small (Part 06.2 goes longer)[/dim]",
        title="[bold green]💾 Kernel 3: Autoregressive Memoization (Module 18 KV-Cache)[/bold green]",
        border_style="green",
        box=box.ROUNDED,
    )
    console.print(box3)

    # -------------------------------------------------------------------------
    # SUMMARY SCORECARD TABLE
    # -------------------------------------------------------------------------
    table = Table(title="🚀 Optimization Tier Acceleration Scorecard", box=box.ROUNDED)
    table.add_column("Kernel / Workload", style="cyan")
    table.add_column("Baseline (Unoptimized)", style="yellow")
    table.add_column("Candidate (TinyTorch)", style="green")
    table.add_column("Measured ratio", style="bold")

    table.add_row(
        "Dense GEMM (32×32)",
        f"{gemm_loop_ms:.2f} ms (loops)",
        f"{gemm_vec_ms:.2f} ms (BLAS)",
        speed_label(gemm_speedup)
    )
    table.add_row(
        "2D Conv (4-ch, 8×8)",
        f"{conv_loop_ms:.2f} ms (Conv2d)",
        f"{conv_im2col_ms:.2f} ms (im2col)",
        speed_label(conv_speedup)
    )
    table.add_row(
        "Autoregressive Decode (16 tokens)",
        f"{uncached_ms:.2f} ms (recompute)",
        f"{cached_ms:.2f} ms (cached)",
        speed_label(cache_speedup)
    )

    console.print(table)

    return {
        'standard_time': gemm_loop_ms,
        'vectorized_time': gemm_vec_ms,
        'gemm_loop_time': gemm_loop_ms,
        'gemm_vec_time': gemm_vec_ms,
        'gemm_speedup': gemm_speedup,
        'conv_loop_time': conv_loop_ms,
        'conv_im2col_time': conv_im2col_ms,
        'conv_speedup': conv_speedup,
        'kv_recompute_time': uncached_ms,
        'kv_cached_time': cached_ms,
        'kv_speedup': cache_speedup,
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

    gate("Benchmark latency statistics are consistent", latency_stats_consistent(bench_result),
         f"mean {mean_latency:.4f}, median {bench_result.median:.4f}, "
         f"min {min_latency:.4f}, max {max_latency:.4f} ms",
         "0 < min <= mean, median <= max",
         "BenchmarkResult must summarize the latencies it was given (Module 19). A mean "
         "or median outside [min, max] was not computed from those values.")

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

def check_loss_forward(student_loss, logits, targets):
    """Gate YOUR CrossEntropyLoss forward on the first training batch."""
    reference = reference_cross_entropy(logits, targets)
    gate("CrossEntropyLoss matches a NumPy reference", loss_matches_reference(student_loss, reference),
         f"YOUR loss {student_loss:.6f}, reference {reference:.6f} (first batch)",
         f"within {LOSS_ATOL:.0e} + {LOSS_RTOL:.0e} x reference",
         "CrossEntropyLoss must return the mean over the batch of -log softmax(logits)[target], "
         "with a max-subtracted (stable) log-softmax (Module 04). Training can still converge "
         "with a wrong forward value because backward carries the gradient, but every loss "
         "you report and every loss curve you read would be wrong.")


def check_benchmark_module(BenchmarkResult, pareto_frontier):
    """Gate Module 19 on hand-worked fixtures before its numbers are displayed."""
    stats_problems = check_benchmark_stats(BenchmarkResult)
    gate("BenchmarkResult statistics match hand-computed values", not stats_problems,
         "; ".join(stats_problems) or "mean 5, median 4, std 3 on [2, 4, 4, 5, 10]",
         "mean 5, median 4, sample std 3, min 2, max 10",
         "BenchmarkResult must compute mean, median, sample standard deviation (n - 1), "
         "min and max of its values (Module 19). Every latency in Step 6 and 7 goes "
         "through it.")
    pareto_problems = check_pareto(pareto_frontier)
    gate("pareto_frontier keeps exactly the non-dominated points", not pareto_problems,
         "; ".join(pareto_problems) or "3 fixtures exact",
         "the hand-worked frontier of each fixture",
         "A point is dominated when another is at least as good on every objective and "
         "strictly better on one, honoring lower_is_better per objective (Module 19). "
         "Step 7's ★ Pareto labels come straight from this function.")


def press_enter_to_continue():
    if ("--non-interactive" in sys.argv or "-y" in sys.argv
            or os.environ.get("TITO_NON_INTERACTIVE") == "1"
            or os.environ.get("TINYTORCH_NON_INTERACTIVE") == "1"
            or os.environ.get("CI") == "true"):
        return
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            console.input("\n[yellow]Press Enter to continue...[/yellow] ")
        except EOFError:
            pass
        console.print()

def cosine_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Measure signal preservation (cosine similarity %) between two representations."""
    a_flat, b_flat = a.flatten(), b.flatten()
    norm_product = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    if norm_product == 0.0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / norm_product * 100.0)


def replay_prefixes(model, tokens, Tensor, cache=None):
    """Replay autoregressive prefix generation through model, returning sequence logits."""
    if cache is not None:
        cache.reset()
    logits = []
    with cache.generation() if cache is not None else nullcontext():
        for position in range(tokens.shape[1]):
            if cache is None:
                output = model(Tensor(tokens[:, :position + 1]))
            else:
                output = model(Tensor(tokens[:, position:position + 1]),
                               start_pos=cache.seq_pos)
                cache.advance()
            logits.append(output.data[:, -1, :].copy())
    return np.stack(logits, axis=1)




def pareto_status(name, frontier):
    """Label a candidate from the computed frontier only, never by name."""
    # 2026-09-28: INT8 and Full Stack were once labeled "Pareto (Peak/Optimal)"
    # by name, whatever pareto_frontier() returned.
    if name in frontier:
        return "[bold green]★ Pareto[/bold green]"
    return "[dim]● Dominated[/dim]"


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Largest elementwise difference between two logit arrays."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def synthesis_lines(mlp: dict, cnn: dict, gpt: dict) -> list:
    """
    Build the cross-division takeaway from measured values only.

    Every number is formatted from a measurement and every Pareto claim is
    read from a computed frontier. The CNN and GPT divisions run randomly
    initialized models, so their quality column is agreement with the FP32
    model's outputs, not accuracy.
    """
    mlp_ratio = mlp['base_bytes'] / mlp['quant_bytes']
    acc_delta = mlp['quant_acc'] - mlp['base_acc']
    lines = [
        f"• [cyan]Dense MLP (Edge):[/cyan] INT8 codes are {mlp_ratio:.2f}× smaller "
        f"({mlp['base_bytes']:,} → {mlp['quant_bytes']:,} B, including scale metadata). "
        f"Test accuracy {mlp['base_acc']:.1f}% → {mlp['quant_acc']:.1f}% "
        f"({acc_delta:+.1f} points). INT8 is "
        f"{'on' if mlp['quant_on_frontier'] else 'not on'} the measured memory-vs-error frontier."
    ]

    cnn_ratio = cnn['base_bytes'] / cnn['quant_bytes']
    lines.append(
        f"• [cyan]Spatial CNN (Vision, untrained weights):[/cyan] INT8 shrinks the footprint "
        f"{cnn_ratio:.2f}×. Its outputs agree with the FP32 baseline model's outputs at "
        f"{cnn['agree_quant']:.2f}% cosine similarity (50% pruned: {cnn['agree_pruned']:.2f}%). "
        f"This measures agreement with a random-weight model, not accuracy. "
        f"Measured frontier: {', '.join(cnn['frontier'])}."
    )

    diff = gpt['cache_max_abs_diff']
    if diff <= gpt.get('exact_tol', 1e-4):
        cache_text = f"KV-cached replay matches recompute logits (max |Δ| = {diff:.1e})"
    else:
        cache_text = f"KV-cached replay does NOT match recompute logits (max |Δ| = {diff:.1e})"
    if gpt['full_stack_on_frontier']:
        stack_text = (f"Quant+Cache is on the measured latency/memory/agreement frontier "
                      f"({len(gpt['frontier'])} of {gpt['n_candidates']} candidates are).")
    else:
        stack_text = ("Quant+Cache is dominated on this run; frontier: "
                      f"{', '.join(gpt['frontier'])}.")
    lines.append(f"• [cyan]Autoregressive GPT (LLM, untrained weights):[/cyan] {cache_text}. {stack_text}")
    return lines


def step_7_triad_and_pareto(mlp_baseline, mlp_quant, mlp_prune, measurements,
                            Profiler, Quantizer, Compressor,
                            SimpleCNN, GPT, Benchmark, BenchmarkResult, pareto_frontier,
                            enable_kv_cache, disable_kv_cache, Tensor, X_test):
    """
    Step 7: The Three MLPerf Benchmark Divisions & Workload-Specific Trade-Offs.

    Evaluates three distinct categories across their natural physical constraints
    using the standardized Module 19 Benchmark harness and actual measurements:
    - Division 1 (Dense MLP): Memory Footprint (KB) vs Test Accuracy (%)
    - Division 2 (Spatial CNN): Memory Footprint (KB) vs agreement with the
      FP32 baseline model's outputs (%); the CNN is untrained, so this is not accuracy
    - Division 3 (Autoregressive GPT): Replay Latency (ms) vs Memory Footprint (KB)
    """
    console.print(Panel(
        "[bold]🏆 STEP 7: THREE MLPERF BENCHMARK DIVISIONS[/bold]\n\n"
        "Three distinct workload categories. Three distinct systems bottlenecks.\n"
        "100% measured results evaluated via Module 19's Benchmark harness.",
        border_style="bright_magenta"
    ))

    # -------------------------------------------------------------------------
    # DIVISION 1: Edge & Embedded Inference: DigitMLP (Dense, Parameter-Bound)
    # -------------------------------------------------------------------------
    mlp_bytes = mlp_baseline['param_bytes']
    mlp_q_bytes = mlp_quant['quant_size']
    mlp_acc = mlp_baseline['baseline_acc']
    mlp_q_acc = mlp_quant['quant_acc']
    mlp_p_acc = mlp_prune['pruned_acc']

    mlp_records = [
        ('Baseline FP32', mlp_bytes, measurements['Baseline'], f"{mlp_acc:.1f}%", mlp_acc, 100.0 - mlp_acc),
        ('INT8 Quantized', mlp_q_bytes, measurements['Rounded weights'], f"{mlp_q_acc:.1f}%", mlp_q_acc, 100.0 - mlp_q_acc),
        ('50% Pruned', mlp_bytes, measurements['Pruned weights'], f"{mlp_p_acc:.1f}%", mlp_p_acc, 100.0 - mlp_p_acc),
    ]
    mlp_pts = {r[0]: (r[1] / 1024.0, r[5]) for r in mlp_records}
    mlp_frontier = set(pareto_frontier(mlp_pts, (True, True)))

    console.print(Panel(
        "[bold cyan]📍 DIVISION 1: Edge & Embedded Inference: DigitMLP (Dense)[/bold cyan]\n"
        "[dim]Primary Bottleneck: Dense weight memory capacity (SRAM/Flash footprint)\n"
        "Harness: Measured via Module 19 Benchmark on held-out TinyDigits test set[/dim]",
        border_style="cyan",
        box=box.ROUNDED,
    ))

    t1 = Table(box=box.ROUNDED)
    t1.add_column("Candidate", style="yellow")
    t1.add_column("Memory", justify="right")
    t1.add_column("Accuracy", justify="center")
    t1.add_column("Latency", justify="right")
    t1.add_column("Status", justify="center")
    for r in mlp_records:
        status_str = pareto_status(r[0], mlp_frontier)
        res = r[2]
        t1.add_row(r[0], f"{r[1]:,} B", r[3], f"{res['mean_latency']:.3f} ms", status_str)
    console.print(t1)


    # -------------------------------------------------------------------------
    # DIVISION 2: Spatial Vision & Compute: SimpleCNN (Spatial, Compute-Bound)
    # -------------------------------------------------------------------------
    cnn_base = SimpleCNN()
    cnn_bytes = sum(p.data.nbytes for p in cnn_base.parameters())

    # INT8 Quantized candidate
    cnn_quant = copy.deepcopy(cnn_base)
    cnn_q_res = Quantizer.quantize_model(cnn_quant)
    check_int8_artifact(list(cnn_quant.parameters()), cnn_q_res, Quantizer, record=False)
    cnn_q_bytes = int8_storage_bytes(cnn_q_res)
    cnn_params = [prm for lyr in cnn_quant.layers for prm in lyr.parameters()]
    for idx, prm in enumerate(cnn_params):
        entry = cnn_q_res['quantized_layers'][f'param_{idx}']
        restored = Quantizer.dequantize_tensor(entry['quantized'], entry['scale'], entry['zero_point'])
        prm.data = restored.data.reshape(entry['original_shape'])

    # 50% Pruned candidate
    cnn_pruned = copy.deepcopy(cnn_base)
    Compressor.magnitude_prune(cnn_pruned, sparsity=0.5)

    cnn_base.name = "Baseline FP32"
    cnn_quant.name = "INT8 Quantized"
    cnn_pruned.name = "50% Pruned"

    # Module 19 Benchmark: Standardized latency measurement over multiple runs
    cnn_bench = Benchmark(models=[cnn_base, cnn_quant, cnn_pruned], datasets=[], warmup_runs=2, measurement_runs=10)
    cnn_lat_results = cnn_bench.run_latency_benchmark(input_shape=(1, 1, 8, 8))

    # SimpleCNN is never trained here, so there is no accuracy to report.
    # Quality = cosine agreement with the FP32 baseline model's outputs.
    cnn_test_x = Tensor(X_test.data[:100].reshape(-1, 1, 8, 8))
    y_base = cnn_base(cnn_test_x).data
    y_quant = cnn_quant(cnn_test_x).data
    y_pruned = cnn_pruned(cnn_test_x).data

    fid_base = 100.0
    fid_quant = cosine_fidelity(y_base, y_quant)
    fid_pruned = cosine_fidelity(y_base, y_pruned)

    cnn_records = [
        ('Baseline FP32', cnn_bytes, cnn_lat_results['Baseline FP32'], "reference", fid_base, 100.0 - fid_base),
        ('INT8 Quantized', cnn_q_bytes, cnn_lat_results['INT8 Quantized'], f"{fid_quant:.2f}%", fid_quant, 100.0 - fid_quant),
        ('50% Pruned', cnn_bytes, cnn_lat_results['50% Pruned'], f"{fid_pruned:.2f}%", fid_pruned, 100.0 - fid_pruned),
    ]
    cnn_pts = {r[0]: (r[1] / 1024.0, r[5]) for r in cnn_records}
    cnn_frontier = set(pareto_frontier(cnn_pts, (True, True)))

    console.print(Panel(
        "[bold magenta]📍 DIVISION 2: Spatial Vision & Compute: SimpleCNN (Spatial)[/bold magenta]\n"
        "[dim]Primary Bottleneck: 2D sliding convolution loops & spatial feature extraction\n"
        "Harness: Measured via Module 19 Benchmark on an UNTRAINED SimpleCNN\n"
        "Quality: cosine agreement with the FP32 baseline model's outputs (not accuracy)[/dim]",
        border_style="magenta",
        box=box.ROUNDED,
    ))

    t2 = Table(box=box.ROUNDED)
    t2.add_column("Candidate", style="yellow")
    t2.add_column("Memory", justify="right")
    t2.add_column("Agreement w/ FP32", justify="center")
    t2.add_column("Latency", justify="right")
    t2.add_column("Status", justify="center")
    for r in cnn_records:
        status_str = pareto_status(r[0], cnn_frontier)
        res = r[2]
        t2.add_row(r[0], f"{r[1]:,} B", r[3], f"{res.mean:.2f} ms", status_str)
    console.print(t2)


    # -------------------------------------------------------------------------
    # DIVISION 3: Generative LLM Serving: TinyGPT (Autoregressive, Prefix-Bound)
    # -------------------------------------------------------------------------
    gpt_base = GPT(vocab_size=28, embed_dim=32, num_layers=2, num_heads=2, max_seq_len=32)
    gpt_bytes = sum(p.data.nbytes for p in gpt_base.parameters())
    tokens = np.random.default_rng(7).integers(0, 28, (1, 16))

    # INT8 Quantized candidate
    gpt_quant = copy.deepcopy(gpt_base)
    q_gpt = Quantizer.quantize_model(gpt_quant)
    check_int8_artifact(list(gpt_quant.parameters()), q_gpt, Quantizer, record=False)
    gpt_q_bytes = int8_storage_bytes(q_gpt)
    for idx, prm in enumerate(gpt_quant.parameters()):
        entry = q_gpt['quantized_layers'][f'param_{idx}']
        restored = Quantizer.dequantize_tensor(entry['quantized'], entry['scale'], entry['zero_point'])
        prm.data = restored.data.reshape(entry['original_shape'])

    # Replay outputs to evaluate signal preservation
    with redirect_stdout(io.StringIO()):
        out_base = replay_prefixes(gpt_base, tokens, Tensor)
        out_quant = replay_prefixes(gpt_quant, tokens, Tensor)

        c_base = enable_kv_cache(gpt_base)
        out_cached = replay_prefixes(gpt_base, tokens, Tensor, cache=c_base)
        cache_bytes = int(round(c_base.get_memory_usage()['total_mb'] * 1024 * 1024))
        disable_kv_cache(gpt_base)

        c_quant = enable_kv_cache(gpt_quant)
        out_qc = replay_prefixes(gpt_quant, tokens, Tensor, cache=c_quant)
        disable_kv_cache(gpt_quant)

    fid_gpt_base = 100.0
    fid_gpt_quant = cosine_fidelity(out_base, out_quant)
    fid_gpt_cached = cosine_fidelity(out_base, out_cached)
    fid_gpt_qc = cosine_fidelity(out_base, out_qc)

    # Module 19 Benchmark: Measure repeated independent sequence generation runs
    def measure_generation_runs(fn, runs=10):
        fn()  # Warmup run
        latencies = []
        for _ in range(runs):
            t_start = time.perf_counter()
            fn()
            latencies.append((time.perf_counter() - t_start) * 1000)
        return latencies

    with redirect_stdout(io.StringIO()):
        r_base = BenchmarkResult("FP32_Recompute", measure_generation_runs(lambda: replay_prefixes(gpt_base, tokens, Tensor)))
        r_quant = BenchmarkResult("INT8_Recompute", measure_generation_runs(lambda: replay_prefixes(gpt_quant, tokens, Tensor)))

        c_base = enable_kv_cache(gpt_base)
        r_cached = BenchmarkResult("KV_Cached", measure_generation_runs(lambda: replay_prefixes(gpt_base, tokens, Tensor, cache=c_base)))
        disable_kv_cache(gpt_base)

        c_quant = enable_kv_cache(gpt_quant)
        r_qc = BenchmarkResult("Quant_Cached", measure_generation_runs(lambda: replay_prefixes(gpt_quant, tokens, Tensor, cache=c_quant)))
        disable_kv_cache(gpt_quant)

    gpt_records = [
        ('Baseline FP32 (Recompute)', gpt_bytes, r_base, "reference", fid_gpt_base, 100.0 - fid_gpt_base),
        ('INT8 Quantized (Recompute)', gpt_q_bytes, r_quant, f"{fid_gpt_quant:.2f}%", fid_gpt_quant, 100.0 - fid_gpt_quant),
        ('KV-Cached (Mod 18)', gpt_bytes + cache_bytes, r_cached, f"{fid_gpt_cached:.2f}%", fid_gpt_cached, 100.0 - fid_gpt_cached),
        ('Full Stack (Quant+Cache)', gpt_q_bytes + cache_bytes, r_qc, f"{fid_gpt_qc:.2f}%", fid_gpt_qc, 100.0 - fid_gpt_qc),
    ]
    gpt_pts = {r[0]: (r[2].mean, r[1] / 1024.0, r[5]) for r in gpt_records}
    gpt_frontier = set(pareto_frontier(gpt_pts, (True, True, True)))

    console.print(Panel(
        "[bold green]📍 DIVISION 3: Generative LLM Serving: TinyGPT (Autoregressive)[/bold green]\n"
        "[dim]Primary Bottleneck: O(N²) causal prefix recomputation & DRAM weight streaming\n"
        "Harness: Measured via Module 19 BenchmarkResult; sequence replay across 10 trials\n"
        "Quality: cosine agreement with the untrained FP32 GPT's logits (not accuracy)[/dim]",
        border_style="green",
        box=box.ROUNDED,
    ))

    t3 = Table(box=box.ROUNDED)
    t3.add_column("Serving Strategy", style="yellow")
    t3.add_column("Memory", justify="right")
    t3.add_column("Agreement w/ FP32", justify="center")
    t3.add_column("Latency", justify="right")
    t3.add_column("Status", justify="center")
    for r in gpt_records:
        status_str = pareto_status(r[0], gpt_frontier)
        res = r[2]
        t3.add_row(r[0], f"{r[1]:,} B", r[3], f"{res.mean:.2f} ms", status_str)
    console.print(t3)


    # -------------------------------------------------------------------------
    # CROSS-DIVISION SYSTEMS SYNTHESIS
    # -------------------------------------------------------------------------
    lines = synthesis_lines(
        mlp={'base_bytes': mlp_bytes, 'quant_bytes': mlp_q_bytes,
             'base_acc': mlp_acc, 'quant_acc': mlp_q_acc,
             'quant_on_frontier': 'INT8 Quantized' in mlp_frontier},
        cnn={'base_bytes': cnn_bytes, 'quant_bytes': cnn_q_bytes,
             'agree_quant': fid_quant, 'agree_pruned': fid_pruned,
             'frontier': [r[0] for r in cnn_records if r[0] in cnn_frontier]},
        gpt={'cache_max_abs_diff': max_abs_diff(out_base, out_cached),
             'full_stack_on_frontier': 'Full Stack (Quant+Cache)' in gpt_frontier,
             'frontier': [r[0] for r in gpt_records if r[0] in gpt_frontier],
             'n_candidates': len(gpt_records)},
    )
    console.print(Panel(
        "[bold]💡 Key Systems Takeaway (measured on this run):[/bold]\n\n" + "\n".join(lines),
        border_style="bright_blue",
        title="🔬 Cross-Division Systems Synthesis"
    ))


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
    console.print(f"INT8 artifact counted from YOUR code arrays (with scale metadata): "
                  f"{quant['quant_size']:,} bytes ({quant['measured_ratio']:.2f}× smaller).")
    console.print(f"Pruned zero fraction: {prune['sparsity_after']:.1%}.")
    console.print('Both candidate models still execute dense float32 operations. '
                  'Packed storage and sparse execution require additional implementations.')
    print_gate_scorecard()
    console.print(f'[bold green]MILESTONE 06 COMPLETE: all {len({g[0] for g in GATE_RESULTS})} pass gates '
                  f'met by YOUR modules.[/bold green]')
    return 0


def print_gate_scorecard():
    """Show every pass gate that ran, with what was measured."""
    table = Table(title="Pass gates (computed from YOUR modules)", box=box.ROUNDED)
    for heading in ['Gate', 'Measured', 'Required', '']:
        table.add_column(heading)
    # A gate recorded more than once (INT8 checks run per parameter) shows once.
    seen = {}
    for name, measured, required, passed in GATE_RESULTS:
        if name not in seen or not passed:
            seen[name] = (measured, required, passed)
    for name, (measured, required, passed) in seen.items():
        table.add_row(name, str(measured), str(required),
                      '[green]✓[/green]' if passed else '[red]✗[/red]')
    console.print(table)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run the Olympics; exit 1 with a teaching message when a pass gate fails."""
    try:
        return run_olympics()
    except MilestoneGateFailure as failure:
        print_gate_scorecard()
        console.print(Panel(
            f"[bold red]✗ Pass gate failed: {failure.gate}[/bold red]\n\n"
            f"Measured: {failure.measured}\n"
            f"Required: {failure.required}\n\n"
            f"[yellow]{failure.lesson}[/yellow]\n\n"
            "[dim]Fix the module named above, re-export it, and run this milestone again.[/dim]",
            title="MILESTONE 06 NOT PASSED",
            border_style="red",
        ))
        return 1


def run_olympics():
    """
    The Optimization Olympics pipeline.

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
        "[bold magenta]🏆 THE OPTIMIZATION OLYMPICS[/bold magenta]\n\n"
        "[yellow]MLPerf 2018: where accuracy meets efficiency[/yellow]\n\n"
        "[cyan]Using YOUR implementations from the optimization modules[/cyan]",
        title="Milestone 06: MLPerf",
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

        from tinytorch.core.spatial import Conv2d
        from tinytorch.perf.acceleration import vectorized_matmul, im2col_conv2d
        console.print("  [green]✓[/green] vectorized_matmul & im2col_conv2d (YOUR Module 17)")

        from tinytorch.perf.memoization import KVCache, enable_kv_cache, disable_kv_cache
        console.print("  [green]✓[/green] KVCache (YOUR Module 18)")

        from tinytorch.perf.benchmarking import Benchmark, BenchmarkResult, pareto_frontier
        console.print("  [green]✓[/green] Benchmark & Pareto Frontier (YOUR Module 19)")

        from tinytorch.core.transformers import GPT
        console.print("  [green]✓[/green] GPT Transformer (YOUR Module 13)")

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
    from networks import DigitMLP, SimpleCNN, MinimalTransformer

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
                if epoch == 0 and i == 0:
                    check_loss_forward(float(np.asarray(loss.data)), output.data, batch_y)

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
    step_4_kv_cache(KVCache, MinimalTransformer, GPT=GPT,
                    enable_kv_cache=enable_kv_cache, disable_kv_cache=disable_kv_cache)
    press_enter_to_continue()

    # Step 5: Acceleration
    step_5_accelerate(
        vectorized_matmul, Tensor,
        im2col_conv2d=im2col_conv2d, Conv2d=Conv2d,
        enable_kv_cache=enable_kv_cache, disable_kv_cache=disable_kv_cache, GPT=GPT
    )
    press_enter_to_continue()

    # Step 6: Benchmark (Module 19 is checked on fixtures before it measures)
    check_benchmark_module(BenchmarkResult, pareto_frontier)
    candidates = {'Baseline': model, 'Rounded weights': quant['model'],
                  'Pruned weights': prune['model']}
    measurements = {name: step_6_benchmark(candidate, X_test, y_test,
                    baseline['baseline_acc'], Benchmark, name)
                    for name, candidate in candidates.items()}
    # Step 7: Architectural Triad & Pareto Frontier
    step_7_triad_and_pareto(
        baseline, quant, prune, measurements,
        Profiler, Quantizer, Compressor,
        SimpleCNN, GPT, Benchmark, BenchmarkResult, pareto_frontier,
        enable_kv_cache, disable_kv_cache, Tensor, X_test
    )
    press_enter_to_continue()

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    return print_final_results(baseline, quant, prune, measurements)


if __name__ == "__main__":
    sys.exit(main())
