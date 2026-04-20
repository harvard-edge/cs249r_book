# MLPerf EDU: Practitioner Workflow Examples
# ==========================================
# These examples mirror how MLPerf is actually used in industry.
# Each follows: Problem → Initial Run → Modification → New Run → Report → Insight
#
# Usage: Run these scripts end-to-end to produce comparison reports.

"""
Example 1: Training Optimization Workflow
=========================================
Problem: "Reduce NanoGPT training time by 20% without sacrificing quality."

This mirrors what ML engineers do at NVIDIA/Google when optimizing
training throughput for a new hardware platform.
"""

import json
import os
import time
import hashlib
import datetime


def example_training_optimization():
    """Workflow 1: Optimize training throughput.
    
    A practitioner profiles NanoGPT training and applies three optimizations:
      1. Increase batch size (8 → 64)
      2. Enable mixed precision (FP32 → FP16)
      3. Optimize data loading (num_workers)
    
    The report comparison shows the latency-accuracy tradeoff.
    """
    print("=" * 60)
    print("Example 1: Training Optimization Workflow")
    print("=" * 60)
    
    # --- Baseline Run ---
    baseline = {
        "workload": "nanogpt-12m",
        "division": "cloud",
        "scenario": "SingleStream",
        "timestamp": datetime.datetime.now().isoformat(),
        "seed": 42,
        "hardware_fingerprint": {
            "cpu": "Apple M1",
            "gpu": "Apple M1 GPU (MPS)",
            "memory_gb": 8,
            "os": "macOS 15.4.1"
        },
        "config": {
            "batch_size": 8,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "precision": "fp32",
            "num_workers": 0
        },
        "metrics": {
            "loss": 2.28,
            "latency_p50_ms": 12.4,
            "latency_p90_ms": 15.1,
            "latency_p99_ms": 18.3,
            "throughput_qps": 80.6,
            "power_avg_watts": 11.2,
            "energy_joules": 1780.0
        },
        "training": {
            "epochs": 25,
            "final_train_loss": 2.28,
            "final_val_loss": 2.35,
            "total_time_s": 178.0,
            "curve": [
                {"epoch": 1, "train": 4.28, "val": 4.31},
                {"epoch": 5, "train": 3.55, "val": 3.61},
                {"epoch": 10, "train": 2.92, "val": 3.01},
                {"epoch": 15, "train": 2.60, "val": 2.71},
                {"epoch": 20, "train": 2.40, "val": 2.49},
                {"epoch": 25, "train": 2.28, "val": 2.35}
            ]
        },
        "compliance": {"target_met": True, "target": "loss < 2.3", "run_count": 3},
        "integrity": {
            "dataset_hash": "a3f2b8c9d4e5f6a7",
            "checkpoint_hash": "1b2c3d4e5f6a7b8c",
            "log_hash": "f0e1d2c3b4a59687"
        }
    }
    
    # --- Optimized Run (batch_size=64, fp16, num_workers=4) ---
    optimized = {
        "workload": "nanogpt-12m",
        "division": "cloud",
        "scenario": "SingleStream",
        "timestamp": datetime.datetime.now().isoformat(),
        "seed": 42,
        "hardware_fingerprint": baseline["hardware_fingerprint"],
        "config": {
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "precision": "fp16",
            "num_workers": 4
        },
        "metrics": {
            "loss": 2.25,
            "latency_p50_ms": 5.90,
            "latency_p90_ms": 7.20,
            "latency_p99_ms": 9.10,
            "throughput_qps": 169.5,
            "power_avg_watts": 12.3,
            "energy_joules": 1094.7
        },
        "training": {
            "epochs": 25,
            "final_train_loss": 2.25,
            "final_val_loss": 2.31,
            "total_time_s": 89.0,
            "curve": [
                {"epoch": 1, "train": 4.25, "val": 4.28},
                {"epoch": 5, "train": 3.45, "val": 3.50},
                {"epoch": 10, "train": 2.82, "val": 2.88},
                {"epoch": 15, "train": 2.50, "val": 2.57},
                {"epoch": 20, "train": 2.32, "val": 2.39},
                {"epoch": 25, "train": 2.25, "val": 2.31}
            ]
        },
        "compliance": {"target_met": True, "target": "loss < 2.3", "run_count": 3},
        "integrity": {
            "dataset_hash": "a3f2b8c9d4e5f6a7",
            "checkpoint_hash": "9e8d7c6b5a4f3e2d",
            "log_hash": "a1b2c3d4e5f6a7b8"
        }
    }
    
    # Save submissions
    os.makedirs("submissions/examples", exist_ok=True)
    
    baseline_path = "submissions/examples/nanogpt_baseline.json"
    optimized_path = "submissions/examples/nanogpt_optimized.json"
    
    with open(baseline_path, 'w') as f:
        json.dump(baseline, f, indent=2)
    with open(optimized_path, 'w') as f:
        json.dump(optimized, f, indent=2)
    
    # Generate comparative report
    from src.mlperf.report import generate_report
    
    report_path = generate_report(
        optimized_path,
        output_path="submissions/examples/nanogpt_optimization_report.html",
        baseline_path=baseline_path
    )
    
    # Print analysis
    speedup = baseline["training"]["total_time_s"] / optimized["training"]["total_time_s"]
    energy_savings = 1.0 - (optimized["metrics"]["energy_joules"] / baseline["metrics"]["energy_joules"])
    throughput_gain = optimized["metrics"]["throughput_qps"] / baseline["metrics"]["throughput_qps"]
    
    print(f"\n--- Analysis ---")
    print(f"Training speedup:  {speedup:.1f}x ({baseline['training']['total_time_s']}s → {optimized['training']['total_time_s']}s)")
    print(f"Throughput gain:   {throughput_gain:.1f}x ({baseline['metrics']['throughput_qps']:.0f} → {optimized['metrics']['throughput_qps']:.0f} QPS)")
    print(f"Energy savings:    {energy_savings*100:.0f}% ({baseline['metrics']['energy_joules']:.0f}J → {optimized['metrics']['energy_joules']:.0f}J)")
    print(f"Quality preserved: loss {baseline['metrics']['loss']:.2f} → {optimized['metrics']['loss']:.2f} (target: <2.3)")
    print(f"\nReport:  {report_path}")
    
    return report_path


def example_bottleneck_analysis():
    """Workflow 4: System Bottleneck Identification.
    
    A system designer investigates why DLRM training is unexpectedly fast
    despite having embedding tables. Discovers the micro-scale table fits
    in cache, illustrating how production bottlenecks differ at scale.
    """
    print("\n" + "=" * 60)
    print("Example 4: System Bottleneck Analysis")
    print("=" * 60)
    
    dlrm_run = {
        "workload": "micro-dlrm-1m",
        "division": "cloud",
        "scenario": "Offline",
        "timestamp": datetime.datetime.now().isoformat(),
        "seed": 42,
        "hardware_fingerprint": {
            "cpu": "Apple M1",
            "gpu": "Apple M1 GPU (MPS)",
            "memory_gb": 8,
            "os": "macOS 15.4.1"
        },
        "config": {
            "batch_size": 256,
            "learning_rate": 0.01,
            "optimizer": "AdamW"
        },
        "metrics": {
            "accuracy": 0.72,
            "latency_p50_ms": 0.03,
            "latency_p90_ms": 0.04,
            "latency_p99_ms": 0.06,
            "throughput_qps": 33333.0
        },
        "training": {
            "epochs": 25,
            "final_train_loss": 0.58,
            "final_val_loss": 0.61,
            "total_time_s": 5.0,
            "curve": [
                {"epoch": 1, "train": 0.69, "val": 0.69},
                {"epoch": 5, "train": 0.65, "val": 0.66},
                {"epoch": 10, "train": 0.62, "val": 0.63},
                {"epoch": 15, "train": 0.60, "val": 0.62},
                {"epoch": 20, "train": 0.59, "val": 0.61},
                {"epoch": 25, "train": 0.58, "val": 0.61}
            ]
        },
        "compliance": {"target_met": True, "target": "accuracy > 0.70", "run_count": 5},
        "integrity": {
            "dataset_hash": "c4d5e6f7a8b9c0d1",
            "checkpoint_hash": "2e3f4a5b6c7d8e9f",
            "log_hash": "b0c1d2e3f4a5b6c7"
        }
    }
    
    os.makedirs("submissions/examples", exist_ok=True)
    dlrm_path = "submissions/examples/dlrm_bottleneck.json"
    with open(dlrm_path, 'w') as f:
        json.dump(dlrm_run, f, indent=2)
    
    from src.mlperf.report import generate_report
    report_path = generate_report(
        dlrm_path,
        output_path="submissions/examples/dlrm_bottleneck_report.html"
    )
    
    print(f"\n--- Bottleneck Analysis ---")
    print(f"DLRM latency: {dlrm_run['metrics']['latency_p50_ms']:.2f}ms (vs NanoGPT ~5.9ms)")
    print(f"Throughput:   {dlrm_run['metrics']['throughput_qps']:.0f} QPS")
    print(f"Embedding size: 943×32 + 1682×32 = 83,200 floats = 336KB")
    print(f"L2 cache: ~4MB (Apple M1) → tables fit entirely in cache")
    print(f"Production DLRM: terabyte-scale embeddings → memory-bandwidth-bound")
    print(f"\nInsight: The architectural bottleneck (sparse vs. dense) is preserved,")
    print(f"but the scale-dependent bottleneck (memory bandwidth) is absent.")
    print(f"\nReport: {report_path}")
    
    return report_path


def example_architecture_comparison():
    """Workflow 2: Dense vs. Sparse Architecture Comparison.
    
    A practitioner evaluates NanoGPT (dense) vs Nano-MoE (sparse) on the
    same dataset to understand the compute/quality tradeoff of expert routing.
    """
    print("\n" + "=" * 60)
    print("Example 2: Architecture Comparison (Dense vs. Sparse)")
    print("=" * 60)
    
    dense_run = {
        "workload": "nanogpt-12m",
        "division": "cloud", "scenario": "SingleStream",
        "timestamp": datetime.datetime.now().isoformat(), "seed": 42,
        "hardware_fingerprint": {"cpu": "Apple M1", "gpu": "MPS", "memory_gb": 8, "os": "macOS"},
        "config": {"batch_size": 16, "learning_rate": 0.001, "optimizer": "AdamW"},
        "metrics": {"loss": 2.25, "latency_p50_ms": 5.90, "throughput_qps": 169.5},
        "training": {"epochs": 25, "final_train_loss": 2.25, "final_val_loss": 2.31, "total_time_s": 89.0},
        "compliance": {"target_met": True, "target": "loss < 2.3", "run_count": 3},
        "integrity": {"dataset_hash": "abc123", "checkpoint_hash": "def456", "log_hash": "ghi789"}
    }
    
    sparse_run = {
        "workload": "nano-moe-12m",
        "division": "cloud", "scenario": "SingleStream",
        "timestamp": datetime.datetime.now().isoformat(), "seed": 42,
        "hardware_fingerprint": {"cpu": "Apple M1", "gpu": "MPS", "memory_gb": 8, "os": "macOS"},
        "config": {"batch_size": 16, "learning_rate": 0.001, "optimizer": "AdamW"},
        "metrics": {"loss": 0.042, "latency_p50_ms": 8.20, "throughput_qps": 122.0},
        "training": {"epochs": 25, "final_train_loss": 0.042, "final_val_loss": 0.048, "total_time_s": 158.0},
        "compliance": {"target_met": True, "target": "loss < 0.05", "run_count": 3},
        "integrity": {"dataset_hash": "abc123", "checkpoint_hash": "jkl012", "log_hash": "mno345"}
    }
    
    os.makedirs("submissions/examples", exist_ok=True)
    with open("submissions/examples/nanogpt_dense.json", 'w') as f:
        json.dump(dense_run, f, indent=2)
    with open("submissions/examples/nanomoe_sparse.json", 'w') as f:
        json.dump(sparse_run, f, indent=2)
    
    from src.mlperf.report import generate_report
    generate_report("submissions/examples/nanogpt_dense.json",
                    output_path="submissions/examples/dense_report.html")
    report = generate_report("submissions/examples/nanomoe_sparse.json",
                    output_path="submissions/examples/sparse_report.html",
                    baseline_path="submissions/examples/nanogpt_dense.json")
    
    print(f"\n--- Architecture Comparison ---")
    print(f"NanoGPT (dense):  85.9M params, loss=2.25, 89s training, 5.9ms inference")
    print(f"Nano-MoE (sparse): 17.4M params, loss=0.042, 158s training, 8.2ms inference")
    print(f"\nInsight: MoE achieves 54x lower loss with 5x fewer parameters,")
    print(f"but at 1.4x inference latency due to routing overhead.")
    print(f"This is the fundamental dense-vs-sparse tradeoff in ML systems.")
    print(f"\nComparison report: {report}")


if __name__ == '__main__':
    print("MLPerf EDU: Industry-Style Workflow Examples")
    print("=" * 60)
    print()
    
    example_training_optimization()
    example_architecture_comparison()
    example_bottleneck_analysis()
    
    print("\n" + "=" * 60)
    print("All example reports generated in submissions/examples/")
    print("Open any _report.html file in a browser to view.")
