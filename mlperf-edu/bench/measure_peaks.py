#!/usr/bin/env python3
"""
MLPerf EDU: Measure peak FLOPS and peak memory bandwidth for the host.

Replaces the hardcoded M1 defaults in src/mlperf/roofline.py. Caches
the result keyed by the hardware fingerprint hash so the measurement
runs at most once per machine. Per Dean's iter-5 sign-off: needed
before iter-6 measures Llama-1B against the ridge.

Methodology:
  - Peak FLOPS: best of 5 large square fp32 matmuls (best of 5 amortizes
    one-time PyTorch dispatch + MPS init).
  - Peak BW: streaming clone of a tensor much larger than LLC; bytes
    moved / time. Uses a 256 MB tensor by default (>> any reasonable LLC).

Run once: python3 bench/measure_peaks.py
"""
import json
import os
import sys
import time
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

CACHE_DIR = Path.home() / ".mlperf-edu"
GEMM_DIM = 2048
BW_BYTES = 256 * 1024 * 1024  # 256 MB; >> any consumer LLC


def _hwfp_short() -> str:
    try:
        from mlperf.hardware import profile_hardware
        fp = profile_hardware()
    except Exception:
        fp = {"system": "unknown"}
    return hashlib.sha256(
        json.dumps(fp, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_peak_flops(device: str, n_warmup: int = 3, n_runs: int = 5) -> float:
    """Best-of-N square fp32 matmul throughput."""
    a = torch.randn(GEMM_DIM, GEMM_DIM, device=device, dtype=torch.float32)
    b = torch.randn(GEMM_DIM, GEMM_DIM, device=device, dtype=torch.float32)
    flops_per_iter = 2 * GEMM_DIM ** 3
    for _ in range(n_warmup):
        torch.matmul(a, b)
    _sync()
    best = 0.0
    for _ in range(n_runs):
        _sync()
        t0 = time.perf_counter()
        for _ in range(10):
            c = torch.matmul(a, b)
        _sync()
        dt = time.perf_counter() - t0
        flops_per_sec = (10 * flops_per_iter) / dt
        best = max(best, flops_per_sec)
    return best


def measure_peak_bw(device: str, n_warmup: int = 2, n_runs: int = 5) -> float:
    """Best-of-N streaming bandwidth (bytes moved / time, in GB/s)."""
    n_floats = BW_BYTES // 4
    x = torch.randn(n_floats, device=device, dtype=torch.float32)
    for _ in range(n_warmup):
        x.clone()
    _sync()
    best = 0.0
    # Read + write: clone reads x and writes y; effective bytes moved = 2 * BW_BYTES.
    bytes_per_iter = 2 * BW_BYTES
    for _ in range(n_runs):
        _sync()
        t0 = time.perf_counter()
        for _ in range(5):
            y = x.clone()
        _sync()
        dt = time.perf_counter() - t0
        bw = (5 * bytes_per_iter) / dt / 1e9
        best = max(best, bw)
    return best


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hwfp = _hwfp_short()
    device = _device()
    cache_path = CACHE_DIR / f"machine_caps_{hwfp}.json"

    print(f"Hardware fingerprint: {hwfp}")
    print(f"Device:               {device}")
    print(f"Cache file:           {cache_path}")
    print()

    print(f"Measuring peak FLOPS ({GEMM_DIM}x{GEMM_DIM} fp32 matmul, best of 5)...")
    peak_flops = measure_peak_flops(device)
    print(f"  peak_FLOPS = {peak_flops/1e12:.2f} TFLOPS")

    print(f"Measuring peak BW (256 MB streaming clone, best of 5)...")
    peak_bw = measure_peak_bw(device)
    print(f"  peak_BW    = {peak_bw:.2f} GB/s")

    ridge = peak_flops / (peak_bw * 1e9)
    print(f"  ridge      = {ridge:.2f} FLOPs/byte")

    payload = {
        "hardware_fingerprint_short": hwfp,
        "device": device,
        "peak_FLOPS": peak_flops,
        "peak_BW_GBps": peak_bw,
        "ridge_FLOPS_per_byte": ridge,
        "method": {
            "flops": f"best of 5 x 10-iter {GEMM_DIM}x{GEMM_DIM} fp32 matmul",
            "bw": f"best of 5 x 5-iter clone of {BW_BYTES//1024//1024} MB tensor",
        },
    }
    cache_path.write_text(json.dumps(payload, indent=2))
    print(f"\nCached to {cache_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
