#!/usr/bin/env python3
"""
Custom Kernels (2024-Present): From Loop Order to Native Silicon
=============================================================================

📚 HISTORICAL CONTEXT:
Between 2021 and 2024, modern machine learning systems crossed the Foreign Function
Interface (FFI) boundary. Eager Python execution was eclipsed by hardware-specialized
kernels and compilers:
1. OpenAI Triton (2021-2024): block-SPMD GPU kernels written in Python syntax that
   compile to PTX and run close to hand-tuned CUDA.
2. Apple Metal Performance Shaders (2020-2024): GPU kernels over unified memory that
   avoid host-to-device copies.
3. C++ SIMD Vectorization: loops compiled to AVX2/AVX-512 or ARM NEON registers that
   execute 8 to 16 floating-point operations per instruction.

Every one of those kernels is built from the same ideas YOU implemented in Module 17:
blocking a matrix multiply into cache-sized tiles, lowering convolution to one GEMM,
and fusing an elementwise chain into a single pass.

🎯 MILESTONE 07: VERIFY YOUR ACCELERATION KERNELS, THEN MEET NATIVE ONES
Part A (the pass gate) runs YOUR Module 17 functions on many shapes, including
sizes that are not a multiple of the tile, and requires them to match NumPy within
float32 tolerance. Part B then runs the native kernels bundled with TinyTorch
(C++ SIMD, and Apple Metal or Triton when the hardware is present) on the same
problems as reference points and reports the measured timings.

The native kernels ship with TinyTorch; they are context, not your work. The
milestone passes when YOUR code is correct, whether or not a compiler or GPU is
available on this machine.

✅ REQUIRED MODULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR strided tensor data structure
  Module 17 (Acceleration)  : YOUR tiled_matmul, fused_gelu, im2col, im2col_conv2d, col2im
  Optional                  : a C++ compiler (bundled SIMD kernel), Apple GPU or CUDA + Triton
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE:
    ┌──────────────────────────────┐
    │ YOUR Module 17 (Python/NumPy)│  Part A: must match NumPy  ← pass gate
    │ tiling · im2col · fusion     │
    └──────────────┬───────────────┘
                   │ same problems, same shapes
                   ▼
    ┌────────────────────────────────────────────────────────┐
    │ Bundled native kernels (reference points, Part B)      │
    ├──────────────────┬──────────────────┬──────────────────┤
    │ C++ SIMD         │ Apple Metal MPS  │ OpenAI Triton    │
    │ AVX2 / ARM NEON  │ Unified Memory   │ GPU PTX          │
    └──────────────────┴──────────────────┴──────────────────┘
"""

from pathlib import Path
import platform
import sys
import time

import numpy as np

# Ensure repository root is on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# float32 tolerance: accumulation order differs from np.matmul, so compare
# relative to the output scale instead of demanding bit equality.
GEMM_RTOL, GEMM_ATOL = 1e-4, 1e-3
ELEMENTWISE_RTOL, ELEMENTWISE_ATOL = 1e-5, 1e-5

# (M, K, N): tile multiples, ragged edges on every axis, degenerate 1-wide
# matrices, and a size smaller than one tile.
MATMUL_SHAPES = [(64, 64, 64), (65, 33, 97), (100, 1, 3), (1, 70, 1), (7, 13, 5), (130, 127, 129)]
TILE_SIZES = [64, 16, 7]

# (N, C, H, W, out_ch, k, stride, padding): odd spatial sizes, stride 2, no padding.
CONV_CASES = [
    (2, 3, 8, 8, 4, 3, 1, 1),
    (1, 2, 9, 7, 5, 3, 2, 0),
    (3, 1, 6, 11, 2, 2, 1, 0),
    (2, 4, 5, 5, 3, 5, 1, 2),
]


def print_banner():
    print("=" * 76)
    print("  TINYTORCH MILESTONE 07: CUSTOM KERNELS (2024)")
    print("  Part A: YOUR Module 17 kernels (pass gate) · Part B: bundled native kernels")
    print("=" * 76)
    print(f"  Platform: {platform.system()} ({platform.machine()}) | Python {sys.version.split()[0]}")


def ratio_text(baseline_ms, candidate_ms):
    """'N× faster' / 'N× slower' from two timings, never a speedup below 1."""
    if candidate_ms <= 0:
        return "n/a"
    r = baseline_ms / candidate_ms
    if r >= 1.05:
        return f"{r:.1f}× faster"
    if r > 1 / 1.05:
        return "about the same"
    return f"{1 / r:.1f}× slower"


def time_ms(fn, repeats=10):
    """Median wall-clock milliseconds of fn() after one warmup call."""
    fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return float(np.median(samples))


def gelu_reference(x):
    """tanh-approximation GELU in float64."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def conv_reference(x, w, b, stride, padding):
    """Direct convolution in float64 with NumPy only (no TinyTorch code)."""
    x = np.pad(np.asarray(x, np.float64), ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out_ch, _, k, _ = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride]          # (N, C, oh, ow, k, k)
    out = np.einsum('nchwij,ocij->nohw', windows, np.asarray(w, np.float64))
    return out + np.asarray(b, np.float64)[None, :, None, None]


# =============================================================================
# 🎓 PART A: YOUR MODULE 17 KERNELS (the pass gate)
# =============================================================================

class Check:
    """Collect pass/fail results for one student function."""

    def __init__(self, name):
        self.name, self.cases, self.failures = name, 0, []

    def expect(self, label, actual, expected, rtol, atol):
        self.cases += 1
        actual = np.asarray(actual)
        if actual.shape != np.shape(expected):
            self.failures.append(f"{label}: shape {actual.shape}, expected {np.shape(expected)}")
            return
        if not np.all(np.isfinite(actual)) or not np.allclose(actual, expected, rtol=rtol, atol=atol):
            err = float(np.nanmax(np.abs(actual - expected))) if actual.size else 0.0
            self.failures.append(f"{label}: max abs error {err:.2e}")

    def crashed(self, label, error):
        self.cases += 1
        self.failures.append(f"{label}: raised {type(error).__name__}: {str(error).splitlines()[0][:80]}")


def check_tiled_matmul(acc, Tensor, rng):
    check = Check("tiled_matmul")
    for (M, K, N) in MATMUL_SHAPES:
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        expected = A.astype(np.float64) @ B.astype(np.float64)
        for tile in TILE_SIZES:
            label = f"({M}x{K}) @ ({K}x{N}), tile_size={tile}"
            try:
                out = acc.tiled_matmul(Tensor(A), Tensor(B), tile_size=tile).data
            except Exception as error:  # a crash on a ragged shape is a failure, not a traceback
                check.crashed(label, error)
                continue
            check.expect(label, out, expected, GEMM_RTOL, GEMM_ATOL)
    return check, ("Blocked matmul must cover the ragged last tile on every axis: use "
                   "min(start + tile_size, limit) for each tile's end, and ACCUMULATE "
                   "(+=) the k-tile products into each output tile.")


def check_fused_gelu(acc, Tensor, rng):
    check = Check("fused_gelu")
    cases = {
        "shape (1000,) ~ N(0, 1)": rng.standard_normal(1000),
        "shape (17, 33) ~ N(0, 3)": 3 * rng.standard_normal((17, 33)),
        "large |x| up to 20": np.linspace(-20, 20, 101),
        "exact zeros": np.zeros((4, 5)),
    }
    for label, x in cases.items():
        x = x.astype(np.float32)
        try:
            out = acc.fused_gelu(Tensor(x)).data
        except Exception as error:
            check.crashed(label, error)
            continue
        check.expect(label, out, gelu_reference(x), ELEMENTWISE_RTOL, ELEMENTWISE_ATOL)
    return check, ("fused_gelu must compute 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))) "
                   "in one expression over the whole array.")


def check_im2col_conv2d(acc, Tensor, rng):
    check = Check("im2col_conv2d")
    for (N, C, H, W, O, k, stride, padding) in CONV_CASES:
        x = rng.standard_normal((N, C, H, W)).astype(np.float32)
        w = rng.standard_normal((O, C, k, k)).astype(np.float32)
        b = rng.standard_normal((O,)).astype(np.float32)
        label = f"x{(N, C, H, W)}, {O} filters {k}x{k}, stride={stride}, padding={padding}"
        try:
            out = acc.im2col_conv2d(Tensor(x), Tensor(w), Tensor(b), stride=stride, padding=padding).data
        except Exception as error:
            check.crashed(label, error)
            continue
        check.expect(label, out, conv_reference(x, w, b, stride, padding), GEMM_RTOL, GEMM_ATOL)
    return check, ("im2col must order each patch row as (channel, kernel row, kernel column) to "
                   "match weight.reshape(out_ch, -1), and positions as (image, out row, out col).")


def check_col2im(acc, Tensor, rng):
    """col2im is the adjoint of im2col: <im2col(x), g> must equal <x, col2im(g)>."""
    check = Check("col2im")
    for (N, C, H, W, _, k, stride, padding) in CONV_CASES:
        x = rng.standard_normal((N, C, H, W))
        label = f"x{(N, C, H, W)}, k={k}, stride={stride}, padding={padding}"
        try:
            cols = np.asarray(acc.im2col(Tensor(x.astype(np.float32)), kernel_size=k,
                                         stride=stride, padding=padding).data, np.float64)
            g = rng.standard_normal(cols.shape)
            back = np.asarray(acc.col2im(Tensor(g.astype(np.float32)), (N, C, H, W), k,
                                         stride=stride, padding=padding).data, np.float64)
        except Exception as error:
            check.crashed(label, error)
            continue
        if back.shape != x.shape:
            check.cases += 1
            check.failures.append(f"{label}: col2im returned shape {back.shape}, expected {x.shape}")
            continue
        check.expect(label, np.sum(back * x), np.sum(cols * g), 1e-4, 1e-3)
    return check, ("col2im must scatter-ADD every patch value back to the pixel it came from, "
                   "so overlapping patches accumulate, then crop the padding.")


def run_part_a(acc, Tensor):
    print("\n" + "-" * 76)
    print("  PART A: YOUR Module 17 kernels vs NumPy (float32 tolerance)")
    print("-" * 76)
    rng = np.random.default_rng(17)
    results = []
    for runner in (check_tiled_matmul, check_fused_gelu, check_im2col_conv2d, check_col2im):
        check, lesson = runner(acc, Tensor, rng)
        status = "PASS" if not check.failures else "FAIL"
        print(f"  [{status}] {check.name:<14} {check.cases - len(check.failures)}/{check.cases} cases match")
        for failure in check.failures[:4]:
            print(f"         ✗ {failure}")
        if check.failures:
            print(f"         → {lesson}")
        results.append(check)
    return results


# =============================================================================
# 📊 PART B: BUNDLED NATIVE KERNELS (reference points, not the pass gate)
# =============================================================================
#
# These kernels ship with TinyTorch in tinytorch/extensions/. They are not
# student code. Each one falls back to NumPy when its native path is missing,
# so a path is reported only when it actually executed natively.

def native_simd(acc, Tensor, A, B, x, bias):
    try:
        from tinytorch.extensions.simd_ops import simd_matmul, simd_fused_bias_gelu, simd_build_info
        info = simd_build_info()
    except Exception as error:
        return {"status": f"not available (import failed: {error})"}
    if not info.get("built"):
        return {"status": f"not built ({info.get('error') or 'no C++ compiler found'})"}
    notes = []
    gemm_ok = np.allclose(simd_matmul(A, B), A @ B, rtol=GEMM_RTOL, atol=GEMM_ATOL)
    gelu_ok = np.allclose(simd_fused_bias_gelu(x, bias), gelu_reference(x + bias),
                          rtol=ELEMENTWISE_RTOL, atol=1e-4)
    if not gemm_ok:
        notes.append("bundled simd_matmul disagrees with NumPy")
    if not gelu_ok:
        notes.append("bundled simd_fused_bias_gelu disagrees with NumPy")
    return {
        "status": "built" + (" with OpenMP" if info.get("openmp") else ", single-threaded"),
        "gemm_ms": time_ms(lambda: simd_matmul(A, B)),
        "gelu_ms": time_ms(lambda: simd_fused_bias_gelu(x, bias)),
        "warnings": notes,
    }


def native_mps(A, B):
    try:
        from tinytorch.extensions.mps_ops import has_mps_support, mps_matmul
        if not has_mps_support():
            return {"status": "not available (no Apple GPU / PyTorch MPS backend)"}
    except Exception as error:
        return {"status": f"not available (import failed: {error})"}
    ok = np.allclose(mps_matmul(A, B), A @ B, rtol=GEMM_RTOL, atol=GEMM_ATOL)
    return {"status": "Apple GPU via PyTorch MPS (torch.matmul, a library call)",
            "gemm_ms": time_ms(lambda: mps_matmul(A, B)),
            "warnings": [] if ok else ["bundled mps_matmul disagrees with NumPy"]}


def native_triton(x, bias):
    try:
        from tinytorch.extensions.triton_gelu import triton_fused_gelu, _HAS_TRITON
        if not _HAS_TRITON:
            return {"status": "not available (no CUDA GPU with Triton)"}
    except Exception as error:
        return {"status": f"not available (import failed: {error})"}
    ok = np.allclose(np.asarray(triton_fused_gelu(x, bias)), gelu_reference(x + bias),
                     rtol=ELEMENTWISE_RTOL, atol=1e-4)
    return {"status": "CUDA GPU via Triton",
            "gelu_ms": time_ms(lambda: triton_fused_gelu(x, bias)),
            "warnings": [] if ok else ["bundled triton_fused_gelu disagrees with NumPy"]}


def run_part_b(acc, Tensor):
    print("\n" + "-" * 76)
    print("  PART B: the same work on bundled native kernels (context, not graded)")
    print("-" * 76)
    rng = np.random.default_rng(21)
    M = K = N = 256
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    x = rng.standard_normal((512, 1024)).astype(np.float32)
    bias = rng.standard_normal((1024,)).astype(np.float32)

    yours_gemm = time_ms(lambda: acc.tiled_matmul(Tensor(A), Tensor(B)), repeats=5)
    numpy_gemm = time_ms(lambda: A @ B)
    yours_gelu = time_ms(lambda: acc.fused_gelu(Tensor(x + bias)), repeats=5)
    numpy_gelu = time_ms(lambda: gelu_reference(x + bias))  # float64 NumPy, for scale

    paths = {"C++ SIMD (bundled)": native_simd(acc, Tensor, A, B, x, bias),
             "Apple Metal MPS (bundled)": native_mps(A, B),
             "OpenAI Triton (bundled)": native_triton(x, bias)}

    print(f"\n  GEMM {M}x{K}x{N} (median ms; ratio is relative to YOUR tiled_matmul)")
    print(f"    {'YOUR tiled_matmul (tile 64)':<32} {yours_gemm:9.3f} ms   baseline")
    print(f"    {'NumPy np.matmul (BLAS)':<32} {numpy_gemm:9.3f} ms   {ratio_text(yours_gemm, numpy_gemm)}")
    for name, key in (("C++ SIMD (bundled)", "gemm_ms"), ("Apple Metal MPS (bundled)", "gemm_ms")):
        if key in paths[name]:
            print(f"    {name:<32} {paths[name][key]:9.3f} ms   {ratio_text(yours_gemm, paths[name][key])}")
    print("    (Each block product in YOUR tiled_matmul is a NumPy BLAS call, so it can beat a")
    print("     simple compiled kernel; what YOU control is the loop order around the blocks.)")
    print(f"\n  Bias + GELU over 512x1024 (median ms; relative to YOUR fused_gelu)")
    print(f"    {'YOUR fused_gelu':<32} {yours_gelu:9.3f} ms   baseline")
    print(f"    {'NumPy float64 reference':<32} {numpy_gelu:9.3f} ms   {ratio_text(yours_gelu, numpy_gelu)}")
    for name in ("C++ SIMD (bundled)", "OpenAI Triton (bundled)"):
        if "gelu_ms" in paths[name]:
            print(f"    {name:<32} {paths[name]['gelu_ms']:9.3f} ms   {ratio_text(yours_gelu, paths[name]['gelu_ms'])}")
    print("    (YOUR fused_gelu timing includes the bias add, as the native kernels do.)")

    print("\n  Native paths on this machine:")
    warnings = []
    for name, result in paths.items():
        print(f"    • {name:<26}: {result['status']}")
        warnings += [f"{name}: {w}" for w in result.get("warnings", [])]
    ran = [name for name, result in paths.items() if "gemm_ms" in result or "gelu_ms" in result]
    return ran, warnings


def main():
    print_banner()
    try:
        from tinytorch.core.tensor import Tensor
        import tinytorch.perf.acceleration as acc
        for name in ("tiled_matmul", "fused_gelu", "im2col", "im2col_conv2d", "col2im"):
            getattr(acc, name)
    except (ImportError, AttributeError) as error:
        print(f"\n  [INCOMPLETE] Module 17 is not exported: {error}")
        print("               Complete and export Module 17 (Acceleration), then re-run.")
        return 1

    checks = run_part_a(acc, Tensor)
    failed = [c.name for c in checks if c.failures]
    if failed:
        print("\n" + "=" * 76)
        print(f"  [NOT PASSED] YOUR {', '.join(failed)} did not match NumPy.")
        print("  A fast kernel that computes the wrong answer is not an optimization.")
        print("  Fix the function(s) above in Module 17, re-export, and run this again.")
        print("=" * 76 + "\n")
        return 1

    ran, warnings = run_part_b(acc, Tensor)

    print("\n" + "=" * 76)
    print("  MILESTONE 07 SUMMARY")
    print("=" * 76)
    total = sum(c.cases for c in checks)
    print(f"  YOUR Module 17 kernels: {total}/{total} cases match NumPy across "
          f"{len(checks)} functions (ragged tiles, strides, padding).")
    if ran:
        print(f"  Bundled native kernels measured for comparison: {', '.join(ran)}.")
    else:
        print("  No bundled native kernel ran here (no C++ compiler or GPU). That is fine:")
        print("  the milestone grades YOUR Python kernels. Install a C++ compiler to see")
        print("  how the same tiling and fusion ideas perform in compiled SIMD code.")
    for warning in warnings:
        print(f"  ⚠ {warning} (a TinyTorch bug, not yours; please report it).")
    print("  [SUCCESS] Milestone 07 complete: YOUR acceleration kernels are correct.")
    print("=" * 76 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
